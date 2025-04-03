import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging
import os

logger = logging.getLogger(__name__)

class TensorBoardProfilerCallback(TrainerCallback):
    def __init__(self, profile_steps=1, profile_warmup=0):
        """
        Callback to profile model execution during evaluation using available hooks.

        Args:
            profile_steps (int): Number of evaluation steps (batches) to profile after warmup.
            profile_warmup (int): Number of evaluation steps (batches) to skip before starting profiling.
        """
        if profile_steps < 1:
            logger.warning(f"profile_steps must be >= 1, setting to 1. Got: {profile_steps}")
            profile_steps = 1

        self.profile_steps = profile_steps
        self.profile_warmup = profile_warmup
        self.profiler = None  # Инициализируется во время первого шага evaluation
        self.log_dir = "logs"
        self.is_profiling_active_in_eval = False # Флаг, что профилирование запущено в текущем evaluate

        logger.info(
            f"TensorBoardProfilerCallback initialized (warmup={self.profile_warmup}, "
            f"active={self.profile_steps}). Will activate during evaluation."
        )

    

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called during each step of evaluation/prediction.
        Handles profiler initialization, start (__enter__), and step.
        """
        # Логируем состояние перед проверкой
        if state.is_world_process_zero: # Логируем только на главном процессе
            is_training = kwargs.get('model', None).training if 'model' in kwargs else 'N/A'
            logger.info(
                f">>> ProfilerCallback: on_prediction_step ENTERED "
                f"(is_world_process_zero={state.is_world_process_zero}, "
                f"has_log_dir={bool(self.log_dir)}, "
                f"model.training={is_training})"
            )

        # Убедимся, что мы в режиме evaluation и директория логов доступна
        if not state.is_world_process_zero or not self.log_dir or kwargs['model'].training:
             # ЛОГ ПРИЧИНЫ ВЫХОДА (только на главном процессе для наглядности)
             if state.is_world_process_zero:
                  reason = []
                  if not self.log_dir: reason.append("no log_dir")
                  if kwargs.get('model', None) and kwargs['model'].training: reason.append("model.training is True")
                  logger.warning(f">>> ProfilerCallback: Skipping on_prediction_step on world_process_zero. Reason(s): {', '.join(reason)}")
             elif not state.is_world_process_zero:
                  # Можно добавить лог для не-главных процессов, если нужно
                  pass
                  # logger.debug(">>> ProfilerCallback: Skipping on_prediction_step on non-world_process_zero.")
             return control

        # --- Если мы дошли сюда, значит if НЕ сработал ---
        logger.info(">>> ProfilerCallback: Condition passed, proceeding to initialize/step profiler.")

        # Инициализация и старт профайлера при первом вызове в evaluation
        if self.profiler is None:
            logger.info(">>> ProfilerCallback: self.profiler is None. Attempting initialization...") # <--- ЛОГ 1
            try:
                logger.info(">>> ProfilerCallback: Before profile(...) call") # <--- ЛОГ 2
                prof = profile( # Используем временную переменную
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(
                        wait=0, warmup=self.profile_warmup, active=self.profile_steps, repeat=1
                    ),
                    on_trace_ready=tensorboard_trace_handler(self.log_dir),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                )
                logger.info(f">>> ProfilerCallback: profile(...) returned object: {type(prof)}") # <--- ЛОГ 3
                logger.info(">>> ProfilerCallback: Before prof.__enter__()") # <--- ЛОГ 4
                prof.__enter__()
                logger.info(">>> ProfilerCallback: After prof.__enter__()") # <--- ЛОГ 5
                # --- Присваиваем ТОЛЬКО после успешного __enter__ ---
                self.profiler = prof
                self.is_profiling_active_in_eval = True
                logger.info(f">>> Profiler started and assigned to self.profiler in {self.log_dir}") # <--- ЛОГ 6
            except Exception as e:
                logger.error(f"!!! ProfilerCallback: FAILED during profiler initialization or __enter__: {e}", exc_info=True) # <--- ЛОГ ОШИБКИ
                self.profiler = None # Убедимся, что он None при ошибке
                self.is_profiling_active_in_eval = False
                return control

        # Вызываем profiler.step() на каждом шаге предсказания, если он активен
        elif self.is_profiling_active_in_eval and self.profiler: # Используем elif для ясности
            # Логируем, что мы именно делаем step
            logger.info(">>> ProfilerCallback: self.profiler exists. Calling profiler.step().") # <--- ЛОГ 7
            self.profiler.step()
        else:
             # Логируем странное состояние, если profiler не None, но флаг False
             if self.profiler is not None:
                  logger.warning(">>> ProfilerCallback: self.profiler exists but is_profiling_active_in_eval is False. Skipping step.")

        return control


    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs):
        """
        Called after the evaluation loop is finished.
        Handles profiler stop (__exit__) and logging results.
        """
        logger.info(">>> ProfilerCallback: on_evaluate called")

        # Останавливаем и логируем только если профайлер был успешно запущен в этом evaluate
        if self.is_profiling_active_in_eval and self.profiler:
            logger.info(">>> Stopping and processing profiler results...")
            try:
                self.profiler.__exit__(None, None, None)
                logger.info("Profiler stopped. Trace should be saved to TensorBoard log directory.")

                key_averages = self.profiler.key_averages()
                if not key_averages:
                    logger.warning("Profiler recorded no key averages. Was the active phase reached/long enough?")
                else:
                    total_cpu_time_ms = sum(evt.cpu_time_total for evt in key_averages) / 1000.0
                    total_cuda_time_ms = sum(evt.cuda_time_total for evt in key_averages) / 1000.0
                    # peak_cpu_mem_mb = sum(evt.cpu_memory_usage for evt in key_averages) / (1024**2)
                    # peak_cuda_mem_mb = sum(evt.cuda_memory_usage for evt in key_averages) / (1024**2)

                    profile_metrics = {
                        "profiler/eval_total_cpu_time_ms": round(total_cpu_time_ms, 2),
                        "profiler/eval_total_cuda_time_ms": round(total_cuda_time_ms, 2),
                        # "profiler/eval_peak_cpu_mem_mb": round(peak_cpu_mem_mb, 2), # Если включите profile_memory
                        # "profiler/eval_peak_cuda_mem_mb": round(peak_cuda_mem_mb, 2),# Если включите profile_memory
                    }

                    # Логируем через trainer, если он доступен (предпочтительно)
                    trainer = kwargs.get('trainer')
                    if trainer:
                        trainer.log(profile_metrics)
                    else:
                        metrics.update(profile_metrics) # Иначе добавляем в стандартные метрики
                    logger.info(f"Profiler summary: {profile_metrics}")

            except Exception as e:
                logger.error(f"Error during profiler exit or logging in on_evaluate: {e}")
            finally:
                # Сбрасываем состояние для следующего возможного вызова evaluate
                self.profiler = None
                self.is_profiling_active_in_eval = False
        elif state.is_world_process_zero:
             logger.info(">>> Profiler was not active during this evaluation run.")

        return control

