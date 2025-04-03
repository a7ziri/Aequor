import logging
import sys
import os


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from unsloth import FastLanguageModel
import datasets
import torch
from unsloth import PatchDPOTrainer
PatchDPOTrainer()
import transformers
from transformers import AutoModelForCausalLM, set_seed
from src.data import DPODataset
from src.yaml_parse import H4ArgumentParser
from src.config import (
    DataArguments,
    ModelArguments,
    DPOConfig,
)
from src.model_utils import (
    get_tokenizer,
    get_peft_config,
    get_checkpoint,
    get_quantization_config,
    auto_find_batch_size,
    get_kbit_device_map,
    is_adapter_model
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer, setup_chat_format
from src.callback.alignment_callback import AlignmentMetricsCallback 
from src.callback.profiler_callback import TensorBoardProfilerCallback



logger = logging.getLogger(__name__)


    
def main():
    

    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    

    
    # Настройка семени для воспроизводимости
    set_seed(training_args.seed)
    
    # Логируем информацию об акселераторе

    
    # Настройка логирования
    # Создаем директорию для логов, если она не существует
    log_dir = os.path.join(training_args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'dpo_train.log')
    
    # Настройка логирования с выводом на экран и в файл
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ],
    )
    
    # Устанавливаем уровень логирования
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Создаем директорию вывода (только в главном процессе)

    
    # Проверка доступности CUDA через PyTorch
    logger.info(f"PyTorch видит CUDA: {torch.cuda.is_available()}")
    logger.info(f"Количество GPU: {torch.cuda.device_count()}")
    logger.info(f"Текущее устройство: {torch.cuda.current_device()}")
    
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load tokenizer
    ################
    logger.info("*** Load tokenizer ***")
    tokenizer = get_tokenizer(model_args, data_args)

    ###############
    # Load datasets
    ###############

    logger.info("*** Load datasets ***")
    raw_datasets = DPODataset(
        data_args,
        tokenizer=tokenizer,
    )
    
    # Логируем информацию через внутренний DatasetDict
    logger.info(
        f"Training datasets sizes: "
        f"Train: {raw_datasets.datasets['train'].num_rows} samples, "
        f"Test: {raw_datasets.datasets['test'].num_rows} samples"
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    logger.info(f"Train dataset: {train_dataset[0]}")
    logger.info(f"Eval dataset: {eval_dataset[0]}")
    
    #######################
    # Load pretrained model
    #######################

    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=None,
            use_cache=False,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
            max_length=data_args.tokenizer_max_seq_length,  
        )
    if is_adapter_model(model_args.model_name_or_path) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path , **model_kwargs)
        model_kwargs = dict(
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=None,
            torch_dtype=torch_dtype,
            use_cache=False,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            **model_kwargs,
        )
        model_kwargs = None

    ref_model = model_args.model_name_or_path

    if model_args.use_peft is True:
        ref_model = None


    if model_args.use_unsloth:
        try:
            logger.info(f"Model kwargs: {model_kwargs}")
            logger.info(f"Torch dtype: {torch_dtype}")
            model, _= FastLanguageModel.from_pretrained(
                dtype=torch_dtype,
                model_name=model_args.model_name_or_path,
                use_gradient_checkpointing='unsloth',
                load_in_8bit=model_args.load_in_8bit,
                load_in_4bit=model_args.load_in_4bit,
                token=data_args.hf_token,
                full_finetuning=model_args.full_finetuning,
                **model_kwargs
            )
            if model_args.use_peft:
                    model = FastLanguageModel.get_peft_model(
                        model, 
                        r=model_args.lora_r,
                        lora_alpha=model_args.lora_alpha,
                        lora_dropout=model_args.lora_dropout,
                        target_modules=model_args.lora_target_modules,
                        use_gradient_checkpointing='unsloth'
                    )
            if tokenizer.chat_template is None:
                model, tokenizer = setup_chat_format(model, tokenizer)
        except ImportError:
            raise ImportError("Unsloth is not installed. Use pip install unsloth")
    else:
        # Загружаем модель при помощи AutoModelForCausalLM вместо передачи строки
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            **model_kwargs
        )
        
        # Если нужно применить PEFT
        if model_args.use_peft:
            from peft import get_peft_model
            peft_config = get_peft_config(model_args)
            model = get_peft_model(model, peft_config)
            
        # Применяем chat template если нужно
        if tokenizer.chat_template is None:
            model, tokenizer = setup_chat_format(model, tokenizer)
        
        if training_args.auto_find_batch_size:
            training_args.per_device_train_batch_size = auto_find_batch_size(training_args, model)
            training_args.per_device_eval_batch_size = training_args.per_device_train_batch_size
            logger.info(f"Automatically selected batch size: {training_args.per_device_train_batch_size}")

    ########################
    # Initialize the Trainer
    ########################
    callbacks = []
    profiler_callback = TensorBoardProfilerCallback(
            profile_steps=10,
            profile_warmup=5
        )
    # if model_args.use_profiler:
    #     profiler_callback = TensorBoardProfilerCallback(
    #         profile_steps=10,
    #         profile_warmup=5
    #     )
    #     logger.info(f"Callback created: {profiler_callback}")
    #     callbacks.append(profiler_callback)
    # else:
    #     profiler_callback = None
    if model_args.use_alignment_metrics:
        alignment_callback = AlignmentMetricsCallback(
            tokenizer=tokenizer,
            eval_dataset=eval_dataset
        )
        logger.info(f"Callback created: {alignment_callback}")
        callbacks.append(alignment_callback)
    else:
        alignment_callback = None

    # Создаем тренер с колбэком
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_num_proc=1,
        learning_rate=training_args.learning_rate,
        peft_config=None if model_args.use_peft else get_peft_config(model_args),
        callbacks=[profiler_callback],
        loss_type=training_args.loss_type,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=2,
        beta=training_args.beta,
        max_prompt_length=512
    )
    logger.info(f"Trainer callbacks: {trainer.callback_handler.callbacks}")

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint


    train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
    if hasattr(train_result, "metrics"):
        logger.info("******************** Сохраняем результаты тренировки ***********************")
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)



    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)

    # Объединяем метрики датасета и обучения

    kwargs = {
        "model_name": model_args.model_name_or_path,
        "dataset_name": list(data_args.dataset_mixer.keys()),
        "tags": list(data_args.dataset_mixer.keys()) + ["alignment"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    if model_args.use_peft:
        # Сохраняем только адаптер
        trainer.model.save_pretrained(training_args.output_dir)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
