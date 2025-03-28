from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import EvalLoopOutput
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import PreTrainedModel
import logging

logger = logging.getLogger(__name__)

class AlignmentMetricsCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset  # Сохраняем eval датасет
        print("=== AlignmentMetricsCallback initialized ===")
        self.metrics_history = {
            "kl_divergence": [],
            "cross_entropy": [],
            "response_length": [],
            "top_k_token_concentration": [],
            "training_steps": []
        }

    def setup(self, args, state, model, **kwargs):
        """Called when the training starts"""
        print("=== AlignmentMetricsCallback setup called ===")
        logger.info("Setup called")
        self.model = model
        


    def on_train_begin(self, args, state, control, **kwargs):
        print("=== Training started ===")
        # Сохраняем ссылку на trainer
        if 'trainer' in kwargs:
            self.trainer = kwargs['trainer']
        return control



    def calculate_metrics(self, model) -> Dict[str, float]:
        try:
            model.eval()
            
            # Инициализируем аккумуляторы для метрик
            total_kl_div = 0.0
            total_entropy = 0.0
            total_response_length = 0.0
            total_top_k_concentration = 0.0
            num_samples = min(len(self.eval_dataset), 1500)  # Ограничиваем количество сэмплов для скорости
            
            with torch.no_grad():
                for i in range(num_samples):
                    # Получаем реальный пример из датасета
                    sample = self.eval_dataset[i]
                    input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(model.device)
                    
                    # Получаем выход модели
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits
                    
                    # Берем логиты только для последнего токена
                    last_token_logits = logits[:, -1, :]
                    
                    # Считаем вероятности
                    probs = torch.softmax(last_token_logits, dim=-1)
                    
                    # KL-дивергенция
                    uniform_probs = torch.ones_like(probs) / probs.size(-1)
                    kl_div = torch.sum(probs * torch.log(probs / uniform_probs))
                    
                    # Энтропия
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    
                    # Длина ответа (разница между входом и выходом)
                    response_length = len(sample['input_ids'])
                    
                    # Концентрация на топ-K токенах
                    k = 10
                    top_k_probs, _ = torch.topk(probs, k)
                    top_k_concentration = torch.sum(top_k_probs)
                    
                    # Аккумулируем метрики
                    total_kl_div += kl_div.item()
                    total_entropy += entropy.item()
                    total_response_length += response_length
                    total_top_k_concentration += top_k_concentration.item()
            
            # Усредняем метрики
            metrics = {
                "kl_divergence": total_kl_div / num_samples,
                "cross_entropy": total_entropy / num_samples,
                "response_length": total_response_length / num_samples,
                "top_k_concentration": total_top_k_concentration / num_samples,
                "samples_evaluated": num_samples
            }
            
            logger.info(f"Calculated metrics on {num_samples} samples: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "kl_divergence": 0.0,
                "cross_entropy": 0.0,
                "response_length": 0.0,
                "top_k_concentration": 0.0,
                "samples_evaluated": 0
            }
        finally:
            model.train()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        try:
            print("=== Evaluation started ===")
            # Получаем модель из trainer напрямую
            model = kwargs.get('model', None)
            if model is None and hasattr(self, 'trainer'):
                model = self.trainer.model
            
            if model is not None:
                eval_metrics = self.calculate_metrics(model)
                print(f"Evaluation metrics: {eval_metrics}")
                
                # Если metrics это словарь, добавляем наши метрики
                if isinstance(metrics, dict):
                    metrics.update(eval_metrics)
            else:
                logger.warning("No model available for evaluation")
            
        except Exception as e:
            print(f"Error in on_evaluate: {str(e)}")
            logger.error(f"Error in on_evaluate: {str(e)}")
        
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logger.info(f"Logging triggered - logs: {logs}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> TrainerControl:
        logger.info("Training ended - callback triggered")
        logger.info(f"Final metrics history: {self.metrics_history}")
        return control


