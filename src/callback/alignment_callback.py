from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class AlignmentMetricsCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset  
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
        logger.info("Setup called for AlignmentMetricsCallback")
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
            num_samples = min(len(self.eval_dataset), 500)  # Ограничиваем количество сэмплов для скорости
            
            with torch.no_grad():
                for i in range(min(num_samples, len(self.eval_dataset))):
                    # Получаем реальный пример из датасета
                    sample = self.eval_dataset[i]
                    
                    # Проверяем, какой тип данных у нас - DPO или обычный
                    if 'prompt_input_ids' in sample:
                        # Это DPO датасет, оцениваем оба ответа
                        
                        # Объединяем промпт и chosen ответ
                        prompt_ids = torch.tensor(sample['prompt_input_ids']).to(model.device)
                        chosen_ids = torch.tensor(sample['chosen_input_ids']).to(model.device)
                        chosen_input_ids = torch.cat([prompt_ids, chosen_ids], dim=0).unsqueeze(0)
                        
                        # Объединяем промпт и rejected ответ
                        rejected_ids = torch.tensor(sample['rejected_input_ids']).to(model.device)
                        rejected_input_ids = torch.cat([prompt_ids, rejected_ids], dim=0).unsqueeze(0)
                        
                        # Оцениваем chosen ответ
                        chosen_outputs = model(input_ids=chosen_input_ids)
                        chosen_logits = chosen_outputs.logits[:, -1, :]
                        chosen_probs = torch.softmax(chosen_logits, dim=-1)
                        
                        # Оцениваем rejected ответ
                        rejected_outputs = model(input_ids=rejected_input_ids)
                        rejected_logits = rejected_outputs.logits[:, -1, :]
                        rejected_probs = torch.softmax(rejected_logits, dim=-1)
                        
                        # Вычисляем метрики для chosen
                        chosen_uniform_probs = torch.ones_like(chosen_probs) / chosen_probs.size(-1)
                        chosen_kl_div = torch.sum(chosen_probs * torch.log(chosen_probs / chosen_uniform_probs))
                        chosen_entropy = -torch.sum(chosen_probs * torch.log(chosen_probs + 1e-10))
                        
                        # Вычисляем метрики для rejected
                        rejected_uniform_probs = torch.ones_like(rejected_probs) / rejected_probs.size(-1)
                        rejected_kl_div = torch.sum(rejected_probs * torch.log(rejected_probs / rejected_uniform_probs))
                        rejected_entropy = -torch.sum(rejected_probs * torch.log(rejected_probs + 1e-10))
                        
                        # Сохраняем метрики
                        kl_div = (chosen_kl_div + rejected_kl_div) / 2  # Среднее значение
                        entropy = (chosen_entropy + rejected_entropy) / 2  # Среднее значение
                        
                        # Определяем, насколько модель предпочитает chosen над rejected
                        # Это должно расти в процессе DPO обучения
                        reward_gap = chosen_logits.mean() - rejected_logits.mean()
                        metrics['dpo_reward_gap'] = reward_gap.item()
                        
                        # Длина ответов
                        chosen_length = len(sample['chosen_input_ids'])
                        rejected_length = len(sample['rejected_input_ids'])
                        response_length = (chosen_length + rejected_length) / 2
                    else:
                        # Обычный датасет - код остается прежним
                        input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(model.device)
                        outputs = model(input_ids=input_ids)
                        logits = outputs.logits
                        last_token_logits = logits[:, -1, :]
                        probs = torch.softmax(last_token_logits, dim=-1)
                        
                        uniform_probs = torch.ones_like(probs) / probs.size(-1)
                        kl_div = torch.sum(probs * torch.log(probs / uniform_probs))
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
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


