import os
from typing import Any, List, Dict, Optional
from datasets import DatasetDict
import logging
from .base_dataset import BaseDataset
from src.config import DataArguments

logger = logging.getLogger(__name__)

class DPODataset(BaseDataset):
    """Датасет для Direct Preference Optimization"""
    
    def __init__(self, data_args: DataArguments, tokenizer: Any):
        super().__init__(data_args=data_args, tokenizer=tokenizer)




    def _load_and_preprocess_data(self) -> DatasetDict:
        datasets = self._load_raw_datasets()
        
        # Обновленный список signature_columns с текстовыми полями

        # Apply chat template
        datasets = datasets.map(
            self._apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer},
        )
        return datasets

    def _apply_chat_template(self, example: Dict, **kwargs) -> Dict:
        """
        Применяет chat template для промпта, chosen и rejected ответов
        """
        if not isinstance(example['messages'], list):
            raise ValueError(f"Expected list of messages, got {type(example['messages'])}")

        # Получаем промпт (все сообщения до chosen и rejected)
        prompt_messages = example['messages'][:-2]
        # Получаем chosen и rejected ответы отдельно
        chosen_response = example['messages'][-2]
        rejected_response = example['messages'][-1]
        
        # Применяем chat template для промпта
        prompt_text = kwargs['tokenizer'].apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Удаляем bos_token если он есть
        if kwargs['tokenizer'].bos_token is not None and prompt_text.startswith(kwargs['tokenizer'].bos_token):
            prompt_text = prompt_text[len(kwargs['tokenizer'].bos_token):]
        
        # Токенизируем промпт + chosen и промпт + rejected вместе
        chosen_full = prompt_text + chosen_response['content']
        rejected_full = prompt_text + rejected_response['content']
        
        chosen_tokens = kwargs['tokenizer'](
            chosen_full,
            truncation=True,
            padding=False,
            max_length=self.data_args.tokenizer_max_seq_length
        )
        
        rejected_tokens = kwargs['tokenizer'](
            rejected_full,
            truncation=True,
            padding=False,
            max_length=self.data_args.tokenizer_max_seq_length
        )
        
        # Находим длину промпта в токенах для правильного разделения
        prompt_tokens = kwargs['tokenizer'](
            prompt_text,
            truncation=True,
            padding=False,
            max_length=self.data_args.tokenizer_max_seq_length,
            add_special_tokens=False  # Важно! Не добавляем специальные токены
        )
        prompt_length = len(prompt_tokens['input_ids'])
        
        return {
            "prompt": prompt_text,  # Текстовый промпт (для отладки)
            "chosen": chosen_response['content'],  # Текстовый chosen (для отладки)
            "rejected": rejected_response['content'],  # Текстовый rejected (для отладки)
            
            # Промпт отдельно (может потребоваться для некоторых реализаций)
            "prompt_input_ids": chosen_tokens["input_ids"][:prompt_length],
            "prompt_attention_mask": chosen_tokens["attention_mask"][:prompt_length],
            
            # Полные последовательности промпт + ответ
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"]
        } 