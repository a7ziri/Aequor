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

    def is_openai_format(self, messages):
        """Проверяет, соответствует ли список сообщений формату OpenAI"""
        if not isinstance(messages, list):
            return False
        if not messages:
            return False
        return all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in messages)

    def _load_and_preprocess_data(self) -> DatasetDict:
        datasets = self._load_raw_datasets()
        
        # Apply chat template and ensure we have the right columns for DPO
        logger.info(f"Before processing, datasets have columns: {datasets['train'].column_names}")
        
        # Add the required columns for DPO training
        datasets = datasets.map(
            self._apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer},  # Remove original columns
        )
        
        logger.info(f"After processing, datasets have columns: {datasets['train'].column_names}")
        
        # Verify required columns exist
        required_columns = ['prompt', 'chosen', 'rejected']
        for split in datasets:
            missing_cols = [col for col in required_columns if col not in datasets[split].column_names]
            if missing_cols:
                logger.error(f"Missing required columns in {split} dataset: {missing_cols}")
                raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
        return datasets

    def _apply_chat_template(self, example: Dict, **kwargs) -> Dict:
        """
        Применяет chat template для промпта, chosen и rejected ответов
        """

        logger.debug(f"Example structure: {list(example.keys())}")

        result = {}
        # add format  if  chosen or rejected is not openai format
        if all(key in example for key in ['prompt', 'chosen', 'rejected']):
            prompt_input = example['prompt']
            chosen_input = example['chosen'] 
            rejected_input = example['rejected'] 
            
            # Применяем chat template в зависимости от формата
            if prompt_input:
                if self.is_openai_format(prompt_input):
                    # Формат с диалогами OpenAI - применяем template ко всем сообщениям
                    for field, input_data in [
                        ('prompt', prompt_input),
                        ('chosen', chosen_input),
                        ('rejected', rejected_input)
                    ]:
                        result[field] = kwargs['tokenizer'].apply_chat_template(
                            input_data,
                            tokenize=False,
                        )
                else:
                    # Формат с последним сообщением как ответом
                    result['prompt'] = kwargs['tokenizer'].apply_chat_template(
                        example["chosen"][:-1],
                        tokenize=False,
                    )
                    # Now we extract the final turn to define chosen/rejected responses
                    result['chosen'] = kwargs['tokenizer'].apply_chat_template(
                        example["chosen"][-1:],
                        tokenize=False,
                    )
                    result['rejected'] = kwargs['tokenizer'].apply_chat_template(
                        example["rejected"][-1:],
                        tokenize=False,
                    )
            else:
                # Если prompt пустой, просто копируем исходные значения
                return  ValueError("Prompt is empty")
                
            
        return result

   