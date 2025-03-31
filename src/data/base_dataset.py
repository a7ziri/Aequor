from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
import logging
from src.config import DataArguments
import os


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
logger = logging.getLogger(__name__)

class BaseDataset(ABC):
    def __init__(self, data_args: DataArguments, tokenizer: Any):

        self.data_args = data_args
        self.tokenizer = tokenizer
        self.converters = {} 
        self.chat_template = self._setup_tokenizer()
        self.datasets = self._load_and_preprocess_data()


    @abstractmethod
    def _apply_chat_template(self, example: Dict, **kwargs) -> Dict:
        """Apply chat template to a single example"""
        raise NotImplementedError("Subclasses must implement this method")
    

    


    def _setup_tokenizer(self) -> str:
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        return self.tokenizer.chat_template

    def _get_converter(self, dataset_path: str) -> Any:
        """Получает или создает конвертер для датасета"""
        if dataset_path not in self.converters:
            self.converters[dataset_path] = self.data_args.create_converter_for_dataset(dataset_path)
        return self.converters[dataset_path]
    @abstractmethod
    def _load_and_preprocess_data(self) -> DatasetDict:
        raise NotImplementedError("Subclasses must implement this method")



    def _load_raw_datasets(self) -> DatasetDict:
        datasets = DatasetDict()
        train_subsets = []
        val_datasets = []

        for ds_name, frac in self.data_args.dataset_mixer.items():
            converter = self._get_converter(ds_name)
            
            for split in self.data_args.dataset_splits:
                try:
                    # Сначала пробуем загрузить локально
                    try:
                        logger.info(f"Attempting to load dataset from local path: {ds_name}")
                        if os.path.isdir(ds_name):
                            # Если указана директория
                            dataset = load_from_disk(ds_name)
                            if split in dataset:
                                dataset = dataset[split]
                            else:
                                logger.warning(f"Split {split} not found in local dataset, skipping...")
                                continue
                        else:
                            # Если указан путь к файлу
                            dataset = load_dataset(
                                'json' if ds_name.endswith('.json') else 'csv',
                                data_files=ds_name,
                                split=split
                            )
                        logger.info(f"Successfully loaded local dataset from {ds_name}")
                    
                    except (FileNotFoundError, ValueError) as e:
                        # Если локально не нашли, пробуем загрузить из HF Hub
                        logger.info(f"Local dataset not found, trying Hugging Face Hub: {ds_name}")
                        dataset = load_dataset(
                            ds_name, 
                            name=self.data_args.data_configs.get(ds_name), 
                            split=split, 
                            token=self.data_args.hf_token
                        )
                        logger.info(f"Successfully loaded dataset from Hugging Face Hub")
                    
                    # Конвертируем данные
                    dataset = dataset.map(
                        lambda x: {"messages": converter.convert_to_chat_format(
                            x, 
                            auto_insert_empty_system_msg=self.data_args.auto_insert_empty_system_msg
                        )},
                        remove_columns=dataset.column_names
                    )
                    
                    logger.info(f"Successfully converted dataset {ds_name}")
                    
                    if "train" in split:
                        subset = dataset.select(range(int(frac * len(dataset))))
                        train_subsets.append(subset)
                    elif "test" in split:
                        val_datasets.append(dataset)
                        
                except Exception as e:
                    logger.error(f"Failed to process dataset {ds_name}: {e}")
                    raise

        # Объединяем датасеты
        if train_subsets:
            datasets["train"] = concatenate_datasets(train_subsets)
        if val_datasets:
            datasets["test"] = concatenate_datasets(val_datasets)
        
        if self.data_args.shuffle:
            datasets = datasets.shuffle(seed=42)
            
        return datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]

    def _validate_conversation_data(self, dataset):
        """Validate that conversation data meets expected format"""
        if self.conversation_column not in dataset.column_names:
            raise ValueError(f"Column {self.conversation_column} not found in dataset")
        
        # Проверка формата на основе примера
        sample = dataset[self.conversation_column][0]
        if not isinstance(sample, list):
            raise ValueError(f"Expected list for conversation data, got {type(sample)}")
        
        # Проверка структуры сообщений
        required_fields = ['role', 'content']
        for msg in sample:
            if not isinstance(msg, dict):
                raise ValueError(f"Expected dict for message, got {type(msg)}")
            for field in required_fields:
                if field not in msg:
                    raise ValueError(f"Required field '{field}' missing from message: {msg}")
        
        logger.info(f"Conversation data validated successfully with {len(sample)} messages")
        return True