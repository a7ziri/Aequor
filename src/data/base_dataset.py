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
    
    @abstractmethod
    def _load_and_preprocess_data(self) -> DatasetDict:
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


    def _load_raw_datasets(self) -> DatasetDict:
        datasets_by_split = {"train": [], "test": []}
        
        for ds_name, frac in self.data_args.dataset_mixer.items():
            logger.info(f"Processing dataset {ds_name} (fraction: {frac})")
            converter = self._get_converter(ds_name)
            logger.info(f"Using converter: {converter.__class__.__name__}")
            
            try:
                # Загрузка датасета
                try:
                    if os.path.isdir(ds_name):
                        dataset = load_from_disk(ds_name)
                    else:
                        dataset = load_dataset(
                            'json' if ds_name.endswith('.json') else 'csv',
                            data_files=ds_name,
                            split='train' # Сначала пробуем загрузить train split
                        )
                    logger.info(f"Successfully loaded local dataset from {ds_name}")
                except (FileNotFoundError, ValueError):
                    dataset = load_dataset(
                        ds_name,
                        name=self.data_args.data_configs.get(ds_name),
                        token=self.data_args.hf_token
                    )
                
                # Обеспечиваем наличие обоих сплитов
                dataset_splits = self._ensure_splits_exist(dataset, ds_name)
                
                # Конвертация данных
                for split_name, split_dataset in dataset_splits.items():
                    if converter.__class__.__name__ == "ChatMessageConverter" or converter.__class__.__name__ == "QAConverter":
                        converted_dataset = split_dataset.map( 
                            lambda x: { 'messages': converter.convert_to_chat_format(x)},
                            remove_columns=split_dataset.column_names
                        )
                    else:
                        converted_dataset = split_dataset.map( 
                            lambda x: converter.convert_to_chat_format(x),
                            remove_columns=split_dataset.column_names
                        )
                    
                    # Применяем фракцию только к тренировочному сплиту
                    if split_name == "train":
                        subset_size = int(frac * len(converted_dataset))
                        converted_dataset = converted_dataset.select(range(subset_size))
                    if split_name == "test":
                        subset_size = int(frac * len(converted_dataset))
                        converted_dataset = converted_dataset.select(range(subset_size))
                    
                    datasets_by_split[split_name].append(converted_dataset)
                    
            except Exception as e:
                logger.error(f"Failed to process dataset {ds_name}: {e}")
                raise

        # Объединяем все сплиты
        final_datasets = DatasetDict({
            split_name: concatenate_datasets(split_datasets) if split_datasets else None
            for split_name, split_datasets in datasets_by_split.items()
        })

        # Перемешиваем данные если нужно
        if self.data_args.shuffle:
            for split_name in final_datasets:
                if final_datasets[split_name] is not None:
                    final_datasets[split_name] = final_datasets[split_name].shuffle(seed=42)
        
        return final_datasets

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

    def _create_train_test_split(self, dataset, test_size=0.1, seed=42):
        """
        Create train/test split if test split is missing
        """
        splits = dataset.train_test_split(test_size=test_size, seed=seed)
        return {
            "train": splits["train"],
            "test": splits["test"]
        }

    def _ensure_splits_exist(self, dataset, dataset_name):
        """
        Ensure both train and test splits exist for the dataset
        """
        logger.info(f"Checking splits for dataset {dataset_name}")
        
        if isinstance(dataset, dict):
            # Если датасет уже имеет структуру словаря сплитов
            has_train = "train" in dataset
            has_test = "test" in dataset
            
            if has_train and has_test:
                logger.info(f"Dataset {dataset_name} already has both train and test splits")
                return dataset
            elif has_train and not has_test:
                logger.info(f"Creating test split for dataset {dataset_name}")
                splits = self._create_train_test_split(dataset["train"])
                return splits
            elif not has_train:
                raise ValueError(f"Dataset {dataset_name} must have at least a train split")
        else:
            # Если датасет представлен как единый набор данных
            logger.info(f"Creating train/test splits for dataset {dataset_name}")
            return self._create_train_test_split(dataset)