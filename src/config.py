from dataclasses import dataclass, field
from typing import List, Optional,Dict, Any

import trl
from dataclasses import dataclass
from typing import Any, List, NewType, Optional

DataClassType = NewType("DataClassType", Any)
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adapters.")},
    )
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Coder-32B-Instruct")
    use_unsloth:Optional[bool]= field(
        default= False,
        metadata={
            "help": (
                "Use  unsloth  for  training"
            )
        },
    


    )


    use_alignment_metrics: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use alignment metrics for training"
        },
    )

    full_finetuning: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use full finetuning or not"
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`."
            )
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can use --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )

    peft_type: Optional[str] = field(
        default="lora",
        metadata={"help": ("PEFT type.")},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": ("LoRA target modules.")},
    )
    lora_task_type: Optional[str] = field(
        default=None,
        metadata={"help": ("LoRA task type.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})



    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})
    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8",
        metadata={"help": "storage type to pack the quanitzed 4-bit prarams."},
    )
    # General quantization parameters
    quantization_method: Optional[str] = field(
        default="bnb",
        metadata={"help": "still not implemented"},
    )
    torch_quantization_scheme: Optional[str] = field(
        default="dynamic",
        metadata={"help": "still not implemented"},
    )
    onnx_quantization_format: Optional[str] = field(
        default="qint8",
        metadata={"help": "still not implemented"},
    )

    def __post_init__(self):
        self._validate_arguments()

        
    def _validate_arguments(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")
        if (self.load_in_8bit or self.load_in_4bit) and self.quantization_method != "bnb":
            raise ValueError("load_in_8bit/4bit работает только с quantization_method='bnb'")
        if self.use_peft and self.full_finetuning:
            raise ValueError("use_peft и full_finetuning не могут быть true одновременно")
        if self.full_finetuning and not(self.use_unsloth):
            raise ValueError("full_finetuning работает только с use_unsloth=True")
        
        


    

@dataclass
class DataArguments:

    dataset_mixer: Dict[str, Any] = field(default_factory=dict)
    data_configs: Optional[Dict[str, str]] = field(
        default_factory=dict,
        metadata={"help": "Словарь конфигов для датасетов (датасет -> конфиг)"},
    )

    dataset_splits: List[str] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": "Сплиты для загрузки"}
    )

    shuffle: Optional[bool] = field(
        default=True,
        metadata={"help": "Перемешивать ли исходные  данные?"}

    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "Шаблон для форматирования чата"}
    )
    subset:Optional[str] = field(
        default=None,
        metadata={"help": "саб сет если такой  есть "}
    )
    
    tokenizer_max_seq_length: int = 2048
    preprocessing_num_workers: int = field(
        default=8,
        metadata={"help": "Количество воркеров для предобработки"}
    )
    truncation_side: Optional[str] = field(
        default=None,
        metadata={"help": "Сторона усечения (left или right)"}
    )
    auto_insert_empty_system_msg: bool = field(default=True)

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Директория для кэша"}
    )
    sample_prop: float = field(
        default=1.0,
        metadata={"help": "Доля случайных семплов для загрузки"}
    )
    hf_token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face токен для доступа к приватным датасетам"}
    )

    converter_config: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "chat_message": {
                "default_system_prompt": "You are a helpful assistant.",
                "input_columns": ["conversations", "dialogue", "messages"],  # возможные колонки
                "format_variants": [
                    # вариант 1
                    {
                        "required_fields": ["from", "value"],
                        "role_field": "from",
                        "content_field": "value"
                    },
                    # вариант 2
                    {
                        "required_fields": ["role", "content"],
                        "role_field": "role",
                        "content_field": "content"
                    }
                ]
            },
            "preferred_answer": {
                "include_rejected": True,
                "default_system_prompt": "You are a helpful assistant.",
                "input_columns": ["prompt", "chosen", "rejected"],
                "format_columns": ["prompt", "chosen", "rejected"]
            },
            "qa": {
                "default_system_prompt": "You are a helpful assistant. Answer the question based on the context provided.",
            }
        },
        metadata={
            "help": "Конфигурация для различных конвертеров данных"
        }
    )

    dataset_formats: Dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Формат каждого датасета (dataset_path -> format_type) и дефолтный формат",
            "choices": ["chat_message", "qa", "preferred_answer"]
        }
    )

    default_system_prompt: str = field(
        default="You are a helpful assistant.",
        metadata={
            "help": "Какой промт  использовать по умолчанию"
        }
    )


    def create_converter_for_dataset(self, dataset_path: str) -> Any:
        """
        Создает конвертер для конкретного датасета
        
        Args:
            dataset_path: Путь к датасету
            
        Returns:
            DataConverter: Сконфигурированный конвертер для данного датасета
        """
        # Импортируем здесь, внутри метода
        from src.data.data_converters import ChatMessageConverter, PreferredAnswerConverter, QAConverter
        
        # Используем формат датасета или дефолтный, если не указан
        format_type = self.dataset_formats.get(dataset_path)
        if format_type is None:
            raise ValueError(f"No format type specified for dataset: {dataset_path}"
                             f"Available types: {list(self.converter_config.keys())}")
        
        config = self.converter_config[format_type]
        
        if format_type == "chat_message":
            return ChatMessageConverter(config)
        elif format_type == "qa":
            return QAConverter(config)
        elif format_type == "preferred_answer":
            return PreferredAnswerConverter(config)
        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def __post_init__(self):
        self._validate_converter_config()
        self._validate_dataset_formats()


    def _validate_dataset_formats(self):
        """Проверяет совместимость форматов датасетов"""
        formats = set(self.dataset_formats.values())
        

        preference_formats = {"preferred_answer"}
        conversation_formats = {"chat_message", "qa"}
        
        # Проверяем пересечение форматов
        if formats.intersection(preference_formats) and formats.intersection(conversation_formats):
            raise ValueError(
                "Preference-based formats (preferred_answer) cannot be mixed with conversation formats "
                f"(chat_message). Current formats: {formats}"
            )




    def _validate_converter_config(self):
        """Проверяет корректность конфигурации конвертеров"""
        # Проверяем все форматы
        for dataset_path, format_type in self.dataset_formats.items():
            if format_type not in self.converter_config:
                raise ValueError(
                    f"Format type '{format_type}' for dataset '{dataset_path}' not found in converter_config. "
                    f"Available types: {list(self.converter_config.keys())}"
                )



@dataclass
class SFTConfig(trl.SFTConfig):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    Also used for the continued pretraining task.
    """
    auto_find_batch_size: bool = field(
        default=False,
        metadata={"help": "Автоматически определять оптимальный размер батча"}
    )

    report_to: List[str] = field(default_factory=lambda: ["wandb"])



@dataclass
class DPOConfig(trl.DPOConfig):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    """

    auto_find_batch_size: bool = field(
        default=False,
        metadata={"help": "Автоматически определять оптимальный размер батча"}
    )
    report_to: List[str] = field(default_factory=lambda: ["wandb"])

    loss_type: Optional[str] = field(
        default="sigmoid",
        metadata={"help": "loss type for DPO"}
    )

    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "beta for DPO"}
    )
