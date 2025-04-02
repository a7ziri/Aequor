import os
from pathlib import Path
from typing import Optional, Union, Dict

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel
from transformers.trainer_utils import get_last_checkpoint

from huggingface_hub import list_repo_files
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from peft import LoraConfig, PeftConfig

from src.config import DataArguments, DPOConfig, ModelArguments, SFTConfig
from src.data.base_dataset import DEFAULT_CHAT_TEMPLATE




def get_current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def estimate_memory_requirements(
    model: PreTrainedModel,
    batch_size: int,
    sequence_length: int
) -> Dict[str, float]:
    """
    Оценивает требования к памяти для обучения модели.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)  # GB
    
    # Примерная оценка памяти для входных данных и градиентов
    dtype_size = 4  # float32
    if next(model.parameters()).dtype == torch.float16:
        dtype_size = 2  # float16
    
    batch_size_memory = (batch_size * sequence_length * model.config.hidden_size * dtype_size) / (1024**3)  # GB
    
    return {
        "model_size_gb": param_size,
        "batch_memory_gb": batch_size_memory,
        "estimated_total_gb": param_size + (batch_size_memory * 3)  # *3 для учета градиентов и оптимизатора
    }



def get_quantization_config(model_args: ModelArguments) -> Optional[BitsAndBytesConfig]:
    """Создает конфигурацию для квантования модели."""
    
    compute_dtype = torch.float32
    if model_args.torch_dtype:
        try:
            compute_dtype = getattr(torch, model_args.torch_dtype)
        except AttributeError:
            raise ValueError(f"Неподдерживаемый тип данных: {model_args.torch_dtype}")
    
    if model_args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage,
        ).to_dict()
    if model_args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True).to_dict()
    
    return None



def get_tokenizer(
    model_args: ModelArguments, 
    data_args: DataArguments, 
    auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Инициализирует и настраивает токенизатор."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path or model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
            token = data_args.hf_token
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки токенизатора: {str(e)}")
    
    # Установка pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Токенизатор не имеет eos_token. Укажите pad_token вручную.")
    
    # Настройка параметров усечения
    if data_args.truncation_side:
        if data_args.truncation_side not in ["left", "right"]:
            raise ValueError("truncation_side должен быть 'left' или 'right'")
        tokenizer.truncation_side = data_args.truncation_side
    
    # Установка шаблона чата
    if data_args.chat_template:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and not tokenizer.chat_template:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    
    return tokenizer


def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
    if model_args.use_peft is False:
        return None
    if model_args.peft_type.lower() == "lora":
        return LoraConfig(
            r=model_args.lora_r,                    
            lora_alpha=model_args.lora_alpha,      
            lora_dropout=model_args.lora_dropout,                            
            task_type=model_args.lora_task_type,                
            target_modules=model_args.lora_target_modules,  
            inference_mode=False,                  
        )
    
    # TODO: add other peft types
    else:
        raise NotImplementedError(f"PEFT type {model_args.peft_type} not implemented")
    



def is_adapter_model(model_name_or_path: str) -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path)
    except (HFValidationError, RepositoryNotFoundError):
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files


def get_checkpoint(training_args: SFTConfig | DPOConfig) -> Path | None:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Dict[str, int] | None:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None

def auto_find_batch_size(
    training_args: Union[SFTConfig, DPOConfig], 
    model: torch.nn.Module
) -> int:
    """Автоматически находит оптимальный размер батча на основе доступной памяти."""
    total_mem = torch.cuda.get_device_properties(get_current_device()).total_memory / (1024**3)
    
    # Эмпирическая формула для расчета батча
    approx_batch_size = int(total_mem * 0.8 / (model.num_parameters() / 1e9))
    return max(1, min(approx_batch_size, training_args.per_device_train_batch_size))

