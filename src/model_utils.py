import os
from pathlib import Path
from typing import Optional, Union, Dict

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint

from accelerate import Accelerator
from huggingface_hub import list_repo_files
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from peft import LoraConfig, PeftConfig

from src.config import DataArguments, DPOConfig, ModelArguments, SFTConfig
from src.data.base_dataset import DEFAULT_CHAT_TEMPLATE




def get_current_device() -> torch.device:
    if torch.cuda.is_available():
        accelerator = Accelerator()
        return torch.device(f"cuda:{accelerator.local_process_index}")
    return torch.device("cpu")



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

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
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
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Dict[str, int] | None:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None

def auto_find_batch_size(
    training_args: Union[SFTConfig, DPOConfig], 
    model: torch.nn.Module
) -> int:
    """Автоматически находит оптимальный размер батча на основе доступной памяти."""
    accelerator = Accelerator()
    total_mem = torch.cuda.get_device_properties(accelerator.device).total_memory / (1024**3)
    
    # Эмпирическая формула для расчета батча
    approx_batch_size = int(total_mem * 0.8 / (model.num_parameters() / 1e9))
    return max(1, min(approx_batch_size, training_args.per_device_train_batch_size))


def print_model_info(model: torch.nn.Module):
    """Выводит подробную информацию о модели."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Архитектура модели: {model.__class__.__name__}")
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Процент обучаемых параметров: {100 * trainable_params / total_params:.2f}%")
    
    if hasattr(model, "hf_device_map"):
        print("\nРаспределение по устройствам:")
        for device, layers in model.hf_device_map.items():
            print(f"{device}: {len(layers)} слоев")