"""
Data module for loading and processing datasets.

Available datasets:
- SFTDataset: Dataset for supervised fine-tuning
- DPODataset: Dataset for Direct Preference Optimization
"""

from .base_dataset import BaseDataset
from .sft_dataset import SFTDataset
from .dpo_dataset import DPODataset
from .data_converters import (
    DataConverter,
    ChatMessageConverter,
    PreferredAnswerConverter,
    QAConverter
)

__all__ = [
    "BaseDataset",
    "DPODataset",
    "SFTDataset",
    "DataConverter",
    "ChatMessageConverter",
    "PreferredAnswerConverter",
    "QAConverter"
]

