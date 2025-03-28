"""
Data module for loading and processing datasets.

Available datasets:
- SFTDataset: Dataset for supervised fine-tuning
- TODO: Add DPO dataset
"""

from .base_dataset import BaseDataset
from .sft_dataset import SFTDataset
from .data_converters import (
    DataConverter,
    ChatMessageConverter,
    PreferredAnswerConverter,
    QAConverter
)

__all__ = [
    "BaseDataset",
    "SFTDataset",
    "DataConverter",
    "ChatMessageConverter",
    "PreferredAnswerConverter",
    "QAConverter"
]

