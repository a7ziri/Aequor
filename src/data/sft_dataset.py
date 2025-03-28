from typing import Any , Dict
import  logging
from .base_dataset import BaseDataset
from src.config import DataArguments


logger = logging.getLogger(__name__)

class SFTDataset(BaseDataset):

    def __init__(self, data_args: DataArguments, tokenizer: Any):
        super().__init__(data_args=data_args, tokenizer=tokenizer)

    def _apply_chat_template(self, example: Dict, **kwargs) -> Dict:

        # Проверка типа сообщений
        if not isinstance(example['messages'], list):
            raise ValueError(f"Expected list of messages, got {type(example['messages'])}")
        

        texts= kwargs['tokenizer'].apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        if  kwargs['tokenizer'].bos_token is not None:
            if texts.startswith(kwargs['tokenizer'].bos_token):  
                texts = texts[len(kwargs['tokenizer'].bos_token):]

        return  kwargs['tokenizer'](texts, truncation=True, padding=True, max_length=self.data_args.tokenizer_max_seq_length)





       