from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class DataConverter(ABC):
    """Base class for converting between different data formats"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def normalize_role(self, role: str) -> str:
        """Normalize role to standard format for case-insensitive comparison"""
        return "assistant" if role.lower() == "assistant" else "user"
    
    def add_system_message(self, messages: List[Dict[str, str]], system_content: Optional[str] = None) -> None:
        """Add system message to the beginning of the list if not already present"""
        if system_content or self.config.get("default_system_prompt"):
            messages.insert(0, {
                "role": "system", 
                "content": system_content or self.config["default_system_prompt"]
            })
    
    def detect_format_variant(self, data: Dict) -> Dict[str, str]:
        """Detect format variant of data message or list of messages from config"""
        if not isinstance(data, (dict, list)):
            raise ValueError(f"Invalid data format: {type(data)}")

        sample = data[0] if isinstance(data, list) and len(data) > 0 else data

        for variant in self.config["format_variants"]:
            if all(field in sample for field in variant["required_fields"]):
                return variant

        raise ValueError(
            f"Could not detect format variant. Sample: {sample}. "
            f"Available variants: {self.config['format_variants']}"
        )

    @abstractmethod
    def convert_to_chat_format(self, data: Any) -> List[Dict[str, str]]:
        """Convert data to standard chat format with role/content"""
        pass

class ChatMessageConverter(DataConverter):
    """Converter for dialog format where each message has a role and content for  json  or  openai  format
    Examples of formats:
    1. [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    2. [{"from": "human", "value": "Hello"}, {"from": "assistant", "value": "Hi"}]
    """
    
    def convert_to_chat_format(self, data: Dict, auto_insert_empty_system_msg: bool = True) -> List[Dict[str, str]]:
        messages = []
        
        # Добавляем системное сообщение, если есть
        if auto_insert_empty_system_msg:
            system_content = data.get("system") or data.get("system_prompt")
            self.add_system_message(messages, system_content)

        # Находим колонку с сообщениями
        input_data = None
        for col in self.config["input_columns"]:
            if col in data:
                input_data = data[col]
                break

        if input_data is None:
            raise ValueError(f"No valid input column found. Expected one of: {self.config['input_columns']}")

        # Определяем формат (role/from, content/value и т.д.)
        format_variant = self.detect_format_variant(input_data)
        
        # Конвертируем сообщения
        for msg in input_data:
            content = msg[format_variant["content_field"]]
            if not content and self.config.get("skip_empty_messages", True):
                continue
                
            role = self.normalize_role(msg[format_variant["role_field"]])
            messages.append({"role": role, "content": content})
        
        return messages

class PreferredAnswerConverter(DataConverter):
    """Конвертер для формата PreferredAnswerConverter с chosen/rejected ответами
    Пример: {"prompt": "Hello", "chosen": "Hi", "rejected": "Go away"}
    """
    
    def convert_to_chat_format(self, data: Dict , auto_insert_empty_system_msg: bool = True) -> List[Dict[str, str]]:
        messages = []
        
        if auto_insert_empty_system_msg:
            self.add_system_message(messages, data.get("system"))
        
        required_fields = self.config["format_columns"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        if data.get("prompt"):
            messages.append({"role": "user", "content": data["prompt"]})
        messages.append({"role": "assistant", "content": data["chosen"]})
        
        # Добавляем rejected ответ, если это требуется
        if self.config.get("include_rejected", True):
            data["__rejected"] = data["rejected"]
            messages.append({"role": "assistant", "content": data["rejected"], "rejected": True})
        
        return messages

class QAConverter(DataConverter):
    """Конвертер для формата вопрос-ответ (Q&A)
    Примеры форматов:
    1. {"question": "What is Python?", "answer": "Python is a programming language"}
    2. {"q": "What is Python?", "a": "Python is a programming language"}
    3. {"input": "What is Python?", "output": "Python is a programming language"}
    4. {"prompt": "What is Python?", "response": "Python is a programming language"}
    """
    
    def convert_to_chat_format(self, data: Dict, auto_insert_empty_system_msg: bool = True) -> List[Dict[str, str]]:
        messages = []
        
        # Добавляем системное сообщение, если нужно
        if auto_insert_empty_system_msg:
            self.add_system_message(messages, data.get("system"))
        
        # Определяем поля вопроса и ответа
        question_fields = ["question", "q", "input", "prompt"]
        answer_fields = ["answer", "a", "output", "response"]
        
        # Ищем вопрос
        question = None
        for field in question_fields:
            if field in data:
                question = data[field]
                break
                
        # Ищем ответ
        answer = None
        for field in answer_fields:
            if field in data:
                answer = data[field]
                break
        
        if question is None or answer is None:
            raise ValueError(
                f"Missing question or answer fields. Expected one of:\n"
                f"Question fields: {question_fields}\n"
                f"Answer fields: {answer_fields}\n"
                f"Got data: {data}"
            )
        
        # Формируем сообщения
        messages.extend([
            {"role": "user", "content": str(question).strip()},
            {"role": "assistant", "content": str(answer).strip()}
        ])
        
        return messages

    