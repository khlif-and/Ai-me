from abc import ABC, abstractmethod
from typing import List
from src.domain.entities.chat_message import ChatMessage


class ModelInterface(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(self, messages: List[ChatMessage]) -> str:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass
