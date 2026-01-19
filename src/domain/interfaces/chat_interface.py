from abc import ABC, abstractmethod


class ChatInterface(ABC):
    @abstractmethod
    def send_message(self, message: str) -> str:
        pass

    @abstractmethod
    def clear_history(self) -> None:
        pass
