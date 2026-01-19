from typing import List
from src.domain.interfaces.chat_interface import ChatInterface
from src.domain.interfaces.model_interface import ModelInterface
from src.domain.entities.chat_message import ChatMessage, Role
from config.prompts import ROBI_SYSTEM_PROMPT


class ChatUseCase(ChatInterface):
    def __init__(self, model: ModelInterface):
        self._model = model
        self._history: List[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=ROBI_SYSTEM_PROMPT)
        ]

    def send_message(self, message: str) -> str:
        user_message = ChatMessage(role=Role.USER, content=message)
        self._history.append(user_message)
        response = self._model.generate(self._history)
        assistant_message = ChatMessage(role=Role.ASSISTANT, content=response)
        self._history.append(assistant_message)
        return response

    def send_message_with_reasoning(self, message: str) -> tuple:
        user_message = ChatMessage(role=Role.USER, content=message)
        self._history.append(user_message)
        response, reasoning, reasoning_time = self._model.generate_with_reasoning(self._history)
        assistant_message = ChatMessage(role=Role.ASSISTANT, content=response)
        self._history.append(assistant_message)
        return response, reasoning, reasoning_time

    def clear_history(self) -> None:
        self._history = [
            ChatMessage(role=Role.SYSTEM, content=ROBI_SYSTEM_PROMPT)
        ]
