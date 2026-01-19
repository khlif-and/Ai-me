import httpx
import time
from typing import List, Tuple
from src.domain.interfaces.model_interface import ModelInterface
from src.domain.entities.chat_message import ChatMessage
from src.domain.entities.model_config import ModelConfig


class GroqModel(ModelInterface):
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, config: ModelConfig, api_key: str):
        self._config = config
        self._api_key = api_key
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def generate(self, messages: List[ChatMessage]) -> str:
        response, _, _ = self.generate_with_reasoning(messages)
        return response

    def generate_with_reasoning(self, messages: List[ChatMessage]) -> Tuple[str, str, float]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        messages_dict = [msg.to_dict() for msg in messages]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._config.model_id,
            "messages": messages_dict,
            "max_completion_tokens": self._config.max_new_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "reasoning_effort": "medium",
            "stream": False,
        }

        start_time = time.time()

        with httpx.Client(timeout=120.0) as client:
            response = client.post(self.API_URL, headers=headers, json=payload)
            if response.status_code != 200:
                error_detail = response.text
                raise RuntimeError(f"Groq API error: {error_detail}")
            data = response.json()

        reasoning_time = time.time() - start_time

        choice = data["choices"][0]
        message = choice["message"]
        
        content = message.get("content", "")
        reasoning = message.get("reasoning", "")

        return content, reasoning, reasoning_time

    def is_loaded(self) -> bool:
        return self._loaded
