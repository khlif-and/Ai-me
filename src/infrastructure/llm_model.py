from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.domain.interfaces.model_interface import ModelInterface
from src.domain.entities.chat_message import ChatMessage
from src.domain.entities.model_config import ModelConfig


class LLMModel(ModelInterface):
    def __init__(self, config: ModelConfig):
        self._config = config
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_id)
        quantization_config = self._build_quantization_config()
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=self._get_torch_dtype(),
        )

    def generate(self, messages: List[ChatMessage]) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        messages_dict = [msg.to_dict() for msg in messages]
        input_ids = self._tokenizer.apply_chat_template(
            messages_dict,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        outputs = self._model.generate(
            input_ids,
            max_new_tokens=self._config.max_new_tokens,
            eos_token_id=self._tokenizer.eos_token_id,
            do_sample=True,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
        )
        return self._tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        )

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if not self._config.load_in_4bit:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    def _get_torch_dtype(self):
        if self._config.load_in_4bit:
            return None
        return torch.float16
