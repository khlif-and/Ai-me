from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_id: str
    max_new_tokens: int
    temperature: float
    top_p: float
