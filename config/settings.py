import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    model_id: str
    groq_api_key: str
    max_new_tokens: int
    temperature: float
    top_p: float
    api_host: str
    api_port: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            model_id=os.getenv("MODEL_ID", "llama-3.1-70b-versatile"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "1024")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            top_p=float(os.getenv("TOP_P", "0.9")),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
        )
