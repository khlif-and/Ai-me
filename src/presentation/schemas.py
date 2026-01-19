from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    clear_history: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    reasoning: Optional[str] = None
    reasoning_time: Optional[float] = None
    success: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
