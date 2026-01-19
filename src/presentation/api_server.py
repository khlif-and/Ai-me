from fastapi import FastAPI, HTTPException
from src.domain.interfaces.model_interface import ModelInterface
from src.usecases.chat_usecase import ChatUseCase
from src.presentation.schemas import ChatRequest, ChatResponse, HealthResponse


class APIServer:
    def __init__(self, chat: ChatUseCase, model: ModelInterface):
        self._chat = chat
        self._model = model
        self._app = FastAPI(
            title="Robi AI API",
            description="Personal AI Assistant API with Reasoning",
            version="1.0.0",
        )
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self._app.get("/health", response_model=HealthResponse)
        async def health_check():
            return HealthResponse(
                status="ok",
                model_loaded=self._model.is_loaded(),
            )

        @self._app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            try:
                if request.clear_history:
                    self._chat.clear_history()
                response, reasoning, reasoning_time = self._chat.send_message_with_reasoning(request.message)
                return ChatResponse(
                    response=response,
                    reasoning=reasoning if reasoning else None,
                    reasoning_time=round(reasoning_time, 2) if reasoning_time else None,
                    success=True,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self._app.post("/clear")
        async def clear_history():
            self._chat.clear_history()
            return {"message": "History cleared", "success": True}

    @property
    def app(self) -> FastAPI:
        return self._app
