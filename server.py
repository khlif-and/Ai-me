import uvicorn
from config.settings import Settings
from src.domain.entities.model_config import ModelConfig
from src.infrastructure.groq_model import GroqModel
from src.infrastructure.console_printer import ConsolePrinter
from src.usecases.chat_usecase import ChatUseCase
from src.presentation.api_server import APIServer


def create_app():
    settings = Settings.from_env()
    printer = ConsolePrinter()

    printer.info("Initializing Robi AI API...")

    if not settings.groq_api_key:
        printer.error("GROQ_API_KEY not set in .env file!")
        printer.warning("Get your free API key at: https://console.groq.com/keys")
        raise ValueError("GROQ_API_KEY is required")

    printer.warning(f"Using model: {settings.model_id}")

    model_config = ModelConfig(
        model_id=settings.model_id,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )

    model = GroqModel(model_config, settings.groq_api_key)
    model.load()
    printer.success("API ready!")

    chat = ChatUseCase(model)
    api_server = APIServer(chat, model)

    return api_server.app


app = create_app()


if __name__ == "__main__":
    settings = Settings.from_env()
    uvicorn.run(
        "server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
