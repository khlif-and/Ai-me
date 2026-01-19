from huggingface_hub import login
from config.settings import Settings
from src.domain.entities.model_config import ModelConfig
from src.infrastructure.llm_model import LLMModel
from src.infrastructure.console_printer import ConsolePrinter
from src.usecases.chat_usecase import ChatUseCase
from src.presentation.cli_app import CLIApp


def main():
    settings = Settings.from_env()
    printer = ConsolePrinter()

    printer.info("Initializing AI CLI...")

    if settings.hf_auth_token:
        login(token=settings.hf_auth_token)

    printer.warning(f"Loading model: {settings.model_id}... (This may take a while)")

    model_config = ModelConfig(
        model_id=settings.model_id,
        load_in_4bit=settings.load_in_4bit,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )

    try:
        model = LLMModel(model_config)
        model.load()
        printer.success("Model loaded successfully!")

        chat = ChatUseCase(model)
        app = CLIApp(chat, printer)
        app.run()
    except Exception as e:
        printer.error(f"Failed to load model: {e}")
        print("Please ensure you have access to the model and a valid Hugging Face token.")


if __name__ == "__main__":
    main()
