from src.domain.interfaces.chat_interface import ChatInterface
from src.infrastructure.console_printer import ConsolePrinter


class CLIApp:
    EXIT_COMMANDS = ("quit", "exit")

    def __init__(self, chat: ChatInterface, printer: ConsolePrinter):
        self._chat = chat
        self._printer = printer

    def run(self) -> None:
        self._printer.hint("Type 'quit' or 'exit' to end the conversation.")
        while True:
            try:
                user_input = self._printer.user_prompt()
                if self._should_exit(user_input):
                    self._printer.info("Goodbye!")
                    break
                response = self._chat.send_message(user_input)
                self._printer.ai_response(response)
            except KeyboardInterrupt:
                self._printer.info("\nGoodbye!")
                break
            except Exception as e:
                self._printer.error(f"Error: {e}")

    def _should_exit(self, user_input: str) -> bool:
        return user_input.lower() in self.EXIT_COMMANDS
