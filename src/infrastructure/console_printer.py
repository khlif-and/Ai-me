from colorama import init, Fore, Style


class ConsolePrinter:
    def __init__(self):
        init()

    def info(self, message: str) -> None:
        print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")

    def warning(self, message: str) -> None:
        print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")

    def success(self, message: str) -> None:
        print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")

    def error(self, message: str) -> None:
        print(f"{Fore.RED}{message}{Style.RESET_ALL}")

    def user_prompt(self) -> str:
        return input(f"{Fore.BLUE}You: {Style.RESET_ALL}")

    def ai_response(self, response: str) -> None:
        print(f"{Fore.GREEN}AI: {Style.RESET_ALL}{response}\n")

    def hint(self, message: str) -> None:
        print(f"{Fore.MAGENTA}{message}{Style.RESET_ALL}\n")
