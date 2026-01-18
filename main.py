import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from colorama import init, Fore, Style

# Initialize colorama
init()

def setup_args():
    parser = argparse.ArgumentParser(description="AI CLI with Llama 3.1 8B Instruct")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID")
    parser.add_argument("--auth_token", type=str, help="Hugging Face API Token")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    return parser.parse_args()

def main():
    args = setup_args()

    print(f"{Fore.CYAN}Initializing AI CLI...{Style.RESET_ALL}")

    if args.auth_token:
        login(token=args.auth_token)
    
    print(f"{Fore.YELLOW}Loading model: {args.model}... (This may take a while){Style.RESET_ALL}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        quantization_config = None
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if not args.load_in_4bit else None
        )
        
        print(f"{Fore.GREEN}Model loaded successfully!{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Type 'quit' or 'exit' to end the conversation.{Style.RESET_ALL}\n")

        chat_history = []

        while True:
            try:
                user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
                if user_input.lower() in ["quit", "exit"]:
                    print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                    break

                chat_history.append({"role": "user", "content": user_input})

                input_ids = tokenizer.apply_chat_template(
                    chat_history,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

                response = inputs = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
                
                print(f"{Fore.GREEN}AI: {Style.RESET_ALL}{response}\n")
                
                chat_history.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Failed to load model: {e}{Style.RESET_ALL}")
        print("Please ensure you have access to the model and a valid Hugging Face token.")

if __name__ == "__main__":
    main()
