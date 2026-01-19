# AI CLI

CLI application for chatting with LLM models using Hugging Face Transformers.

## Project Structure

```
AI/
├── src/
│   ├── domain/          # Entities and interfaces
│   ├── infrastructure/  # External implementations
│   ├── usecases/        # Business logic
│   └── presentation/    # CLI interface
├── config/              # Configuration management
├── main.py              # Entry point
├── .env                 # Environment variables
└── requirements.txt     # Dependencies
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_ID | Hugging Face model ID | Qwen/Qwen2.5-1.5B-Instruct |
| HF_AUTH_TOKEN | Hugging Face API token | - |
| LOAD_IN_4BIT | Enable 4-bit quantization | false |
| MAX_NEW_TOKENS | Maximum tokens to generate | 512 |
| TEMPERATURE | Sampling temperature | 0.6 |
| TOP_P | Top-p sampling value | 0.9 |

## Usage

```bash
python main.py
```

Type your message and press Enter. Type `quit` or `exit` to end.
