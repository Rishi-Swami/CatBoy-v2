"""
Chat REPL for CatBoy-v2.

- Loads model and tokenizer from disk
- Runs generation with reasonable defaults
- Prints assistant responses
"""

import argparse
from pathlib import Path
import torch
from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPTModel
from .generation import generate


DEFAULT_MODEL_PATH = "saved/gpt_chatbot_final.pth"
DEFAULT_TOKENIZER_PATH = "saved/tokenizer.json"
DEFAULT_CONFIG_PATH = "configs/model_config.json"


def load_tokenizer(tokenizer_path: str) -> BPETokenizer:
    t = BPETokenizer()
    t.load(tokenizer_path)
    return t


def load_model(config: dict, model_path: str, device=None) -> GPTModel:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = GPTModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def chat_repl(
    model: GPTModel,
    tokenizer: BPETokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.0,
    repetition_penalty: float = 1.1,
):
    print("CatBoy-v2: Ready to chat. Type 'exit' or 'quit' or 'bye' to leave.")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "bye"):
            print("CatBoy-v2: Bye!")
            break

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=user,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        print("CatBoy-v2:", response)


def main():
    parser = argparse.ArgumentParser(description="Chat with CatBoy-v2")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import json
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(config, args.model, device=args.device)

    chat_repl(model, tokenizer)


if __name__ == "__main__":
    main()
