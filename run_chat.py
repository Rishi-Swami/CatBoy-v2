import json
import torch
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPTModel
from inference.chat import chat_repl


CONFIG_PATH = "configs/model_config.json"
MODEL_PATH = "saved/gpt_chatbot_final.pth"
TOKENIZER_PATH = "saved/tokenizer.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------ Load config ------------------
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # ------------------ Load tokenizer ------------------
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    # ------------------ Load model ------------------
    model = GPTModel(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # ------------------ Chat ------------------
    chat_repl(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.get("max_new_tokens", 128),
        temperature=config.get("temperature", 0.9),
        top_k=config.get("top_k", 50),
        top_p=config.get("top_p", 0.0),
        repetition_penalty=config.get("repetition_penalty", 1.1),
    )


if __name__ == "__main__":
    main()
