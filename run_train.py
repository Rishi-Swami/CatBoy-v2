import json
from pathlib import Path
import torch

from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.train_bpe import train_bpe_tokenizer
from model.gpt import GPTModel
from training.dataset import CausalLMDataset
from training.trainer import train


CONFIG_PATH = "configs/model_config.json"
DATASET_PATH = "dataset/training_data.txt"
SAVE_DIR = Path("saved")
TOKENIZER_PATH = SAVE_DIR / "tokenizer.json"
MODEL_PATH = SAVE_DIR / "gpt_chatbot_final.pth"


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load config ----------
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------- Tokenizer ----------
    tokenizer = BPETokenizer()

    if not TOKENIZER_PATH.exists():
        print("[INFO] Training BPE tokenizer...")
        train_bpe_tokenizer(
            DATASET_PATH,
            config["vocab_size"],
            str(TOKENIZER_PATH),
        )
        print("[INFO] Tokenizer trained.")

    tokenizer.load(str(TOKENIZER_PATH))

    # ---------- Dataset ----------
    dataset = CausalLMDataset(
        file_path=DATASET_PATH,
        tokenizer=tokenizer,
        max_length=config["block_size"],
    )

    # ---------- Model ----------
    model = GPTModel(config).to(device)
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- Training ----------
    print("[INFO] Starting training...")
    train(
        model=model,
        dataset=dataset,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        lr=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
        save_path=str(MODEL_PATH),
    )

    print("[INFO] Training complete.")
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()

