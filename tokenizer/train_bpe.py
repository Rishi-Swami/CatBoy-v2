import argparse
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer

def train_bpe_tokenizer(data_path, vocab_size, output_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=vocab_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train CatBoy-v2 BPE Tokenizer")
    parser.add_argument("--data", type=str, required=True, help="Path to training text file")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--output", type=str, default="saved/tokenizer.json")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=args.vocab_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    print(f"Tokenizer trained and saved to {output_path}")
    print(f"Vocab size: {len(tokenizer.vocab)}")


if __name__ == "__main__":
    main()
