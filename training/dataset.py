import torch
from torch.utils.data import Dataset


class CausalLMDataset(Dataset):
    """
    Dataset for causal language modeling.

    Each line is treated as independent text.
    Input:  tokens[:-1]
    Target: tokens[1:]
    """

    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.lines[idx])

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        pad_id = self.tokenizer.vocab[self.tokenizer.pad_token]

        input_ids = tokens[:-1]
        labels = tokens[1:]

        if len(input_ids) < self.max_length - 1:
            pad_len = (self.max_length - 1) - len(input_ids)
            input_ids += [pad_id] * pad_len
            labels += [pad_id] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
