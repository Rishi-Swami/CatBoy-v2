import torch.nn as nn

from .embeddings import RotaryEmbedding
from .transformer_block import TransformerBlock, RMSNorm


class GPTModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.vocab_size = config["vocab_size"]
        self.embed_dim = config["embed_dim"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.max_positions = config["max_positions"]

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        rope = RotaryEmbedding(self.embed_dim // self.num_heads, self.max_positions)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    rope=rope,
                    ff_mult=config["ff_mult"],
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, past_kv=None):
        B, T = input_ids.shape

        x = self.token_embedding(input_ids)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            pkv = None if past_kv is None else past_kv[i]
            x, kv = block(x, pkv)
            new_kvs.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_kvs
