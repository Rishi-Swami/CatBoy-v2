import torch
import torch.nn as nn
from .attention import CausalSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(torch.nn.functional.silu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, rope, ff_mult=4):
        super().__init__()

        self.attn_norm = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, rope)

        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, embed_dim * ff_mult)

    def forward(self, x, past_kv=None):
        attn_out, kv = self.attn(self.attn_norm(x), past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv
