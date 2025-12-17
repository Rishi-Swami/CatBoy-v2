import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_positions=None):
        super().__init__()
        self.dim = dim
        # Only half of dim used for cos/sin interleaving
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("cos", torch.cos(inv_freq))  # shape [D_head/2]
        self.register_buffer("sin", torch.sin(inv_freq))  # shape [D_head/2]

    def forward(self, x):
        # x: [B, H, T, D_head]
        B, H, T, D_head = x.shape
        assert D_head == self.dim, f"RoPE head dim mismatch: {D_head} vs {self.dim}"

        # Take first D_head//2 for cosine and sine
        cos = self.cos[:D_head//2].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,D_head//2]
        sin = self.sin[:D_head//2].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,D_head//2]

        # Expand to match q/k shape
        cos = cos.expand(1, H, T, D_head//2)
        sin = sin.expand(1, H, T, D_head//2)

        # Split x into even and odd dimensions for interleaving
        x_even = x[..., ::2]  # [B,H,T,D_head/2]
        x_odd  = x[..., 1::2] # [B,H,T,D_head/2]

        # Apply rotary transformation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd  = x_even * sin + x_odd * cos

        # Interleave back to original shape
        x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)

        return x_out
