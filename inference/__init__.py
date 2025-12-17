"""
Inference utilities for CatBoy-v2.

Provides:
- Sampler: top-k, top-p (nucleus), temperature, repetition penalty
- Simple KV-cache helper (present but generation uses full-context to ensure correctness)
- Generation loop that produces tokens autoregressively (streaming-capable)
- Chat REPL using the generation loop
"""
from .sampler import sample_next_token
from .generation import generate
from .chat import chat_repl

__all__ = ["sample_next_token", "generate", "chat_repl"]
