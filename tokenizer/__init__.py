"""
CatBoy-v2 Tokenizer Package

Implements a Byte Pair Encoding (BPE) tokenizer suitable for
causal language models (GPT-style).
"""

from .bpe_tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]
