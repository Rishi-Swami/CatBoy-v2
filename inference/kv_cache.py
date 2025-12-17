"""
KV-cache helpers.

NOTE:
- The current CatBoy-v2 model (as sealed earlier) computes RoPE in a per-forward fashion and
  does not accept a position offset for cached keys. To avoid incorrect rotary positions,
  the generation implementation re-evaluates the full context each step (no incremental cache).
- This module provides lightweight structures to store/inspect cache for future model upgrades.
"""

from typing import List, Optional, Tuple
import torch


KV = Optional[Tuple[torch.Tensor, torch.Tensor]]  # (k, v) per layer


def init_empty_cache(num_layers: int) -> List[KV]:
    """Create an empty cache (None per layer)."""
    return [None] * num_layers


def cache_to(device, cache: Optional[List[KV]]) -> Optional[List[KV]]:
    """Move cache tensors to device (no-op if None)."""
    if cache is None:
        return None
    out = []
    for kv in cache:
        if kv is None:
            out.append(None)
        else:
            k, v = kv
            out.append((k.to(device), v.to(device)))
    return out


def merge_cache(old_cache: List[KV], new_cache: List[KV]) -> List[KV]:
    """
    Merge two caches. This is a convenience helper: if model returns full (k,v) per layer,
    new_cache will replace old_cache for each layer. If more sophisticated incremental
    merging is required, update here.
    """
    if old_cache is None:
        return new_cache
    if new_cache is None:
        return old_cache
    merged = []
    for a, b in zip(old_cache, new_cache):
        # prefer new if provided
        merged.append(b if b is not None else a)
    return merged
