"""
Sampling utilities for CatBoy-v2.

Functions:
- sample_next_token: given logits (1D tensor), sample a next token id using
  temperature, top-k, top-p (nucleus), and repetition penalty.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def top_k_filter(logits: torch.Tensor, k: int):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_value = values[-1]
    return torch.where(logits < min_value, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: torch.Tensor, p: float):
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find cutoff where cumulative_probs > p
    cutoff_idx = torch.searchsorted(cumulative_probs, p)
    # mask tokens beyond cutoff
    cutoff_val = sorted_logits[cutoff_idx] if cutoff_idx < sorted_logits.size(0) else sorted_logits[-1]
    return torch.where(logits < cutoff_val, torch.full_like(logits, float("-inf")), logits)


def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: Optional[List[int]], penalty: float):
    """
    Implements repetition penalty from CTRL / Transformer-XL. For tokens in generated_tokens,
    adjust logits: if logit < 0 -> multiply by penalty, else divide by penalty.
    """
    if penalty == 1.0 or not generated_tokens:
        return logits
    logits = logits.clone()
    for tok in set(generated_tokens):
        if tok < 0 or tok >= logits.size(0):
            continue
        if logits[tok] < 0:
            logits[tok] = logits[tok] * penalty
        else:
            logits[tok] = logits[tok] / penalty
    return logits


def sample_next_token(
    logits: torch.Tensor,
    generated_tokens: Optional[List[int]] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> int:
    """
    Sample next token id from logits.

    Args:
        logits: 1D tensor (vocab,)
        generated_tokens: list of previously generated token ids (for repetition penalty)
        temperature: scaling factor (>0)
        top_k: keep only top_k tokens (0 = disabled)
        top_p: nucleus sampling probability (0.0 = disabled)
        repetition_penalty: >1.0 to discourage repetition
        device: torch device

    Returns:
        int token_id
    """
    device = device or logits.device
    logits = logits.to(device)

    # apply repetition penalty
    logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

    # temperature
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature

    # top-k
    logits = top_k_filter(logits, top_k)

    # top-p
    logits = top_p_filter(logits, top_p)

    # convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # numerical safety: if all -inf or NaN, fallback to uniform
    if torch.isnan(probs).any() or torch.isinf(probs).all():
        probs = torch.ones_like(probs) / probs.size(0)

    # sample
    next_token = torch.multinomial(probs, num_samples=1).item()
    return int(next_token)
