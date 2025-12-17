"""
Generation utilities for CatBoy-v2.

This generation loop is architecturally correct and simple:
- Uses full-context autoregressive generation each step (no incremental KV cache used),
  ensuring Rotary embeddings are applied consistently (important given the sealed model).
- Supports temperature, top-k, top-p, repetition_penalty, and early stop on EOS.
"""

from typing import List, Optional
import torch
from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPTModel
from .sampler import sample_next_token

DEFAULT_DEVICE = None  # auto-detect in functions


def _ensure_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def generate(
    model: GPTModel,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    eos_token: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    Autoregressive generation (full context each step).

    Args:
        model: loaded GPTModel (in eval mode)
        tokenizer: trained BPETokenizer
        prompt: raw text prompt
        max_new_tokens: maximum tokens to generate
        temperature, top_k, top_p, repetition_penalty: sampling params
        eos_token: optional string token (defaults to tokenizer.eos_token)
        device: torch device

    Returns:
        generated text (decoded)
    """
    device = _ensure_device(device)
    model.to(device)
    model.eval()

    eos_token = eos_token or tokenizer.eos_token
    eos_id = tokenizer.vocab.get(eos_token, None)

    # prepare initial token ids (do not double-add BOS/EOS)
    # The tokenizer.encode default expects add_special_tokens True; call False and add BOS manually.
    try:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    except TypeError:
        # backward compatibility if encode signature differs
        prompt_ids = tokenizer.encode(prompt)

    # Prepend BOS if present in vocab
    if tokenizer.bos_token in tokenizer.vocab:
        bos_id = tokenizer.vocab[tokenizer.bos_token]
        context_ids: List[int] = [bos_id] + prompt_ids
    else:
        context_ids = prompt_ids.copy()

    generated_ids: List[int] = context_ids.copy()

    # We will re-evaluate the full context each step to keep RoPE consistent.
    with torch.no_grad():
        for step in range(max_new_tokens):
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
            logits, _ = model(input_ids)  # logits: (1, seq_len, vocab)
            next_logits = logits[0, -1, :].detach().cpu()

            next_id = sample_next_token(
                next_logits,
                generated_tokens=generated_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=next_logits.device,
            )

            generated_ids.append(int(next_id))

            if eos_id is not None and next_id == eos_id:
                break

    # decode (skip special tokens)
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded
