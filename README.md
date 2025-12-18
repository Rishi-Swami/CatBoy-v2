# 🐾 CatBoy-v2

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

A compact, educational implementation of a **transformer-based causal language model (mini-GPT)** built entirely from scratch in **PyTorch**.  
Created for learning, experimentation, and a deeper understanding of how GPT-style models work.
**CatBoy-v2** is the next-generation conversational AI inspired by CatBoy-v1. It retains the charm of the original while introducing advanced features for more fluid and context-aware dialogue.

---

## 🚀 Key Improvements

| Feature | CatBoy-v1 | CatBoy-v2 |
|---------|-----------|-----------|
| **Model Architecture** | Decoder-only Transformer | Decoder-only Transformer (optimized) |
| **Parameters** | ~12M | 16.4M |
| **Tokenizer** | Basic BPE | Custom BPE with larger vocab & improved tokenization |
| **Context Length** | 128 tokens | 128 tokens (optimized for multi-turn dialogue) |
| **Training Dataset** | ~1.6k lines | Same dataset, now trained on GPU for faster convergence |
| **Training Device** | CPU | GPU / TPU compatible |
| **Dropout / Regularization** | Basic | Improved dropout & attention dropout for better generalization |
| **RMSNorm & RoPE** | Not included | Included for stable training and better sequence modeling |
| **Tie Embeddings** | Not tied | Embeddings tied for memory efficiency |
| **Advanced Sampling** | Basic greedy / top-k | Top-k, Top-p, temperature & repetition penalty configurable |
| **Response Quality** | Basic text | More coherent, context-aware responses with reduced token fragmentation |

---

## ⚡ Advanced Features in v2

- **RoPE (Rotary Positional Embeddings):** Handles longer sequences and multi-turn contexts.  
- **RMSNorm:** More stable normalization than traditional layer norm.  
- **Tied embeddings:** Reduces memory usage and improves embedding consistency.  
- **Flexible sampling:** Control generation behavior via `temperature`, `top_k`, `top_p`, and `repetition_penalty`.  
- **GPU/TPU Training:** Drastically faster training even on small datasets.  
- **Better Tokenization:** Custom BPE trained on full dataset prevents broken words seen in v1.

---

## ⚖️ Comparison with Industry Standards

| Model | Parameters | Dataset Size | Context Handling | Notes |
|-------|------------|--------------|-----------------|-------|
| **CatBoy-v2** | 16.4M | 1.6k lines | Multi-turn, 128 tokens | Lightweight, optimized for small-scale training and experimentation |
| **GPT-2 Small** | 117M | WebText (8M docs) | Multi-turn, 1024 tokens | Industry-standard, high-quality generation but heavy for local training |
| **DialoGPT Small** | 117M | Reddit conversations | Multi-turn, variable | High-quality dialogue but requires large dataset and GPU for training |

> **Takeaway:** CatBoy-v2 offers a **lightweight, GPU/TPU-friendly alternative** suitable for small datasets while incorporating techniques used in industry-standard models, like RoPE, RMSNorm, and advanced sampling.

---

## 🛠 Usage

For usage instructions, refer to the **CatBoy-v1 README**. CatBoy-v2 is fully backward-compatible with previous interaction scripts.

---

## ⚖️ Notes

- Trained on a small dataset (~1.6k lines), so responses may still have occasional quirks.  
- Designed for experimentation, multi-turn conversations, and rapid GPU/TPU training.  

---

> CatBoy-v2 | Meow responsibly. Keep coding curiously.

