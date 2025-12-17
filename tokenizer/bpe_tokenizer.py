import json
import re
from collections import Counter
from typing import List, Dict


# noinspection PyTypeChecker
class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer.

    Design goals:
    - Deterministic
    - Minimal dependencies
    - Fully serializable
    - GPT-compatible token flow
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.bpe_merges: List[tuple] = []

        # Special tokens (fixed IDs for safety)
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self._initialized = False

    # ------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------

    def _basic_tokenize(self, text: str) -> List[str]:
        """
        Initial tokenization using regex.
        Splits text into words and punctuation.
        """
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def _get_stats(self, corpus: List[List[str]]) -> Counter:
        """
        Count symbol pair frequencies.
        """
        pairs = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _merge_pair(self, pair, corpus):
        """
        Merge a frequent pair in the corpus.
        """
        merged = []
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(rf"(?<!\S){bigram}(?!\S)")

        for word in corpus:
            joined = " ".join(word)
            joined = pattern.sub("".join(pair), joined)
            merged.append(joined.split())
        return merged

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    def train(self, texts: List[str], vocab_size: int = 30000):
        """
        Train BPE tokenizer from raw texts.
        """
        assert vocab_size > len(self.special_tokens), "vocab_size too small"

        # Step 1: initial corpus (character-level words)
        corpus = []
        for text in texts:
            for token in self._basic_tokenize(text):
                corpus.append(list(token) + ["</w>"])

        # Step 2: BPE merges
        while True:
            stats = self._get_stats(corpus)
            if not stats:
                break

            best = stats.most_common(1)[0][0]
            corpus = self._merge_pair(best, corpus)
            self.bpe_merges.append(best)

            # Stop if vocab target reached
            vocab = set()
            for word in corpus:
                vocab.update(word)
            if len(vocab) + len(self.special_tokens) >= vocab_size:
                break

        # Step 3: Build vocabulary
        tokens = set()
        for word in corpus:
            tokens.update(word)

        tokens = sorted(tokens)
        full_vocab = self.special_tokens + tokens

        self.vocab = {tok: idx for idx, tok in enumerate(full_vocab)}
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}

        self._initialized = True

    # ------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs.
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer not trained or loaded")

        tokens = []
        if add_special_tokens:
            tokens.append(self.vocab[self.bos_token])

        for word in self._basic_tokenize(text):
            chars = list(word) + ["</w>"]
            for merge in self.bpe_merges:
                i = 0
                while i < len(chars) - 1:
                    if (chars[i], chars[i + 1]) == merge:
                        chars[i:i + 2] = ["".join(merge)]
                    else:
                        i += 1
            for token in chars:
                tokens.append(self.vocab.get(token, self.vocab[self.unk_token]))

        if add_special_tokens:
            tokens.append(self.vocab[self.eos_token])

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer not trained or loaded")

        tokens = []
        for idx in token_ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        text = "".join(tokens)
        return text.replace("</w>", " ").strip()

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------

    def save(self, path: str):
        """
        Save tokenizer to disk.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "bpe_merges": self.bpe_merges,
                    "special_tokens": self.special_tokens,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str):
        """
        Load tokenizer from disk.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = {k: int(v) for k, v in data["vocab"].items()}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.bpe_merges = [tuple(pair) for pair in data["bpe_merges"]]
        self.special_tokens = data["special_tokens"]

        self.pad_token, self.unk_token, self.bos_token, self.eos_token = self.special_tokens
        self._initialized = True
