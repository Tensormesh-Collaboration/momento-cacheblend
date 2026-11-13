"""Tokenizers and chunk hashing shared by the generator and the simulator.

The live replay sends chat messages and lets vLLM apply the model's real chat
template. Everything offline (token counts, context clearing, the simulator)
renders messages the way Qwen-family ChatML templates render plain-text
messages, one message at a time:

    <|im_start|>{role}\n{content}<|im_end|>\n

Per-message rendering keeps message boundaries on token boundaries, which the
real template also does (the markers are special tokens).
"""

from __future__ import annotations

from functools import lru_cache
import hashlib

import numpy as np

CHUNK = 256  # LMCache / CacheBlend chunk size (fixed for CacheBlend)


class Tokenizer:
    """Hugging Face tokenizer when available, else tiktoken cl100k as a proxy."""

    def __init__(self, name: str | None):
        self.name = name or "cl100k_base"
        self.exact = False
        self._hf = None
        self._tk = None
        self._chars = name == "chars"  # one token per character: offline tests only
        if self._chars:
            return
        if name and name != "cl100k_base":
            try:
                from transformers import AutoTokenizer  # type: ignore

                self._hf = AutoTokenizer.from_pretrained(name)
                self.exact = True
            except Exception as exc:  # noqa: BLE001 - fall back, but say so
                print(f"[tokens] could not load tokenizer {name!r} ({exc}); using cl100k_base as a proxy")
        if self._hf is None:
            import tiktoken  # type: ignore

            self._tk = tiktoken.get_encoding("cl100k_base")
            self.name = "cl100k_base"

    def encode(self, text: str) -> list[int]:
        if self._chars:
            return [ord(c) for c in text]
        if self._hf is not None:
            return self._hf.encode(text, add_special_tokens=False)
        return self._tk.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        if self._chars:
            return "".join(map(chr, ids))
        if self._hf is not None:
            return self._hf.decode(ids)
        return self._tk.decode(ids)

    def message(self, role: str, content: str) -> list[int]:
        """Tokens of one rendered ChatML message."""
        return self.encode(f"<|im_start|>{role}\n") + self.encode(content) + self.encode("<|im_end|>\n")


# --- chunk hashing -----------------------------------------------------------
#
# Polynomial hash of every CHUNK-token window, vectorised with uint64
# wrap-around arithmetic: H(i) = sum_j t[i+j] * B^(C-1-j)  (mod 2^64).
# With S[k] = sum_{j<k} t[j] * Binv^j, H(i) = B^(i+C-1) * (S[i+C] - S[i]).
# B is odd, so it is invertible mod 2^64.

_B = np.uint64(0x9E3779B97F4A7C15)


def _inv_mod_2_64(b: int) -> int:
    return pow(b, -1, 1 << 64)


_BINV = np.uint64(_inv_mod_2_64(int(_B)))


@lru_cache(maxsize=8)
def _powers(n: int) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore"):
        bp = np.cumprod(np.full(n + 1, _B, dtype=np.uint64))
        bp = np.concatenate(([np.uint64(1)], bp[:-1]))
        bi = np.cumprod(np.full(n + 1, _BINV, dtype=np.uint64))
        bi = np.concatenate(([np.uint64(1)], bi[:-1]))
    return bp, bi


def window_hashes(tokens: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """Hash of every length-`chunk` window; empty if the sequence is shorter."""
    n = len(tokens)
    if n < chunk:
        return np.zeros(0, dtype=np.uint64)
    size = 1 << max(12, (n + 1).bit_length())  # cache powers by size class
    bp, bi = _powers(size)
    t = tokens.astype(np.uint64) + np.uint64(1)  # avoid all-zero windows hashing to 0
    with np.errstate(over="ignore"):
        s = np.concatenate(([np.uint64(0)], np.cumsum(t * bi[:n], dtype=np.uint64)))
        diff = s[chunk:] - s[: n - chunk + 1]
        return diff * bp[chunk - 1 : n]


def aligned_chunk_hashes(tokens: np.ndarray, chunk: int = CHUNK) -> list[int]:
    """Content hashes of the stored chunks: windows at 0, C, 2C, ... (full chunks only)."""
    wh = window_hashes(tokens, chunk)
    return [int(wh[i]) for i in range(0, len(wh), chunk)] if len(wh) else []


def chain(prev: int, content: int) -> int:
    """Prefix-chained chunk key (the same text after a different prefix is a different key)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(prev.to_bytes(8, "little"))
    h.update(content.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")
