from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

TokenizerStrategy = Literal["word", "char_ngram", "mixed"]

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_]+")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


class LexicalTokenizerConfig(BaseModel):
    strategy: TokenizerStrategy = "mixed"
    lowercase: bool = True
    min_n: int = 2
    max_n: int = 4
    ngram_cjk_only: bool = True
    deduplicate: bool = False


class LexicalTokenizer:
    def __init__(self, config: LexicalTokenizerConfig | None = None):
        self.config = config or LexicalTokenizerConfig()

    def tokenize(self, text: str) -> list[str]:
        if self.config.strategy == "word":
            return maybe_deduplicate(
                word_tokens(text, lowercase=self.config.lowercase),
                deduplicate=self.config.deduplicate,
            )
        if self.config.strategy == "char_ngram":
            return maybe_deduplicate(
                char_ngram_tokens(
                    text,
                    min_n=self.config.min_n,
                    max_n=self.config.max_n,
                    lowercase=self.config.lowercase,
                    cjk_only=self.config.ngram_cjk_only,
                ),
                deduplicate=self.config.deduplicate,
            )
        if self.config.strategy == "mixed":
            return maybe_deduplicate(
                word_tokens(text, lowercase=self.config.lowercase)
                + char_ngram_tokens(
                    text,
                    min_n=self.config.min_n,
                    max_n=self.config.max_n,
                    lowercase=self.config.lowercase,
                    cjk_only=self.config.ngram_cjk_only,
                ),
                deduplicate=self.config.deduplicate,
            )
        raise ValueError(f"Unsupported tokenizer strategy: {self.config.strategy}")


def word_tokens(text: str, lowercase: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if lowercase:
        return [token.lower() for token in tokens]
    return tokens


def char_ngram_tokens(
    text: str,
    min_n: int = 2,
    max_n: int = 4,
    lowercase: bool = True,
    cjk_only: bool = True,
) -> list[str]:
    if min_n <= 0 or max_n < min_n:
        raise ValueError("Expected 0 < min_n <= max_n")
    tokens = word_tokens(text, lowercase=lowercase)
    ngrams: list[str] = []
    for token in tokens:
        if cjk_only and not CJK_RE.search(token):
            continue
        ngrams.extend(token_ngrams(token, min_n=min_n, max_n=max_n))
    return ngrams


def token_ngrams(token: str, min_n: int, max_n: int) -> list[str]:
    results = []
    upper = min(max_n, len(token))
    for size in range(min_n, upper + 1):
        for start in range(0, len(token) - size + 1):
            results.append(token[start : start + size])
    return results


def stable_unique(tokens: list[str]) -> list[str]:
    seen = set()
    unique = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique


def maybe_deduplicate(tokens: list[str], deduplicate: bool) -> list[str]:
    return stable_unique(tokens) if deduplicate else tokens
