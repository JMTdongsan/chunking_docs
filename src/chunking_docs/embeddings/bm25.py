from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from chunking_docs.models import DocumentChunk

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25LexicalIndex:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(self.tokens)

    def search(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.chunks[index], float(score)) for index, score in ranked]

    def dump_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = [
            {"chunk_id": chunk.chunk_id, "tokens": tokens}
            for chunk, tokens in zip(self.chunks, self.tokens)
        ]
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
