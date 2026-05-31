from __future__ import annotations

import json
import math
from pathlib import Path

from rank_bm25 import BM25Okapi

from chunking_docs.embeddings.records import asset_text_parts
from chunking_docs.embeddings.tokenizers import LexicalTokenizer, LexicalTokenizerConfig
from chunking_docs.models import DocumentChunk, VisualAsset


class BM25LexicalIndex:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        tokenizer_config: LexicalTokenizerConfig | None = None,
        texts: list[str] | None = None,
    ):
        if texts is not None and len(texts) != len(chunks):
            raise ValueError("BM25 text count must match chunk count")
        self.chunks = chunks
        self.texts = texts or [chunk.text for chunk in chunks]
        self.tokenizer = LexicalTokenizer(tokenizer_config)
        self.tokenizer_config = self.tokenizer.config
        self.tokens = [self.tokenizer.tokenize(text) for text in self.texts]
        self.index = BM25Okapi(self.tokens)

    def search(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        query_tokens = self.tokenizer.tokenize(query)
        query_token_set = set(query_tokens)
        scores = self.index.get_scores(query_tokens)
        adjusted_scores = [
            finite_score(score) + lexical_overlap(query_token_set, tokens)
            for score, tokens in zip(scores, self.tokens)
        ]
        ranked = sorted(
            [(index, score) for index, score in enumerate(adjusted_scores) if score > 0],
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [(self.chunks[index], float(score)) for index, score in ranked]

    def dump_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "tokenizer": self.tokenizer_config.model_dump(),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text_char_count": len(text),
                    "tokens": tokens,
                }
                for chunk, text, tokens in zip(self.chunks, self.texts, self.tokens)
            ],
        }
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_lexical_texts(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset] | None = None,
) -> list[str]:
    asset_by_id = {asset.asset_id: asset for asset in assets or []}
    return [chunk_lexical_text(chunk, asset_by_id) for chunk in chunks]


def chunk_lexical_text(
    chunk: DocumentChunk,
    asset_by_id: dict[str, VisualAsset],
) -> str:
    parts = [chunk.text]
    for asset_id in chunk.asset_ids:
        asset = asset_by_id.get(asset_id)
        if asset is None:
            continue
        parts.extend(asset_text_parts(asset))
    return "\n".join(deduplicate_text_parts(parts))


def deduplicate_text_parts(parts: list[str]) -> list[str]:
    selected = []
    seen = set()
    for part in parts:
        normalized = " ".join(part.split())
        if not normalized or normalized in seen:
            continue
        selected.append(part)
        seen.add(normalized)
    return selected


def lexical_overlap(query_tokens: set[str], document_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = query_tokens.intersection(document_tokens)
    return len(overlap) / len(query_tokens)


def finite_score(score: float) -> float:
    value = float(score)
    return value if math.isfinite(value) and value > 0.0 else 0.0
