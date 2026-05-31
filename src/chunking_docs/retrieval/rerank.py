from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from chunking_docs.embeddings.tokenizers import LexicalTokenizer, LexicalTokenizerConfig
from chunking_docs.models import DocumentChunk


class Reranker(Protocol):
    source: str

    def score(self, query: str, chunks: list[DocumentChunk]) -> list[float]:
        """Return relevance scores for chunks in the provided order."""


class LexicalOverlapReranker:
    source = "rerank:lexical"

    def __init__(self, tokenizer_config: LexicalTokenizerConfig | None = None):
        self.tokenizer = LexicalTokenizer(tokenizer_config)

    def score(self, query: str, chunks: list[DocumentChunk]) -> list[float]:
        query_tokens = set(self.tokenizer.tokenize(query))
        if not query_tokens:
            return [0.0 for _ in chunks]
        scores = []
        for chunk in chunks:
            chunk_tokens = set(self.tokenizer.tokenize(chunk.text))
            overlap = query_tokens & chunk_tokens
            if not overlap:
                scores.append(0.0)
                continue
            query_coverage = len(overlap) / len(query_tokens)
            jaccard = len(overlap) / len(query_tokens | chunk_tokens)
            scores.append(query_coverage + jaccard)
        return scores


class SentenceTransformerCrossEncoderReranker:
    source = "rerank:cross_encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        max_length: int | None = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional package.
            raise RuntimeError("Install chunking-docs[embeddings] to use cross-encoder reranking") from exc

        kwargs = {"device": device}
        if max_length is not None:
            kwargs["max_length"] = max_length
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = CrossEncoder(model_name, **kwargs)

    def score(self, query: str, chunks: list[DocumentChunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        return [float(score) for score in scores]


def rerank_hits(query: str, hits: list, reranker: Reranker | None, top_k: int) -> list:
    if reranker is None:
        return hits[:top_k]
    chunks = [hit.chunk for hit in hits if getattr(hit, "chunk", None) is not None]
    scores = reranker.score(query, chunks)
    chunk_score_pairs = iter(scores)
    scored_hits = []
    for index, hit in enumerate(hits):
        chunk = getattr(hit, "chunk", None)
        if chunk is None:
            score = float("-inf")
        else:
            score = next(chunk_score_pairs)
        scored_hits.append((score, index, with_rerank_score(hit, score, reranker.source)))
    scored_hits.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [hit for score, _, hit in scored_hits[:top_k] if score != float("-inf")]


def with_rerank_score(hit, score: float, source: str):
    sources = list(getattr(hit, "sources", []))
    if source not in sources:
        sources.append(source)
    return replace(hit, score=score, sources=sources)
