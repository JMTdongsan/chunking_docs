from __future__ import annotations

import math
from dataclasses import dataclass

from chunking_docs.embeddings.bm25 import BM25LexicalIndex
from chunking_docs.embeddings.interfaces import DenseTextEmbedder
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.graph.export import related_terms
from chunking_docs.models import DocumentChunk, GraphTriple
from chunking_docs.retrieval.fusion import RankedHit, reciprocal_rank_fusion


@dataclass(frozen=True)
class HybridSearchHit:
    chunk: DocumentChunk
    score: float
    sources: list[str]


class LocalHybridSearcher:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedder: DenseTextEmbedder,
        triples: list[GraphTriple] | None = None,
        tokenizer_config: LexicalTokenizerConfig | None = None,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25LexicalIndex(chunks, tokenizer_config=tokenizer_config)
        self.chunk_vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        self.triples = triples or []

    def search(
        self,
        query: str,
        top_k: int = 10,
        graph_expand: bool = False,
    ) -> list[HybridSearchHit]:
        expanded_query = self._expanded_query(query) if graph_expand else query
        dense_hits = self._dense_hits(expanded_query, top_k=max(top_k * 3, 20))
        bm25_hits = self._bm25_hits(expanded_query, top_k=max(top_k * 3, 20))
        result_sets = [dense_hits, bm25_hits]
        if graph_expand:
            graph_hits = self._graph_hits(query, top_k=max(top_k * 3, 20))
            result_sets.append(graph_hits)
        fused = reciprocal_rank_fusion(result_sets, top_k=top_k)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        return [
            HybridSearchHit(chunk=chunk_by_id[item_id], score=score, sources=sources)
            for item_id, score, sources in fused
            if item_id in chunk_by_id
        ]

    def _dense_hits(self, query: str, top_k: int) -> list[RankedHit]:
        query_vector = self.embedder.embed_texts([query])[0]
        scored = [
            (chunk.chunk_id, cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.chunk_vectors)
        ]
        ranked = sorted(
            [(chunk_id, score) for chunk_id, score in scored if score > 0],
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [
            RankedHit(item_id=chunk_id, rank=index + 1, score=score, source="dense")
            for index, (chunk_id, score) in enumerate(ranked)
        ]

    def _bm25_hits(self, query: str, top_k: int) -> list[RankedHit]:
        results = self.bm25.search(query, top_k=top_k)
        return [
            RankedHit(item_id=chunk.chunk_id, rank=index + 1, score=score, source="bm25")
            for index, (chunk, score) in enumerate(results)
        ]

    def _graph_hits(self, query: str, top_k: int) -> list[RankedHit]:
        query_lower = query.lower()
        scored: dict[str, int] = {}
        for triple in self.triples:
            haystack = " ".join([triple.subject, triple.predicate, triple.object]).lower()
            score = sum(1 for token in query_lower.split() if token in haystack)
            if score:
                scored[triple.chunk_id] = max(scored.get(triple.chunk_id, 0), score)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RankedHit(item_id=chunk_id, rank=index + 1, score=float(score), source="graph")
            for index, (chunk_id, score) in enumerate(ranked)
        ]

    def _expanded_query(self, query: str) -> str:
        terms = related_terms(self.triples, query)
        if not terms:
            return query
        return query + " " + " ".join(terms)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
