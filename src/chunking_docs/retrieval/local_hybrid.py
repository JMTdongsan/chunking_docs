from __future__ import annotations

import math
from dataclasses import dataclass, field

from chunking_docs.embeddings.bm25 import BM25LexicalIndex, chunk_lexical_texts
from chunking_docs.embeddings.interfaces import DenseTextEmbedder
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.graph.export import related_terms
from chunking_docs.graph.provenance import chunk_asset_ids, ordered_unique, triple_asset_ids
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval.fusion import RankedHit, reciprocal_rank_fusion
from chunking_docs.retrieval.hierarchy import collapse_ranked_hits, merge_evidence_maps
from chunking_docs.retrieval.rerank import Reranker, rerank_hits


@dataclass(frozen=True)
class HybridSearchHit:
    chunk: DocumentChunk
    score: float
    sources: list[str]
    evidence_chunks: list[DocumentChunk] = field(default_factory=list)


class LocalHybridSearcher:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedder: DenseTextEmbedder,
        triples: list[GraphTriple] | None = None,
        tokenizer_config: LexicalTokenizerConfig | None = None,
        assets: list[VisualAsset] | None = None,
    ):
        self.chunks = chunks
        self.assets = assets or []
        self.embedder = embedder
        self.bm25 = BM25LexicalIndex(
            chunks,
            tokenizer_config=tokenizer_config,
            texts=chunk_lexical_texts(chunks, self.assets),
        )
        self.chunk_vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        self.triples = triples or []
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_id_by_alias = chunk_id_alias_map(chunks)
        self.chunk_ids_by_asset_id = chunk_ids_by_asset_id(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        graph_expand: bool = False,
        collapse_hierarchical: bool = False,
        use_dense: bool = True,
        use_bm25: bool = True,
        use_graph: bool | None = None,
        fusion_weights: dict[str, float] | None = None,
        reranker: Reranker | None = None,
        rerank_top_k: int | None = None,
    ) -> list[HybridSearchHit]:
        use_graph = graph_expand if use_graph is None else use_graph
        expanded_query = self._expanded_query(query) if graph_expand else query
        result_sets = []
        evidence_by_item = {}
        candidate_k = max(rerank_top_k or top_k, top_k)

        if use_dense:
            dense_hits = self._dense_hits(expanded_query, top_k=max(candidate_k * 3, 20))
            dense_hits, dense_evidence = collapse_ranked_hits(
                dense_hits,
                self.chunk_by_id,
                collapse_hierarchical=collapse_hierarchical,
            )
            result_sets.append(dense_hits)
            evidence_by_item = merge_evidence_maps(evidence_by_item, dense_evidence)

        if use_bm25:
            bm25_hits = self._bm25_hits(expanded_query, top_k=max(candidate_k * 3, 20))
            bm25_hits, bm25_evidence = collapse_ranked_hits(
                bm25_hits,
                self.chunk_by_id,
                collapse_hierarchical=collapse_hierarchical,
            )
            result_sets.append(bm25_hits)
            evidence_by_item = merge_evidence_maps(evidence_by_item, bm25_evidence)

        if use_graph:
            graph_hits = self._graph_hits(query, top_k=max(candidate_k * 3, 20))
            graph_hits, graph_evidence = collapse_ranked_hits(
                graph_hits,
                self.chunk_by_id,
                collapse_hierarchical=collapse_hierarchical,
            )
            evidence_by_item = merge_evidence_maps(evidence_by_item, graph_evidence)
            result_sets.append(graph_hits)
        fused = reciprocal_rank_fusion(result_sets, top_k=candidate_k, source_weights=fusion_weights)
        hits = [
            HybridSearchHit(
                chunk=self.chunk_by_id[item_id],
                score=score,
                sources=sources,
                evidence_chunks=[
                    self.chunk_by_id[evidence_id]
                    for evidence_id in evidence_by_item.get(item_id, [])
                    if evidence_id in self.chunk_by_id
                ],
            )
            for item_id, score, sources in fused
            if item_id in self.chunk_by_id
        ]
        return rerank_hits(query, hits, reranker, top_k=top_k)

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
        query_normalized = normalize_graph_match_text(query)
        scored: dict[str, float] = {}
        for triple in self.triples:
            score = graph_match_score(query_normalized, triple)
            if score:
                for chunk_id in graph_candidate_chunk_ids(
                    triple,
                    self.chunk_by_id,
                    self.chunk_id_by_alias,
                    self.chunk_ids_by_asset_id,
                ):
                    scored[chunk_id] = max(scored.get(chunk_id, 0), score)
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


def chunk_id_alias_map(chunks: list[DocumentChunk]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for chunk in chunks:
        aliases.setdefault(chunk.chunk_id, chunk.chunk_id)
        for key in ("source_chunk_id", "parent_chunk_id"):
            value = chunk.metadata.get(key)
            if isinstance(value, str):
                aliases.setdefault(value, chunk.chunk_id)
    return aliases


def chunk_ids_by_asset_id(chunks: list[DocumentChunk]) -> dict[str, list[str]]:
    indexed: dict[str, list[str]] = {}
    for chunk in chunks:
        for asset_id in chunk_asset_ids(chunk):
            indexed.setdefault(asset_id, []).append(chunk.chunk_id)
    return {asset_id: ordered_unique(chunk_ids) for asset_id, chunk_ids in indexed.items()}


def graph_candidate_chunk_ids(
    triple: GraphTriple,
    chunk_by_id: dict[str, DocumentChunk],
    chunk_id_by_alias: dict[str, str],
    chunk_ids_by_asset: dict[str, list[str]],
) -> list[str]:
    candidates = []
    chunk_id = chunk_id_by_alias.get(triple.chunk_id, triple.chunk_id)
    if chunk_id in chunk_by_id:
        candidates.append(chunk_id)
    for asset_id in sorted(triple_asset_ids(triple)):
        candidates.extend(chunk_ids_by_asset.get(asset_id, []))
    return ordered_unique(candidates)


def graph_match_score(query_normalized: str, triple: GraphTriple) -> float:
    components = [
        normalize_graph_match_text(triple.subject),
        normalize_graph_match_text(triple.predicate),
        normalize_graph_match_text(triple.object),
    ]
    haystack = " ".join(component for component in components if component)
    if not query_normalized or not haystack:
        return 0.0

    score = 0.0
    if query_normalized in haystack:
        score += 8.0

    query_tokens = set(query_normalized.split())
    haystack_tokens = set(haystack.split())
    score += len(query_tokens & haystack_tokens)

    for component in components:
        if not component:
            continue
        component_tokens = set(component.split())
        if component in query_normalized:
            score += 4.0
        elif component_tokens and component_tokens <= query_tokens:
            score += 2.0

    return score


def normalize_graph_match_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())
