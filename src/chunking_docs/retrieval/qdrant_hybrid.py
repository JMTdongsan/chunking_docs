from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chunking_docs.embeddings.bm25 import BM25LexicalIndex, chunk_lexical_texts
from chunking_docs.embeddings.interfaces import DenseTextEmbedder
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.graph.export import related_terms
from chunking_docs.graph.provenance import (
    chunk_asset_ids,
    chunk_id_alias_map,
    chunk_ids_by_asset_id,
    string_values,
)
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval.fusion import RankedHit, reciprocal_rank_fusion
from chunking_docs.retrieval.hierarchy import (
    canonical_chunk_id,
    collapse_ranked_hits,
    merge_evidence_maps,
)
from chunking_docs.retrieval.local_hybrid import (
    graph_candidate_chunk_ids,
    graph_match_score,
    normalize_graph_match_text,
)
from chunking_docs.retrieval.rerank import Reranker, rerank_hits
from chunking_docs.storage.qdrant_store import QdrantChunkStore
from chunking_docs.storage.records import VectorSearchHit


@dataclass(frozen=True)
class QdrantHybridSearchHit:
    item_id: str
    score: float
    sources: list[str]
    chunk: DocumentChunk | None = None
    payloads: list[dict] = field(default_factory=list)
    evidence_chunks: list[DocumentChunk] = field(default_factory=list)


class QdrantHybridSearcher:
    def __init__(
        self,
        store: QdrantChunkStore,
        chunks: list[DocumentChunk],
        assets: list[VisualAsset],
        embedder: DenseTextEmbedder,
        vector_embedders: dict[str, DenseTextEmbedder] | None = None,
        triples: list[GraphTriple] | None = None,
        tokenizer_config: LexicalTokenizerConfig | None = None,
    ):
        self.store = store
        self.chunks = chunks
        self.assets = assets
        self.embedder = embedder
        self.vector_embedders = vector_embedders or {}
        self.triples = triples or []
        self.bm25 = BM25LexicalIndex(
            chunks,
            tokenizer_config=tokenizer_config,
            texts=chunk_lexical_texts(chunks, assets),
        )
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_id_by_alias = chunk_id_alias_map(chunks)
        self.chunk_ids_by_asset_id = chunk_ids_by_asset_id(chunks)
        self.asset_to_chunk_id = {
            asset_id: chunk_ids[0]
            for asset_id, chunk_ids in self.chunk_ids_by_asset_id.items()
        }

    def search(
        self,
        query: str,
        vector_names: list[str] | None = None,
        top_k: int = 10,
        graph_expand: bool = False,
        doc_id: str | None = None,
        payload_filter: dict[str, Any] | None = None,
        collapse_hierarchical: bool = False,
        fusion_weights: dict[str, float] | None = None,
        reranker: Reranker | None = None,
        rerank_top_k: int | None = None,
    ) -> list[QdrantHybridSearchHit]:
        vector_names = vector_names or ["text_dense", "caption_dense"]
        expanded_query = self._expanded_query(query) if graph_expand else query
        filters = {**(payload_filter or {})}
        if doc_id:
            filters["doc_id"] = doc_id
        filters = filters or None
        allowed_chunk_ids = self._matching_chunk_ids(filters or {})
        qdrant_hits_by_item: dict[str, list[VectorSearchHit]] = {}
        evidence_maps = []
        result_sets = []
        candidate_k = max(rerank_top_k or top_k, top_k)
        for vector_name in vector_names:
            query_vector = self._query_vector(expanded_query, vector_name)
            hits = self.store.query_vector(
                vector=query_vector,
                vector_name=vector_name,
                top_k=max(candidate_k * 3, 20),
                must_payload=filters,
            )
            ranked_hits = []
            qdrant_evidence: dict[str, list[str]] = {}
            for rank, hit in enumerate(hits, start=1):
                raw_item_id = self._asset_canonical_item_id(hit)
                item_id = canonical_chunk_id(
                    raw_item_id,
                    self.chunk_by_id,
                    collapse_hierarchical=collapse_hierarchical,
                ) if raw_item_id else None
                if not item_id or (allowed_chunk_ids is not None and item_id not in allowed_chunk_ids):
                    continue
                if raw_item_id != item_id:
                    qdrant_evidence.setdefault(item_id, []).append(raw_item_id)
                qdrant_hits_by_item.setdefault(item_id, []).append(hit)
                ranked_hits.append(
                    RankedHit(
                        item_id=item_id,
                        rank=rank,
                        score=hit.score,
                        source=f"qdrant:{vector_name}",
                    )
                )
            ranked_hits, collapsed_evidence = collapse_ranked_hits(
                ranked_hits,
                self.chunk_by_id,
                collapse_hierarchical=collapse_hierarchical,
            )
            evidence_maps.append(merge_evidence_maps(qdrant_evidence, collapsed_evidence))
            result_sets.append(ranked_hits)

        bm25_hits, bm25_evidence = collapse_ranked_hits(
            self._filter_ranked_hits(
                self._bm25_hits(expanded_query, top_k=max(candidate_k * 3, 20)),
                allowed_chunk_ids,
            ),
            self.chunk_by_id,
            collapse_hierarchical=collapse_hierarchical,
        )
        evidence_maps.append(bm25_evidence)
        result_sets.append(bm25_hits)
        if graph_expand:
            graph_hits, graph_evidence = collapse_ranked_hits(
                self._filter_ranked_hits(
                    self._graph_hits(query, top_k=max(candidate_k * 3, 20)),
                    allowed_chunk_ids,
                ),
                self.chunk_by_id,
                collapse_hierarchical=collapse_hierarchical,
            )
            evidence_maps.append(graph_evidence)
            result_sets.append(graph_hits)

        evidence_by_item = merge_evidence_maps(*evidence_maps)
        fused = reciprocal_rank_fusion(result_sets, top_k=candidate_k, source_weights=fusion_weights)
        hits = [
            QdrantHybridSearchHit(
                item_id=item_id,
                score=score,
                sources=sources,
                chunk=self.chunk_by_id.get(item_id),
                payloads=[hit.payload for hit in qdrant_hits_by_item.get(item_id, [])],
                evidence_chunks=[
                    self.chunk_by_id[evidence_id]
                    for evidence_id in evidence_by_item.get(item_id, [])
                    if evidence_id in self.chunk_by_id
                ],
            )
            for item_id, score, sources in fused
        ]
        return rerank_hits(query, hits, reranker, top_k=top_k)

    def _asset_canonical_item_id(self, hit: VectorSearchHit) -> str | None:
        for asset_id in string_values(hit.payload.get("asset_id")):
            if asset_id in self.asset_to_chunk_id:
                return self.asset_to_chunk_id[asset_id]
        return hit.chunk_id

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

    def _query_vector(self, query: str, vector_name: str) -> list[float]:
        embedder = self.vector_embedders.get(vector_name, self.embedder)
        return embedder.embed_texts([query])[0]

    def _matching_chunk_ids(self, filters: dict[str, Any]) -> set[str] | None:
        if not filters:
            return None
        return {
            chunk.chunk_id
            for chunk in self.chunks
            if all(chunk_filter_value(chunk, key, expected) for key, expected in filters.items())
        }

    def _filter_ranked_hits(self, hits: list[RankedHit], allowed_chunk_ids: set[str] | None) -> list[RankedHit]:
        if allowed_chunk_ids is None:
            return hits
        return [hit for hit in hits if hit.item_id in allowed_chunk_ids]


def chunk_filter_value(chunk: DocumentChunk, key: str, expected: Any) -> bool:
    if key == "doc_id":
        actual = chunk.doc_id
    elif key == "chunk_id":
        actual = chunk.chunk_id
    elif key == "kind":
        actual = str(chunk.kind)
    elif key == "asset_id":
        return match_payload_value(chunk_asset_ids(chunk), expected)
    elif key == "page_no":
        return match_page_value(chunk.page_start, chunk.page_end, expected)
    elif key == "page_start":
        actual = chunk.page_start
    elif key == "page_end":
        actual = chunk.page_end
    elif key.startswith("section."):
        actual = getattr(chunk.section, key.split(".", 1)[1], None)
    else:
        actual = chunk.metadata.get(key)
    return match_payload_value(actual, expected)


def match_page_value(page_start: int, page_end: int, expected: Any) -> bool:
    if isinstance(expected, dict):
        return match_payload_value(page_start, expected) or match_payload_value(page_end, expected)
    return page_start <= expected <= page_end


def match_payload_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "any" in expected:
            return match_any(actual, expected["any"])
        range_values = {
            bound: expected[bound]
            for bound in ("gt", "gte", "lt", "lte")
            if bound in expected and expected[bound] is not None
        }
        if range_values:
            return match_range(actual, range_values)
        if "match" in expected:
            return match_payload_value(actual, expected["match"])
    if isinstance(actual, list):
        return expected in actual
    return actual == expected


def match_any(actual: Any, expected_values: list[Any]) -> bool:
    if isinstance(actual, list):
        return any(value in actual for value in expected_values)
    return actual in expected_values


def match_range(actual: Any, range_values: dict[str, Any]) -> bool:
    if actual is None:
        return False
    if "gt" in range_values and not actual > range_values["gt"]:
        return False
    if "gte" in range_values and not actual >= range_values["gte"]:
        return False
    if "lt" in range_values and not actual < range_values["lt"]:
        return False
    if "lte" in range_values and not actual <= range_values["lte"]:
        return False
    return True
