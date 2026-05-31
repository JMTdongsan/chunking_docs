from __future__ import annotations

from dataclasses import dataclass, field

from chunking_docs.embeddings.bm25 import BM25LexicalIndex
from chunking_docs.embeddings.interfaces import DenseTextEmbedder
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.graph.export import related_terms
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval.fusion import RankedHit, reciprocal_rank_fusion
from chunking_docs.retrieval.hierarchy import (
    canonical_chunk_id,
    collapse_ranked_hits,
    merge_evidence_maps,
)
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
        triples: list[GraphTriple] | None = None,
        tokenizer_config: LexicalTokenizerConfig | None = None,
    ):
        self.store = store
        self.chunks = chunks
        self.assets = assets
        self.embedder = embedder
        self.triples = triples or []
        self.bm25 = BM25LexicalIndex(chunks, tokenizer_config=tokenizer_config)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.asset_to_chunk_id = {
            asset_id: chunk.chunk_id
            for chunk in chunks
            for asset_id in chunk.asset_ids
        }

    def search(
        self,
        query: str,
        vector_names: list[str] | None = None,
        top_k: int = 10,
        graph_expand: bool = False,
        doc_id: str | None = None,
        collapse_hierarchical: bool = False,
    ) -> list[QdrantHybridSearchHit]:
        vector_names = vector_names or ["text_dense", "caption_dense"]
        expanded_query = self._expanded_query(query) if graph_expand else query
        query_vector = self.embedder.embed_texts([expanded_query])[0]
        filters = {"doc_id": doc_id} if doc_id else None
        qdrant_hits_by_item: dict[str, list[VectorSearchHit]] = {}
        evidence_maps = []
        result_sets = []
        for vector_name in vector_names:
            hits = self.store.query_vector(
                vector=query_vector,
                vector_name=vector_name,
                top_k=max(top_k * 3, 20),
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
                if not item_id:
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
            self._bm25_hits(expanded_query, top_k=max(top_k * 3, 20)),
            self.chunk_by_id,
            collapse_hierarchical=collapse_hierarchical,
        )
        evidence_maps.append(bm25_evidence)
        result_sets.append(bm25_hits)
        if graph_expand:
            graph_hits, graph_evidence = collapse_ranked_hits(
                self._graph_hits(query, top_k=max(top_k * 3, 20)),
                self.chunk_by_id,
                collapse_hierarchical=collapse_hierarchical,
            )
            evidence_maps.append(graph_evidence)
            result_sets.append(graph_hits)

        evidence_by_item = merge_evidence_maps(*evidence_maps)
        fused = reciprocal_rank_fusion(result_sets, top_k=top_k)
        return [
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

    def _asset_canonical_item_id(self, hit: VectorSearchHit) -> str | None:
        asset_id = hit.payload.get("asset_id")
        if asset_id and asset_id in self.asset_to_chunk_id:
            return self.asset_to_chunk_id[asset_id]
        return hit.chunk_id

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
