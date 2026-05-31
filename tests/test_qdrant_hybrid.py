from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval.qdrant_hybrid import QdrantHybridSearcher
from chunking_docs.storage.records import VectorSearchHit


class FakeQdrantStore:
    def query_vector(self, vector, vector_name, top_k, must_payload=None, score_threshold=None):
        if vector_name == "caption_dense":
            return [
                VectorSearchHit(
                    point_id="asset-point",
                    score=0.9,
                    vector_name=vector_name,
                    chunk_id="asset-1",
                    doc_id="doc",
                    payload={
                        "asset_id": "asset-1",
                        "doc_id": "doc",
                        "page_no": 5,
                        "caption": "river corridor diagram",
                    },
                )
            ]
        return []


class FakeHierarchicalQdrantStore:
    def query_vector(self, vector, vector_name, top_k, must_payload=None, score_threshold=None):
        return [
            VectorSearchHit(
                point_id="child-point",
                score=0.92,
                vector_name=vector_name,
                chunk_id="child",
                doc_id="doc",
                payload={
                    "chunk_id": "child",
                    "doc_id": "doc",
                    "page_start": 8,
                    "page_end": 8,
                    "text": "station access child evidence",
                },
            )
        ]


def test_qdrant_hybrid_maps_asset_hits_to_parent_chunk():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=5,
        page_end=5,
        kind=ChunkKind.TEXT,
        text="base text",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=5,
        kind=AssetKind.FIGURE,
        caption="river corridor diagram",
    )

    searcher = QdrantHybridSearcher(
        store=FakeQdrantStore(),
        chunks=[chunk],
        assets=[asset],
        embedder=HashingTextEmbedder(embedding_dim=8),
    )
    hits = searcher.search("river corridor", vector_names=["caption_dense"], top_k=1)

    assert hits[0].item_id == "chunk-1"
    assert hits[0].chunk == chunk
    assert "qdrant:caption_dense" in hits[0].sources
    assert hits[0].payloads[0]["asset_id"] == "asset-1"


def test_qdrant_hybrid_can_include_graph_hits():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=5,
        page_end=5,
        kind=ChunkKind.TEXT,
        text="base text",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="policy",
        predicate="uses_axis",
        object="river corridor",
    )

    searcher = QdrantHybridSearcher(
        store=FakeQdrantStore(),
        chunks=[chunk],
        assets=[],
        embedder=HashingTextEmbedder(embedding_dim=8),
        triples=[triple],
    )
    hits = searcher.search("policy", vector_names=["text_dense"], top_k=1, graph_expand=True)

    assert hits[0].item_id == "chunk-1"
    assert "graph" in hits[0].sources


def test_qdrant_hybrid_can_collapse_hierarchical_child_to_parent():
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=8,
        page_end=8,
        kind=ChunkKind.PAGE_SUMMARY,
        text="page summary",
        metadata={"retrieval_role": "parent"},
    )
    child = DocumentChunk(
        chunk_id="child",
        doc_id="doc",
        page_start=8,
        page_end=8,
        kind=ChunkKind.TEXT,
        text="station access child evidence",
        metadata={
            "retrieval_role": "child",
            "hierarchical_parent_chunk_id": "parent",
        },
    )

    searcher = QdrantHybridSearcher(
        store=FakeHierarchicalQdrantStore(),
        chunks=[parent, child],
        assets=[],
        embedder=HashingTextEmbedder(embedding_dim=8),
    )
    hits = searcher.search(
        "station access",
        vector_names=["text_dense"],
        top_k=1,
        collapse_hierarchical=True,
    )

    assert hits[0].item_id == "parent"
    assert hits[0].chunk == parent
    assert [chunk.chunk_id for chunk in hits[0].evidence_chunks] == ["child"]
    assert hits[0].payloads[0]["chunk_id"] == "child"
