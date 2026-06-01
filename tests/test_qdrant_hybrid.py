import pytest
import typer

from chunking_docs.cli import (
    build_payload_filter,
    build_qdrant_query_embedders,
    build_reranker,
    parse_fusion_weights,
    qdrant_query_encoder_details,
    resolve_qdrant_query_backend_options,
    validate_qdrant_query_encoder_dimensions,
)
from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval.qdrant_hybrid import QdrantHybridSearcher
from chunking_docs.retrieval.rerank import LexicalOverlapReranker
from chunking_docs.storage.records import VectorSearchHit


class RecordingTextEmbedder:
    def __init__(self, embedding_dim=8, value=1.0):
        self.embedding_dim = embedding_dim
        self.value = value
        self.calls = []

    def embed_texts(self, texts):
        self.calls.extend(texts)
        return [[self.value] * self.embedding_dim for _ in texts]


class FakeQdrantStore:
    def __init__(self):
        self.queries = []

    def query_vector(self, vector, vector_name, top_k, must_payload=None, score_threshold=None):
        self.queries.append((vector_name, vector))
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
                        "kind": "figure",
                        "caption": "river corridor diagram",
                    },
                )
            ]
        if vector_name == "image_dense":
            return [
                VectorSearchHit(
                    point_id="image-point",
                    score=0.8,
                    vector_name=vector_name,
                    chunk_id="asset-1",
                    doc_id="doc",
                    payload={
                        "asset_id": "asset-1",
                        "doc_id": "doc",
                        "page_no": 5,
                    },
                )
            ]
        if vector_name == "object_dense":
            return [
                VectorSearchHit(
                    point_id="object-point",
                    score=0.85,
                    vector_name=vector_name,
                    chunk_id="asset-1",
                    doc_id="doc",
                    payload={
                        "record_kind": "visual_object",
                        "object_id": "asset-1:object:0",
                        "asset_id": "asset-1",
                        "doc_id": "doc",
                        "page_no": 5,
                        "kind": "figure",
                        "label": "route marker",
                        "text": "Object: route marker",
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


class FilteringQdrantStore:
    def __init__(self):
        self.must_payload = None

    def query_vector(self, vector, vector_name, top_k, must_payload=None, score_threshold=None):
        self.must_payload = must_payload
        return [
            VectorSearchHit(
                point_id="old-point",
                score=0.9,
                vector_name=vector_name,
                chunk_id="old",
                doc_id="doc",
                payload={"chunk_id": "old", "doc_id": "doc", "page_start": 1, "page_end": 1},
            ),
            VectorSearchHit(
                point_id="recent-point",
                score=0.8,
                vector_name=vector_name,
                chunk_id="recent",
                doc_id="doc",
                payload={"chunk_id": "recent", "doc_id": "doc", "page_start": 12, "page_end": 12},
            ),
        ]


class TriplePayloadFilteringQdrantStore:
    def __init__(self):
        self.must_payload = None

    def query_vector(self, vector, vector_name, top_k, must_payload=None, score_threshold=None):
        self.must_payload = must_payload
        return [
            VectorSearchHit(
                point_id="other-triple-point",
                score=0.95,
                vector_name=vector_name,
                chunk_id="target",
                doc_id="doc",
                payload={
                    "record_kind": "graph_triple",
                    "triple_id": "other-triple",
                    "chunk_id": "target",
                    "doc_id": "doc",
                    "page_start": 9,
                    "page_end": 9,
                },
            ),
            VectorSearchHit(
                point_id="target-triple-point",
                score=0.9,
                vector_name=vector_name,
                chunk_id="target",
                doc_id="doc",
                payload={
                    "record_kind": "graph_triple",
                    "triple_id": "target-triple",
                    "chunk_id": "target",
                    "doc_id": "doc",
                    "page_start": 9,
                    "page_end": 9,
                },
            ),
        ]


class RerankQdrantStore:
    def query_vector(self, vector, vector_name, top_k, must_payload=None, score_threshold=None):
        return [
            VectorSearchHit(
                point_id="weak-point",
                score=0.95,
                vector_name=vector_name,
                chunk_id="weak",
                doc_id="doc",
                payload={"chunk_id": "weak", "doc_id": "doc", "page_start": 1, "page_end": 1},
            ),
            VectorSearchHit(
                point_id="strong-point",
                score=0.7,
                vector_name=vector_name,
                chunk_id="strong",
                doc_id="doc",
                payload={"chunk_id": "strong", "doc_id": "doc", "page_start": 2, "page_end": 2},
            ),
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

    store = FakeQdrantStore()
    searcher = QdrantHybridSearcher(
        store=store,
        chunks=[chunk],
        assets=[asset],
        embedder=HashingTextEmbedder(embedding_dim=8),
    )
    hits = searcher.search("river corridor", vector_names=["caption_dense"], top_k=1)

    assert hits[0].item_id == "chunk-1"
    assert hits[0].chunk == chunk
    assert "qdrant:caption_dense" in hits[0].sources
    assert hits[0].payloads[0]["asset_id"] == "asset-1"


def test_qdrant_hybrid_maps_asset_hits_to_source_ref_chunk():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=5,
        page_end=5,
        kind=ChunkKind.TEXT,
        text="base text",
        source_refs=["asset:asset-1"],
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
    assert hits[0].payloads[0]["asset_id"] == "asset-1"


def test_qdrant_hybrid_maps_object_hits_to_parent_chunk():
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
        metadata={"objects": [{"label": "route marker"}]},
    )

    searcher = QdrantHybridSearcher(
        store=FakeQdrantStore(),
        chunks=[chunk],
        assets=[asset],
        embedder=HashingTextEmbedder(embedding_dim=8),
    )
    hits = searcher.search("route marker", vector_names=["object_dense"], top_k=1)

    assert hits[0].item_id == "chunk-1"
    assert hits[0].chunk == chunk
    assert "qdrant:object_dense" in hits[0].sources
    assert hits[0].payloads[0]["object_id"] == "asset-1:object:0"


def test_qdrant_hybrid_bm25_can_match_linked_visual_asset_text():
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
    hits = searcher.search("river corridor", vector_names=["text_dense"], top_k=1)

    assert hits[0].item_id == "chunk-1"
    assert hits[0].sources == ["bm25"]


def test_qdrant_hybrid_payload_filter_matches_source_ref_asset_id():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=5,
        page_end=5,
        kind=ChunkKind.TEXT,
        text="base text",
        source_refs=["asset:asset-1"],
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
    hits = searcher.search(
        "river corridor",
        vector_names=["text_dense"],
        top_k=1,
        payload_filter={"asset_id": "asset-1"},
    )

    assert hits[0].item_id == "chunk-1"
    assert hits[0].sources == ["bm25"]


def test_qdrant_hybrid_preserves_visual_payload_kind_filtered_hits():
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
    hits = searcher.search(
        "river corridor",
        vector_names=["caption_dense"],
        top_k=1,
        payload_filter={"kind": "figure"},
    )

    assert hits[0].item_id == "chunk-1"
    assert hits[0].sources == ["qdrant:caption_dense"]
    assert hits[0].payloads[0]["kind"] == "figure"


def test_qdrant_hybrid_can_rerank_fused_candidates():
    weak = DocumentChunk(
        chunk_id="weak",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="unrelated",
    )
    strong = DocumentChunk(
        chunk_id="strong",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="river station corridor",
    )

    searcher = QdrantHybridSearcher(
        store=RerankQdrantStore(),
        chunks=[weak, strong],
        assets=[],
        embedder=HashingTextEmbedder(embedding_dim=8),
    )
    hits = searcher.search(
        "river station",
        vector_names=["text_dense"],
        top_k=2,
        reranker=LexicalOverlapReranker(),
        rerank_top_k=2,
    )

    assert [hit.item_id for hit in hits] == ["strong", "weak"]
    assert "rerank:lexical" in hits[0].sources


def test_qdrant_hybrid_respects_fusion_weights():
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
    hits = searcher.search(
        "river corridor",
        vector_names=["caption_dense"],
        top_k=1,
        fusion_weights={"qdrant": 0.0, "bm25": 0.0},
    )

    assert hits == []


def test_qdrant_hybrid_uses_per_vector_query_embedders():
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
    store = FakeQdrantStore()
    text_embedder = RecordingTextEmbedder(embedding_dim=3, value=1.0)
    image_query_embedder = RecordingTextEmbedder(embedding_dim=5, value=2.0)

    searcher = QdrantHybridSearcher(
        store=store,
        chunks=[chunk],
        assets=[asset],
        embedder=text_embedder,
        vector_embedders={"image_dense": image_query_embedder},
    )
    searcher.search("river corridor", vector_names=["text_dense", "image_dense"], top_k=1)

    assert store.queries[0] == ("text_dense", [1.0, 1.0, 1.0])
    assert store.queries[1] == ("image_dense", [2.0, 2.0, 2.0, 2.0, 2.0])
    assert text_embedder.calls == ["river corridor"]
    assert image_query_embedder.calls == ["river corridor"]


def test_qdrant_query_embedder_requires_image_query_backend_for_mismatched_size():
    with pytest.raises(typer.BadParameter):
        build_qdrant_query_embedders(
            selected_vectors=["text_dense", "image_dense"],
            vector_sizes={"text_dense": 3, "image_dense": 5},
            default_embedder=RecordingTextEmbedder(embedding_dim=3),
            image_query_backend="none",
            image_query_model="clip-model",
            device="cpu",
            hashing_dim=3,
        )


def test_validate_qdrant_query_encoder_dimensions_accepts_matching_vectors():
    validate_qdrant_query_encoder_dimensions(
        selected_vectors=["text_dense", "caption_dense", "image_dense"],
        vector_sizes={"text_dense": 3, "caption_dense": 3, "image_dense": 5},
        default_embedder=RecordingTextEmbedder(embedding_dim=3),
        vector_embedders={"image_dense": RecordingTextEmbedder(embedding_dim=5)},
        image_query_backend="clip",
    )


def test_validate_qdrant_query_encoder_dimensions_rejects_text_mismatch():
    with pytest.raises(typer.BadParameter) as exc_info:
        validate_qdrant_query_encoder_dimensions(
            selected_vectors=["text_dense"],
            vector_sizes={"text_dense": 1024},
            default_embedder=RecordingTextEmbedder(embedding_dim=384),
            vector_embedders={},
            vector_notes={"text_dense": "SentenceTransformerTextEmbedder model=BAAI/bge-m3 device=cuda."},
        )

    message = str(exc_info.value)
    assert "text_dense expects 1024 dimensions" in message
    assert "default text query encoder produces 384" in message
    assert "--text-backend sentence-transformers" in message
    assert "BAAI/bge-m3" in message


def test_validate_qdrant_query_encoder_dimensions_rejects_image_query_mismatch():
    with pytest.raises(typer.BadParameter) as exc_info:
        validate_qdrant_query_encoder_dimensions(
            selected_vectors=["image_dense"],
            vector_sizes={"image_dense": 768},
            default_embedder=RecordingTextEmbedder(embedding_dim=1024),
            vector_embedders={"image_dense": RecordingTextEmbedder(embedding_dim=512)},
            image_query_backend="clip",
        )

    message = str(exc_info.value)
    assert "image_dense expects 768 dimensions" in message
    assert "clip image query encoder produces 512" in message
    assert "--image-query-backend" in message


def test_qdrant_query_encoder_details_records_backend_model_and_dimension():
    default = RecordingTextEmbedder(embedding_dim=1024)
    image = RecordingTextEmbedder(embedding_dim=768)

    details = qdrant_query_encoder_details(
        selected_vectors=["text_dense", "caption_dense", "image_dense"],
        vector_embedders={"image_dense": image},
        default_embedder=default,
        text_backend="sentence-transformers",
        text_model="BAAI/bge-m3",
        image_query_backend="clip",
        image_query_model="openai/clip-vit-large-patch14",
    )

    assert details["text_dense"] == {
        "encoder": "default text query encoder",
        "backend": "sentence-transformers",
        "model": "BAAI/bge-m3",
        "dimension": 1024,
    }
    assert details["caption_dense"]["model"] == "BAAI/bge-m3"
    assert details["caption_dense"]["dimension"] == 1024
    assert details["image_dense"] == {
        "encoder": "clip image query encoder",
        "backend": "clip",
        "model": "openai/clip-vit-large-patch14",
        "dimension": 768,
    }


def test_resolve_qdrant_query_backend_options_reads_embedding_manifest(tmp_path):
    (tmp_path / "embedding_manifest.json").write_text(
        """
{
  "vectors": {
    "text_dense": {
      "dimension": 1024,
      "embedding": {"backend": "sentence-transformers", "model": "BAAI/bge-m3"}
    },
    "caption_dense": {
      "dimension": 1024,
      "embedding": {"same_as": "text_dense"}
    },
    "image_dense": {
      "dimension": 768,
      "embedding": {"backend": "clip", "model": "openai/clip-vit-large-patch14"}
    }
  }
}
""",
        encoding="utf-8",
    )

    options = resolve_qdrant_query_backend_options(
        package_dir=tmp_path,
        selected_vectors=["caption_dense", "image_dense"],
        text_backend="auto",
        text_model="fallback-text",
        image_query_backend="auto",
        image_query_model="fallback-image",
    )

    assert options == {
        "text_backend": "sentence-transformers",
        "text_model": "BAAI/bge-m3",
        "image_query_backend": "clip",
        "image_query_model": "openai/clip-vit-large-patch14",
    }


def test_build_payload_filter_parses_exact_and_range_filters():
    filters = build_payload_filter(
        doc_id="doc",
        filter_specs=["kind=map", "page_start<=12", "page_end>=12"],
    )

    assert filters == {
        "doc_id": "doc",
        "kind": "map",
        "page_start": {"lte": 12},
        "page_end": {"gte": 12},
    }


def test_parse_fusion_weights():
    weights = parse_fusion_weights(["bm25=1.3", "qdrant:caption_dense=1.5"])

    assert weights == {"bm25": 1.3, "qdrant:caption_dense": 1.5}


def test_build_reranker_supports_lexical_backend():
    built = build_reranker("lexical")

    assert isinstance(built, LexicalOverlapReranker)


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


def test_qdrant_hybrid_graph_hits_resolve_source_chunk_alias():
    chunk = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=5,
        page_end=5,
        kind=ChunkKind.PAGE_SUMMARY,
        text="summary",
        metadata={"source_chunk_id": "source-a"},
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="source-a",
        subject="north district",
        predicate="uses_axis",
        object="riverfront axis",
    )

    searcher = QdrantHybridSearcher(
        store=FakeQdrantStore(),
        chunks=[chunk],
        assets=[],
        embedder=HashingTextEmbedder(embedding_dim=8),
        triples=[triple],
    )
    hits = searcher.search("north district", vector_names=["text_dense"], top_k=1, graph_expand=True)

    assert hits[0].item_id == "parent"
    assert hits[0].sources == ["graph"]


def test_qdrant_hybrid_graph_hits_resolve_visual_asset_provenance():
    chunk = DocumentChunk(
        chunk_id="chunk-asset",
        doc_id="doc",
        page_start=5,
        page_end=5,
        kind=ChunkKind.TEXT,
        text="summary",
        asset_ids=["asset-map"],
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="annotation-only",
        subject="station access map",
        predicate="shows",
        object="riverfront axis",
        qualifiers={"source": "visual_annotation", "asset_id": "asset-map"},
    )

    searcher = QdrantHybridSearcher(
        store=FakeQdrantStore(),
        chunks=[chunk],
        assets=[],
        embedder=HashingTextEmbedder(embedding_dim=8),
        triples=[triple],
    )
    hits = searcher.search(
        "station access map",
        vector_names=["text_dense"],
        top_k=1,
        graph_expand=True,
        fusion_weights={"qdrant": 0.0, "bm25": 0.0, "graph": 1.0},
    )

    assert hits[0].item_id == "chunk-asset"
    assert hits[0].sources == ["graph"]


def test_qdrant_hybrid_graph_hits_prioritize_exact_triple_components():
    generic = DocumentChunk(
        chunk_id="generic",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="unrelated generic note",
    )
    target = DocumentChunk(
        chunk_id="target",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="unrelated target note",
    )
    triples = [
        GraphTriple(
            triple_id="generic-triple",
            doc_id="doc",
            chunk_id="generic",
            subject="center system",
            predicate="core",
            object="generic hub",
        ),
        GraphTriple(
            triple_id="target-triple",
            doc_id="doc",
            chunk_id="target",
            subject="center system",
            predicate="core",
            object="specific hub",
        ),
    ]

    searcher = QdrantHybridSearcher(
        store=FakeQdrantStore(),
        chunks=[generic, target],
        assets=[],
        embedder=HashingTextEmbedder(embedding_dim=8),
        triples=triples,
    )
    hits = searcher.search(
        "center system core specific hub",
        vector_names=["text_dense"],
        top_k=1,
        graph_expand=True,
        fusion_weights={"qdrant": 0.0, "bm25": 0.0, "graph": 1.0},
    )

    assert hits[0].item_id == "target"
    assert hits[0].sources == ["graph"]


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

    child_filtered_hits = searcher.search(
        "station access",
        vector_names=["text_dense"],
        top_k=1,
        collapse_hierarchical=True,
        payload_filter={"retrieval_role": "child"},
    )

    assert child_filtered_hits[0].item_id == "parent"
    assert [chunk.chunk_id for chunk in child_filtered_hits[0].evidence_chunks] == ["child"]


def test_qdrant_hybrid_applies_payload_filter_to_qdrant_and_local_hits():
    old = DocumentChunk(
        chunk_id="old",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="station access",
    )
    recent = DocumentChunk(
        chunk_id="recent",
        doc_id="doc",
        page_start=12,
        page_end=12,
        kind=ChunkKind.TEXT,
        text="station access",
    )
    store = FilteringQdrantStore()
    searcher = QdrantHybridSearcher(
        store=store,
        chunks=[old, recent],
        assets=[],
        embedder=HashingTextEmbedder(embedding_dim=8),
    )

    hits = searcher.search(
        "station access",
        vector_names=["text_dense"],
        top_k=2,
        payload_filter={"page_start": {"gte": 10}},
    )

    assert store.must_payload == {"page_start": {"gte": 10}}
    assert [hit.item_id for hit in hits] == ["recent"]


def test_qdrant_hybrid_preserves_payload_only_filtered_graph_vector_hits():
    target = DocumentChunk(
        chunk_id="target",
        doc_id="doc",
        page_start=9,
        page_end=9,
        kind=ChunkKind.TEXT,
        text="station access map evidence",
    )
    store = TriplePayloadFilteringQdrantStore()
    searcher = QdrantHybridSearcher(
        store=store,
        chunks=[target],
        assets=[],
        embedder=HashingTextEmbedder(embedding_dim=8),
    )

    hits = searcher.search(
        "station access map",
        vector_names=["triple_dense"],
        top_k=1,
        payload_filter={"record_kind": "graph_triple", "triple_id": "target-triple"},
    )

    assert store.must_payload == {"record_kind": "graph_triple", "triple_id": "target-triple"}
    assert [hit.item_id for hit in hits] == ["target"]
    assert hits[0].sources == ["qdrant:triple_dense"]
    assert hits[0].payloads[0]["triple_id"] == "target-triple"
