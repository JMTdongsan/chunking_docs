from chunking_docs.chunking.page_chunker import chunk_id
from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from pathlib import Path
import uuid

from chunking_docs.embeddings.records import (
    make_caption_embedding_records,
    make_image_embedding_records,
    make_text_embedding_records,
    make_triple_embedding_records,
    triple_text,
)
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval import RankedHit, reciprocal_rank_fusion


def make_chunk(text: str, page_no: int = 1):
    return DocumentChunk(
        chunk_id=chunk_id("doc", page_no, page_no, ChunkKind.TEXT),
        doc_id="doc",
        page_start=page_no,
        page_end=page_no,
        kind=ChunkKind.TEXT,
        text=text,
    )


def test_make_text_embedding_records():
    chunks = [
        make_chunk("technical policy transit corridor").model_copy(
            update={"asset_ids": ["asset-1"], "source_refs": ["asset:asset-2"]}
        )
    ]
    records = make_text_embedding_records(chunks, HashingTextEmbedder(embedding_dim=16))

    assert len(records) == 1
    assert records[0].vector_name == "text_dense"
    assert len(records[0].vector) == 16
    assert records[0].payload["chunk_id"] == chunks[0].chunk_id
    assert records[0].payload["asset_id"] == ["asset-1", "asset-2"]
    assert records[0].payload["asset_ids"] == ["asset-1", "asset-2"]
    assert records[0].payload["source_refs"] == ["asset:asset-2"]
    assert str(uuid.UUID(records[0].point_id)) == records[0].point_id


def test_make_image_embedding_records():
    class FakeImageEmbedder:
        embedding_dim = 2

        def embed_images(self, image_paths):
            return [[1.0, 0.0] for _ in image_paths]

    records = make_image_embedding_records(
        [
            VisualAsset(
                asset_id="asset-1",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.MAP,
                path=Path("page.png"),
            )
        ],
        FakeImageEmbedder(),
    )

    assert len(records) == 1
    assert records[0].vector_name == "image_dense"
    assert records[0].payload["asset_id"] == "asset-1"


def test_make_caption_embedding_records():
    records = make_caption_embedding_records(
        [
            VisualAsset(
                asset_id="asset-1",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.MAP,
                caption="north district development map",
            )
        ],
        HashingTextEmbedder(embedding_dim=8),
    )

    assert len(records) == 1
    assert records[0].vector_name == "caption_dense"
    assert records[0].payload["text"] == "north district development map"


def test_make_triple_embedding_records():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=3,
        page_end=4,
        kind=ChunkKind.TEXT,
        text="chunk text",
        asset_ids=["asset-2"],
        metadata={"chunking_strategy": "semantic"},
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="map panel",
        predicate="contains_object",
        object="station marker",
        qualifiers={"asset_id": "asset-1", "source_field": "objects"},
        confidence=0.8,
    )

    records = make_triple_embedding_records(
        [triple],
        HashingTextEmbedder(embedding_dim=8),
        chunks=[chunk],
    )

    assert len(records) == 1
    assert records[0].vector_name == "triple_dense"
    assert records[0].chunk_id == "chunk-1"
    assert records[0].payload["triple_id"] == "triple-1"
    assert records[0].payload["record_kind"] == "graph_triple"
    assert records[0].payload["kind"] == "text"
    assert records[0].payload["page_start"] == 3
    assert records[0].payload["page_end"] == 4
    assert records[0].payload["asset_id"] == ["asset-1", "asset-2"]
    assert records[0].payload["chunking_strategy"] == "semantic"
    assert records[0].payload["text"] == "map panel contains object station marker source field objects"
    assert triple_text(triple) == records[0].payload["text"]


def test_make_triple_embedding_records_resolves_asset_backed_chunk():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=5,
        page_end=5,
        kind=ChunkKind.TEXT,
        text="visual evidence chunk",
        source_refs=["asset:asset-1"],
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="vlm-annotation",
        subject="map panel",
        predicate="contains_object",
        object="station marker",
        qualifiers={"asset_id": "asset-1"},
    )

    records = make_triple_embedding_records(
        [triple],
        HashingTextEmbedder(embedding_dim=8),
        chunks=[chunk],
    )

    assert records[0].chunk_id == "chunk-1"
    assert records[0].payload["chunk_id"] == "chunk-1"
    assert records[0].payload["source_triple_chunk_id"] == "vlm-annotation"
    assert records[0].payload["kind"] == "text"
    assert records[0].payload["page_start"] == 5


def test_reciprocal_rank_fusion_combines_sources():
    fused = reciprocal_rank_fusion(
        [
            [RankedHit(item_id="a", rank=1, score=0.9, source="dense")],
            [RankedHit(item_id="a", rank=2, score=8.0, source="bm25")],
            [RankedHit(item_id="b", rank=1, score=0.8, source="graph")],
        ]
    )

    assert fused[0][0] == "a"
    assert fused[0][2] == ["bm25", "dense"]


def test_reciprocal_rank_fusion_applies_source_weights():
    fused = reciprocal_rank_fusion(
        [
            [RankedHit(item_id="a", rank=1, score=0.9, source="bm25")],
            [RankedHit(item_id="b", rank=1, score=0.9, source="qdrant:caption_dense")],
        ],
        source_weights={"bm25": 0.1, "qdrant": 2.0},
    )

    assert fused[0][0] == "b"
    assert fused[0][2] == ["qdrant:caption_dense"]
