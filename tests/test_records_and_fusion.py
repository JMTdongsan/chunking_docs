from chunking_docs.chunking.page_chunker import chunk_id
from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from pathlib import Path
import uuid

from chunking_docs.embeddings.records import (
    make_caption_embedding_records,
    make_image_embedding_records,
    make_text_embedding_records,
)
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, VisualAsset
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
    chunks = [make_chunk("technical policy transit corridor")]
    records = make_text_embedding_records(chunks, HashingTextEmbedder(embedding_dim=16))

    assert len(records) == 1
    assert records[0].vector_name == "text_dense"
    assert len(records[0].vector) == 16
    assert records[0].payload["chunk_id"] == chunks[0].chunk_id
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
