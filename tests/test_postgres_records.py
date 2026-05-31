from pathlib import Path

from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    TextQuality,
    VisualAsset,
)
from chunking_docs.storage.postgres_records import asset_row, chunk_row, document_row, page_row, triple_row
from chunking_docs.storage.postgres_store import manifest_rows


def test_postgres_row_transforms_are_json_ready():
    document = SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf"))
    page = PageProfile(
        doc_id="doc",
        page_no=1,
        width=1,
        height=2,
        char_count=3,
        line_count=4,
        text_block_count=5,
        image_block_count=6,
        embedded_image_count=7,
        drawing_count=8,
        text_quality=TextQuality.DEGRADED,
    )
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="hello",
        asset_ids=["asset"],
    )
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        path=Path("/tmp/assets/page.png"),
        bbox=(0, 1, 2, 3),
    )
    triple = GraphTriple(
        triple_id="triple",
        doc_id="doc",
        chunk_id="chunk",
        subject="a",
        predicate="b",
        object="c",
    )

    assert document_row(document)["doc_id"] == "doc"
    assert page_row(page)["profile"]["text_quality"] == "degraded"
    assert chunk_row(chunk)["metadata"]["asset_ids"] == ["asset"]
    assert asset_row(asset, base_dir=Path("/tmp"))["path"] == "assets/page.png"
    assert triple_row(triple)["object"] == "c"


def test_manifest_rows_counts():
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        profiles=[],
        chunks=[],
        assets=[],
        triples=[],
    )

    rows = manifest_rows(manifest)

    assert rows["document"]["doc_id"] == "doc"
    assert rows["pages"] == []
