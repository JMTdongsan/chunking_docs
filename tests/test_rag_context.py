import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.io import write_jsonl
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval.context import build_context_bundle
from chunking_docs.retrieval.local_hybrid import HybridSearchHit


def test_build_context_bundle_includes_evidence_assets_and_triples(tmp_path):
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=3,
        page_end=3,
        kind=ChunkKind.PAGE_SUMMARY,
        text="summary context",
    )
    child = DocumentChunk(
        chunk_id="child",
        doc_id="doc",
        page_start=3,
        page_end=3,
        kind=ChunkKind.TEXT,
        text="station access evidence " * 20,
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.MAP,
        caption="station access map",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="child",
        subject="station access",
        predicate="shown_by",
        object="map",
    )
    hit = HybridSearchHit(chunk=parent, score=0.5, sources=["dense"], evidence_chunks=[child])

    bundle = build_context_bundle(
        query="station access",
        hits=[hit],
        assets=[asset],
        triples=[triple],
        max_chars_per_chunk=80,
    )

    assert [chunk.role for chunk in bundle.chunks] == ["hit", "evidence"]
    assert bundle.chunks[1].text.endswith("...")
    assert bundle.assets[0].asset_id == "asset-1"
    assert bundle.triples[0].triple_id == "triple-1"
    assert bundle.metadata["asset_count"] == 1


def test_build_rag_context_cli_writes_bundle(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    output = tmp_path / "context.json"
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="station access corridor evidence",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.FIGURE,
            caption="station access diagram",
        )
    ]
    triples = [
        GraphTriple(
            triple_id="triple-1",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="station access",
            predicate="uses",
            object="corridor",
        )
    ]
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", triples)

    result = CliRunner().invoke(
        app,
        [
            "build-rag-context",
            "station access",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["query"] == "station access"
    assert payload["chunks"][0]["chunk_id"] == "chunk-1"
    assert payload["assets"][0]["asset_id"] == "asset-1"
    assert payload["triples"][0]["triple_id"] == "triple-1"
