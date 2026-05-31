import json

from typer.testing import CliRunner

import chunking_docs.cli as cli_module
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


def test_build_context_bundle_adds_neighbor_chunks():
    chunks = [
        DocumentChunk(
            chunk_id="prev",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="previous page context",
        ),
        DocumentChunk(
            chunk_id="hit",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="station access evidence",
        ),
        DocumentChunk(
            chunk_id="next",
            doc_id="doc",
            page_start=3,
            page_end=3,
            kind=ChunkKind.TEXT,
            text="next page context",
        ),
    ]
    triple = GraphTriple(
        triple_id="neighbor-triple",
        doc_id="doc",
        chunk_id="next",
        subject="next context",
        predicate="supports",
        object="answer",
    )
    hit = HybridSearchHit(chunk=chunks[1], score=0.5, sources=["bm25"])

    bundle = build_context_bundle(
        query="station access",
        hits=[hit],
        chunks=chunks,
        triples=[triple],
        neighbor_window=1,
    )

    assert [chunk.chunk_id for chunk in bundle.chunks] == ["hit", "prev", "next"]
    assert [chunk.role for chunk in bundle.chunks] == ["hit", "neighbor", "neighbor"]
    assert bundle.chunks[1].metadata["neighbor_source_chunk_id"] == "hit"
    assert bundle.chunks[1].metadata["neighbor_offset"] == -1
    assert bundle.triples[0].triple_id == "neighbor-triple"
    assert bundle.metadata["neighbor_window"] == 1


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
            "--neighbor-window",
            "0",
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


def test_qdrant_rag_context_cli_writes_bundle(monkeypatch, tmp_path):
    output = tmp_path / "qdrant_context.json"
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="station access corridor evidence",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.FIGURE,
        caption="station access diagram",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="station access",
        predicate="uses",
        object="corridor",
    )

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            assert kwargs["vector_names"] == ["text_dense"]
            return [HybridSearchHit(chunk=chunk, score=0.7, sources=["qdrant:text_dense"])]

    def fake_prepare(**kwargs):
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": "documents",
            "selected_vectors": ["text_dense"],
            "query_encoders": {"text_dense": "default_text"},
            "upserted": 1,
            "chunks": [chunk],
            "assets": [asset],
            "triples": [triple],
        }

    monkeypatch.setattr(cli_module, "prepare_qdrant_hybrid_search", fake_prepare)

    result = CliRunner().invoke(
        app,
        [
            "qdrant-rag-context",
            "station access",
            "--package-dir",
            str(tmp_path),
            "--vector-names",
            "text_dense",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["collection"] == "documents"
    assert payload["chunks"][0]["sources"] == ["qdrant:text_dense"]
    assert payload["assets"][0]["asset_id"] == "asset-1"
