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
    assert len(bundle.chunks[1].text) <= 80
    assert bundle.assets[0].asset_id == "asset-1"
    assert bundle.triples[0].triple_id == "triple-1"
    assert bundle.metadata["asset_count"] == 1
    assert bundle.metadata["chunk_count"] == 2
    assert bundle.metadata["hit_chunk_count"] == 1
    assert bundle.metadata["evidence_chunk_count"] == 1
    assert bundle.metadata["source_family_counts"] == {"dense_text": 2}
    assert bundle.metadata["pages"] == [3]
    assert bundle.metadata["page_count"] == 1
    assert bundle.metadata["has_dense_text_context"] is True
    assert bundle.metadata["has_visual_context"] is True
    assert bundle.metadata["has_graph_context"] is True


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
    assert bundle.metadata["neighbor_chunk_count"] == 2
    assert bundle.metadata["source_family_counts"] == {"lexical": 1, "neighbor": 2}
    assert bundle.metadata["role_counts"] == {"hit": 1, "neighbor": 2}
    assert bundle.metadata["pages"] == [1, 2, 3]
    assert bundle.metadata["has_lexical_context"] is True


def test_build_context_bundle_trims_visual_asset_text_with_metadata():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="visual evidence",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=2,
        kind=AssetKind.FIGURE,
        caption="short caption",
        ocr_text="long ocr text " * 8,
        vlm_summary="long vlm summary " * 8,
    )
    hit = HybridSearchHit(chunk=chunk, score=0.7, sources=["dense"])

    bundle = build_context_bundle(
        query="visual evidence",
        hits=[hit],
        assets=[asset],
        max_chars_per_asset_text=40,
    )

    context_asset = bundle.assets[0]
    assert context_asset.caption == "short caption"
    assert context_asset.ocr_text.endswith("...")
    assert context_asset.vlm_summary.endswith("...")
    assert len(context_asset.ocr_text) <= 40
    assert len(context_asset.vlm_summary) <= 40
    assert context_asset.metadata["context_text"]["max_chars_per_field"] == 40
    assert context_asset.metadata["context_text"]["truncated_fields"] == [
        "ocr_text",
        "vlm_summary",
    ]
    assert bundle.metadata["max_chars_per_asset_text"] == 40
    assert bundle.metadata["asset_text_truncated_count"] == 1
    assert bundle.metadata["asset_text_truncated_fields"] == {
        "ocr_text": 1,
        "vlm_summary": 1,
    }
    assert bundle.metadata["asset_context_char_count"] < bundle.metadata["asset_text_char_count"]


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
            ocr_text="station access diagram ocr text " * 10,
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
            "--max-chars-per-asset-text",
            "50",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["query"] == "station access"
    assert payload["chunks"][0]["chunk_id"] == "chunk-1"
    assert payload["assets"][0]["asset_id"] == "asset-1"
    assert payload["assets"][0]["ocr_text"].endswith("...")
    assert payload["assets"][0]["metadata"]["context_text"]["truncated_fields"] == ["ocr_text"]
    assert payload["triples"][0]["triple_id"] == "triple-1"
    assert payload["metadata"]["max_chars_per_asset_text"] == 50
    assert payload["metadata"]["asset_text_truncated_count"] == 1
    assert payload["metadata"]["source_family_counts"] == {"dense_text": 1, "lexical": 1}
    assert payload["metadata"]["has_visual_context"] is True
    assert payload["metadata"]["has_graph_context"] is True


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
            "query_encoder_details": {
                "text_dense": {
                    "encoder": "default text query encoder",
                    "backend": "sentence-transformers",
                    "model": "BAAI/bge-m3",
                    "dimension": 1024,
                }
            },
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
    assert payload["metadata"]["source_family_counts"] == {"dense_text": 1}
    assert payload["metadata"]["query_encoder_details"]["text_dense"]["model"] == "BAAI/bge-m3"
    assert payload["metadata"]["query_encoder_details"]["text_dense"]["dimension"] == 1024
    assert payload["metadata"]["has_dense_text_context"] is True
    assert payload["metadata"]["has_visual_context"] is True
    assert payload["chunks"][0]["sources"] == ["qdrant:text_dense"]
    assert payload["assets"][0]["asset_id"] == "asset-1"
