import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.readiness import build_ingestion_readiness_report
from chunking_docs.io import write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    TextQuality,
    VisualAsset,
)
from chunking_docs.storage.records import EmbeddingRecord


def test_ingestion_readiness_passes_ready_package(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(package_dir, manifest)

    assert report.passed is True
    assert report.package_counts == {"pages": 1, "chunks": 1, "assets": 1, "triples": 0}
    assert report.artifact_presence["bm25_tokens.json"] is True
    assert report.postgres_row_counts["embedding_artifacts"] == 1
    assert report.failed_components == []


def test_ingestion_readiness_cli_reports_missing_required_artifact(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    (package_dir / "bm25_tokens.json").unlink()
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "bm25_tokens" in payload["failed_components"]


def write_ready_package(tmp_path: Path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    doc = SourceDocument(
        doc_id="doc",
        title="Reference Document",
        local_path=tmp_path / "reference.pdf",
    )
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=100,
            height=100,
            char_count=120,
            line_count=4,
            text_block_count=1,
            image_block_count=1,
            embedded_image_count=0,
            drawing_count=0,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="reference retrieval evidence",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            path=package_dir / "assets/page.png",
            caption="reference visual evidence",
            metadata={"requires_ocr": False, "requires_vlm": False},
        )
    ]
    manifest = ProcessingManifest(doc=doc, profiles=profiles, chunks=chunks, assets=assets)
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", profiles)
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", [])
    (package_dir / "bm25_tokens.json").write_text(json.dumps({"tokenizer": {"strategy": "mixed"}}), encoding="utf-8")
    (package_dir / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "document_chunks",
                "named_vectors": {"text_dense": {"size": 2, "distance": "Cosine"}},
                "payload_indexes": [
                    {"field": "doc_id", "schema": "keyword"},
                    {"field": "chunk_id", "schema": "keyword"},
                    {"field": "asset_id", "schema": "keyword"},
                    {"field": "kind", "schema": "keyword"},
                    {"field": "page_no", "schema": "integer"},
                    {"field": "page_start", "schema": "integer"},
                    {"field": "page_end", "schema": "integer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        package_dir / "qdrant_text_records.jsonl",
        [
            EmbeddingRecord(
                point_id="00000000-0000-0000-0000-000000000001",
                chunk_id="chunk-1",
                doc_id="doc",
                vector_name="text_dense",
                vector=[0.1, 0.2],
                payload={
                    "chunk_id": "chunk-1",
                    "doc_id": "doc",
                    "page_start": 1,
                    "page_end": 1,
                    "kind": "text",
                    "text": "reference retrieval evidence",
                },
            )
        ],
    )
    (package_dir / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": "document_chunks",
                "vectors": {
                    "text_dense": {
                        "file": "qdrant_text_records.jsonl",
                        "record_count": 1,
                        "dimension": 2,
                        "distance": "Cosine",
                        "exists": True,
                        "bytes": 1,
                        "sha256": "a" * 64,
                    }
                },
                "payload_indexes": [{"field": "doc_id", "schema": "keyword"}],
            }
        ),
        encoding="utf-8",
    )
    return package_dir, manifest
