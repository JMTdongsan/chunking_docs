import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.experiment import build_experiment_report
from chunking_docs.evaluation.retrieval import RetrievalCase
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


def test_build_experiment_report_summarizes_artifacts_and_candidates(tmp_path):
    package_dir, manifest = write_minimal_package(tmp_path)
    cases = [RetrievalCase(query="retrieval benchmark", expected_pages=[1])]

    report = build_experiment_report(
        package_dir=package_dir,
        manifest=manifest,
        candidates={"current": package_dir / "chunks.jsonl"},
        retrieval_cases=cases,
        min_chars=10,
    )

    artifacts = {artifact.path: artifact for artifact in report.artifacts}
    assert report.package_counts == {"pages": 1, "chunks": 1, "assets": 1, "triples": 0}
    assert artifacts["chunks.jsonl"].record_count == 1
    assert artifacts["chunks.jsonl"].sha256 is not None
    assert artifacts["ingestion_readiness.final.json"].exists is True
    assert artifacts["retrieval_gate.final.json"].exists is True
    assert report.qdrant_collection["collection"] == "document_chunks"
    assert report.bm25_tokenizer["strategy"] == "mixed"
    assert report.candidate_files == {"current": str(package_dir / "chunks.jsonl")}
    assert report.comparison is not None
    assert report.comparison.best_by_retrieval == "current"


def test_write_experiment_report_cli_writes_json(tmp_path):
    package_dir, _ = write_minimal_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "experiment_report.json"
    write_jsonl(cases_path, [RetrievalCase(query="retrieval benchmark", expected_pages=[1])])

    result = CliRunner().invoke(
        app,
        [
            "write-experiment-report",
            "--package-dir",
            str(package_dir),
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparison"]["best_by_retrieval"] == "current"
    assert payload["artifacts"][0]["sha256"]


def write_minimal_package(tmp_path):
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
            char_count=128,
            line_count=6,
            text_block_count=1,
            image_block_count=1,
            embedded_image_count=0,
            drawing_count=2,
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
            text="retrieval benchmark visual evidence table policy",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.TABLE,
            caption="benchmark table",
            vlm_summary="visual evidence table",
        )
    ]
    manifest = ProcessingManifest(
        doc=doc,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=[],
        metadata={"profile_summary": {"page_count": 1, "degraded_page_count": 0}},
    )
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", profiles)
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", [])
    (package_dir / "bm25_tokens.json").write_text(
        json.dumps({"tokenizer": {"strategy": "mixed"}}, indent=2),
        encoding="utf-8",
    )
    (package_dir / "ingestion_readiness.final.json").write_text(
        json.dumps({"passed": True}, indent=2),
        encoding="utf-8",
    )
    (package_dir / "retrieval_gate.final.json").write_text(
        json.dumps({"passed": True}, indent=2),
        encoding="utf-8",
    )
    (package_dir / "qdrant_collection.json").write_text(
        json.dumps({"collection": "document_chunks", "named_vectors": {"text_dense": {"size": 384}}}),
        encoding="utf-8",
    )
    return package_dir, manifest
