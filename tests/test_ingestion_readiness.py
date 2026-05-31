import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.ablation import (
    QdrantVectorAblationMode,
    QdrantVectorAblationReport,
    QdrantVectorAblationRow,
)
from chunking_docs.evaluation.compare import ChunkingComparison, ChunkingComparisonRow
from chunking_docs.evaluation.readiness import build_ingestion_readiness_report
from chunking_docs.evaluation.retrieval import RetrievalCase, evaluate_search_results
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
from chunking_docs.retrieval.local_hybrid import HybridSearchHit
from chunking_docs.storage.records import EmbeddingRecord


def test_ingestion_readiness_passes_ready_package(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(package_dir, manifest)

    assert report.passed is True
    assert report.package_counts == {"pages": 1, "chunks": 1, "assets": 1, "triples": 0}
    assert report.artifact_presence["bm25_tokens.json"] is True
    assert report.postgres_row_counts["embedding_artifacts"] == 1
    assert report.failed_components == []


def test_ingestion_readiness_includes_retrieval_cases_and_chunking_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    cases = [
        RetrievalCase(query="reference evidence", expected_pages=[1]),
        RetrievalCase(query="visual evidence", expected_asset_ids=["asset-1"]),
    ]
    comparison = chunking_comparison()

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_cases=cases,
        retrieval_case_options={"min_case_count": 2, "min_page_cases": 1, "min_asset_cases": 1},
        chunking_comparison=comparison,
        chunking_gate_options={
            "candidate": "candidate",
            "baseline_candidate": "baseline",
            "min_quality_score": 0.8,
            "min_recall_at_k": 0.8,
            "max_recall_drop": 0.05,
        },
    )

    assert report.passed is True
    assert report.retrieval_case_audit is not None
    assert report.retrieval_case_audit.target_counts["asset"] == 1
    assert report.chunking_comparison_gate is not None
    assert report.chunking_comparison_gate.candidate == "candidate"
    assert report.failed_components == []


def test_ingestion_readiness_includes_qdrant_vector_ablation_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        qdrant_vector_ablation=qdrant_vector_ablation_report(),
        qdrant_vector_ablation_mode="text",
        qdrant_vector_ablation_gate_options={
            "min_recall_at_k": 1.0,
            "min_target_coverage_at_k": 1.0,
            "max_failed_queries": 0,
            "min_source_family_target_coverage": {"dense_text": 1.0},
            "require_best_by_recall": True,
        },
    )

    assert report.passed is True
    assert report.qdrant_vector_ablation_gate is not None
    assert report.qdrant_vector_ablation_gate.mode == "text"
    assert report.qdrant_vector_ablation_gate.metrics["failed_query_count"] == 0.0
    assert (
        report.qdrant_vector_ablation_gate.source_family_metrics["dense_text"][
            "target_coverage_at_k"
        ]
        == 1.0
    )
    assert report.failed_components == []


def test_ingestion_readiness_requires_qdrant_vector_ablation(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_qdrant_vector_ablation=True,
    )

    assert report.passed is False
    assert "qdrant_vector_ablation_gate" in report.failed_components


def test_ingestion_readiness_can_gate_visual_quality_from_assets(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.assets[0] = manifest.assets[0].model_copy(
        update={
            "ocr_text": "recognized visual text",
            "vlm_summary": "structured visual summary",
            "metadata": {
                "requires_ocr": True,
                "requires_vlm": True,
                "vlm_parse_status": "json_object",
            },
        }
    )

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_visual_annotations=True,
        require_visual_quality=True,
        visual_quality_options={
            "min_completion_rate": 1.0,
            "min_ocr_text_coverage": 1.0,
            "min_vlm_summary_coverage": 1.0,
            "min_vlm_json_parse_rate": 1.0,
        },
    )
    visual_component = next(component for component in report.components if component.name == "visual_quality")

    assert report.passed is True
    assert report.visual_quality is not None
    assert report.visual_quality.vlm_json_parse_rate == 1.0
    assert visual_component.metadata["source"] == "assets"


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


def test_ingestion_readiness_cli_requires_retrieval_cases(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--require-retrieval-cases",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "retrieval_case_audit" in payload["failed_components"]


def test_ingestion_readiness_cli_can_gate_visual_quality_from_assets(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            path=package_dir / "assets/page.png",
            ocr_text="recognized visual text",
            vlm_summary="structured visual summary",
            metadata={
                "requires_ocr": True,
                "requires_vlm": True,
                "vlm_parse_status": "json_object",
            },
        )
    ]
    write_jsonl(package_dir / "assets.jsonl", assets)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--require-visual-annotations",
            "--require-visual-quality",
            "--min-visual-completion-rate",
            "1",
            "--min-ocr-text-coverage",
            "1",
            "--min-vlm-summary-coverage",
            "1",
            "--min-vlm-json-parse-rate",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    visual_component = next(
        component for component in payload["components"] if component["name"] == "visual_quality"
    )
    assert visual_component["metadata"]["source"] == "assets"


def test_ingestion_readiness_cli_can_gate_qdrant_vector_ablation(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    ablation_path = tmp_path / "qdrant_vector_ablation.json"
    output = tmp_path / "readiness.json"
    ablation_path.write_text(
        qdrant_vector_ablation_report().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--qdrant-vector-ablation",
            str(ablation_path),
            "--qdrant-vector-mode",
            "text",
            "--min-qdrant-vector-recall-at-k",
            "1.0",
            "--min-qdrant-vector-target-coverage-at-k",
            "1.0",
            "--max-qdrant-vector-failed-queries",
            "0",
            "--min-qdrant-vector-source-family-target-coverage",
            "dense_text=1.0",
            "--require-qdrant-vector-best-by-recall",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "qdrant_vector_ablation_gate"
    )
    assert component["metadata"]["mode"] == "text"
    assert component["metadata"]["metrics"]["recall_at_k"] == 1.0
    assert component["metadata"]["source_family_metrics"]["dense_text"]["target_coverage_at_k"] == 1.0


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
    record_content = (package_dir / "qdrant_text_records.jsonl").read_bytes()
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
                        "bytes": len(record_content),
                        "sha256": hashlib.sha256(record_content).hexdigest(),
                    }
                },
                "payload_indexes": [{"field": "doc_id", "schema": "keyword"}],
            }
        ),
        encoding="utf-8",
    )
    return package_dir, manifest


def chunking_comparison():
    return ChunkingComparison(
        rows=[
            chunking_row("candidate", quality_score=0.9, recall=0.9),
            chunking_row("baseline", quality_score=0.85, recall=0.88),
        ],
        best_by_quality="candidate",
        best_by_retrieval="candidate",
        fastest_by_mean_latency="candidate",
    )


def chunking_row(name: str, quality_score: float, recall: float):
    return ChunkingComparisonRow(
        name=name,
        chunk_count=1,
        quality_score=quality_score,
        retrieval_hit_rate=recall,
        retrieval_recall_at_k=recall,
        retrieval_mrr=recall,
        retrieval_target_coverage_at_k=recall,
        retrieval_mean_target_ndcg_at_k=recall,
        retrieval_mean_precision_at_k=recall,
        retrieval_mean_latency_ms=5.0,
        retrieval_p95_latency_ms=7.0,
        failed_queries=[],
        page_coverage_ratio=1.0,
        visual_annotation_ratio=1.0,
        chunks_under_min_chars=0,
        chunks_over_max_chars=0,
        issue_codes=[],
    )


def qdrant_vector_ablation_report():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="reference retrieval evidence",
        asset_ids=["asset-1"],
    )
    cases = [RetrievalCase(query="reference evidence", expected_asset_ids=["asset-1"])]
    passing_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=chunk,
                score=0.9,
                sources=["qdrant:text_dense"],
            )
        ],
        top_k=5,
    )
    failing_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [],
        top_k=5,
    )
    return QdrantVectorAblationReport(
        rows=[
            QdrantVectorAblationRow(
                mode=QdrantVectorAblationMode(name="text", vector_names=["text_dense"]),
                evaluation=passing_evaluation,
            ),
            QdrantVectorAblationRow(
                mode=QdrantVectorAblationMode(
                    name="caption",
                    vector_names=["caption_dense"],
                ),
                evaluation=failing_evaluation,
            ),
        ],
        best_by_recall="text",
        best_by_target_coverage="text",
        best_by_target_ndcg="text",
        best_by_mrr="text",
        fastest_by_mean_latency="caption",
    )
