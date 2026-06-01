import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.embeddings.bm25 import BM25LexicalIndex, chunk_lexical_texts
from chunking_docs.evaluation.ablation import (
    QdrantVectorAblationMode,
    QdrantVectorAblationReport,
    QdrantVectorAblationRow,
    RetrievalAblationMode,
    RetrievalAblationReport,
    RetrievalAblationRow,
)
from chunking_docs.evaluation.compare import ChunkingComparison, ChunkingComparisonRow
from chunking_docs.evaluation.readiness import build_ingestion_readiness_report, chunks_with_linked_asset_text
from chunking_docs.evaluation.retrieval import RetrievalCase, evaluate_search_results
from chunking_docs.io import read_jsonl, write_jsonl
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
from chunking_docs.vision.compare import compare_visual_runs
from chunking_docs.vision.jobs import VisualJobRunResult
from chunking_docs.vision.manual_annotations import AssetAnnotation


def test_ingestion_readiness_passes_ready_package(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(package_dir, manifest)

    assert report.passed is True
    assert report.package_counts == {"pages": 1, "chunks": 1, "assets": 1, "triples": 0}
    assert report.artifact_presence["bm25_tokens.json"] is True
    assert report.postgres_row_counts["embedding_artifacts"] == 1
    bm25_component = next(component for component in report.components if component.name == "bm25_tokens")
    assert bm25_component.metadata["chunks_with_linked_asset_text"] == 1
    assert bm25_component.metadata["indexed_linked_asset_text_chunk_count"] == 1
    assert report.failed_components == []


def test_chunks_with_linked_asset_text_counts_source_refs():
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        chunks=[
            DocumentChunk(
                chunk_id="chunk-1",
                doc_id="doc",
                page_start=1,
                page_end=1,
                kind=ChunkKind.TEXT,
                text="visual context",
                source_refs=["asset:asset-1"],
            )
        ],
        assets=[
            VisualAsset(
                asset_id="asset-1",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.MAP,
                caption="visual caption",
            )
        ],
    )

    assert chunks_with_linked_asset_text(manifest) == ["chunk-1"]


def test_ingestion_readiness_can_require_embedding_vectors(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        required_vectors=["text_dense"],
    )

    component = next(component for component in report.components if component.name == "embedding_vectors")
    assert report.passed is True
    assert component.metadata["required_vectors"] == ["text_dense"]
    assert component.metadata["required_vector_details"]["text_dense"]["record_count"] == 1
    assert component.metadata["required_vector_details"]["text_dense"]["dimension"] == 2


def test_ingestion_readiness_flags_missing_required_embedding_vector(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        required_vectors=["text_dense", "caption_dense"],
    )

    component = next(component for component in report.components if component.name == "embedding_vectors")
    assert report.passed is False
    assert "embedding_vectors" in report.failed_components
    assert component.metadata["missing_collection_vectors"] == ["caption_dense"]


def test_ingestion_readiness_detects_stale_bm25_visual_asset_text(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    stale_index = BM25LexicalIndex(manifest.chunks, texts=[chunk.text for chunk in manifest.chunks])
    stale_index.dump_manifest(package_dir / "bm25_tokens.json")

    report = build_ingestion_readiness_report(package_dir, manifest)

    assert report.passed is False
    assert "bm25_tokens" in report.failed_components
    bm25_component = next(component for component in report.components if component.name == "bm25_tokens")
    assert bm25_component.metadata["missing_linked_asset_text_chunk_count"] == 1
    assert bm25_component.metadata["text_char_count_mismatch_count"] == 1


def test_ingestion_readiness_can_gate_package_visual_text_coverage(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        min_visual_text_coverage_ratio=0.8,
    )

    assert report.passed is False
    assert "visual_text_coverage" in report.failed_components
    component = next(component for component in report.components if component.name == "visual_text_coverage")
    assert component.metadata["visual_text_asset_count"] == 1
    assert component.metadata["visual_text_covered_asset_count"] == 0
    assert component.metadata["visual_text_coverage_ratio"] == 0.0
    assert component.metadata["missing_asset_ids"] == ["asset-1"]


def test_ingestion_readiness_reports_standalone_visual_text_chunks(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    standalone_chunk = DocumentChunk(
        chunk_id="visual-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.MAP,
        text="standalone visual evidence",
        asset_ids=["asset-2"],
        source_refs=["asset:asset-2"],
        metadata={
            "chunking_strategy": "visual_asset_text",
            "visual_asset_unlinked": True,
        },
    )
    standalone_asset = VisualAsset(
        asset_id="asset-2",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="standalone visual evidence",
    )
    manifest.chunks.append(standalone_chunk)
    manifest.assets.append(standalone_asset)
    write_jsonl(package_dir / "chunks.jsonl", manifest.chunks)
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)
    write_bm25_manifest(package_dir, manifest.chunks, manifest.assets)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        min_visual_text_coverage_ratio=0.5,
    )

    component = next(component for component in report.components if component.name == "visual_text_coverage")
    assert component.metadata["standalone_visual_chunk_count"] == 1
    assert component.metadata["standalone_visual_text_asset_count"] == 1
    assert component.metadata["standalone_visual_text_asset_ids"] == ["asset-2"]


def test_ingestion_readiness_includes_retrieval_cases_and_chunking_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    cases = [
        RetrievalCase(query="reference evidence", expected_pages=[1]),
        RetrievalCase(
            query="visual evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe", "object_probe_visual_only": True},
        ),
    ]
    comparison = chunking_comparison()

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_cases=cases,
        retrieval_case_options={
            "min_case_count": 2,
            "min_page_cases": 1,
            "min_asset_cases": 1,
            "require_visual_only_object_probes": True,
        },
        chunking_comparison=comparison,
        chunking_gate_options={
            "candidate": "candidate",
            "baseline_candidate": "baseline",
            "min_quality_score": 0.8,
            "min_recall_at_k": 0.8,
            "min_visual_text_coverage_ratio": 0.8,
            "min_target_type_coverage": {"asset": 0.8, "triple": 0.8},
            "min_source_family_target_coverage": {"lexical": 0.8},
            "max_recall_drop": 0.05,
        },
    )

    assert report.passed is True
    assert report.retrieval_case_audit is not None
    assert report.retrieval_case_audit.target_counts["asset"] == 1
    assert report.retrieval_case_audit.visual_only_object_probe_count == 1
    retrieval_component = next(
        component for component in report.components if component.name == "retrieval_case_audit"
    )
    assert retrieval_component.metadata["visual_object_probe_count"] == 1
    assert retrieval_component.metadata["non_visual_only_object_probe_count"] == 0
    assert report.chunking_comparison_gate is not None
    assert report.chunking_comparison_gate.candidate == "candidate"
    assert report.chunking_comparison_gate.metrics["visual_text_coverage_ratio"] == 0.9
    assert report.chunking_comparison_gate.metrics["target_type.asset.coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.metrics["target_type.triple.coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.metrics["source_family.lexical.target_coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.target_metrics["asset"]["coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.source_family_metrics["lexical"][
        "target_coverage_at_k"
    ] == 0.9
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
            "min_target_type_coverage": {"asset": 1.0},
            "min_source_family_target_coverage": {"dense_text": 1.0},
            "min_case_group_target_coverage": {"case_source:visual_object_probe": 1.0},
            "require_best_by_recall": True,
        },
    )

    assert report.passed is True
    assert report.qdrant_vector_ablation_gate is not None
    assert report.qdrant_vector_ablation_gate.mode == "text"
    assert report.qdrant_vector_ablation_gate.metrics["failed_query_count"] == 0.0
    assert report.qdrant_vector_ablation_gate.target_metrics["asset"]["coverage_at_k"] == 1.0
    assert (
        report.qdrant_vector_ablation_gate.source_family_metrics["dense_text"][
            "target_coverage_at_k"
        ]
        == 1.0
    )
    assert report.qdrant_vector_ablation_gate.case_group_metrics["case_source"][
        "visual_object_probe"
    ]["target_coverage_at_k"] == 1.0
    assert report.failed_components == []


def test_ingestion_readiness_includes_retrieval_ablation_lift_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_ablation=retrieval_ablation_report(),
        retrieval_ablation_mode="bm25_visual",
        retrieval_ablation_baseline_mode="bm25_text",
        retrieval_ablation_gate_options={
            "min_recall_at_k": 1.0,
            "min_recall_lift": 1.0,
            "min_target_coverage_lift": 1.0,
            "min_target_type_coverage": {"asset": 1.0},
            "min_source_family_target_coverage": {"lexical": 1.0},
            "min_case_group_target_coverage": {"case_source:visual_lexical_probe": 1.0},
            "require_best_by_recall": True,
        },
    )

    assert report.passed is True
    assert report.retrieval_ablation_gate is not None
    assert report.retrieval_ablation_gate.mode == "bm25_visual"
    assert report.retrieval_ablation_gate.baseline_mode == "bm25_text"
    assert report.retrieval_ablation_gate.metrics["recall_at_k"] == 1.0
    assert report.retrieval_ablation_gate.baseline_metrics["recall_at_k"] == 0.0
    component = next(
        component for component in report.components if component.name == "retrieval_ablation_gate"
    )
    assert component.metadata["metrics"]["target_type.asset.coverage_at_k"] == 1.0
    assert (
        component.metadata["metrics"][
            "case_group.case_source.visual_lexical_probe.target_coverage_at_k"
        ]
        == 1.0
    )
    assert component.metadata["baseline_metrics"]["target_coverage_at_k"] == 0.0
    assert report.failed_components == []


def test_ingestion_readiness_requires_retrieval_ablation(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_retrieval_ablation=True,
    )

    assert report.passed is False
    assert "retrieval_ablation_gate" in report.failed_components


def test_ingestion_readiness_requires_qdrant_vector_ablation(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_qdrant_vector_ablation=True,
    )

    assert report.passed is False
    assert "qdrant_vector_ablation_gate" in report.failed_components


def test_ingestion_readiness_can_gate_retrieval_source_family(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_evaluation=qdrant_vector_ablation_report().rows[0].evaluation,
        retrieval_gate_options={
            "min_recall_at_k": 1.0,
            "min_source_family_target_coverage": {"dense_text": 1.0},
        },
    )

    assert report.passed is True
    assert report.retrieval_gate is not None
    assert report.retrieval_gate.source_family_metrics["dense_text"]["target_coverage_at_k"] == 1.0
    component = next(component for component in report.components if component.name == "retrieval_gate")
    assert component.metadata["source_family_metrics"]["dense_text"]["target_coverage_at_k"] == 1.0
    assert report.failed_components == []


def test_ingestion_readiness_can_gate_retrieval_target_type(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_evaluation=qdrant_vector_ablation_report().rows[0].evaluation,
        retrieval_gate_options={
            "min_recall_at_k": 1.0,
            "min_target_type_coverage": {"asset": 1.0},
        },
    )

    assert report.passed is True
    assert report.retrieval_gate is not None
    assert report.retrieval_gate.target_metrics["asset"]["coverage_at_k"] == 1.0
    component = next(component for component in report.components if component.name == "retrieval_gate")
    assert component.metadata["target_metrics"]["asset"]["coverage_at_k"] == 1.0
    assert report.failed_components == []


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
    write_bm25_manifest(package_dir, manifest.chunks, manifest.assets)

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


def test_ingestion_readiness_can_gate_visual_run_comparison(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    comparison = compare_visual_runs(
        {
            "raw": visual_run_results("raw_text", triple=False),
            "structured": visual_run_results("json_object", triple=True),
        }
    )

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        visual_run_comparison=comparison,
        visual_run_comparison_options={
            "require_same_jobs": True,
            "min_run_count": 2,
            "min_shared_job_count": 1,
            "expected_best_by_quality": "structured",
        },
    )

    component = next(
        component for component in report.components if component.name == "visual_run_comparison"
    )
    assert report.passed is True
    assert report.visual_run_comparison is not None
    assert component.metadata["job_set_mismatch"] is False
    assert component.metadata["best_by_quality"] == "structured"


def test_ingestion_readiness_flags_visual_run_job_mismatch(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    comparison = compare_visual_runs(
        {
            "raw": visual_run_results("raw_text", triple=False, job_id="job-1"),
            "structured": visual_run_results("json_object", triple=True, job_id="job-2"),
        }
    )

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        visual_run_comparison=comparison,
        visual_run_comparison_options={"require_same_jobs": True},
    )

    component = next(
        component for component in report.components if component.name == "visual_run_comparison"
    )
    assert report.passed is False
    assert "visual_run_comparison" in report.failed_components
    assert component.metadata["failed_checks"] == ["same_job_set"]
    assert component.metadata["job_set_mismatch"] is True


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


def test_ingestion_readiness_cli_can_require_visual_only_object_probes(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output = tmp_path / "readiness.json"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="broad object target",
                expected_asset_ids=["asset-1"],
                metadata={
                    "case_source": "visual_object_probe",
                    "object_probe_visual_only": False,
                },
            )
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-cases",
            str(cases_path),
            "--require-visual-only-object-probes",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "retrieval_case_audit" in payload["failed_components"]
    component = next(
        component for component in payload["components"] if component["name"] == "retrieval_case_audit"
    )
    assert component["metadata"]["failed_checks"] == ["require_visual_only_object_probes"]
    assert component["metadata"]["visual_object_probe_count"] == 1
    assert component["metadata"]["non_visual_only_object_probe_count"] == 1


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
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    write_bm25_manifest(package_dir, chunks, assets)
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


def test_ingestion_readiness_cli_can_gate_package_visual_text_coverage(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--min-visual-text-coverage-ratio",
            "0.8",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "visual_text_coverage" in payload["failed_components"]
    component = next(
        component for component in payload["components"] if component["name"] == "visual_text_coverage"
    )
    assert component["metadata"]["visual_text_coverage_ratio"] == 0.0
    assert component["metadata"]["missing_asset_ids"] == ["asset-1"]


def test_ingestion_readiness_cli_can_gate_visual_run_comparison(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    comparison_path = tmp_path / "visual_run_comparison.json"
    output = tmp_path / "readiness.json"
    comparison = compare_visual_runs(
        {
            "raw": visual_run_results("raw_text", triple=False),
            "structured": visual_run_results("json_object", triple=True),
        }
    )
    comparison_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--visual-run-comparison",
            str(comparison_path),
            "--require-visual-run-same-jobs",
            "--min-visual-run-count",
            "2",
            "--min-visual-run-shared-jobs",
            "1",
            "--visual-run-best-by-quality",
            "structured",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "visual_run_comparison"
    )
    assert payload["passed"] is True
    assert component["metadata"]["job_set_mismatch"] is False
    assert component["metadata"]["best_by_quality"] == "structured"


def test_ingestion_readiness_cli_can_require_embedding_vectors(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--required-vector",
            "text_dense",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    component = next(
        component for component in payload["components"] if component["name"] == "embedding_vectors"
    )
    assert payload["passed"] is True
    assert component["metadata"]["required_vectors"] == ["text_dense"]
    assert component["metadata"]["required_vector_details"]["text_dense"]["record_count"] == 1


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
            "--min-qdrant-vector-target-type-coverage",
            "asset=1.0",
            "--min-qdrant-vector-source-family-target-coverage",
            "dense_text=1.0",
            "--min-qdrant-vector-case-group-target-coverage",
            "case_source:visual_object_probe=1.0",
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
    assert component["metadata"]["target_metrics"]["asset"]["coverage_at_k"] == 1.0
    assert component["metadata"]["source_family_metrics"]["dense_text"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["case_group_metrics"]["case_source"]["visual_object_probe"][
        "target_coverage_at_k"
    ] == 1.0


def test_ingestion_readiness_cli_can_gate_retrieval_ablation_lift(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    ablation_path = tmp_path / "retrieval_ablation.json"
    output = tmp_path / "readiness.json"
    ablation_path.write_text(
        retrieval_ablation_report().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-ablation",
            str(ablation_path),
            "--retrieval-ablation-mode",
            "bm25_visual",
            "--retrieval-ablation-baseline-mode",
            "bm25_text",
            "--min-retrieval-ablation-recall-at-k",
            "1.0",
            "--min-retrieval-ablation-recall-lift",
            "1.0",
            "--min-retrieval-ablation-target-coverage-lift",
            "1.0",
            "--min-retrieval-ablation-target-type-coverage",
            "asset=1.0",
            "--min-retrieval-ablation-source-family-target-coverage",
            "lexical=1.0",
            "--min-retrieval-ablation-case-group-target-coverage",
            "case_source:visual_lexical_probe=1.0",
            "--require-retrieval-ablation-best-by-recall",
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
        if component["name"] == "retrieval_ablation_gate"
    )
    assert component["metadata"]["mode"] == "bm25_visual"
    assert component["metadata"]["baseline_mode"] == "bm25_text"
    assert component["metadata"]["metrics"]["recall_at_k"] == 1.0
    assert component["metadata"]["baseline_metrics"]["recall_at_k"] == 0.0
    assert component["metadata"]["case_group_metrics"]["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ] == 1.0


def test_ingestion_readiness_cli_can_gate_chunking_target_coverage(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    comparison_path = tmp_path / "chunking_comparison.json"
    output = tmp_path / "readiness.json"
    comparison_path.write_text(
        chunking_comparison().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--chunking-comparison",
            str(comparison_path),
            "--chunking-candidate",
            "candidate",
            "--baseline-chunking-candidate",
            "baseline",
            "--min-chunking-recall-at-k",
            "0.8",
            "--min-chunking-visual-text-coverage-ratio",
            "0.8",
            "--min-chunking-target-type-coverage",
            "asset=0.8",
            "--min-chunking-target-type-coverage",
            "triple=0.8",
            "--min-chunking-source-family-target-coverage",
            "lexical=0.8",
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
        if component["name"] == "chunking_comparison_gate"
    )
    assert component["metadata"]["candidate"] == "candidate"
    assert component["metadata"]["metrics"]["visual_text_coverage_ratio"] == 0.9
    assert component["metadata"]["metrics"]["target_type.asset.coverage_at_k"] == 0.9
    assert component["metadata"]["metrics"]["target_type.triple.coverage_at_k"] == 0.9
    assert component["metadata"]["metrics"]["source_family.lexical.target_coverage_at_k"] == 0.9
    assert component["metadata"]["target_metrics"]["asset"]["coverage_at_k"] == 0.9
    assert component["metadata"]["source_family_metrics"]["lexical"]["target_coverage_at_k"] == 0.9


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
    write_bm25_manifest(package_dir, chunks, assets)
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


def write_bm25_manifest(
    package_dir: Path,
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
) -> None:
    bm25 = BM25LexicalIndex(
        chunks,
        texts=chunk_lexical_texts(chunks, assets),
    )
    bm25.dump_manifest(package_dir / "bm25_tokens.json")


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
        target_metrics={
            "asset": {"coverage_at_k": recall},
            "triple": {"coverage_at_k": recall},
        },
        source_family_metrics={"lexical": {"target_coverage_at_k": recall}},
        failed_queries=[],
        page_coverage_ratio=1.0,
        visual_annotation_ratio=1.0,
        visual_text_asset_count=10,
        visual_text_covered_asset_count=round(10 * recall),
        visual_text_coverage_ratio=recall,
        chunks_under_min_chars=0,
        chunks_over_max_chars=0,
        issue_codes=[],
    )


def retrieval_ablation_report():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="reference retrieval evidence",
        asset_ids=["asset-1"],
    )
    cases = [
        RetrievalCase(
            query="reference evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_lexical_probe"},
        )
    ]
    passing_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=chunk,
                score=0.9,
                sources=["bm25"],
            )
        ],
        top_k=5,
    )
    failing_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [],
        top_k=5,
    )
    return RetrievalAblationReport(
        rows=[
            RetrievalAblationRow(
                mode=RetrievalAblationMode(
                    name="bm25_visual",
                    use_dense=False,
                    use_bm25=True,
                    include_asset_text=True,
                ),
                evaluation=passing_evaluation,
            ),
            RetrievalAblationRow(
                mode=RetrievalAblationMode(
                    name="bm25_text",
                    use_dense=False,
                    use_bm25=True,
                    include_asset_text=False,
                ),
                evaluation=failing_evaluation,
            ),
        ],
        best_by_recall="bm25_visual",
        best_by_target_coverage="bm25_visual",
        best_by_target_ndcg="bm25_visual",
        best_by_mrr="bm25_visual",
        fastest_by_mean_latency="bm25_text",
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
    cases = [
        RetrievalCase(
            query="reference evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe"},
        )
    ]
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


def visual_run_results(
    parse_status: str,
    triple: bool,
    job_id: str = "job-1",
) -> list[VisualJobRunResult]:
    triples = (
        [{"subject": "subject", "predicate": "relates_to", "object": "object"}]
        if triple
        else []
    )
    return [
        VisualJobRunResult(
            job_id=job_id,
            asset_id="asset-1",
            page_no=1,
            status="completed",
            annotation=AssetAnnotation(
                asset_id="asset-1",
                page_no=1,
                ocr_text="recognized visual text",
                vlm_summary="structured visual evidence" if triple else "plain visual evidence",
                triples=triples,
                metadata={"vlm_parse_status": parse_status},
            ),
            metadata={
                "operations": ["ocr", "vlm"],
                "ocr_text_chars": 22,
                "vlm_parse_status": parse_status,
                "total_duration_ms": 40.0 if triple else 20.0,
            },
        )
    ]
