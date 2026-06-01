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
    assert artifacts["qdrant_eval.final.json"].exists is True
    assert artifacts["chunking_comparison_gate.final.json"].exists is True
    assert artifacts["graph_audit.final.json"].exists is True
    assert artifacts["visual_gate.final.json"].exists is True
    assert artifacts["visual_run_comparison.json"].exists is True
    validations = {validation.path: validation for validation in report.validation_summaries}
    assert validations["ingestion_readiness.final.json"].passed is True
    assert validations["ingestion_readiness.final.json#retrieval_gate"].metrics["recall_at_k"] == 1.0
    assert validations["retrieval_gate.final.json"].metrics["recall_at_k"] == 1.0
    assert validations["retrieval_gate.final.json"].metrics["mrr"] == 1.0
    assert validations["retrieval_gate.final.json"].metrics["p95_latency_ms"] == 12.0
    assert validations["retrieval_gate.final.json"].metrics[
        "chunk_strategy.visual_asset_text.target_coverage_at_k"
    ] == 1.0
    assert validations["retrieval_gate.final.json"].metrics[
        "retrieval_role.child.target_coverage_at_k"
    ] == 1.0
    assert validations["retrieval_gate.final.json"].metrics[
        "case_group.case_source.visual_object_probe.target_coverage_at_k"
    ] == 1.0
    assert validations["qdrant_eval.final.json"].metrics["target_coverage_at_k"] == 1.0
    assert validations["qdrant_eval.final.json"].metrics["mean_latency_ms"] == 8.0
    assert validations["qdrant_eval.final.json"].metrics["case_count"] == 1.0
    assert validations["qdrant_eval.final.json"].metrics[
        "case_group.modality.vision_object.ndcg_at_k"
    ] == 0.8
    assert validations["chunking_comparison_gate.final.json"].metrics[
        "case_group.case_source.visual_object_probe.target_coverage_at_k"
    ] == 0.9
    assert validations["chunking_comparison_gate.final.json"].metrics[
        "pairwise_target_ndcg_delta_ci_low"
    ] == 0.03
    assert validations["chunking_comparison_gate.final.json"].metrics[
        "pairwise_candidate_win_rate"
    ] == 0.7
    assert validations["graph_audit.final.json"].metrics["orphan_count"] == 0.0
    assert validations["visual_gate.final.json"].metrics["vlm_summary_coverage"] == 1.0
    assert (
        validations["ingestion_readiness.final.json#visual_text_coverage"]
        .metrics["visual_text_coverage_ratio"]
        == 1.0
    )
    assert validations["visual_run_comparison.json"].passed is True
    assert validations["visual_run_comparison.json"].candidate == "structured"
    assert validations["visual_run_comparison.json"].metrics["run_count"] == 2.0
    assert validations["visual_run_comparison.json"].metrics["shared_job_count"] == 1.0
    assert validations["visual_run_comparison.json"].metrics["job_set_mismatch"] == 0.0
    assert validations["visual_run_comparison.json"].metrics["best_quality_score"] == 0.92
    assert report.qdrant_collection["collection"] == "document_chunks"
    assert report.bm25_tokenizer["strategy"] == "mixed"
    assert report.candidate_files == {"current": str(package_dir / "chunks.jsonl")}
    assert report.comparison is not None
    assert report.comparison.best_by_retrieval == "current"
    assert report.comparison.rows[0].visual_text_coverage_ratio == 1.0
    assert report.comparison.rows[0].chunk_strategy_metrics["visual_asset_text"][
        "target_coverage_at_k"
    ] == 1.0
    assert report.comparison.rows[0].retrieval_role_metrics["child"][
        "target_coverage_at_k"
    ] == 1.0


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
    assert payload["validation_summaries"][0]["passed"] is True
    assert any(
        summary["path"] == "ingestion_readiness.final.json#retrieval_gate"
        and summary["metrics"]["recall_at_k"] == 1.0
        for summary in payload["validation_summaries"]
    )
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
            metadata={
                "chunking_strategy": "visual_asset_text",
                "retrieval_role": "child",
            },
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
        json.dumps(
            {
                "passed": True,
                "failed_components": [],
                "components": [
                    {
                        "name": "retrieval_gate",
                        "passed": True,
                        "metadata": {
                            "failed_checks": [],
                            "metrics": {
                                "recall_at_k": 1.0,
                                "target_type.asset.coverage_at_k": 1.0,
                            },
                        },
                    },
                    {
                        "name": "visual_text_coverage",
                        "passed": True,
                        "metadata": {
                            "visual_text_asset_count": 1,
                            "visual_text_covered_asset_count": 1,
                            "visual_text_coverage_ratio": 1.0,
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "retrieval_gate.final.json").write_text(
        json.dumps(
            {
                "passed": True,
                "failed_checks": [],
                "metrics": {
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "p95_latency_ms": 12.0,
                    "target_type.asset.coverage_at_k": 1.0,
                    "chunk_strategy.visual_asset_text.target_coverage_at_k": 1.0,
                    "retrieval_role.child.target_coverage_at_k": 1.0,
                },
                "case_group_metrics": {
                    "case_source": {
                        "visual_object_probe": {
                            "target_coverage_at_k": 1.0,
                            "ndcg_at_k": 1.0,
                        }
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "qdrant_eval.final.json").write_text(
        json.dumps(
            {
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "target_coverage_at_k": 1.0,
                "mean_target_ndcg_at_k": 1.0,
                "mean_latency_ms": 8.0,
                "case_count": 1,
                "case_group_metrics": {
                    "modality": {
                        "vision_object": {
                            "target_coverage_at_k": 1.0,
                            "ndcg_at_k": 0.8,
                        }
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "chunking_comparison_gate.final.json").write_text(
        json.dumps(
            {
                "passed": True,
                "failed_checks": [],
                "candidate": "multimodal",
                "metrics": {"retrieval_target_coverage_at_k": 0.9},
                "case_group_metrics": {
                    "case_source": {
                        "visual_object_probe": {
                            "target_coverage_at_k": 0.9,
                            "precision_at_k": 0.75,
                        }
                    }
                },
                "pairwise_metrics": {
                    "pairwise_candidate_win_rate": 0.7,
                    "pairwise_mean_target_ndcg_delta": 0.05,
                    "pairwise_target_ndcg_delta_ci_low": 0.03,
                    "pairwise_target_ndcg_delta_ci_high": 0.08,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "graph_audit.final.json").write_text(
        json.dumps({"triple_count": 0, "orphan_count": 0, "duplicate_count": 0}, indent=2),
        encoding="utf-8",
    )
    (package_dir / "visual_gate.final.json").write_text(
        json.dumps(
            {
                "passed": True,
                "failed_checks": [],
                "vlm_summary_coverage": 1.0,
                "ocr_text_coverage": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "visual_run_comparison.json").write_text(
        json.dumps(
            {
                "best_by_quality": "structured",
                "fastest_by_total_latency": "raw",
                "best_by_triple_density": "structured",
                "union_job_count": 1,
                "shared_job_count": 1,
                "job_set_mismatch": False,
                "rows": [
                    {
                        "name": "structured",
                        "quality_score": 0.92,
                        "triples_per_vlm_job": 1.0,
                        "total_mean_latency_ms": 42.0,
                    },
                    {
                        "name": "raw",
                        "quality_score": 0.55,
                        "triples_per_vlm_job": 0.0,
                        "total_mean_latency_ms": 18.0,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "qdrant_collection.json").write_text(
        json.dumps({"collection": "document_chunks", "named_vectors": {"text_dense": {"size": 384}}}),
        encoding="utf-8",
    )
    return package_dir, manifest
