import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.chunking_gate import (
    gate_chunking_comparison,
    load_chunking_comparison,
)
from chunking_docs.evaluation.compare import ChunkingComparison, ChunkingComparisonRow


def test_gate_chunking_comparison_passes_selected_candidate_against_baseline():
    comparison = comparison_report()

    report = gate_chunking_comparison(
        comparison,
        candidate="strong",
        baseline_candidate="weak",
        require_retrieval=True,
        min_quality_score=0.8,
        min_recall_at_k=0.8,
        min_target_coverage_at_k=0.8,
        min_target_ndcg_at_k=0.75,
        min_precision_at_k=0.5,
        max_failed_queries=0,
        max_recall_drop=0.05,
        max_mean_latency_ratio=2.0,
    )

    assert report.passed is True
    assert report.candidate == "strong"
    assert report.baseline_candidate == "weak"
    assert report.failed_checks == []
    assert report.metrics["retrieval_recall_at_k"] == 0.9


def test_gate_chunking_comparison_flags_retrieval_regressions():
    comparison = comparison_report()

    report = gate_chunking_comparison(
        comparison,
        candidate="weak",
        baseline_candidate="strong",
        require_retrieval=True,
        min_recall_at_k=0.8,
        min_target_coverage_at_k=0.75,
        max_failed_queries=0,
        max_recall_drop=0.1,
        max_mean_latency_ratio=1.5,
    )

    assert report.passed is False
    assert "min_recall_at_k" in report.failed_checks
    assert "min_target_coverage_at_k" in report.failed_checks
    assert "max_failed_queries" in report.failed_checks
    assert "max_recall_at_k_drop" in report.failed_checks
    assert "max_mean_latency_ms_ratio" in report.failed_checks


def test_load_chunking_comparison_accepts_nested_report_shape(tmp_path):
    path = tmp_path / "chunking_sweep.json"
    path.write_text(json.dumps({"comparison": comparison_report().model_dump()}), encoding="utf-8")

    loaded = load_chunking_comparison(path)

    assert loaded.best_by_retrieval == "strong"
    assert loaded.rows[0].name == "strong"


def test_gate_chunking_comparison_cli_writes_json_and_fails(tmp_path):
    comparison_path = tmp_path / "comparison.json"
    output_path = tmp_path / "gate.json"
    comparison_path.write_text(comparison_report().model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "gate-chunking-comparison",
            str(comparison_path),
            "--candidate",
            "weak",
            "--baseline-candidate",
            "strong",
            "--require-retrieval",
            "--min-recall-at-k",
            "0.8",
            "--max-recall-drop",
            "0.1",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate"] == "weak"
    assert "min_recall_at_k" in payload["failed_checks"]
    assert "max_recall_at_k_drop" in payload["failed_checks"]


def comparison_report() -> ChunkingComparison:
    return ChunkingComparison(
        rows=[
            row(
                name="strong",
                quality_score=0.92,
                recall=0.9,
                target_coverage=0.85,
                target_ndcg=0.82,
                precision=0.6,
                mean_latency=12.0,
                p95_latency=20.0,
                failed_queries=[],
            ),
            row(
                name="weak",
                quality_score=0.7,
                recall=0.5,
                target_coverage=0.4,
                target_ndcg=0.35,
                precision=0.25,
                mean_latency=30.0,
                p95_latency=55.0,
                failed_queries=["missing target"],
            ),
        ],
        best_by_quality="strong",
        best_by_retrieval="strong",
        fastest_by_mean_latency="strong",
    )


def row(
    name: str,
    quality_score: float,
    recall: float,
    target_coverage: float,
    target_ndcg: float,
    precision: float,
    mean_latency: float,
    p95_latency: float,
    failed_queries: list[str],
) -> ChunkingComparisonRow:
    return ChunkingComparisonRow(
        name=name,
        chunk_count=12,
        quality_score=quality_score,
        retrieval_hit_rate=recall,
        retrieval_recall_at_k=recall,
        retrieval_mrr=0.75,
        retrieval_target_coverage_at_k=target_coverage,
        retrieval_mean_target_ndcg_at_k=target_ndcg,
        retrieval_mean_precision_at_k=precision,
        retrieval_mean_latency_ms=mean_latency,
        retrieval_p95_latency_ms=p95_latency,
        failed_queries=failed_queries,
        page_coverage_ratio=1.0,
        visual_annotation_ratio=0.5,
        chunks_under_min_chars=0,
        chunks_over_max_chars=0,
        issue_codes=[],
    )
