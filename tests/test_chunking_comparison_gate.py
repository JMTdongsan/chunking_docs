import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.chunking_gate import (
    gate_chunking_comparison,
    load_chunking_comparison,
)
from chunking_docs.evaluation.compare import (
    ChunkingComparison,
    ChunkingComparisonRow,
    ChunkingPairwiseComparison,
)


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
        min_visual_text_coverage_ratio=0.8,
        min_target_type_coverage={"asset": 0.8},
        min_source_family_target_coverage={"lexical": 0.8},
        min_chunk_strategy_target_coverage={"visual_asset_text": 0.8},
        min_retrieval_role_target_coverage={"child": 0.8},
        max_failed_queries=0,
        max_recall_drop=0.05,
        max_mean_latency_ratio=2.0,
    )

    assert report.passed is True
    assert report.candidate == "strong"
    assert report.baseline_candidate == "weak"
    assert report.failed_checks == []
    assert report.metrics["retrieval_recall_at_k"] == 0.9
    assert report.metrics["visual_text_coverage_ratio"] == 0.9
    assert report.metrics["target_type.asset.coverage_at_k"] == 0.85
    assert report.metrics["source_family.lexical.target_coverage_at_k"] == 0.9
    assert report.metrics["chunk_strategy.visual_asset_text.target_coverage_at_k"] == 0.9
    assert report.metrics["retrieval_role.child.target_coverage_at_k"] == 0.9
    assert report.target_metrics["asset"]["coverage_at_k"] == 0.85
    assert report.source_family_metrics["lexical"]["target_coverage_at_k"] == 0.9
    assert report.chunk_strategy_metrics["visual_asset_text"]["target_coverage_at_k"] == 0.9
    assert report.retrieval_role_metrics["child"]["target_coverage_at_k"] == 0.9


def test_gate_chunking_comparison_flags_retrieval_regressions():
    comparison = comparison_report()

    report = gate_chunking_comparison(
        comparison,
        candidate="weak",
        baseline_candidate="strong",
        require_retrieval=True,
        min_recall_at_k=0.8,
        min_target_coverage_at_k=0.75,
        min_visual_text_coverage_ratio=0.8,
        min_target_type_coverage={"asset": 0.8},
        min_source_family_target_coverage={"lexical": 0.8},
        min_chunk_strategy_target_coverage={"visual_asset_text": 0.8},
        min_retrieval_role_target_coverage={"child": 0.8},
        max_failed_queries=0,
        max_recall_drop=0.1,
        max_mean_latency_ratio=1.5,
    )

    assert report.passed is False
    assert "min_recall_at_k" in report.failed_checks
    assert "min_target_coverage_at_k" in report.failed_checks
    assert "min_visual_text_coverage_ratio" in report.failed_checks
    assert "min_target_type_coverage:asset" in report.failed_checks
    assert "min_source_family_target_coverage:lexical" in report.failed_checks
    assert "min_chunk_strategy_target_coverage:visual_asset_text" in report.failed_checks
    assert "min_retrieval_role_target_coverage:child" in report.failed_checks
    assert "max_failed_queries" in report.failed_checks
    assert "max_recall_at_k_drop" in report.failed_checks
    assert "max_mean_latency_ms_ratio" in report.failed_checks


def test_gate_chunking_comparison_checks_pairwise_lift():
    comparison = comparison_report()

    report = gate_chunking_comparison(
        comparison,
        candidate="strong",
        baseline_candidate="weak",
        min_pairwise_shared_queries=10,
        min_pairwise_win_rate=0.6,
        min_pairwise_target_coverage_lift=0.2,
        min_pairwise_target_ndcg_lift=0.2,
        min_pairwise_mrr_lift=0.2,
        min_pairwise_precision_lift=0.2,
        max_pairwise_mean_latency_delta_ms=10.0,
    )

    assert report.passed is True
    assert report.pairwise_metrics["pairwise_shared_query_count"] == 10.0
    assert report.pairwise_metrics["pairwise_candidate_win_rate"] == 0.7
    assert report.pairwise_metrics["pairwise_mean_target_ndcg_delta"] == 0.25


def test_gate_chunking_comparison_flags_missing_pairwise_lift():
    comparison = comparison_report()

    report = gate_chunking_comparison(
        comparison,
        candidate="weak",
        baseline_candidate="strong",
        min_pairwise_win_rate=0.6,
        min_pairwise_target_coverage_lift=0.0,
        min_pairwise_target_ndcg_lift=0.0,
        max_pairwise_mean_latency_delta_ms=0.0,
    )

    assert report.passed is False
    assert report.pairwise_metrics["pairwise_candidate_win_rate"] == 0.2
    assert "min_pairwise_win_rate" in report.failed_checks
    assert "min_pairwise_target_coverage_lift" in report.failed_checks
    assert "min_pairwise_target_ndcg_lift" in report.failed_checks
    assert "max_pairwise_mean_latency_delta_ms" in report.failed_checks


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
            "--min-target-type-coverage",
            "asset=0.8",
            "--min-visual-text-coverage-ratio",
            "0.8",
            "--min-source-family-target-coverage",
            "lexical=0.8",
            "--min-chunk-strategy-target-coverage",
            "visual_asset_text=0.8",
            "--min-retrieval-role-target-coverage",
            "child=0.8",
            "--max-recall-drop",
            "0.1",
            "--min-pairwise-win-rate",
            "0.6",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate"] == "weak"
    assert "min_recall_at_k" in payload["failed_checks"]
    assert "min_visual_text_coverage_ratio" in payload["failed_checks"]
    assert "min_target_type_coverage:asset" in payload["failed_checks"]
    assert "min_source_family_target_coverage:lexical" in payload["failed_checks"]
    assert "min_chunk_strategy_target_coverage:visual_asset_text" in payload["failed_checks"]
    assert "min_retrieval_role_target_coverage:child" in payload["failed_checks"]
    assert payload["target_metrics"]["asset"]["coverage_at_k"] == 0.2
    assert payload["source_family_metrics"]["lexical"]["target_coverage_at_k"] == 0.1
    assert payload["chunk_strategy_metrics"]["visual_asset_text"]["target_coverage_at_k"] == 0.2
    assert payload["retrieval_role_metrics"]["child"]["target_coverage_at_k"] == 0.1
    assert "max_recall_at_k_drop" in payload["failed_checks"]
    assert "min_pairwise_win_rate" in payload["failed_checks"]
    assert payload["pairwise_metrics"]["pairwise_candidate_win_rate"] == 0.2


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
                target_metrics={"asset": {"coverage_at_k": 0.85}},
                source_family_metrics={"lexical": {"target_coverage_at_k": 0.9}},
                chunk_strategy_metrics={"visual_asset_text": {"target_coverage_at_k": 0.9}},
                retrieval_role_metrics={"child": {"target_coverage_at_k": 0.9}},
                visual_text_coverage=0.9,
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
                target_metrics={"asset": {"coverage_at_k": 0.2}},
                source_family_metrics={"lexical": {"target_coverage_at_k": 0.1}},
                chunk_strategy_metrics={"visual_asset_text": {"target_coverage_at_k": 0.2}},
                retrieval_role_metrics={"child": {"target_coverage_at_k": 0.1}},
                visual_text_coverage=0.2,
            ),
        ],
        best_by_quality="strong",
        best_by_retrieval="strong",
        fastest_by_mean_latency="strong",
        pairwise=[
            ChunkingPairwiseComparison(
                candidate="strong",
                baseline="weak",
                shared_query_count=10,
                candidate_win_count=7,
                baseline_win_count=1,
                tie_count=2,
                candidate_win_rate=0.7,
                baseline_win_rate=0.1,
                mean_reciprocal_rank_delta=0.25,
                mean_target_coverage_delta=0.3,
                mean_target_ndcg_delta=0.25,
                mean_precision_delta=0.2,
                mean_latency_delta_ms=-18.0,
            ),
            ChunkingPairwiseComparison(
                candidate="weak",
                baseline="strong",
                shared_query_count=10,
                candidate_win_count=2,
                baseline_win_count=7,
                tie_count=1,
                candidate_win_rate=0.2,
                baseline_win_rate=0.7,
                mean_reciprocal_rank_delta=-0.25,
                mean_target_coverage_delta=-0.3,
                mean_target_ndcg_delta=-0.25,
                mean_precision_delta=-0.2,
                mean_latency_delta_ms=18.0,
            ),
        ],
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
    target_metrics: dict[str, dict[str, float]] | None = None,
    source_family_metrics: dict[str, dict[str, float]] | None = None,
    chunk_strategy_metrics: dict[str, dict[str, float]] | None = None,
    retrieval_role_metrics: dict[str, dict[str, float]] | None = None,
    visual_text_coverage: float = 1.0,
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
        target_metrics=target_metrics or {},
        source_family_metrics=source_family_metrics or {},
        chunk_strategy_metrics=chunk_strategy_metrics or {},
        retrieval_role_metrics=retrieval_role_metrics or {},
        failed_queries=failed_queries,
        page_coverage_ratio=1.0,
        visual_annotation_ratio=0.5,
        visual_text_asset_count=10,
        visual_text_covered_asset_count=round(10 * visual_text_coverage),
        visual_text_coverage_ratio=visual_text_coverage,
        chunks_under_min_chars=0,
        chunks_over_max_chars=0,
        issue_codes=[],
    )
