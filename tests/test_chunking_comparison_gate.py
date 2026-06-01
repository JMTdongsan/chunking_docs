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
        max_mean_target_rank=2.0,
        min_result_stability_rate=1.0,
        max_unstable_result_count=0,
        min_visual_text_coverage_ratio=0.8,
        min_visual_text_part_coverage_ratio=0.8,
        min_target_type_coverage={"asset": 0.8},
        min_source_target_coverage={"bm25": 0.8},
        min_source_family_target_coverage={"lexical": 0.8},
        min_chunk_strategy_target_coverage={"visual_asset_text": 0.8},
        min_retrieval_role_target_coverage={"child": 0.8},
        min_case_group_target_coverage={"case_source:visual_lexical_probe": 0.8},
        min_case_group_source_target_coverage={
            "case_source:visual_lexical_probe:bm25": 0.8
        },
        min_case_group_source_family_target_coverage={
            "case_source:visual_lexical_probe:lexical": 0.8
        },
        max_failed_queries=0,
        max_total_chunk_chars=2000,
        max_embedding_text_kchars=2.0,
        min_retrieval_score_per_embedding_kchar=0.5,
        min_target_coverage_per_embedding_kchar=0.5,
        min_target_ndcg_per_embedding_kchar=0.5,
        min_retrieval_score_per_mean_latency_ms=0.05,
        min_target_coverage_per_mean_latency_ms=0.05,
        min_target_ndcg_per_mean_latency_ms=0.05,
        min_retrieval_score_per_p95_latency_ms=0.03,
        min_target_coverage_per_p95_latency_ms=0.03,
        min_target_ndcg_per_p95_latency_ms=0.03,
        max_recall_drop=0.05,
        max_mean_latency_ratio=2.0,
    )

    assert report.passed is True
    assert report.candidate == "strong"
    assert report.baseline_candidate == "weak"
    assert report.failed_checks == []
    assert report.metrics["retrieval_recall_at_k"] == 0.9
    assert report.metrics["retrieval_mean_target_rank"] == 1.0
    assert report.metrics["retrieval_result_stability_rate"] == 1.0
    assert report.metrics["retrieval_unstable_result_count"] == 0.0
    assert report.metrics["total_chunk_chars"] == 1200.0
    assert report.metrics["embedding_text_kchars"] == 1.2
    assert report.metrics["retrieval_score_per_embedding_kchar"] > 0.5
    assert report.metrics["retrieval_score_per_mean_latency_ms"] > 0.05
    assert report.metrics["target_coverage_per_mean_latency_ms"] > 0.05
    assert report.metrics["target_ndcg_per_mean_latency_ms"] > 0.05
    assert report.metrics["retrieval_score_per_p95_latency_ms"] > 0.03
    assert report.metrics["target_coverage_per_p95_latency_ms"] > 0.03
    assert report.metrics["target_ndcg_per_p95_latency_ms"] > 0.03
    assert report.metrics["visual_text_coverage_ratio"] == 0.9
    assert report.metrics["visual_text_part_coverage_ratio"] == 0.9
    assert report.metrics["target_type.asset.coverage_at_k"] == 0.85
    assert report.metrics["source.bm25.target_coverage_at_k"] == 0.9
    assert report.metrics["source_family.lexical.target_coverage_at_k"] == 0.9
    assert report.metrics["chunk_strategy.visual_asset_text.target_coverage_at_k"] == 0.9
    assert report.metrics["retrieval_role.child.target_coverage_at_k"] == 0.9
    assert report.metrics["case_group.case_source.visual_lexical_probe.target_coverage_at_k"] == 0.9
    assert (
        report.metrics[
            "case_group_source.case_source.visual_lexical_probe.bm25.target_coverage_at_k"
        ]
        == 0.9
    )
    assert (
        report.metrics[
            "case_group_source_family.case_source.visual_lexical_probe.lexical.target_coverage_at_k"
        ]
        == 0.9
    )
    assert report.target_metrics["asset"]["coverage_at_k"] == 0.85
    assert report.source_metrics["bm25"]["target_coverage_at_k"] == 0.9
    assert report.source_family_metrics["lexical"]["target_coverage_at_k"] == 0.9
    assert report.chunk_strategy_metrics["visual_asset_text"]["target_coverage_at_k"] == 0.9
    assert report.retrieval_role_metrics["child"]["target_coverage_at_k"] == 0.9
    assert report.case_group_metrics["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ] == 0.9
    assert report.case_group_source_metrics["case_source"]["visual_lexical_probe"][
        "bm25"
    ]["target_coverage_at_k"] == 0.9
    assert report.case_group_source_family_metrics["case_source"][
        "visual_lexical_probe"
    ]["lexical"]["target_coverage_at_k"] == 0.9


def test_gate_chunking_comparison_flags_retrieval_regressions():
    comparison = comparison_report()

    report = gate_chunking_comparison(
        comparison,
        candidate="weak",
        baseline_candidate="strong",
        require_retrieval=True,
        min_recall_at_k=0.8,
        min_target_coverage_at_k=0.75,
        max_mean_target_rank=2.0,
        min_result_stability_rate=1.0,
        max_unstable_result_count=0,
        min_visual_text_coverage_ratio=0.8,
        min_visual_text_part_coverage_ratio=0.8,
        min_target_type_coverage={"asset": 0.8},
        min_source_target_coverage={"bm25": 0.8},
        min_source_family_target_coverage={"lexical": 0.8},
        min_chunk_strategy_target_coverage={"visual_asset_text": 0.8},
        min_retrieval_role_target_coverage={"child": 0.8},
        min_case_group_target_coverage={"case_source:visual_lexical_probe": 0.8},
        min_case_group_source_target_coverage={
            "case_source:visual_lexical_probe:bm25": 0.8
        },
        min_case_group_source_family_target_coverage={
            "case_source:visual_lexical_probe:lexical": 0.8
        },
        max_failed_queries=0,
        max_total_chunk_chars=2000,
        max_embedding_text_kchars=2.0,
        min_retrieval_score_per_embedding_kchar=0.5,
        min_target_coverage_per_embedding_kchar=0.5,
        min_target_ndcg_per_embedding_kchar=0.5,
        min_retrieval_score_per_mean_latency_ms=0.05,
        min_target_coverage_per_mean_latency_ms=0.05,
        min_target_ndcg_per_mean_latency_ms=0.05,
        min_retrieval_score_per_p95_latency_ms=0.03,
        min_target_coverage_per_p95_latency_ms=0.03,
        min_target_ndcg_per_p95_latency_ms=0.03,
        max_recall_drop=0.1,
        max_mean_latency_ratio=1.5,
    )

    assert report.passed is False
    assert "min_recall_at_k" in report.failed_checks
    assert "min_target_coverage_at_k" in report.failed_checks
    assert "max_mean_target_rank" in report.failed_checks
    assert "min_result_stability_rate" in report.failed_checks
    assert "max_unstable_result_count" in report.failed_checks
    assert "min_visual_text_coverage_ratio" in report.failed_checks
    assert "min_visual_text_part_coverage_ratio" in report.failed_checks
    assert "min_target_type_coverage:asset" in report.failed_checks
    assert "min_source_target_coverage:bm25" in report.failed_checks
    assert "min_source_family_target_coverage:lexical" in report.failed_checks
    assert "min_chunk_strategy_target_coverage:visual_asset_text" in report.failed_checks
    assert "min_retrieval_role_target_coverage:child" in report.failed_checks
    assert (
        "min_case_group_target_coverage:case_source:visual_lexical_probe"
        in report.failed_checks
    )
    assert (
        "min_case_group_source_target_coverage:case_source:visual_lexical_probe:bm25"
        in report.failed_checks
    )
    assert (
        "min_case_group_source_family_target_coverage:"
        "case_source:visual_lexical_probe:lexical"
        in report.failed_checks
    )
    assert "max_failed_queries" in report.failed_checks
    assert "max_total_chunk_chars" in report.failed_checks
    assert "max_embedding_text_kchars" in report.failed_checks
    assert "min_retrieval_score_per_embedding_kchar" in report.failed_checks
    assert "min_target_coverage_per_embedding_kchar" in report.failed_checks
    assert "min_target_ndcg_per_embedding_kchar" in report.failed_checks
    assert "min_retrieval_score_per_mean_latency_ms" in report.failed_checks
    assert "min_target_coverage_per_mean_latency_ms" in report.failed_checks
    assert "min_target_ndcg_per_mean_latency_ms" in report.failed_checks
    assert "min_retrieval_score_per_p95_latency_ms" in report.failed_checks
    assert "min_target_coverage_per_p95_latency_ms" in report.failed_checks
    assert "min_target_ndcg_per_p95_latency_ms" in report.failed_checks
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
        min_pairwise_target_coverage_ci_low=0.1,
        min_pairwise_target_ndcg_ci_low=0.1,
        min_pairwise_mrr_ci_low=0.1,
        min_pairwise_precision_ci_low=0.1,
        max_pairwise_mean_first_relevant_rank_delta=0.0,
        max_pairwise_mean_target_rank_delta=0.0,
        max_pairwise_first_relevant_rank_delta_ci_high=0.0,
        max_pairwise_target_rank_delta_ci_high=0.0,
        max_pairwise_mean_latency_delta_ms=10.0,
    )

    assert report.passed is True
    assert report.pairwise_metrics["pairwise_shared_query_count"] == 10.0
    assert report.pairwise_metrics["pairwise_candidate_win_rate"] == 0.7
    assert report.pairwise_metrics["pairwise_mean_target_ndcg_delta"] == 0.25
    assert report.pairwise_metrics["pairwise_mean_target_rank_delta"] == -2.0
    assert report.pairwise_metrics["pairwise_target_rank_delta_ci_high"] == -0.5
    assert report.pairwise_metrics["pairwise_target_coverage_delta_ci_low"] == 0.12
    assert report.pairwise_metrics["pairwise_bootstrap_samples"] == 1000.0


def test_gate_chunking_comparison_flags_missing_pairwise_lift():
    comparison = comparison_report()

    report = gate_chunking_comparison(
        comparison,
        candidate="weak",
        baseline_candidate="strong",
        min_pairwise_win_rate=0.6,
        min_pairwise_target_coverage_lift=0.0,
        min_pairwise_target_ndcg_lift=0.0,
        min_pairwise_target_coverage_ci_low=0.0,
        min_pairwise_target_ndcg_ci_low=0.0,
        max_pairwise_mean_target_rank_delta=0.0,
        max_pairwise_target_rank_delta_ci_high=0.0,
        max_pairwise_mean_latency_delta_ms=0.0,
    )

    assert report.passed is False
    assert report.pairwise_metrics["pairwise_candidate_win_rate"] == 0.2
    assert "min_pairwise_win_rate" in report.failed_checks
    assert "min_pairwise_target_coverage_lift" in report.failed_checks
    assert "min_pairwise_target_ndcg_lift" in report.failed_checks
    assert "min_pairwise_target_coverage_ci_low" in report.failed_checks
    assert "min_pairwise_target_ndcg_ci_low" in report.failed_checks
    assert "max_pairwise_mean_target_rank_delta" in report.failed_checks
    assert "max_pairwise_target_rank_delta_ci_high" in report.failed_checks
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
            "--max-mean-target-rank",
            "2.0",
            "--min-result-stability-rate",
            "1.0",
            "--max-unstable-result-count",
            "0",
            "--max-total-chunk-chars",
            "2000",
            "--min-retrieval-score-per-embedding-kchar",
            "0.5",
            "--min-retrieval-score-per-mean-latency-ms",
            "0.05",
            "--min-target-coverage-per-mean-latency-ms",
            "0.05",
            "--min-target-ndcg-per-p95-latency-ms",
            "0.03",
            "--min-target-type-coverage",
            "asset=0.8",
            "--min-source-target-coverage",
            "bm25=0.8",
            "--min-visual-text-coverage-ratio",
            "0.8",
            "--min-visual-text-part-coverage-ratio",
            "0.8",
            "--min-source-family-target-coverage",
            "lexical=0.8",
            "--min-chunk-strategy-target-coverage",
            "visual_asset_text=0.8",
            "--min-retrieval-role-target-coverage",
            "child=0.8",
            "--min-case-group-target-coverage",
            "case_source:visual_lexical_probe=0.8",
            "--min-case-group-source-target-coverage",
            "case_source:visual_lexical_probe:bm25=0.8",
            "--min-case-group-source-family-target-coverage",
            "case_source:visual_lexical_probe:lexical=0.8",
            "--max-recall-drop",
            "0.1",
            "--min-pairwise-win-rate",
            "0.6",
            "--min-pairwise-target-coverage-ci-low",
            "0.0",
            "--max-pairwise-mean-target-rank-delta",
            "0.0",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate"] == "weak"
    assert "min_recall_at_k" in payload["failed_checks"]
    assert "max_mean_target_rank" in payload["failed_checks"]
    assert "min_result_stability_rate" in payload["failed_checks"]
    assert "max_unstable_result_count" in payload["failed_checks"]
    assert "max_total_chunk_chars" in payload["failed_checks"]
    assert "min_retrieval_score_per_embedding_kchar" in payload["failed_checks"]
    assert "min_retrieval_score_per_mean_latency_ms" in payload["failed_checks"]
    assert "min_target_coverage_per_mean_latency_ms" in payload["failed_checks"]
    assert "min_target_ndcg_per_p95_latency_ms" in payload["failed_checks"]
    assert "min_visual_text_coverage_ratio" in payload["failed_checks"]
    assert "min_visual_text_part_coverage_ratio" in payload["failed_checks"]
    assert "min_target_type_coverage:asset" in payload["failed_checks"]
    assert "min_source_target_coverage:bm25" in payload["failed_checks"]
    assert "min_source_family_target_coverage:lexical" in payload["failed_checks"]
    assert "min_chunk_strategy_target_coverage:visual_asset_text" in payload["failed_checks"]
    assert "min_retrieval_role_target_coverage:child" in payload["failed_checks"]
    assert (
        "min_case_group_target_coverage:case_source:visual_lexical_probe"
        in payload["failed_checks"]
    )
    assert (
        "min_case_group_source_target_coverage:case_source:visual_lexical_probe:bm25"
        in payload["failed_checks"]
    )
    assert (
        "min_case_group_source_family_target_coverage:"
        "case_source:visual_lexical_probe:lexical"
        in payload["failed_checks"]
    )
    assert payload["target_metrics"]["asset"]["coverage_at_k"] == 0.2
    assert payload["source_metrics"]["bm25"]["target_coverage_at_k"] == 0.1
    assert payload["source_family_metrics"]["lexical"]["target_coverage_at_k"] == 0.1
    assert payload["chunk_strategy_metrics"]["visual_asset_text"]["target_coverage_at_k"] == 0.2
    assert payload["retrieval_role_metrics"]["child"]["target_coverage_at_k"] == 0.1
    assert payload["case_group_metrics"]["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ] == 0.2
    assert payload["case_group_source_metrics"]["case_source"]["visual_lexical_probe"][
        "bm25"
    ]["target_coverage_at_k"] == 0.2
    assert payload["case_group_source_family_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["lexical"]["target_coverage_at_k"] == 0.2
    assert "max_recall_at_k_drop" in payload["failed_checks"]
    assert "min_pairwise_win_rate" in payload["failed_checks"]
    assert "min_pairwise_target_coverage_ci_low" in payload["failed_checks"]
    assert "max_pairwise_mean_target_rank_delta" in payload["failed_checks"]
    assert payload["pairwise_metrics"]["pairwise_candidate_win_rate"] == 0.2
    assert payload["pairwise_metrics"]["pairwise_mean_target_rank_delta"] == 2.0


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
                mean_target_rank=1.0,
                failed_queries=[],
                target_metrics={"asset": {"coverage_at_k": 0.85}},
                source_metrics={"bm25": {"target_coverage_at_k": 0.9}},
                source_family_metrics={"lexical": {"target_coverage_at_k": 0.9}},
                chunk_strategy_metrics={"visual_asset_text": {"target_coverage_at_k": 0.9}},
                retrieval_role_metrics={"child": {"target_coverage_at_k": 0.9}},
                case_group_metrics={
                    "case_source": {"visual_lexical_probe": {"target_coverage_at_k": 0.9}}
                },
                case_group_source_metrics={
                    "case_source": {
                        "visual_lexical_probe": {
                            "bm25": {"target_coverage_at_k": 0.9}
                        }
                    }
                },
                case_group_source_family_metrics={
                    "case_source": {
                        "visual_lexical_probe": {
                            "lexical": {"target_coverage_at_k": 0.9}
                        }
                    }
                },
                visual_text_coverage=0.9,
                total_chunk_chars=1200.0,
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
                mean_target_rank=6.0,
                failed_queries=["missing target"],
                target_metrics={"asset": {"coverage_at_k": 0.2}},
                source_metrics={"bm25": {"target_coverage_at_k": 0.1}},
                source_family_metrics={"lexical": {"target_coverage_at_k": 0.1}},
                chunk_strategy_metrics={"visual_asset_text": {"target_coverage_at_k": 0.2}},
                retrieval_role_metrics={"child": {"target_coverage_at_k": 0.1}},
                case_group_metrics={
                    "case_source": {"visual_lexical_probe": {"target_coverage_at_k": 0.2}}
                },
                case_group_source_metrics={
                    "case_source": {
                        "visual_lexical_probe": {
                            "bm25": {"target_coverage_at_k": 0.2}
                        }
                    }
                },
                case_group_source_family_metrics={
                    "case_source": {
                        "visual_lexical_probe": {
                            "lexical": {"target_coverage_at_k": 0.2}
                        }
                    }
                },
                visual_text_coverage=0.2,
                result_stability=0.5,
                unstable_results=1.0,
                total_chunk_chars=9000.0,
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
                mean_first_relevant_rank_delta=-1.0,
                mean_target_rank_delta=-2.0,
                mean_latency_delta_ms=-18.0,
                bootstrap_samples=1000,
                confidence_level=0.95,
                reciprocal_rank_delta_ci_low=0.11,
                reciprocal_rank_delta_ci_high=0.39,
                target_coverage_delta_ci_low=0.12,
                target_coverage_delta_ci_high=0.48,
                target_ndcg_delta_ci_low=0.1,
                target_ndcg_delta_ci_high=0.4,
                precision_delta_ci_low=0.1,
                precision_delta_ci_high=0.32,
                first_relevant_rank_delta_ci_low=-2.0,
                first_relevant_rank_delta_ci_high=-0.2,
                target_rank_delta_ci_low=-3.0,
                target_rank_delta_ci_high=-0.5,
                latency_delta_ci_low_ms=-24.0,
                latency_delta_ci_high_ms=-10.0,
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
                mean_first_relevant_rank_delta=1.0,
                mean_target_rank_delta=2.0,
                mean_latency_delta_ms=18.0,
                bootstrap_samples=1000,
                confidence_level=0.95,
                reciprocal_rank_delta_ci_low=-0.39,
                reciprocal_rank_delta_ci_high=-0.11,
                target_coverage_delta_ci_low=-0.48,
                target_coverage_delta_ci_high=-0.12,
                target_ndcg_delta_ci_low=-0.4,
                target_ndcg_delta_ci_high=-0.1,
                precision_delta_ci_low=-0.32,
                precision_delta_ci_high=-0.1,
                first_relevant_rank_delta_ci_low=0.2,
                first_relevant_rank_delta_ci_high=2.0,
                target_rank_delta_ci_low=0.5,
                target_rank_delta_ci_high=3.0,
                latency_delta_ci_low_ms=10.0,
                latency_delta_ci_high_ms=24.0,
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
    mean_target_rank: float,
    failed_queries: list[str],
    target_metrics: dict[str, dict[str, float]] | None = None,
    source_metrics: dict[str, dict[str, float]] | None = None,
    source_family_metrics: dict[str, dict[str, float]] | None = None,
    chunk_strategy_metrics: dict[str, dict[str, float]] | None = None,
    retrieval_role_metrics: dict[str, dict[str, float]] | None = None,
    case_group_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    case_group_source_metrics: dict[
        str, dict[str, dict[str, dict[str, float]]]
    ] | None = None,
    case_group_source_family_metrics: dict[
        str, dict[str, dict[str, dict[str, float]]]
    ] | None = None,
    visual_text_coverage: float = 1.0,
    visual_text_part_coverage: float | None = None,
    result_stability: float = 1.0,
    unstable_results: float = 0.0,
    total_chunk_chars: float = 1200.0,
) -> ChunkingComparisonRow:
    visual_text_part_coverage = (
        visual_text_coverage if visual_text_part_coverage is None else visual_text_part_coverage
    )
    retrieval_score = (recall + target_coverage + target_ndcg + precision) / 4
    embedding_text_kchars = total_chunk_chars / 1000.0
    return ChunkingComparisonRow(
        name=name,
        chunk_count=12,
        total_chunk_chars=total_chunk_chars,
        mean_chunk_chars=total_chunk_chars / 12,
        p95_chunk_chars=total_chunk_chars / 12,
        embedding_text_kchars=embedding_text_kchars,
        quality_score=quality_score,
        retrieval_score=retrieval_score,
        retrieval_score_per_embedding_kchar=retrieval_score / embedding_text_kchars,
        target_coverage_per_embedding_kchar=target_coverage / embedding_text_kchars,
        target_ndcg_per_embedding_kchar=target_ndcg / embedding_text_kchars,
        retrieval_score_per_mean_latency_ms=retrieval_score / mean_latency,
        target_coverage_per_mean_latency_ms=target_coverage / mean_latency,
        target_ndcg_per_mean_latency_ms=target_ndcg / mean_latency,
        retrieval_score_per_p95_latency_ms=retrieval_score / p95_latency,
        target_coverage_per_p95_latency_ms=target_coverage / p95_latency,
        target_ndcg_per_p95_latency_ms=target_ndcg / p95_latency,
        retrieval_hit_rate=recall,
        retrieval_recall_at_k=recall,
        retrieval_mrr=0.75,
        retrieval_target_coverage_at_k=target_coverage,
        retrieval_mean_target_ndcg_at_k=target_ndcg,
        retrieval_mean_precision_at_k=precision,
        retrieval_mean_latency_ms=mean_latency,
        retrieval_p95_latency_ms=p95_latency,
        retrieval_mean_first_relevant_rank=mean_target_rank,
        retrieval_p95_first_relevant_rank=mean_target_rank,
        retrieval_mean_target_rank=mean_target_rank,
        retrieval_p95_target_rank=mean_target_rank,
        retrieval_ranked_expected_case_count=10.0,
        retrieval_ranked_target_count=10.0,
        retrieval_result_stability_rate=result_stability,
        retrieval_unstable_result_count=unstable_results,
        target_metrics=target_metrics or {},
        source_metrics=source_metrics or {},
        source_family_metrics=source_family_metrics or {},
        chunk_strategy_metrics=chunk_strategy_metrics or {},
        retrieval_role_metrics=retrieval_role_metrics or {},
        case_group_metrics=case_group_metrics or {},
        case_group_source_metrics=case_group_source_metrics or {},
        case_group_source_family_metrics=case_group_source_family_metrics or {},
        failed_queries=failed_queries,
        page_coverage_ratio=1.0,
        visual_annotation_ratio=0.5,
        visual_text_asset_count=10,
        visual_text_covered_asset_count=round(10 * visual_text_coverage),
        visual_text_coverage_ratio=visual_text_coverage,
        visual_text_part_count=20,
        visual_text_covered_part_count=round(20 * visual_text_part_coverage),
        visual_text_part_coverage_ratio=visual_text_part_coverage,
        chunks_under_min_chars=0,
        chunks_over_max_chars=0,
        issue_codes=[],
    )
