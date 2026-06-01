import json

import pytest
from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.gate import gate_retrieval_evaluation
from chunking_docs.evaluation.retrieval import (
    RetrievalCaseGroupMetric,
    RetrievalCaseResult,
    RetrievalEvaluation,
    RetrievalSourceMetric,
    RetrievalTargetMetric,
)


def make_evaluation(
    recall: float = 0.9,
    target_coverage: float = 0.85,
    target_ndcg: float = 0.8,
    precision: float = 0.7,
    mean_latency: float = 12.0,
    p95_latency: float = 20.0,
    target_type_coverage: dict[str, float] | None = None,
    source_coverage: dict[str, float] | None = None,
    source_family_coverage: dict[str, float] | None = None,
    chunk_strategy_coverage: dict[str, float] | None = None,
    retrieval_role_coverage: dict[str, float] | None = None,
    case_group_coverage: dict[str, dict[str, float]] | None = None,
) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        case_count=10,
        expected_case_count=10,
        passed_count=int(recall * 10),
        failed_count=10 - int(recall * 10),
        hit_rate=recall,
        recall_at_k=recall,
        mrr=0.75,
        target_coverage_at_k=target_coverage,
        mean_target_ndcg_at_k=target_ndcg,
        mean_precision_at_k=precision,
        top_k=5,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
        target_metrics={
            target_type: target_type_metric(coverage)
            for target_type, coverage in (target_type_coverage or {}).items()
        },
        source_metrics={
            source: source_family_metric(coverage)
            for source, coverage in (source_coverage or {}).items()
        },
        source_family_metrics={
            family: source_family_metric(coverage)
            for family, coverage in (source_family_coverage or {}).items()
        },
        chunk_strategy_metrics={
            strategy: source_family_metric(coverage)
            for strategy, coverage in (chunk_strategy_coverage or {}).items()
        },
        retrieval_role_metrics={
            role: source_family_metric(coverage)
            for role, coverage in (retrieval_role_coverage or {}).items()
        },
        case_group_metrics={
            group_name: {
                group_value: case_group_metric(coverage)
                for group_value, coverage in group_values.items()
            }
            for group_name, group_values in (case_group_coverage or {}).items()
        },
        failed_queries=[],
        results=[],
    )


def test_retrieval_gate_passes_absolute_thresholds():
    report = gate_retrieval_evaluation(
        make_evaluation(),
        min_recall_at_k=0.8,
        min_target_coverage_at_k=0.8,
        min_target_ndcg_at_k=0.75,
        min_precision_at_k=0.6,
        max_mean_latency_ms=20.0,
        max_p95_latency_ms=25.0,
    )

    assert report.passed is True
    assert report.failed_checks == []
    assert report.metrics["recall_at_k"] == 0.9


def test_retrieval_gate_checks_target_rank_metrics():
    evaluation = make_evaluation_with_rank_results()

    report = gate_retrieval_evaluation(
        evaluation,
        max_mean_first_relevant_rank=2.0,
        max_p95_first_relevant_rank=3.0,
        max_mean_target_rank=3.0,
        max_p95_target_rank=5.6,
    )

    assert report.passed is True
    assert report.metrics["mean_first_relevant_rank"] == 2.0
    assert report.metrics["p95_first_relevant_rank"] == pytest.approx(2.9)
    assert report.metrics["mean_target_rank"] == 3.0
    assert report.metrics["p95_target_rank"] == pytest.approx(5.6)
    assert report.metrics["ranked_expected_case_count"] == 2.0
    assert report.metrics["ranked_target_count"] == 3.0

    failed = gate_retrieval_evaluation(
        evaluation,
        max_mean_first_relevant_rank=1.5,
        max_p95_first_relevant_rank=2.0,
        max_mean_target_rank=2.5,
        max_p95_target_rank=5.0,
    )

    assert failed.passed is False
    assert failed.failed_checks == [
        "max_mean_first_relevant_rank",
        "max_p95_first_relevant_rank",
        "max_mean_target_rank",
        "max_p95_target_rank",
    ]


def test_retrieval_gate_checks_target_type_coverage():
    evaluation = make_evaluation(target_type_coverage={"asset": 1.0, "triple": 0.8})

    report = gate_retrieval_evaluation(
        evaluation,
        min_target_type_coverage={"asset": 1.0, "triple": 0.75},
    )

    assert report.passed is True
    assert report.metrics["target_type.asset.coverage_at_k"] == 1.0
    assert report.target_metrics["triple"]["coverage_at_k"] == 0.8

    failed = gate_retrieval_evaluation(
        evaluation,
        min_target_type_coverage={"triple": 0.9},
    )

    assert failed.passed is False
    assert failed.failed_checks == ["min_target_type_coverage:triple"]


def test_retrieval_gate_checks_source_family_target_coverage():
    evaluation = make_evaluation(source_family_coverage={"visual": 0.75, "lexical": 1.0})

    report = gate_retrieval_evaluation(
        evaluation,
        min_source_family_target_coverage={"visual": 0.7, "lexical": 1.0},
    )

    assert report.passed is True
    assert report.metrics["source_family.visual.target_coverage_at_k"] == 0.75
    assert report.source_family_metrics["lexical"]["target_coverage_at_k"] == 1.0

    failed = gate_retrieval_evaluation(
        evaluation,
        min_source_family_target_coverage={"visual": 0.8},
    )

    assert failed.passed is False
    assert failed.failed_checks == ["min_source_family_target_coverage:visual"]


def test_retrieval_gate_checks_exact_source_target_coverage():
    evaluation = make_evaluation(
        source_coverage={"qdrant:caption_dense": 1.0, "qdrant:image_dense": 0.5}
    )

    report = gate_retrieval_evaluation(
        evaluation,
        min_source_target_coverage={"qdrant:caption_dense": 1.0},
    )

    assert report.passed is True
    assert report.metrics["source.qdrant:caption_dense.target_coverage_at_k"] == 1.0
    assert report.source_metrics["qdrant:image_dense"]["target_coverage_at_k"] == 0.5

    failed = gate_retrieval_evaluation(
        evaluation,
        min_source_target_coverage={"qdrant:image_dense": 0.75},
    )

    assert failed.passed is False
    assert failed.failed_checks == ["min_source_target_coverage:qdrant:image_dense"]


def test_retrieval_gate_checks_chunk_strategy_and_role_coverage():
    evaluation = make_evaluation(
        chunk_strategy_coverage={"visual_asset_text": 1.0, "hierarchical_child": 0.75},
        retrieval_role_coverage={"child": 0.75},
    )

    report = gate_retrieval_evaluation(
        evaluation,
        min_chunk_strategy_target_coverage={"visual_asset_text": 1.0},
        min_retrieval_role_target_coverage={"child": 0.7},
    )

    assert report.passed is True
    assert report.metrics["chunk_strategy.visual_asset_text.target_coverage_at_k"] == 1.0
    assert report.metrics["retrieval_role.child.target_coverage_at_k"] == 0.75
    assert report.chunk_strategy_metrics["hierarchical_child"]["target_coverage_at_k"] == 0.75

    failed = gate_retrieval_evaluation(
        evaluation,
        min_chunk_strategy_target_coverage={"hierarchical_child": 0.8},
        min_retrieval_role_target_coverage={"child": 0.8},
    )

    assert failed.passed is False
    assert failed.failed_checks == [
        "min_chunk_strategy_target_coverage:hierarchical_child",
        "min_retrieval_role_target_coverage:child",
    ]


def test_retrieval_gate_checks_case_group_target_coverage():
    evaluation = make_evaluation(
        case_group_coverage={
            "case_source": {"visual_lexical_probe": 0.75, "page": 1.0},
        }
    )

    report = gate_retrieval_evaluation(
        evaluation,
        min_case_group_target_coverage={"case_source:visual_lexical_probe": 0.7},
    )

    assert report.passed is True
    assert (
        report.metrics["case_group.case_source.visual_lexical_probe.target_coverage_at_k"]
        == 0.75
    )
    assert report.case_group_metrics["case_source"]["page"]["target_coverage_at_k"] == 1.0

    failed = gate_retrieval_evaluation(
        evaluation,
        min_case_group_target_coverage={"case_source:visual_lexical_probe": 0.8},
    )

    assert failed.passed is False
    assert failed.failed_checks == [
        "min_case_group_target_coverage:case_source:visual_lexical_probe"
    ]


def test_retrieval_gate_flags_baseline_regressions():
    baseline = make_evaluation(recall=0.95, target_coverage=0.9, target_ndcg=0.85, precision=0.75)
    candidate = make_evaluation(
        recall=0.7,
        target_coverage=0.75,
        target_ndcg=0.6,
        precision=0.5,
        mean_latency=30.0,
        p95_latency=60.0,
    )

    report = gate_retrieval_evaluation(
        candidate,
        baseline=baseline,
        min_recall_at_k=0.8,
        max_recall_drop=0.1,
        max_target_coverage_drop=0.1,
        max_target_ndcg_drop=0.1,
        max_precision_drop=0.1,
        max_mean_latency_ratio=2.0,
        max_p95_latency_ratio=2.0,
    )

    assert report.passed is False
    assert "min_recall_at_k" in report.failed_checks
    assert "max_recall_at_k_drop" in report.failed_checks
    assert "max_mean_target_ndcg_at_k_drop" in report.failed_checks
    assert "max_mean_latency_ms_ratio" in report.failed_checks
    assert report.baseline_metrics["recall_at_k"] == 0.95


def test_gate_retrieval_cli_exits_nonzero_on_failed_gate(tmp_path):
    evaluation_path = tmp_path / "retrieval_eval.json"
    evaluation_path.write_text(make_evaluation(recall=0.4).model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["gate-retrieval", str(evaluation_path), "--min-recall-at-k", "0.8"],
    )

    assert result.exit_code == 1
    assert "min_recall_at_k" in result.output


def test_gate_retrieval_cli_checks_source_family_target_coverage(tmp_path):
    evaluation_path = tmp_path / "retrieval_eval.json"
    output = tmp_path / "retrieval_gate.json"
    evaluation_path.write_text(
        make_evaluation(source_family_coverage={"visual": 0.5}).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-retrieval",
            str(evaluation_path),
            "--min-source-family-target-coverage",
            "visual=0.8",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "min_source_family_target_coverage:visual" in result.output
    payload = output.read_text(encoding="utf-8")
    assert "source_family.visual.target_coverage_at_k" in payload


def test_gate_retrieval_cli_checks_exact_source_target_coverage(tmp_path):
    evaluation_path = tmp_path / "retrieval_eval.json"
    output = tmp_path / "retrieval_gate.json"
    evaluation_path.write_text(
        make_evaluation(source_coverage={"qdrant:image_dense": 0.5}).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-retrieval",
            str(evaluation_path),
            "--min-source-target-coverage",
            "qdrant:image_dense=0.8",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "min_source_target_coverage:qdrant:image_dense" in result.output
    payload = output.read_text(encoding="utf-8")
    assert "source.qdrant:image_dense.target_coverage_at_k" in payload


def test_gate_retrieval_cli_checks_chunk_strategy_target_coverage(tmp_path):
    evaluation_path = tmp_path / "retrieval_eval.json"
    output = tmp_path / "retrieval_gate.json"
    evaluation_path.write_text(
        make_evaluation(chunk_strategy_coverage={"visual_asset_text": 0.5}).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-retrieval",
            str(evaluation_path),
            "--min-chunk-strategy-target-coverage",
            "visual_asset_text=0.8",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "min_chunk_strategy_target_coverage:visual_asset_text" in result.output
    payload = output.read_text(encoding="utf-8")
    assert "chunk_strategy.visual_asset_text.target_coverage_at_k" in payload


def test_gate_retrieval_cli_checks_case_group_target_coverage(tmp_path):
    evaluation_path = tmp_path / "retrieval_eval.json"
    output = tmp_path / "retrieval_gate.json"
    evaluation_path.write_text(
        make_evaluation(
            case_group_coverage={"case_source": {"visual_lexical_probe": 0.5}}
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-retrieval",
            str(evaluation_path),
            "--min-case-group-target-coverage",
            "case_source:visual_lexical_probe=0.8",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "min_case_group_target_coverage:case_source:visual_lexical_probe" in result.output
    payload = output.read_text(encoding="utf-8")
    assert "case_group.case_source.visual_lexical_probe.target_coverage_at_k" in payload


def test_gate_retrieval_cli_checks_target_type_coverage(tmp_path):
    evaluation_path = tmp_path / "retrieval_eval.json"
    output = tmp_path / "retrieval_gate.json"
    evaluation_path.write_text(
        make_evaluation(target_type_coverage={"asset": 0.5}).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-retrieval",
            str(evaluation_path),
            "--min-target-type-coverage",
            "asset=1.0",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "min_target_type_coverage:asset" in result.output
    payload = output.read_text(encoding="utf-8")
    assert "target_type.asset.coverage_at_k" in payload


def test_gate_retrieval_cli_checks_target_rank_metrics(tmp_path):
    evaluation_path = tmp_path / "retrieval_eval.json"
    output = tmp_path / "retrieval_gate.json"
    evaluation_path.write_text(make_evaluation_with_rank_results().model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "gate-retrieval",
            str(evaluation_path),
            "--max-mean-target-rank",
            "2.5",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "max_mean_target_rank" in result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["mean_target_rank"] == 3.0
    assert payload["failed_checks"] == ["max_mean_target_rank"]


def make_evaluation_with_rank_results() -> RetrievalEvaluation:
    return RetrievalEvaluation(
        case_count=2,
        expected_case_count=2,
        passed_count=2,
        failed_count=0,
        hit_rate=1.0,
        recall_at_k=1.0,
        mrr=0.75,
        target_coverage_at_k=2 / 3,
        mean_target_ndcg_at_k=0.7,
        mean_precision_at_k=0.5,
        top_k=5,
        mean_latency_ms=10.0,
        p95_latency_ms=12.0,
        target_metrics={},
        source_family_metrics={},
        failed_queries=[],
        results=[
            case_result(
                query="first target",
                expected_pages=[1],
                expected_asset_ids=[],
                matched_rank=1,
                target_key_matched_ranks={"page:1": 1},
            ),
            case_result(
                query="partial target",
                expected_pages=[2],
                expected_asset_ids=["asset-2"],
                matched_rank=3,
                target_key_matched_ranks={"page:2": 2},
            ),
        ],
    )


def case_result(
    query: str,
    expected_pages: list[int],
    expected_asset_ids: list[str],
    matched_rank: int,
    target_key_matched_ranks: dict[str, int],
) -> RetrievalCaseResult:
    expected_target_count = len(expected_pages) + len(expected_asset_ids)
    return RetrievalCaseResult(
        query=query,
        passed=True,
        top_pages=[],
        top_chunk_ids=[],
        expected_pages=expected_pages,
        expected_chunk_ids=[],
        expected_asset_ids=expected_asset_ids,
        expected_triple_ids=[],
        expected_target_count=expected_target_count,
        matched_target_count=len(target_key_matched_ranks),
        matched_rank=matched_rank,
        reciprocal_rank=1 / matched_rank,
        target_key_matched_ranks=target_key_matched_ranks,
    )


def target_type_metric(target_coverage: float) -> RetrievalTargetMetric:
    return RetrievalTargetMetric(
        expected_count=10,
        passed_count=int(target_coverage * 10),
        recall_at_k=target_coverage,
        mrr=target_coverage,
        target_count=10,
        matched_target_count=int(target_coverage * 10),
        coverage_at_k=target_coverage,
        ndcg_at_k=target_coverage,
    )


def source_family_metric(target_coverage: float) -> RetrievalSourceMetric:
    return RetrievalSourceMetric(
        query_count=10,
        relevant_query_count=int(target_coverage * 10),
        hit_count=50,
        relevant_hit_count=int(target_coverage * 10),
        expected_target_count=10,
        matched_target_count=int(target_coverage * 10),
        precision_at_hits=target_coverage,
        target_coverage_at_k=target_coverage,
        mean_relevant_rank=1.0,
    )


def case_group_metric(target_coverage: float) -> RetrievalCaseGroupMetric:
    return RetrievalCaseGroupMetric(
        case_count=10,
        expected_case_count=10,
        passed_count=int(target_coverage * 10),
        failed_count=10 - int(target_coverage * 10),
        recall_at_k=target_coverage,
        mrr=target_coverage,
        target_count=10,
        matched_target_count=int(target_coverage * 10),
        target_coverage_at_k=target_coverage,
        ndcg_at_k=target_coverage,
        precision_at_k=target_coverage,
        mean_latency_ms=10.0,
    )
