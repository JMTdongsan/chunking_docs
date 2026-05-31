from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.gate import gate_retrieval_evaluation
from chunking_docs.evaluation.retrieval import (
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
    source_family_coverage: dict[str, float] | None = None,
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
        source_family_metrics={
            family: source_family_metric(coverage)
            for family, coverage in (source_family_coverage or {}).items()
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
