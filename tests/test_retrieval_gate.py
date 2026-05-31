from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.gate import gate_retrieval_evaluation
from chunking_docs.evaluation.retrieval import RetrievalEvaluation


def make_evaluation(
    recall: float = 0.9,
    target_coverage: float = 0.85,
    target_ndcg: float = 0.8,
    precision: float = 0.7,
    mean_latency: float = 12.0,
    p95_latency: float = 20.0,
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
