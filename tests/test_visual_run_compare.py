import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.retrieval import RetrievalCaseGroupMetric, RetrievalEvaluation
from chunking_docs.io import write_jsonl
from chunking_docs.vision.compare import compare_visual_runs
from chunking_docs.vision.jobs import VisualJobRunResult
from chunking_docs.vision.manual_annotations import AssetAnnotation


def test_compare_visual_runs_ranks_by_quality_and_tracks_latency():
    comparison = compare_visual_runs(
        {
            "raw": raw_text_results(),
            "json": json_results(),
        }
    )

    assert comparison.best_by_quality == "json"
    assert comparison.fastest_by_total_latency == "raw"
    assert comparison.best_by_triple_density == "json"
    assert comparison.rows[0].name == "json"
    assert comparison.rows[0].vlm_json_parse_rate == 1.0
    assert comparison.rows[0].triples_per_vlm_job == 1.0
    assert comparison.rows[0].total_mean_latency_ms == 40.0
    assert comparison.rows[-1].failed_count == 1
    assert comparison.job_set_mismatch is True
    assert comparison.union_job_count == 2
    assert comparison.shared_job_count == 1
    assert comparison.missing_job_ids_by_run["json"] == ["job-2"]
    assert comparison.unshared_job_ids_by_run["raw"] == ["job-2"]


def test_compare_visual_runs_attaches_retrieval_evaluations():
    comparison = compare_visual_runs(
        {
            "raw": raw_text_results(),
            "json": json_results(),
        },
        retrieval_evaluations={
            "raw": retrieval_evaluation(target_coverage=0.9, object_probe_coverage=0.8),
            "json": retrieval_evaluation(target_coverage=0.6, object_probe_coverage=0.4),
        },
    )

    assert comparison.best_by_quality == "json"
    assert comparison.best_by_retrieval == "raw"
    assert comparison.retrieval_evaluation_run_count == 2
    assert comparison.missing_retrieval_evaluation_runs == []
    raw_row = next(row for row in comparison.rows if row.name == "raw")
    assert raw_row.retrieval_case_count == 3
    assert raw_row.retrieval_target_coverage_at_k == 0.9
    assert raw_row.retrieval_visual_object_probe_target_coverage_at_k == 0.8
    assert raw_row.retrieval_score == 0.74


def test_compare_visual_runs_cli_writes_json(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    json_path = tmp_path / "json.jsonl"
    output = tmp_path / "visual_run_comparison.json"
    write_jsonl(raw_path, raw_text_results())
    write_jsonl(json_path, json_results())

    result = CliRunner().invoke(
        app,
        [
            "compare-visual-runs",
            "--run",
            f"raw={raw_path}",
            "--run",
            f"json={json_path}",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["best_by_quality"] == "json"
    assert payload["fastest_by_total_latency"] == "raw"
    assert payload["job_set_mismatch"] is True
    assert payload["missing_job_ids_by_run"]["json"] == ["job-2"]
    assert payload["rows"][0]["name"] == "json"


def test_compare_visual_runs_cli_accepts_retrieval_eval_files(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    json_path = tmp_path / "json.jsonl"
    raw_eval_path = tmp_path / "raw_retrieval.json"
    json_eval_path = tmp_path / "json_retrieval.json"
    output = tmp_path / "visual_run_comparison.json"
    write_jsonl(raw_path, raw_text_results())
    write_jsonl(json_path, json_results())
    raw_eval_path.write_text(
        retrieval_evaluation(target_coverage=0.9, object_probe_coverage=0.8).model_dump_json(indent=2),
        encoding="utf-8",
    )
    json_eval_path.write_text(
        retrieval_evaluation(target_coverage=0.6, object_probe_coverage=0.4).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "compare-visual-runs",
            "--run",
            f"raw={raw_path}",
            "--run",
            f"json={json_path}",
            "--retrieval-eval",
            f"raw={raw_eval_path}",
            "--retrieval-eval",
            f"json={json_eval_path}",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["best_by_retrieval"] == "raw"
    assert payload["retrieval_evaluation_run_count"] == 2
    assert payload["rows"][1]["retrieval_visual_object_probe_target_coverage_at_k"] == 0.8


def test_compare_visual_runs_cli_can_fail_on_job_set_mismatch(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    json_path = tmp_path / "json.jsonl"
    write_jsonl(raw_path, raw_text_results())
    write_jsonl(json_path, json_results())

    result = CliRunner().invoke(
        app,
        [
            "compare-visual-runs",
            "--run",
            f"raw={raw_path}",
            "--run",
            f"json={json_path}",
            "--require-same-jobs",
        ],
    )

    assert result.exit_code == 1


def json_results() -> list[VisualJobRunResult]:
    return [
        VisualJobRunResult(
            job_id="job-1",
            asset_id="asset-1",
            page_no=1,
            status="completed",
            annotation=AssetAnnotation(
                asset_id="asset-1",
                page_no=1,
                ocr_text="recognized words",
                vlm_summary="structured visual summary with key evidence",
                triples=[{"subject": "subject", "predicate": "relates_to", "object": "object"}],
                metadata={"vlm_parse_status": "json_object"},
            ),
            metadata={
                "operations": ["ocr", "vlm"],
                "ocr_text_chars": 16,
                "vlm_parse_status": "json_object",
                "total_duration_ms": 40.0,
                "ocr_duration_ms": 10.0,
                "vlm_duration_ms": 30.0,
            },
        )
    ]


def raw_text_results() -> list[VisualJobRunResult]:
    return [
        VisualJobRunResult(
            job_id="job-1",
            asset_id="asset-1",
            page_no=1,
            status="completed",
            annotation=AssetAnnotation(
                asset_id="asset-1",
                page_no=1,
                ocr_text="recognized words",
                vlm_summary="plain visual summary",
                metadata={"vlm_parse_status": "raw_text"},
            ),
            metadata={
                "operations": ["ocr", "vlm"],
                "ocr_text_chars": 16,
                "vlm_parse_status": "raw_text",
                "total_duration_ms": 20.0,
                "ocr_duration_ms": 8.0,
                "vlm_duration_ms": 12.0,
            },
        ),
        VisualJobRunResult(
            job_id="job-2",
            asset_id="asset-2",
            page_no=2,
            status="failed",
            error="model error",
            metadata={"operations": ["vlm"]},
        ),
    ]


def retrieval_evaluation(
    target_coverage: float,
    object_probe_coverage: float,
) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        metadata={"mode": "test"},
        case_count=3,
        expected_case_count=3,
        passed_count=2,
        failed_count=1,
        hit_rate=0.7,
        recall_at_k=0.7,
        mrr=0.5,
        target_coverage_at_k=target_coverage,
        mean_target_ndcg_at_k=0.7,
        mean_precision_at_k=0.6,
        top_k=5,
        mean_latency_ms=12.0,
        failed_queries=["missed query"],
        results=[],
        case_group_metrics={
            "case_source": {
                "visual_object_probe": RetrievalCaseGroupMetric(
                    case_count=1,
                    expected_case_count=1,
                    target_count=1,
                    matched_target_count=1 if object_probe_coverage else 0,
                    target_coverage_at_k=object_probe_coverage,
                )
            }
        },
    )
