import json

from typer.testing import CliRunner

from chunking_docs.cli import app
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
    assert payload["rows"][0]["name"] == "json"


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
