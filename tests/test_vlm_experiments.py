import json

from typer.testing import CliRunner

from chunking_docs.models import AssetKind
from chunking_docs.cli import app
from chunking_docs.vision.experiments import build_vlm_experiment_plan, parse_profile_list
from chunking_docs.vision.jobs import VisualAnnotationJob


def test_build_vlm_experiment_plan_writes_profile_commands(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    jobs = package_dir / "visual_jobs.priority.jsonl"
    jobs.write_text("", encoding="utf-8")

    plan = build_vlm_experiment_plan(
        package_dir=package_dir,
        jobs_file=jobs,
        profiles=["qwen2_5_vl_7b", "llava-next-7b"],
        limit=2,
        vlm_torch_dtype="float16",
    )

    assert plan.profiles == ["qwen2_5_vl_7b", "llava_next_7b"]
    assert plan.job_summary.exists is True
    assert plan.job_summary.total_job_count == 0
    assert len(plan.recipes) == 2
    assert plan.recipes[0].model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert plan.recipes[0].torch_dtype == "float16"
    assert plan.recipes[0].metadata["min_gpu_memory_mib"] == 24576
    assert plan.recipes[0].doctor_command.endswith(
        "--vlm-profile qwen2_5_vl_7b --vlm-memory-margin-ratio 0.1"
    )
    assert "--require-ocr" not in plan.recipes[0].doctor_command
    assert "--ocr none" in plan.recipes[0].command
    assert "--vlm-profile qwen2_5_vl_7b" in plan.recipes[0].command
    assert "--limit 2" in plan.recipes[0].command
    assert "--apply" not in plan.recipes[0].command
    assert "visual_job_results.qwen2_5_vl_7b.jsonl" in plan.compare_command
    assert "visual_run_comparison.json" in plan.compare_command
    assert "--require-same-jobs" in plan.compare_command


def test_build_vlm_experiment_plan_summarizes_selected_jobs(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    jobs = package_dir / "visual_jobs.priority.jsonl"
    jobs.write_text(
        "\n".join(
            [
                VisualAnnotationJob(
                    job_id="job-a",
                    asset_id="asset-a",
                    doc_id="doc",
                    page_no=2,
                    kind=AssetKind.MAP,
                    asset_path=tmp_path / "a.png",
                    operations=["ocr", "vlm"],
                    priority=1200,
                    reason="missing annotations",
                    metadata={"asset_scope": "tile"},
                ).model_dump_json(),
                VisualAnnotationJob(
                    job_id="job-b",
                    asset_id="asset-b",
                    doc_id="doc",
                    page_no=3,
                    kind=AssetKind.TABLE,
                    asset_path=tmp_path / "b.png",
                    operations=["vlm"],
                    priority=1000,
                    reason="missing VLM",
                ).model_dump_json(),
                VisualAnnotationJob(
                    job_id="job-c",
                    asset_id="asset-c",
                    doc_id="doc",
                    page_no=5,
                    kind=AssetKind.FIGURE,
                    asset_path=tmp_path / "c.png",
                    operations=["ocr"],
                    priority=800,
                    reason="missing OCR",
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plan = build_vlm_experiment_plan(
        package_dir=package_dir,
        jobs_file=jobs,
        profiles=["phi3_5_vision"],
        limit=2,
        batch_size=1,
        vlm_max_new_tokens=512,
        vlm_memory_margin_ratio=0.2,
    )

    assert plan.batch_size == 1
    assert len(plan.batches) == 2
    assert plan.batches[0].batch_id == "batch_001"
    assert plan.batches[0].offset == 0
    assert plan.batches[0].limit == 1
    assert plan.batches[0].asset_kind_counts == {"map": 1}
    assert plan.batches[1].batch_id == "batch_002"
    assert plan.batches[1].offset == 1
    assert plan.batches[1].operation_counts == {"vlm": 1}
    assert plan.job_summary.total_job_count == 3
    assert plan.job_summary.selected_job_count == 2
    assert plan.job_summary.skipped_by_limit_count == 1
    assert plan.job_summary.operation_counts == {"ocr": 1, "vlm": 2}
    assert plan.job_summary.asset_kind_counts == {"map": 1, "table": 1}
    assert plan.job_summary.asset_scope_counts == {"asset": 1, "tile": 1}
    assert plan.job_summary.page_min == 2
    assert plan.job_summary.page_max == 3
    assert plan.recipes[0].metadata["selected_vlm_job_count"] == 2
    assert plan.recipes[0].metadata["selected_ocr_job_count"] == 1
    assert plan.recipes[0].metadata["requested_ocr_backend"] == "paddleocr"
    assert plan.recipes[0].metadata["effective_ocr_backend"] == "paddleocr"
    assert plan.recipes[0].metadata["max_generation_tokens_upper_bound"] == 1024
    assert plan.recipes[0].metadata["batch_count"] == 2
    assert len(plan.recipes[0].batch_commands) == 2
    assert "visual_job_results.phi3_5_vision.batch_001.jsonl" in plan.recipes[0].batch_commands[0]
    assert "--limit 1" in plan.recipes[0].batch_commands[0]
    assert "--offset 1" in plan.recipes[0].batch_commands[1]
    assert "--limit 1" in plan.recipes[0].batch_commands[1]
    assert "merge-visual-results" in plan.recipes[0].merge_command
    assert "visual_job_results.phi3_5_vision.batch_001.jsonl" in plan.recipes[0].merge_command
    assert "visual_job_results.phi3_5_vision.batch_002.jsonl" in plan.recipes[0].merge_command
    assert "visual_job_results.phi3_5_vision.jsonl" in plan.recipes[0].merge_command
    assert "visual_annotations.phi3_5_vision.jsonl" in plan.recipes[0].merge_command
    assert len(plan.batch_compare_commands) == 2
    assert "visual_run_comparison.batch_001.json" in plan.batch_compare_commands[0]
    assert "--require-ocr" in plan.recipes[0].doctor_command
    assert plan.recipes[0].doctor_command.endswith(
        "--vlm-profile phi3_5_vision --vlm-memory-margin-ratio 0.2"
    )
    assert "--ocr paddleocr" in plan.recipes[0].command


def test_parse_profile_list_normalizes_names():
    assert parse_profile_list("qwen2-5-vl-7b, llava-next-7b") == [
        "qwen2_5_vl_7b",
        "llava_next_7b",
    ]


def test_plan_vlm_experiments_cli_writes_json(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    jobs = package_dir / "jobs.jsonl"
    output = tmp_path / "plan.json"
    jobs.write_text("", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "plan-vlm-experiments",
            "--package-dir",
            str(package_dir),
            "--jobs",
            str(jobs),
            "--profiles",
            "qwen2_5_vl_7b,phi3_5_vision",
            "--output",
            str(output),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["profiles"] == ["qwen2_5_vl_7b", "phi3_5_vision"]
    assert payload["job_summary"]["exists"] is True
    assert payload["recipes"][1]["model_class"] == "causal-lm"
    assert payload["recipes"][1]["doctor_command"].startswith("chunking-docs doctor")
    assert "compare-visual-runs" in payload["compare_command"]
    assert "--require-same-jobs" in payload["compare_command"]
