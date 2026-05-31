import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.vision.experiments import build_vlm_experiment_plan, parse_profile_list


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
    assert len(plan.recipes) == 2
    assert plan.recipes[0].model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert plan.recipes[0].torch_dtype == "float16"
    assert "--vlm-profile qwen2_5_vl_7b" in plan.recipes[0].command
    assert "--limit 2" in plan.recipes[0].command
    assert "--apply" not in plan.recipes[0].command
    assert "visual_job_results.qwen2_5_vl_7b.jsonl" in plan.compare_command
    assert "visual_run_comparison.json" in plan.compare_command


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
    assert payload["recipes"][1]["model_class"] == "causal-lm"
    assert "compare-visual-runs" in payload["compare_command"]
