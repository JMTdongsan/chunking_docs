from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import chunking_docs.cli as cli_module
from chunking_docs.cli import app
from chunking_docs.io import write_jsonl
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, SectionPath, VisualAsset
from chunking_docs.vision.jobs import (
    VisualJobRunResult,
    completed_annotations,
    plan_visual_jobs,
    run_visual_jobs,
)
from chunking_docs.vision.manual_annotations import AssetAnnotation
from chunking_docs.vision.quality import evaluate_visual_results
from chunking_docs.vision.report import summarize_visual_results


class FakeOCR:
    def recognize(self, image_path: Path, language: str = "kor+eng"):
        return f"ocr:{image_path.name}:{language}"

    def metadata(self):
        return {"name": "fake-ocr", "languages": ["kor", "eng"]}


class FakeVLM:
    def summarize(self, image_path: Path, prompt: str):
        return f"vlm:{image_path.name}:{prompt[:4]}"

    def metadata(self):
        return {"name": "fake-vlm", "max_new_tokens": 64}


class JsonVLM:
    def summarize(self, image_path: Path, prompt: str):
        return """
        {
          "title": "River Corridor Diagram",
          "summary": "Shows connected hubs.",
          "triples": [
            {"subject": "corridor", "predicate": "connects", "object": "hub"}
          ]
        }
        """


def test_plan_visual_jobs_prioritizes_maps_and_missing_annotations(tmp_path):
    map_path = tmp_path / "map.png"
    page_path = tmp_path / "page.png"
    map_path.write_bytes(b"map")
    page_path.write_bytes(b"page")
    assets = [
        VisualAsset(
            asset_id="page",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            path=page_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        ),
        VisualAsset(
            asset_id="map",
            doc_id="doc",
            page_no=2,
            kind=AssetKind.MAP,
            path=map_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        ),
    ]

    jobs = plan_visual_jobs(assets)

    assert [job.asset_id for job in jobs] == ["map", "page"]
    assert jobs[0].operations == ["ocr", "vlm"]


def test_plan_visual_jobs_filters_by_asset_kind(tmp_path):
    map_path = tmp_path / "map.png"
    page_path = tmp_path / "page.png"
    map_path.write_bytes(b"map")
    page_path.write_bytes(b"page")
    assets = [
        VisualAsset(
            asset_id="page",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            path=page_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        ),
        VisualAsset(
            asset_id="map",
            doc_id="doc",
            page_no=2,
            kind=AssetKind.MAP,
            path=map_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        ),
    ]

    jobs = plan_visual_jobs(assets, kinds={AssetKind.MAP})

    assert [job.asset_id for job in jobs] == ["map"]


def test_run_visual_jobs_returns_asset_annotations(tmp_path):
    image_path = tmp_path / "map.png"
    image_path.write_bytes(b"map")
    assets = [
        VisualAsset(
            asset_id="map",
            doc_id="doc",
            page_no=2,
            kind=AssetKind.MAP,
            path=image_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        )
    ]
    jobs = plan_visual_jobs(assets)

    results = run_visual_jobs(jobs, assets, ocr_backend=FakeOCR(), vlm_backend=FakeVLM())
    annotations = completed_annotations(results)

    assert results[0].status == "completed"
    assert results[0].metadata["ocr_duration_ms"] >= 0
    assert results[0].metadata["vlm_duration_ms"] >= 0
    assert results[0].metadata["total_duration_ms"] >= 0
    assert results[0].metadata["ocr_language"] == "kor+eng"
    assert results[0].metadata["ocr_backend_config"]["name"] == "fake-ocr"
    assert results[0].metadata["vlm_backend_config"]["name"] == "fake-vlm"
    assert results[0].metadata["vlm_prompt_name"] == "map_summary_ko"
    assert len(results[0].metadata["vlm_prompt_sha256"]) == 64
    assert results[0].metadata["vlm_prompt_chars"] > 0
    assert annotations[0].asset_id == "map"
    assert annotations[0].ocr_text.startswith("ocr:")
    assert annotations[0].vlm_summary.startswith("vlm:")
    assert annotations[0].metadata["vlm_prompt_name"] == "map_summary_ko"
    assert annotations[0].metadata["vlm_backend_config"]["max_new_tokens"] == 64


def test_run_visual_jobs_parses_json_vlm_triples(tmp_path):
    image_path = tmp_path / "map.png"
    image_path.write_bytes(b"map")
    assets = [
        VisualAsset(
            asset_id="map",
            doc_id="doc",
            page_no=2,
            kind=AssetKind.MAP,
            path=image_path,
            metadata={"requires_ocr": False, "requires_vlm": True},
        )
    ]
    jobs = plan_visual_jobs(assets, include_ocr=False)

    results = run_visual_jobs(jobs, assets, vlm_backend=JsonVLM())
    annotation = completed_annotations(results)[0]

    assert annotation.caption == "River Corridor Diagram"
    assert annotation.triples[0]["predicate"] == "connects"
    assert annotation.metadata["vlm_parse_status"] == "json_object"


def test_summarize_visual_results_reports_backend_metrics(tmp_path):
    image_path = tmp_path / "map.png"
    image_path.write_bytes(b"map")
    assets = [
        VisualAsset(
            asset_id="map",
            doc_id="doc",
            page_no=2,
            kind=AssetKind.MAP,
            path=image_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        )
    ]
    jobs = plan_visual_jobs(assets)
    results = run_visual_jobs(
        jobs,
        assets,
        ocr_backend=FakeOCR(),
        vlm_backend=JsonVLM(),
        ocr_backend_name="fake-ocr",
        vlm_backend_name="json-vlm",
    )

    summary = summarize_visual_results(results)

    assert summary.status_counts == {"completed": 1}
    assert summary.operation_counts == {"ocr": 1, "vlm": 1}
    assert summary.triple_count == 1
    assert next(iter(summary.vlm_prompt_counts)).startswith("map_summary_ko:")
    summaries = {(item.operation, item.backend): item for item in summary.operation_summaries}
    assert summaries[("ocr", "fake-ocr")].duration.count == 1
    assert summaries[("vlm", "json-vlm")].output_chars > 0


def test_summarize_visual_results_cli_writes_json(tmp_path):
    output = tmp_path / "visual_summary.json"
    result_path = tmp_path / "results.jsonl"
    write_jsonl(
        result_path,
        [
            VisualJobRunResult(
                job_id="job",
                asset_id="asset",
                page_no=1,
                status="completed",
                metadata={
                    "operations": ["vlm"],
                    "vlm_backend": "model-a",
                    "vlm_duration_ms": 12.5,
                    "vlm_output_chars": 80,
                },
            )
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "summarize-visual-results",
            "--results",
            str(result_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "model-a" in output.read_text(encoding="utf-8")


def test_evaluate_visual_results_gates_annotation_quality():
    results = [
        VisualJobRunResult(
            job_id="job-1",
            asset_id="asset-1",
            page_no=1,
            status="completed",
            annotation=AssetAnnotation(
                asset_id="asset-1",
                page_no=1,
                ocr_text="recognized text",
                vlm_summary="structured visual summary",
                triples=[{"subject": "a", "predicate": "relates_to", "object": "b"}],
                metadata={"vlm_parse_status": "json_object"},
            ),
            metadata={
                "operations": ["ocr", "vlm"],
                "ocr_text_chars": 15,
                "vlm_output_chars": 120,
                "vlm_parse_status": "json_object",
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

    report = evaluate_visual_results(
        results,
        min_completion_rate=0.8,
        min_ocr_text_coverage=1.0,
        min_vlm_summary_coverage=1.0,
        min_vlm_json_parse_rate=0.8,
        min_triples_per_vlm_job=0.5,
        max_failed_count=0,
    )

    assert report.passed is False
    assert report.completion_rate == 0.5
    assert report.ocr_text_coverage == 1.0
    assert report.vlm_summary_coverage == 0.5
    assert report.vlm_json_parse_rate == 0.5
    assert report.triples_per_vlm_job == 0.5
    assert "min_completion_rate" in report.failed_checks
    assert "max_failed_count" in report.failed_checks
    assert report.issues[0].code == "visual_job_failed"


def test_gate_visual_results_cli_writes_report(tmp_path):
    result_path = tmp_path / "results.jsonl"
    output = tmp_path / "visual_quality.json"
    write_jsonl(
        result_path,
        [
            VisualJobRunResult(
                job_id="job-1",
                asset_id="asset-1",
                page_no=1,
                status="completed",
                annotation=AssetAnnotation(
                    asset_id="asset-1",
                    page_no=1,
                    ocr_text="recognized text",
                    vlm_summary="visual summary",
                    metadata={"vlm_parse_status": "raw_text"},
                ),
                metadata={"operations": ["ocr", "vlm"], "ocr_text_chars": 15},
            )
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-visual-results",
            "--results",
            str(result_path),
            "--min-completion-rate",
            "1",
            "--min-vlm-summary-coverage",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "'passed': True" in result.output
    assert "visual summary" not in output.read_text(encoding="utf-8")


def test_plan_visual_jobs_cli_filters_kind(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    map_path = tmp_path / "map.png"
    page_path = tmp_path / "page.png"
    map_path.write_bytes(b"map")
    page_path.write_bytes(b"page")
    write_jsonl(
        package_dir / "assets.jsonl",
        [
            VisualAsset(
                asset_id="page",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.PAGE_IMAGE,
                path=page_path,
                metadata={"requires_ocr": True, "requires_vlm": True},
            ),
            VisualAsset(
                asset_id="map",
                doc_id="doc",
                page_no=2,
                kind=AssetKind.MAP,
                path=map_path,
                metadata={"requires_ocr": True, "requires_vlm": True},
            ),
        ],
    )
    output = tmp_path / "jobs.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "plan-visual-jobs",
            "--package-dir",
            str(package_dir),
            "--kind",
            "map",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "'filtered_kinds': ['map']" in result.output
    assert "page_image" not in result.output
    assert output.read_text(encoding="utf-8").count("\n") == 1


def test_build_vlm_backend_passes_hf_runtime_options(monkeypatch):
    captured = {}

    class FakeHFBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("chunking_docs.vision.hf_vlm.HuggingFaceVLMBackend", FakeHFBackend)

    backend, name = cli_module.build_vlm_backend(
        "hf",
        "model-id",
        device_map="cuda:0",
        torch_dtype="bfloat16",
        max_new_tokens=256,
        attn_implementation="sdpa",
        model_class="vision2seq",
    )

    assert isinstance(backend, FakeHFBackend)
    assert name == "hf:model-id"
    assert captured == {
        "model_name": "model-id",
        "device_map": "cuda:0",
        "torch_dtype": "bfloat16",
        "max_new_tokens": 256,
        "attn_implementation": "sdpa",
        "model_class": "vision2seq",
        "profile": "",
    }


def test_build_vlm_backend_applies_named_profile(monkeypatch):
    captured = {}

    class FakeHFBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("chunking_docs.vision.hf_vlm.HuggingFaceVLMBackend", FakeHFBackend)

    backend, name = cli_module.build_vlm_backend("hf", "", profile="qwen2_5_vl_7b")

    assert isinstance(backend, FakeHFBackend)
    assert name == "hf:qwen2_5_vl_7b"
    assert captured == {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "max_new_tokens": 768,
        "attn_implementation": "",
        "model_class": "image-text-to-text",
        "profile": "qwen2_5_vl_7b",
    }


def test_build_vlm_backend_rejects_unknown_profile():
    with pytest.raises(typer.BadParameter, match="Unsupported VLM profile"):
        cli_module.build_vlm_backend("hf", "", profile="unknown")


def test_build_ocr_backend_passes_paddle_options(monkeypatch):
    captured = {}

    class FakePaddleBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("chunking_docs.vision.paddle_ocr.PaddleOCRBackend", FakePaddleBackend)

    backend, name = cli_module.build_ocr_backend(
        "paddleocr",
        model_lang="korean",
        device="gpu:0",
        engine="paddle_static",
        min_confidence=0.4,
        use_gpu=True,
    )

    assert isinstance(backend, FakePaddleBackend)
    assert name == "paddleocr:korean"
    assert captured == {
        "lang": "korean",
        "device": "gpu:0",
        "engine": "paddle_static",
        "min_confidence": 0.4,
        "use_gpu": True,
    }


def test_parse_page_numbers_accepts_ranges():
    assert cli_module.parse_page_numbers("1,3-5,8") == {1, 3, 4, 5, 8}


def test_apply_chunk_section_labels_to_visual_assets():
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.MAP,
        metadata={"asset_scope": "tile"},
    )
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=2,
        page_end=4,
        kind=ChunkKind.TEXT,
        text="text",
        section=SectionPath(chapter="Chapter", section="Section"),
    )

    updated = cli_module.apply_chunk_section_labels([asset], [chunk])

    assert updated[0].metadata["section_label"] == "Chapter > Section"
