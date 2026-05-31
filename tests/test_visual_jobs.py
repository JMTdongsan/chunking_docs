from pathlib import Path

from typer.testing import CliRunner

import chunking_docs.cli as cli_module
from chunking_docs.cli import app
from chunking_docs.io import write_jsonl
from chunking_docs.models import AssetKind, VisualAsset
from chunking_docs.vision.jobs import (
    VisualJobRunResult,
    completed_annotations,
    plan_visual_jobs,
    run_visual_jobs,
)
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
    )

    assert isinstance(backend, FakeHFBackend)
    assert name == "hf:model-id"
    assert captured == {
        "model_name": "model-id",
        "device_map": "cuda:0",
        "torch_dtype": "bfloat16",
        "max_new_tokens": 256,
        "attn_implementation": "sdpa",
    }


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
