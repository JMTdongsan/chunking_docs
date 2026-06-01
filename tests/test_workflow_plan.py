import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.analysis.characterize import characterize_package
from chunking_docs.analysis.workflow import build_ingestion_workflow_plan
from chunking_docs.cli import app
from chunking_docs.io import write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    TextQuality,
    VisualAsset,
)


def test_build_ingestion_workflow_plan_orders_runtime_visual_embedding_and_readiness(tmp_path):
    package_dir, manifest = make_workflow_package(tmp_path)
    cases = tmp_path / "retrieval_cases.jsonl"
    characteristics = characterize_package(
        profiles=manifest.profiles,
        chunks=manifest.chunks,
        assets=manifest.assets,
        triples=manifest.triples,
        package_dir=package_dir,
    )

    plan = build_ingestion_workflow_plan(
        characteristics,
        package_dir=package_dir,
        retrieval_cases=cases,
        vlm_profiles=["qwen2_5_vl_7b", "llava_next_7b"],
    )

    step_ids = [step.step_id for step in plan.steps]
    assert step_ids[:4] == [
        "runtime_check",
        "characterize_package",
        "build_page_tiles",
        "visual_annotations",
    ]
    assert "build_embedding_artifacts" in step_ids
    assert step_ids[-2] == "refresh_package_metadata"
    assert step_ids[-1] == "ingestion_readiness"
    assert plan.metadata["required_step_count"] >= 5
    runtime_command = plan.steps[0].commands[0]
    assert "--require-ocr" in runtime_command
    assert "--vlm-profile qwen2_5_vl_7b" in runtime_command
    visual_commands = next(step.commands for step in plan.steps if step.step_id == "visual_annotations")
    assert any("plan-vlm-experiments" in command for command in visual_commands)
    assert any("--profiles qwen2_5_vl_7b,llava_next_7b" in command for command in visual_commands)
    assert any("--ocr paddleocr --vlm hf" in command for command in visual_commands)
    assert any(
        "apply-chunking-sweep" in command
        for step in plan.steps
        for command in step.commands
    )
    assert all("outputs/package" not in command for step in plan.steps for command in step.commands)
    assert any(str(cases) in command for step in plan.steps for command in step.commands)


def test_plan_ingestion_workflow_cli_writes_json(tmp_path):
    package_dir, _ = make_workflow_package(tmp_path)
    output = tmp_path / "workflow.json"
    cases = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "plan-ingestion-workflow",
            "--package-dir",
            str(package_dir),
            "--retrieval-cases",
            str(cases),
            "--vlm-profiles",
            "qwen2_5_vl_7b,llava_next_7b",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["package_dir"] == str(package_dir)
    assert payload["retrieval_cases"] == str(cases)
    assert payload["vlm_profiles"] == ["qwen2_5_vl_7b", "llava_next_7b"]
    assert payload["steps"][0]["step_id"] == "runtime_check"
    assert payload["steps"][-1]["step_id"] == "ingestion_readiness"
    assert any(step["step_id"] == "visual_annotations" for step in payload["steps"])


def make_workflow_package(tmp_path: Path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    doc = SourceDocument(
        doc_id="doc",
        title="Reference Document",
        local_path=tmp_path / "reference.pdf",
    )
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=100,
            height=100,
            char_count=0,
            line_count=0,
            text_block_count=0,
            image_block_count=1,
            embedded_image_count=1,
            drawing_count=30,
            text_quality=TextQuality.EMPTY,
            text_quality_reasons=["empty_text"],
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="visual planning area",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            path=image_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        )
    ]
    manifest = ProcessingManifest(doc=doc, profiles=profiles, chunks=chunks, assets=assets, triples=[])
    (package_dir / "manifest.json").write_text(
        json.dumps({"doc": doc.model_dump(mode="json")}),
        encoding="utf-8",
    )
    write_jsonl(package_dir / "pages.jsonl", profiles)
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", [])
    return package_dir, manifest
