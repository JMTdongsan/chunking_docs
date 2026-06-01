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
    assert "validate_qdrant_rag_context" in step_ids
    assert step_ids.index("refresh_package_indexes") < step_ids.index("build_embedding_artifacts")
    assert step_ids.index("build_embedding_artifacts") < step_ids.index("validate_qdrant_rag_context")
    assert step_ids[-2] == "refresh_package_metadata"
    assert step_ids[-1] == "ingestion_readiness"
    assert plan.metadata["required_step_count"] >= 5
    runtime_command = plan.steps[0].commands[0]
    assert "--require-ocr" in runtime_command
    assert "--vlm-profile qwen2_5_vl_7b" in runtime_command
    visual_commands = next(step.commands for step in plan.steps if step.step_id == "visual_annotations")
    assert any("plan-vlm-experiments" in command for command in visual_commands)
    assert any("--profiles qwen2_5_vl_7b,llava_next_7b" in command for command in visual_commands)
    assert any("--batch-size 25" in command for command in visual_commands)
    assert any("--ocr paddleocr --vlm hf" in command for command in visual_commands)
    assert any(
        "apply-chunking-sweep" in command
        for step in plan.steps
        for command in step.commands
    )
    chunking_commands = next(
        step.commands
        for step in plan.steps
        if step.step_id == "compare_multimodal_hierarchical_chunking"
    )
    assert "--selection-min-retrieval-score-per-embedding-kchar 0.0008" in chunking_commands[0]
    assert "--selection-min-retrieval-score-per-mean-latency-ms 0.0005" in chunking_commands[0]
    assert "--selection-min-target-coverage-per-p95-latency-ms 0.0005" in chunking_commands[0]
    assert "gate-chunking-comparison" in chunking_commands[1]
    assert "--require-retrieval" in chunking_commands[1]
    assert "--min-retrieval-score-per-embedding-kchar 0.0008" in chunking_commands[1]
    assert "--min-retrieval-score-per-mean-latency-ms 0.0005" in chunking_commands[1]
    assert "--min-target-coverage-per-p95-latency-ms 0.0005" in chunking_commands[1]
    qdrant_commands = next(
        step.commands
        for step in plan.steps
        if step.step_id == "validate_qdrant_rag_context"
    )
    assert "sweep-qdrant-fusion" in qdrant_commands[1]
    assert "--vector-names text_dense,caption_dense,image_dense" in qdrant_commands[1]
    assert "export-qdrant-retrieval-config" in qdrant_commands[2]
    assert "eval-qdrant-retrieval-config" in qdrant_commands[3]
    assert "eval-qdrant-rag-context-config" in qdrant_commands[4]
    assert "gate-rag-context" in qdrant_commands[5]
    readiness_command = plan.steps[-1].commands[0]
    assert "--chunking-comparison" in readiness_command
    assert "--min-chunking-retrieval-score-per-embedding-kchar 0.0008" in readiness_command
    assert "--min-chunking-retrieval-score-per-mean-latency-ms 0.0005" in readiness_command
    assert "--min-chunking-target-coverage-per-p95-latency-ms 0.0005" in readiness_command
    assert "--qdrant-retrieval-config" in readiness_command
    assert "--require-qdrant-retrieval-config" in readiness_command
    assert "--rag-context-evaluation" in readiness_command
    assert "--require-rag-context-evaluation" in readiness_command
    assert "--max-rag-context-mean-context-char-count 12000" in readiness_command
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


def test_workflow_plan_rebuilds_embeddings_after_index_refresh_for_indexed_text_package(tmp_path):
    package_dir, manifest = make_indexed_text_package(tmp_path)
    cases = tmp_path / "cases.jsonl"
    characteristics = characterize_package(
        profiles=manifest.profiles,
        chunks=manifest.chunks,
        assets=manifest.assets,
        triples=manifest.triples,
        package_dir=package_dir,
    )

    recommendation_codes = [item.code for item in characteristics.recommendations]
    assert "build_embedding_artifacts" not in recommendation_codes
    assert "evaluate_visual_vectors" not in recommendation_codes
    assert "validate_qdrant_rag_context" in recommendation_codes

    plan = build_ingestion_workflow_plan(
        characteristics,
        package_dir=package_dir,
        retrieval_cases=cases,
        vlm_profiles=["qwen2_5_vl_7b"],
    )

    step_ids = [step.step_id for step in plan.steps]
    assert "rebuild_embedding_artifacts" in step_ids
    assert step_ids.index("refresh_package_indexes") < step_ids.index("rebuild_embedding_artifacts")
    assert step_ids.index("rebuild_embedding_artifacts") < step_ids.index("validate_qdrant_rag_context")
    rebuild_command = next(
        step.commands[0] for step in plan.steps if step.step_id == "rebuild_embedding_artifacts"
    )
    assert "embed-package" in rebuild_command
    assert "--caption-backend same-as-text" in rebuild_command
    readiness_command = plan.steps[-1].commands[0]
    assert "--require-qdrant-retrieval-config" in readiness_command
    assert "--require-rag-context-evaluation" in readiness_command


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


def make_indexed_text_package(tmp_path: Path):
    package_dir = tmp_path / "indexed_text_package"
    package_dir.mkdir()
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
            char_count=240,
            line_count=8,
            text_block_count=2,
            image_block_count=0,
            embedded_image_count=0,
            drawing_count=0,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="redevelopment policy corridor and public housing plan",
        )
    ]
    manifest = ProcessingManifest(doc=doc, profiles=profiles, chunks=chunks, assets=[], triples=[])
    (package_dir / "manifest.json").write_text(
        json.dumps({"doc": doc.model_dump(mode="json")}),
        encoding="utf-8",
    )
    write_jsonl(package_dir / "pages.jsonl", profiles)
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", [])
    write_jsonl(package_dir / "triples.jsonl", [])
    (package_dir / "bm25_tokens.json").write_text("{}", encoding="utf-8")
    (package_dir / "embedding_manifest.json").write_text(
        json.dumps({"records": []}),
        encoding="utf-8",
    )
    (package_dir / "qdrant_text_records.jsonl").write_text(
        json.dumps(
            {
                "id": "chunk-1",
                "vector_name": "text_dense",
                "vector": [0.1, 0.2],
                "payload": {"chunk_id": "chunk-1", "doc_id": "doc"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return package_dir, manifest
