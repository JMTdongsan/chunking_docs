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
    GraphTriple,
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
    assert any("visual_job_results.qwen2_5_vl_7b.jsonl" in command for command in visual_commands)
    assert any("visual_annotations.qwen2_5_vl_7b.jsonl" in command for command in visual_commands)
    assert any("visual_job_results.llava_next_7b.jsonl" in command for command in visual_commands)
    assert any("visual_annotations.llava_next_7b.jsonl" in command for command in visual_commands)
    assert any("runtime_doctor.qwen2_5_vl_7b.json" in command for command in visual_commands)
    assert any("runtime_doctor.llava_next_7b.json" in command for command in visual_commands)
    vlm_gate_commands = [
        command for command in visual_commands if "gate-vlm-experiment-plan" in command
    ]
    assert len(vlm_gate_commands) == 2
    assert "vlm_experiment_plan_gate.runtime.json" in vlm_gate_commands[0]
    assert "--require-doctor-outputs" in vlm_gate_commands[0]
    assert "--require-results" not in vlm_gate_commands[0]
    assert "vlm_experiment_plan_gate.results.json" in vlm_gate_commands[1]
    assert "--require-doctor-outputs" in vlm_gate_commands[1]
    assert "--require-results" in vlm_gate_commands[1]
    assert "--require-annotations" in vlm_gate_commands[1]
    assert "--min-completed-result-profiles 2" in vlm_gate_commands[1]
    assert "--require-same-result-jobs" in vlm_gate_commands[1]
    assert any("apply-annotations" in command for command in visual_commands)
    assert any("visual_annotations.qwen2_5_vl_7b.jsonl" in command and "apply-annotations" in command for command in visual_commands)
    assert any("compare-visual-runs" in command for command in visual_commands)
    assert any("qwen2_5_vl_7b=" in command for command in visual_commands)
    assert any("llava_next_7b=" in command for command in visual_commands)
    assert any("visual_run_comparison.json" in command for command in visual_commands)
    gate_visual_command = next(command for command in visual_commands if "gate-visual-results" in command)
    assert "visual_job_results.qwen2_5_vl_7b.jsonl" in gate_visual_command
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
    assert "--min-source-precision-at-hits qdrant:text_dense=0.5" in qdrant_commands[1]
    assert "--min-source-precision-at-hits qdrant:caption_dense=0.5" in qdrant_commands[1]
    assert "--min-source-precision-at-hits qdrant:image_dense=0.5" in qdrant_commands[1]
    assert "--min-source-family-precision-at-hits dense_text=0.5" in qdrant_commands[1]
    assert "--min-source-family-precision-at-hits visual=0.5" in qdrant_commands[1]
    assert "export-qdrant-retrieval-config" in qdrant_commands[2]
    assert "eval-qdrant-retrieval-config" in qdrant_commands[3]
    assert "--image-query-backend clip" in qdrant_commands[3]
    assert "eval-qdrant-rag-context-config" in qdrant_commands[4]
    assert "--image-query-backend clip" in qdrant_commands[4]
    assert "gate-rag-context" in qdrant_commands[5]
    readiness_command = plan.steps[-1].commands[0]
    assert "--runtime-report" in readiness_command
    assert "runtime_doctor.json" in readiness_command
    assert "--require-runtime-report" in readiness_command
    assert "--require-visual-annotations" in readiness_command
    assert "--require-visual-quality" in readiness_command
    assert "--min-vlm-json-parse-rate 0.9" in readiness_command
    assert "--visual-run-comparison" in readiness_command
    assert "--require-visual-run-comparison" in readiness_command
    assert "--require-visual-run-same-jobs" in readiness_command
    assert "--min-visual-run-count 2" in readiness_command
    assert "--max-retrieval-expected-targets-per-case 5" in readiness_command
    assert "--chunking-comparison" in readiness_command
    assert "--min-chunking-retrieval-score-per-embedding-kchar 0.0008" in readiness_command
    assert "--min-chunking-retrieval-score-per-mean-latency-ms 0.0005" in readiness_command
    assert "--min-chunking-target-coverage-per-p95-latency-ms 0.0005" in readiness_command
    assert "--qdrant-retrieval-config" in readiness_command
    assert "--require-qdrant-retrieval-config" in readiness_command
    assert "--retrieval-evaluation" in readiness_command
    assert "qdrant_retrieval_config_eval.json" in readiness_command
    assert "--require-retrieval-evaluation" in readiness_command
    assert "--min-target-coverage-at-k 0.8" in readiness_command
    assert "--min-target-ndcg-at-k 0.7" in readiness_command
    assert "--max-retrieval-failed-queries 3" in readiness_command
    assert "--min-retrieval-source-precision-at-hits qdrant:text_dense=0.5" in readiness_command
    assert "--min-retrieval-source-precision-at-hits qdrant:caption_dense=0.5" in readiness_command
    assert "--min-retrieval-source-precision-at-hits qdrant:image_dense=0.5" in readiness_command
    assert "--min-retrieval-source-family-precision-at-hits dense_text=0.5" in readiness_command
    assert "--min-retrieval-source-family-precision-at-hits visual=0.5" in readiness_command
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
    assert "--require-retrieval-evaluation" in readiness_command
    assert "--require-rag-context-evaluation" in readiness_command
    assert "--require-visual-run-comparison" not in readiness_command


def test_workflow_plan_exports_adaptive_qdrant_route_for_visual_object_graph_package(tmp_path):
    package_dir, manifest = make_routed_qdrant_package(tmp_path)
    cases = tmp_path / "cases.jsonl"
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
        vlm_profiles=["qwen2_5_vl_7b"],
    )

    qdrant_commands = next(
        step.commands
        for step in plan.steps
        if step.step_id == "validate_qdrant_rag_context"
    )
    assert "--vector-names text_dense,caption_dense,object_dense,image_dense,triple_dense" in (
        qdrant_commands[1]
    )
    assert "--min-source-precision-at-hits qdrant:object_dense=0.3" in qdrant_commands[1]
    assert "--min-source-precision-at-hits qdrant:triple_dense=0.5" in qdrant_commands[1]
    assert "--min-source-family-precision-at-hits graph=0.5" in qdrant_commands[1]
    assert "export-qdrant-retrieval-config" in qdrant_commands[2]
    assert "--route-preset adaptive" in qdrant_commands[2]
    assert "eval-qdrant-retrieval-config" in qdrant_commands[3]
    assert "qdrant_retrieval_config.json" in qdrant_commands[3]
    assert "eval-qdrant-rag-context-config" in qdrant_commands[4]
    readiness_command = plan.steps[-1].commands[0]
    assert "--qdrant-retrieval-config" in readiness_command
    assert "qdrant_retrieval_config.json" in readiness_command
    assert "--retrieval-evaluation" in readiness_command
    assert "qdrant_retrieval_config_eval.json" in readiness_command
    assert "--rag-context-evaluation" in readiness_command
    assert "qdrant_rag_context_config_eval.json" in readiness_command
    assert (
        "--min-retrieval-case-group-target-coverage retrieval_route:graph_triple=0.7"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-target-coverage retrieval_route:visual_object=0.7"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-source-target-coverage "
        "retrieval_route:graph_triple:qdrant:triple_dense=0.7"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-source-target-coverage "
        "retrieval_route:visual_object:qdrant:object_dense=0.3"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-source-family-target-coverage "
        "retrieval_route:graph_triple:graph=0.7"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-source-family-target-coverage "
        "retrieval_route:visual_object:visual=0.3"
        in readiness_command
    )
    assert (
        "--min-retrieval-source-precision-at-hits qdrant:object_dense=0.3"
        in readiness_command
    )
    assert (
        "--min-retrieval-source-precision-at-hits qdrant:triple_dense=0.5"
        in readiness_command
    )
    assert (
        "--min-retrieval-source-family-precision-at-hits graph=0.5"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-source-precision-at-hits "
        "retrieval_route:graph_triple:qdrant:triple_dense=0.5"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-source-precision-at-hits "
        "retrieval_route:visual_object:qdrant:object_dense=0.3"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-source-family-precision-at-hits "
        "retrieval_route:graph_triple:graph=0.5"
        in readiness_command
    )
    assert (
        "--min-retrieval-case-group-source-family-precision-at-hits "
        "retrieval_route:visual_object:visual=0.3"
        in readiness_command
    )
    assert (
        "--min-rag-context-case-group-target-coverage retrieval_route:graph_triple=0.7"
        in readiness_command
    )
    assert (
        "--min-rag-context-case-group-target-coverage retrieval_route:visual_object=0.7"
        in readiness_command
    )


def test_workflow_plan_uses_ocr_only_visual_step_when_no_vlm_jobs_pending(tmp_path):
    package_dir, manifest = make_ocr_only_visual_package(tmp_path)
    cases = tmp_path / "cases.jsonl"
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

    visual_commands = next(step.commands for step in plan.steps if step.step_id == "visual_annotations")
    assert visual_commands[0].endswith("--no-include-vlm")
    assert not any("plan-vlm-experiments" in command for command in visual_commands)
    assert not any("compare-visual-runs" in command for command in visual_commands)
    assert any("--vlm none" in command for command in visual_commands)
    assert any("visual_job_results.ocr.jsonl" in command for command in visual_commands)
    assert any("visual_annotations.ocr.jsonl" in command for command in visual_commands)
    assert any("apply-annotations" in command for command in visual_commands)
    readiness_command = plan.steps[-1].commands[0]
    assert "--require-visual-quality" in readiness_command
    assert "--min-vlm-summary-coverage" not in readiness_command
    assert "--require-visual-run-comparison" not in readiness_command


def test_workflow_plan_skips_visual_jobs_when_annotations_are_not_pending(tmp_path):
    package_dir, manifest = make_completed_visual_package(tmp_path)
    cases = tmp_path / "cases.jsonl"
    characteristics = characterize_package(
        profiles=manifest.profiles,
        chunks=manifest.chunks,
        assets=manifest.assets,
        triples=manifest.triples,
        package_dir=package_dir,
    )

    assert "prioritize_visual_annotations" not in {
        item.code for item in characteristics.recommendations
    }
    plan = build_ingestion_workflow_plan(
        characteristics,
        package_dir=package_dir,
        retrieval_cases=cases,
        vlm_profiles=["qwen2_5_vl_7b", "llava_next_7b"],
    )

    step_ids = [step.step_id for step in plan.steps]
    assert "visual_annotations" not in step_ids
    readiness_command = plan.steps[-1].commands[0]
    assert "--require-visual-quality" in readiness_command
    assert "--min-vlm-summary-coverage" in readiness_command
    assert "--require-visual-run-comparison" not in readiness_command


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


def make_completed_visual_package(tmp_path: Path):
    package_dir, manifest = make_ocr_only_visual_package(tmp_path)
    manifest.assets[0].ocr_text = ""
    manifest.assets[0].metadata["ocr_backend"] = "paddleocr"
    manifest.assets[0].metadata["ocr_text_chars"] = 0
    manifest.assets[0].metadata["vlm_parse_status"] = "json_object"
    manifest.assets[0].metadata["objects"] = [{"label": "station area"}]
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)
    return package_dir, manifest


def make_routed_qdrant_package(tmp_path: Path):
    package_dir, manifest = make_completed_visual_package(tmp_path)
    manifest.triples = [
        GraphTriple(
            triple_id="triple-1",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="station area",
            predicate="supports",
            object="redevelopment plan",
            qualifiers={"source": "visual_annotation", "asset_id": "asset-1"},
        )
    ]
    write_jsonl(package_dir / "triples.jsonl", manifest.triples)
    return package_dir, manifest


def make_ocr_only_visual_package(tmp_path: Path):
    package_dir = tmp_path / "ocr_only_package"
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
            vlm_summary="already summarized map",
            metadata={"requires_ocr": True, "requires_vlm": True, "vlm_parse_status": "json_object"},
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
