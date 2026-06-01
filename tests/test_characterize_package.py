import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.analysis.characterize import characterize_package, chunk_characteristics
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


def test_characterize_package_reports_strategy_observations(tmp_path):
    package_dir, manifest = make_characteristic_package(tmp_path)

    report = characterize_package(
        manifest.profiles,
        manifest.chunks,
        manifest.assets,
        manifest.triples,
        package_dir=package_dir,
    )

    observation_codes = {observation.code for observation in report.observations}
    recommendation_codes = {recommendation.code for recommendation in report.recommendations}
    assert report.text_layer.degraded_or_empty_ratio == 0.5
    assert report.visual.asset_kind_counts["map"] == 1
    assert report.visual.pages_requiring_ocr_count == 1
    assert report.visual.vlm_object_asset_count == 1
    assert report.visual.vlm_object_count == 2
    assert report.visual.vlm_object_bbox_count == 1
    assert report.graph.visual_triple_count == 1
    assert "text_layer_degraded" in observation_codes
    assert "visual_retrieval_required" in observation_codes
    assert "visual_annotation_pending" in observation_codes
    assert "vlm_objects_available" in observation_codes
    assert "graph_triples_missing" not in observation_codes
    assert "prioritize_visual_annotations" in recommendation_codes
    assert "evaluate_visual_vectors" in recommendation_codes
    assert "generate_visual_object_probe_cases" in recommendation_codes
    assert "compare_multimodal_hierarchical_chunking" in recommendation_codes
    assert "maintain_retrieval_benchmark" in recommendation_codes
    object_probe_recommendation = next(
        item for item in report.recommendations if item.code == "generate_visual_object_probe_cases"
    )
    object_probe_audit_command = object_probe_recommendation.commands[1]
    assert "--query-mode salient_terms" in object_probe_recommendation.commands[0]
    assert "--min-case-group-count case_source:visual_object_probe=2" in object_probe_audit_command
    assert "--min-distinct-asset-targets 1" in object_probe_audit_command
    assert "--max-asset-cases-per-target 3" in object_probe_audit_command
    assert "--min-query-terms-per-case 3" in object_probe_audit_command
    assert "--require-visual-only-object-probes" in object_probe_audit_command
    assert object_probe_recommendation.metadata["recommended_object_probe_case_threshold"] == 2
    assert object_probe_recommendation.metadata["recommended_distinct_asset_threshold"] == 1
    benchmark_recommendation = next(
        item for item in report.recommendations if item.code == "maintain_retrieval_benchmark"
    )
    assert "--min-query-terms-per-case 3" in benchmark_recommendation.commands[0]
    assert "--max-duplicate-queries 0" in benchmark_recommendation.commands[0]


def test_characterize_package_cli_writes_json(tmp_path):
    package_dir, _ = make_characteristic_package(tmp_path)
    output = tmp_path / "characteristics.json"

    result = CliRunner().invoke(
        app,
        [
            "characterize-package",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--max-pages",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["visual"]["asset_kind_counts"]["map"] == 1
    assert payload["visual"]["vlm_object_count"] == 2
    assert any(item["code"] == "visual_retrieval_required" for item in payload["observations"])
    assert any(item["code"] == "evaluate_visual_vectors" for item in payload["recommendations"])
    assert any(
        item["code"] == "generate_visual_object_probe_cases"
        for item in payload["recommendations"]
    )
    object_probe_recommendation = next(
        item for item in payload["recommendations"] if item["code"] == "generate_visual_object_probe_cases"
    )
    assert "--require-visual-only-object-probes" in object_probe_recommendation["commands"][1]
    assert object_probe_recommendation["metadata"]["recommended_object_probe_case_threshold"] == 2


def test_chunk_characteristics_counts_source_ref_visual_links():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="visual context",
            source_refs=["asset:asset-1"],
        )
    ]

    report = chunk_characteristics(chunks)

    assert report.chunks_with_assets == 1


def make_characteristic_package(tmp_path: Path):
    package_dir = tmp_path / "package"
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
            char_count=0,
            line_count=0,
            text_block_count=0,
            image_block_count=1,
            embedded_image_count=1,
            drawing_count=30,
            text_quality=TextQuality.EMPTY,
        ),
        PageProfile(
            doc_id="doc",
            page_no=2,
            width=100,
            height=100,
            char_count=200,
            line_count=5,
            text_block_count=1,
            image_block_count=0,
            embedded_image_count=0,
            drawing_count=0,
            text_quality=TextQuality.GOOD,
        ),
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.PAGE_SUMMARY,
            text="visual page",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            caption="map",
            metadata={
                "requires_ocr": True,
                "requires_vlm": True,
                "objects": [
                    {"label": "station marker", "bbox": [0.1, 0.2, 0.3, 0.4]},
                    {"label": "corridor line"},
                ],
            },
        )
    ]
    triples = [
        GraphTriple(
            triple_id="triple-1",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="a",
            predicate="relates_to",
            object="b",
            qualifiers={"source": "visual_annotation"},
        )
    ]
    manifest = ProcessingManifest(doc=doc, profiles=profiles, chunks=chunks, assets=assets, triples=triples)
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", profiles)
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", triples)
    (package_dir / "bm25_tokens.json").write_text("{}", encoding="utf-8")
    (package_dir / "embedding_manifest.json").write_text("{}", encoding="utf-8")
    (package_dir / "qdrant_text_records.jsonl").write_text("", encoding="utf-8")
    return package_dir, manifest
