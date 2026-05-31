import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.delta import compare_processing_packages
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
from chunking_docs.storage.records import EmbeddingRecord


def test_compare_processing_packages_reports_annotation_delta(tmp_path):
    before_dir, before = write_delta_package(tmp_path / "before", annotated=False)
    after_dir, after = write_delta_package(tmp_path / "after", annotated=True)

    report = compare_processing_packages(before, after, before_dir, after_dir)

    assert report.count_delta["annotated_assets"] == 1
    assert report.count_delta["chunks_with_visual_annotations"] == 1
    assert report.count_delta["visual_triples"] == 1
    assert report.changed_ids["chunks"] == ["chunk-1"]
    assert report.changed_ids["assets"] == ["asset-1"]
    assert report.qdrant_record_count_delta["text_dense"] == 1
    assert {observation.code for observation in report.observations} == {
        "annotations_added",
        "chunks_enriched",
        "qdrant_records_added",
        "visual_triples_added",
    }


def test_compare_packages_cli_writes_json(tmp_path):
    before_dir, _ = write_delta_package(tmp_path / "before", annotated=False)
    after_dir, _ = write_delta_package(tmp_path / "after", annotated=True)
    output = tmp_path / "delta.json"

    result = CliRunner().invoke(
        app,
        [
            "compare-packages",
            str(before_dir),
            str(after_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["count_delta"]["annotated_assets"] == 1
    assert payload["changed_ids"]["chunks"] == ["chunk-1"]
    assert payload["observations"][0]["code"] == "annotations_added"


def write_delta_package(path: Path, annotated: bool):
    path.mkdir()
    doc = SourceDocument(
        doc_id="doc",
        title="Reference Document",
        local_path=path / "reference.pdf",
    )
    profile = PageProfile(
        doc_id="doc",
        page_no=1,
        width=100,
        height=100,
        char_count=0,
        line_count=0,
        text_block_count=0,
        image_block_count=1,
        embedded_image_count=1,
        drawing_count=10,
        text_quality=TextQuality.EMPTY,
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base visual page" + ("\n\n[VLM page 1 map]\nvisual summary" if annotated else ""),
        asset_ids=["asset-1"],
        metadata={"has_visual_annotations": True} if annotated else {},
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="map",
        ocr_text="recognized text" if annotated else None,
        vlm_summary="visual summary" if annotated else None,
    )
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
    ] if annotated else []
    manifest = ProcessingManifest(doc=doc, profiles=[profile], chunks=[chunk], assets=[asset], triples=triples)
    (path / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(path / "pages.jsonl", [profile])
    write_jsonl(path / "chunks.jsonl", [chunk])
    write_jsonl(path / "assets.jsonl", [asset])
    write_jsonl(path / "triples.jsonl", triples)
    write_jsonl(
        path / "qdrant_text_records.jsonl",
        [
            EmbeddingRecord(
                point_id="00000000-0000-0000-0000-000000000001",
                chunk_id="chunk-1",
                doc_id="doc",
                vector_name="text_dense",
                vector=[0.1, 0.2],
                payload={"chunk_id": "chunk-1", "doc_id": "doc", "page_start": 1, "page_end": 1, "kind": "text", "text": "base"},
            ),
            *(
                [
                    EmbeddingRecord(
                        point_id="00000000-0000-0000-0000-000000000002",
                        chunk_id="chunk-2",
                        doc_id="doc",
                        vector_name="text_dense",
                        vector=[0.2, 0.3],
                        payload={
                            "chunk_id": "chunk-2",
                            "doc_id": "doc",
                            "page_start": 1,
                            "page_end": 1,
                            "kind": "text",
                            "text": "visual summary",
                        },
                    )
                ]
                if annotated
                else []
            ),
        ],
    )
    return path, manifest
