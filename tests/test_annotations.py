from pathlib import Path

from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, VisualAsset
from typer.testing import CliRunner

from chunking_docs.evaluation.chunking_quality import visual_text_coverage_stats
from chunking_docs.io import write_jsonl
from chunking_docs.models import ProcessingManifest, SourceDocument
from chunking_docs.vision.annotate import (
    annotate_assets,
    merge_asset_annotations_into_chunks,
    repair_visual_text_chunks,
)


class FakeOCR:
    def recognize(self, image_path: Path, language: str = "kor+eng"):
        return f"OCR:{image_path.name}:{language}"


class FakeVLM:
    def summarize(self, image_path: Path, prompt: str):
        return f"VLM:{image_path.name}:{prompt[:8]}"


def test_annotate_assets_and_merge_into_chunks():
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=10,
        kind=AssetKind.MAP,
        path=Path("page_0010.png"),
        metadata={"requires_ocr": True},
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=10,
        page_end=10,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base",
        asset_ids=["asset-1"],
    )

    annotated_assets = annotate_assets([asset], ocr_backend=FakeOCR(), vlm_backend=FakeVLM())
    merged_chunks = merge_asset_annotations_into_chunks([chunk], annotated_assets)

    assert annotated_assets[0].ocr_text.startswith("OCR:")
    assert annotated_assets[0].vlm_summary.startswith("VLM:")
    assert "[OCR page 10]" in merged_chunks[0].text
    assert "[VLM page 10 map]" in merged_chunks[0].text
    assert "OCR:" in merged_chunks[0].text
    assert "VLM:" in merged_chunks[0].text
    assert merged_chunks[0].metadata["has_visual_annotations"] is True
    assert merged_chunks[0].metadata["annotation_asset_count"] == 1
    assert merged_chunks[0].metadata["annotation_asset_ids"] == ["asset-1"]


def test_merge_asset_annotations_into_source_ref_chunks():
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=10,
        kind=AssetKind.MAP,
        ocr_text="source ref OCR",
        vlm_summary="source ref VLM",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=10,
        page_end=10,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base",
        source_refs=["asset:asset-1"],
    )

    merged_chunks = merge_asset_annotations_into_chunks([chunk], [asset])

    assert "[OCR page 10]" in merged_chunks[0].text
    assert "[VLM page 10 map]" in merged_chunks[0].text
    assert merged_chunks[0].metadata["annotation_asset_count"] == 1
    assert merged_chunks[0].metadata["annotation_asset_ids"] == ["asset-1"]


def test_repair_visual_text_chunks_adds_structured_metadata_parts():
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=10,
        kind=AssetKind.MAP,
        caption="north map",
        ocr_text="station corridor labels",
        vlm_summary="shows redevelopment axis",
        metadata={
            "page_type": "map",
            "entities": ["station area", "redevelopment zone"],
            "visual_elements": ["blue axis"],
        },
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=10,
        page_end=10,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base\n\nnorth map",
        source_refs=["asset:asset-1"],
    )

    repaired, report = repair_visual_text_chunks([chunk], [asset])
    coverage = visual_text_coverage_stats(repaired, [asset])

    assert report.updated_chunks == 1
    assert report.repaired_asset_count == 1
    assert "Entities: station area; redevelopment zone" in repaired[0].text
    assert "Visual elements: blue axis" in repaired[0].text
    assert repaired[0].metadata["visual_text_repair"] is True
    assert coverage["part_coverage_ratio"] == 1.0


def test_repair_visual_text_cli_writes_repaired_chunks(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    doc = SourceDocument(doc_id="doc", title="Doc", local_path=tmp_path / "doc.pdf")
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        metadata={"entities": ["station area"]},
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="base",
        asset_ids=["asset-1"],
    )
    manifest = ProcessingManifest(doc=doc, chunks=[chunk], assets=[asset])
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", [])
    write_jsonl(package_dir / "chunks.jsonl", [chunk])
    write_jsonl(package_dir / "assets.jsonl", [asset])
    write_jsonl(package_dir / "triples.jsonl", [])

    from chunking_docs.cli import app

    result = CliRunner().invoke(app, ["repair-visual-text", "--package-dir", str(package_dir)])

    assert result.exit_code == 0, result.output
    assert (package_dir / "chunks.visual_text_repaired.jsonl").exists()
    assert '"added_text_part_count": 1' in result.output
