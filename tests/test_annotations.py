from pathlib import Path

from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, VisualAsset
from chunking_docs.vision.annotate import annotate_assets, merge_asset_annotations_into_chunks


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
