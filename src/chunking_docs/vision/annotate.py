from __future__ import annotations

from chunking_docs.models import AssetKind, DocumentChunk, VisualAsset
from chunking_docs.vision.interfaces import OCRBackend, VLMBackend
from chunking_docs.vision.prompts import MAP_SUMMARY_PROMPT_KO, PAGE_SUMMARY_PROMPT_KO


def annotate_assets(
    assets: list[VisualAsset],
    ocr_backend: OCRBackend | None = None,
    vlm_backend: VLMBackend | None = None,
    pages: set[int] | None = None,
    limit: int | None = None,
    ocr_language: str = "kor+eng",
) -> list[VisualAsset]:
    annotated: list[VisualAsset] = []
    processed = 0
    for asset in assets:
        if pages is not None and asset.page_no not in pages:
            annotated.append(asset)
            continue
        if limit is not None and processed >= limit:
            annotated.append(asset)
            continue
        if asset.path is None:
            annotated.append(asset)
            continue

        updates = {}
        if ocr_backend and asset.metadata.get("requires_ocr", True):
            updates["ocr_text"] = ocr_backend.recognize(asset.path, language=ocr_language)
        if vlm_backend:
            updates["vlm_summary"] = vlm_backend.summarize(asset.path, prompt_for_asset(asset))

        if updates:
            processed += 1
            annotated.append(asset.model_copy(update=updates))
        else:
            annotated.append(asset)
    return annotated


def prompt_for_asset(asset: VisualAsset) -> str:
    if asset.kind == AssetKind.MAP:
        return MAP_SUMMARY_PROMPT_KO
    return PAGE_SUMMARY_PROMPT_KO


def merge_asset_annotations_into_chunks(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
) -> list[DocumentChunk]:
    assets_by_id = {asset.asset_id: asset for asset in assets}
    merged: list[DocumentChunk] = []
    for chunk in chunks:
        additions = []
        for asset_id in chunk.asset_ids:
            asset = assets_by_id.get(asset_id)
            if asset is None:
                continue
            if asset.ocr_text:
                additions.append(f"[OCR page {asset.page_no}]\n{asset.ocr_text}")
            if asset.vlm_summary:
                additions.append(f"[VLM page {asset.page_no} {asset.kind}]\n{asset.vlm_summary}")
        if additions:
            text = chunk.text.rstrip() + "\n\n" + "\n\n".join(additions)
            metadata = {
                **chunk.metadata,
                "has_visual_annotations": True,
                "annotation_asset_count": len(additions),
            }
            merged.append(chunk.model_copy(update={"text": text, "metadata": metadata}))
        else:
            merged.append(chunk)
    return merged
