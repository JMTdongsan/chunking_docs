from __future__ import annotations

import re
from dataclasses import dataclass

from chunking_docs.embeddings.records import asset_text_parts
from chunking_docs.graph.provenance import chunk_asset_ids
from chunking_docs.models import AssetKind, DocumentChunk, VisualAsset
from chunking_docs.vision.interfaces import OCRBackend, VLMBackend
from chunking_docs.vision.prompts import CHART_SUMMARY_PROMPT_KO
from chunking_docs.vision.prompts import FIGURE_SUMMARY_PROMPT_KO
from chunking_docs.vision.prompts import MAP_SUMMARY_PROMPT_KO
from chunking_docs.vision.prompts import PAGE_SUMMARY_PROMPT_KO
from chunking_docs.vision.prompts import TABLE_SUMMARY_PROMPT_KO


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
    if asset.kind == AssetKind.TABLE:
        return TABLE_SUMMARY_PROMPT_KO
    if asset.kind == AssetKind.CHART:
        return CHART_SUMMARY_PROMPT_KO
    if asset.kind == AssetKind.FIGURE:
        return FIGURE_SUMMARY_PROMPT_KO
    return PAGE_SUMMARY_PROMPT_KO


def prompt_name_for_asset(asset: VisualAsset) -> str:
    if asset.kind == AssetKind.MAP:
        return "map_summary_ko"
    if asset.kind == AssetKind.TABLE:
        return "table_summary_ko"
    if asset.kind == AssetKind.CHART:
        return "chart_summary_ko"
    if asset.kind == AssetKind.FIGURE:
        return "figure_summary_ko"
    return "page_summary_ko"


def merge_asset_annotations_into_chunks(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
) -> list[DocumentChunk]:
    merged, _ = repair_visual_text_chunks(chunks, assets)
    return merged


@dataclass(frozen=True)
class VisualTextRepairReport:
    input_chunks: int
    output_chunks: int
    updated_chunks: int
    linked_visual_text_asset_count: int
    repaired_asset_count: int
    added_text_part_count: int
    skipped_unlinked_asset_count: int
    skipped_unlinked_asset_ids: list[str]


def repair_visual_text_chunks(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
) -> tuple[list[DocumentChunk], VisualTextRepairReport]:
    """Append missing linked asset text parts to chunks for dense and lexical retrieval."""

    assets_by_id = {asset.asset_id: asset for asset in assets}
    linked_asset_ids: set[str] = set()
    for chunk in chunks:
        for asset_id in chunk_asset_ids(chunk):
            asset = assets_by_id.get(asset_id)
            if asset is not None and asset_text_parts(asset):
                linked_asset_ids.add(asset_id)
    visual_text_asset_ids = {
        asset.asset_id
        for asset in assets
        if asset_text_parts(asset)
    }
    skipped_unlinked_asset_ids = sorted(visual_text_asset_ids - linked_asset_ids)
    merged: list[DocumentChunk] = []
    updated_chunks = 0
    repaired_asset_ids: set[str] = set()
    added_text_part_count = 0
    for chunk in chunks:
        additions = []
        annotated_asset_ids = set()
        chunk_added_part_count = 0
        for asset_id in chunk_asset_ids(chunk):
            asset = assets_by_id.get(asset_id)
            if asset is None:
                continue
            missing_parts = missing_visual_text_parts(chunk.text, additions, asset_text_parts(asset))
            if not missing_parts:
                continue
            additions.append(
                f"[Visual asset page {asset.page_no} {asset.kind} {asset.asset_id}]\n"
                + "\n".join(format_visual_text_part(asset, part) for part in missing_parts)
            )
            annotated_asset_ids.add(asset.asset_id)
            repaired_asset_ids.add(asset.asset_id)
            added_text_part_count += len(missing_parts)
            chunk_added_part_count += len(missing_parts)
        if additions:
            text = chunk.text.rstrip() + "\n\n" + "\n\n".join(additions)
            metadata = {
                **chunk.metadata,
                "has_visual_annotations": True,
                "annotation_asset_count": len(merged_annotation_asset_ids(chunk, annotated_asset_ids)),
                "annotation_asset_ids": merged_annotation_asset_ids(chunk, annotated_asset_ids),
                "visual_text_repair": True,
                "visual_text_repair_added_parts": chunk_added_part_count,
            }
            merged.append(chunk.model_copy(update={"text": text, "metadata": metadata}))
            updated_chunks += 1
        else:
            merged.append(chunk)
    return merged, VisualTextRepairReport(
        input_chunks=len(chunks),
        output_chunks=len(merged),
        updated_chunks=updated_chunks,
        linked_visual_text_asset_count=len(linked_asset_ids),
        repaired_asset_count=len(repaired_asset_ids),
        added_text_part_count=added_text_part_count,
        skipped_unlinked_asset_count=len(skipped_unlinked_asset_ids),
        skipped_unlinked_asset_ids=skipped_unlinked_asset_ids[:50],
    )


def missing_visual_text_parts(
    chunk_text: str,
    pending_additions: list[str],
    parts: list[str],
) -> list[str]:
    text = "\n\n".join([chunk_text, *pending_additions])
    return [part for part in parts if not visual_text_part_present(part, text)]


def visual_text_part_present(part: str, text: str) -> bool:
    normalized_part = normalize_for_visual_text_repair(part)
    if not normalized_part:
        return True
    normalized_text = normalize_for_visual_text_repair(text)
    return normalized_part in normalized_text or (
        len(normalized_part) > 80 and normalized_part[:80] in normalized_text
    )


def normalize_for_visual_text_repair(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def format_visual_text_part(asset: VisualAsset, part: str) -> str:
    if asset.ocr_text and part == asset.ocr_text:
        return f"[OCR page {asset.page_no}]\n{part}"
    if asset.vlm_summary and part == asset.vlm_summary:
        return f"[VLM page {asset.page_no} {asset.kind}]\n{part}"
    return part


def merged_annotation_asset_ids(chunk: DocumentChunk, added_asset_ids: set[str]) -> list[str]:
    existing = chunk.metadata.get("annotation_asset_ids")
    values = [item for item in existing if isinstance(item, str)] if isinstance(existing, list) else []
    values.extend(sorted(added_asset_ids))
    return sorted(dict.fromkeys(values))
