from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from chunking_docs.chunking.section_map import SectionRange, section_for_page
from chunking_docs.models import AssetKind, PageProfile, TextQuality, VisualAsset


def make_asset_id(doc_id: str, page_no: int, kind: AssetKind, suffix: str) -> str:
    raw = f"{doc_id}:{page_no}:{kind}:{suffix}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def page_kind(profile: PageProfile) -> AssetKind:
    if profile.page_no <= 3:
        return AssetKind.PAGE_IMAGE

    if profile.embedded_image_count + profile.image_block_count >= 4:
        return AssetKind.MAP
    if profile.drawing_count >= 20:
        return AssetKind.TABLE
    if profile.embedded_image_count + profile.image_block_count >= 1:
        return AssetKind.FIGURE
    return AssetKind.PAGE_IMAGE


def should_render_page(profile: PageProfile) -> bool:
    visual_count = profile.embedded_image_count + profile.image_block_count
    return (
        profile.text_quality in {TextQuality.EMPTY, TextQuality.DEGRADED}
        or visual_count > 0
        or profile.drawing_count >= 20
    )


def build_page_assets(
    pdf_path: Path,
    doc_id: str,
    profiles: list[PageProfile],
    output_dir: Path,
    zoom: float = 2.0,
    section_ranges: list[SectionRange] | None = None,
) -> list[VisualAsset]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_map = {profile.page_no: profile for profile in profiles}
    assets: list[VisualAsset] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_no = page_index + 1
            profile = profile_map[page_no]
            if not should_render_page(profile):
                continue

            section = section_for_page(page_no, section_ranges)
            kind = page_kind(profile)
            asset_id = make_asset_id(doc_id, page_no, kind, "page")
            path = output_dir / f"{asset_id}_page_{page_no:04d}.png"
            if not path.exists():
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                pixmap.save(path)

            assets.append(
                VisualAsset(
                    asset_id=asset_id,
                    doc_id=doc_id,
                    page_no=page_no,
                    kind=kind,
                    path=path,
                    bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                    caption=f"Full page render for page {page_no}",
                    metadata={
                        "asset_scope": "page",
                        "text_quality": profile.text_quality,
                        "image_block_count": profile.image_block_count,
                        "embedded_image_count": profile.embedded_image_count,
                        "drawing_count": profile.drawing_count,
                        "section_label": section.label(),
                        "requires_ocr": profile.text_quality != TextQuality.GOOD,
                        "requires_vlm": True,
                    },
                )
            )
    return assets


def attach_assets_to_chunks(chunks, assets: list[VisualAsset]):
    assets_by_page: dict[int, list[str]] = {}
    for asset in assets:
        assets_by_page.setdefault(asset.page_no, []).append(asset.asset_id)

    updated = []
    for chunk in chunks:
        asset_ids = []
        for page_no in range(chunk.page_start, chunk.page_end + 1):
            asset_ids.extend(assets_by_page.get(page_no, []))
        updated.append(chunk.model_copy(update={"asset_ids": sorted(set(chunk.asset_ids + asset_ids))}))
    return updated
