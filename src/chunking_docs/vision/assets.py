from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

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


class TileSpec(NamedTuple):
    index: int
    row: int
    col: int
    bbox: tuple[float, float, float, float]


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


def build_page_tile_assets(
    pdf_path: Path,
    doc_id: str,
    profiles: list[PageProfile],
    output_dir: Path,
    pages: set[int] | None = None,
    rows: int = 2,
    cols: int = 2,
    overlap_ratio: float = 0.08,
    zoom: float = 2.0,
    section_ranges: list[SectionRange] | None = None,
) -> list[VisualAsset]:
    validate_tile_grid(rows=rows, cols=cols, overlap_ratio=overlap_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_map = {profile.page_no: profile for profile in profiles}
    assets: list[VisualAsset] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_no = page_index + 1
            if pages is not None and page_no not in pages:
                continue
            profile = profile_map.get(page_no)
            if profile is None or not should_render_page(profile):
                continue

            section = section_for_page(page_no, section_ranges)
            kind = page_kind(profile)
            parent_asset_id = make_asset_id(doc_id, page_no, kind, "page")
            for tile in page_tiles(
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                rows=rows,
                cols=cols,
                overlap_ratio=overlap_ratio,
            ):
                asset_id = make_asset_id(
                    doc_id,
                    page_no,
                    kind,
                    f"tile:{rows}:{cols}:{overlap_ratio:.4f}:{tile.row}:{tile.col}",
                )
                path = output_dir / f"{asset_id}_page_{page_no:04d}_tile_{tile.row + 1}_{tile.col + 1}.png"
                if not path.exists():
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom),
                        clip=fitz.Rect(*tile.bbox),
                        alpha=False,
                    )
                    pixmap.save(path)

                assets.append(
                    VisualAsset(
                        asset_id=asset_id,
                        doc_id=doc_id,
                        page_no=page_no,
                        kind=kind,
                        path=path,
                        bbox=tile.bbox,
                        caption=(
                            f"Page {page_no} tile {tile.row + 1},{tile.col + 1} "
                            f"of {rows}x{cols}"
                        ),
                        metadata={
                            "asset_scope": "tile",
                            "parent_asset_id": parent_asset_id,
                            "tile_index": tile.index,
                            "tile_row": tile.row,
                            "tile_col": tile.col,
                            "tile_rows": rows,
                            "tile_cols": cols,
                            "tile_overlap_ratio": overlap_ratio,
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


def page_tiles(
    page_width: float,
    page_height: float,
    rows: int = 2,
    cols: int = 2,
    overlap_ratio: float = 0.08,
) -> list[TileSpec]:
    validate_tile_grid(rows=rows, cols=cols, overlap_ratio=overlap_ratio)
    tile_width = page_width / cols
    tile_height = page_height / rows
    x_overlap = tile_width * overlap_ratio
    y_overlap = tile_height * overlap_ratio
    tiles = []
    for row in range(rows):
        for col in range(cols):
            x0 = max(0.0, col * tile_width - x_overlap)
            y0 = max(0.0, row * tile_height - y_overlap)
            x1 = min(page_width, (col + 1) * tile_width + x_overlap)
            y1 = min(page_height, (row + 1) * tile_height + y_overlap)
            tiles.append(TileSpec(index=len(tiles), row=row, col=col, bbox=(x0, y0, x1, y1)))
    return tiles


def validate_tile_grid(rows: int, cols: int, overlap_ratio: float) -> None:
    if rows <= 0:
        raise ValueError("tile rows must be positive")
    if cols <= 0:
        raise ValueError("tile cols must be positive")
    if overlap_ratio < 0.0 or overlap_ratio >= 0.5:
        raise ValueError("tile overlap ratio must be between 0.0 and 0.5")


def merge_visual_assets(existing_assets: list[VisualAsset], new_assets: list[VisualAsset]) -> list[VisualAsset]:
    merged = {asset.asset_id: asset for asset in existing_assets}
    for asset in new_assets:
        merged[asset.asset_id] = asset
    return sorted(merged.values(), key=lambda asset: (asset.page_no, asset.metadata.get("tile_index", -1), asset.asset_id))


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
