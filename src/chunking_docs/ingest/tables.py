from __future__ import annotations

import contextlib
import hashlib
import io
import unicodedata
from pathlib import Path
from typing import Any

import fitz
from pydantic import BaseModel, Field

from chunking_docs.chunking.page_chunker import chunk_id
from chunking_docs.chunking.section_map import SectionRange, section_for_page
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, VisualAsset
from chunking_docs.vision.assets import make_asset_id


class ExtractedTable(BaseModel):
    table_id: str
    doc_id: str
    page_no: int
    bbox: tuple[float, float, float, float]
    cells: list[list[str]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.cells)

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.cells), default=0)


def extract_pdf_tables(
    pdf_path: Path,
    doc_id: str,
    output_dir: Path | None = None,
    section_ranges: list[SectionRange] | None = None,
    pages: set[int] | None = None,
    min_rows: int = 2,
    min_cols: int = 2,
    max_control_char_ratio: float = 0.02,
    zoom: float = 2.0,
) -> tuple[list[VisualAsset], list[DocumentChunk]]:
    assets: list[VisualAsset] = []
    chunks: list[DocumentChunk] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_no = page_index + 1
            if pages is not None and page_no not in pages:
                continue

            section = section_for_page(page_no, section_ranges)
            for table_index, table in enumerate(find_tables(page), start=1):
                cells = normalize_table_cells(table.extract())
                quality = table_text_quality(cells)
                if not useful_table(
                    cells,
                    min_rows=min_rows,
                    min_cols=min_cols,
                    max_control_char_ratio=max_control_char_ratio,
                    quality=quality,
                ):
                    continue

                bbox = normalize_bbox(table.bbox)
                asset_id = table_asset_id(doc_id, page_no, table_index, bbox)
                asset_path = render_table_asset(
                    page=page,
                    bbox=bbox,
                    output_dir=output_dir,
                    asset_id=asset_id,
                    page_no=page_no,
                    zoom=zoom,
                )
                text = table_to_markdown(cells)
                metadata = {
                    "asset_scope": "table",
                    "source": "pdf_table_detection",
                    "table_index": table_index,
                    "table_rows": len(cells),
                    "table_cols": max((len(row) for row in cells), default=0),
                    "table_text_quality": quality,
                    "bbox": bbox,
                    "section_label": section.label(),
                    "requires_ocr": False,
                    "requires_vlm": False,
                }
                assets.append(
                    VisualAsset(
                        asset_id=asset_id,
                        doc_id=doc_id,
                        page_no=page_no,
                        kind=AssetKind.TABLE,
                        path=asset_path,
                        bbox=bbox,
                        caption=f"Extracted table on page {page_no}",
                        ocr_text=text,
                        metadata=metadata,
                    )
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id(doc_id, page_no, page_no, ChunkKind.TABLE, index=table_index),
                        doc_id=doc_id,
                        page_start=page_no,
                        page_end=page_no,
                        kind=ChunkKind.TABLE,
                        text=text,
                        section=section,
                        asset_ids=[asset_id],
                        metadata={
                            **metadata,
                            "table_asset_id": asset_id,
                        },
                    )
                )
    return assets, chunks


def find_tables(page) -> list:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        finder = page.find_tables()
    return list(getattr(finder, "tables", []))


def normalize_table_cells(rows: list[list[Any]]) -> list[list[str]]:
    normalized = []
    for row in rows:
        normalized.append([clean_cell(cell) for cell in row])
    return trim_empty_edges(normalized)


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def trim_empty_edges(rows: list[list[str]]) -> list[list[str]]:
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return []
    max_cols = max(len(row) for row in rows)
    padded = [row + [""] * (max_cols - len(row)) for row in rows]
    keep_cols = [index for index in range(max_cols) if any(row[index] for row in padded)]
    return [[row[index] for index in keep_cols] for row in padded]


def useful_table(
    cells: list[list[str]],
    min_rows: int = 2,
    min_cols: int = 2,
    max_control_char_ratio: float = 0.02,
    quality: dict[str, Any] | None = None,
) -> bool:
    if len(cells) < min_rows:
        return False
    if max((len(row) for row in cells), default=0) < min_cols:
        return False
    if sum(1 for row in cells for cell in row if cell) < min_rows * min_cols:
        return False
    quality = quality or table_text_quality(cells)
    return float(quality["control_char_ratio"]) <= max_control_char_ratio


def table_text_quality(cells: list[list[str]]) -> dict[str, Any]:
    chars = [
        char
        for row in cells
        for cell in row
        for char in cell
        if not char.isspace()
    ]
    control_count = sum(1 for char in chars if unicodedata.category(char).startswith("C"))
    return {
        "char_count": len(chars),
        "control_char_count": control_count,
        "control_char_ratio": control_count / len(chars) if chars else 0.0,
    }


def table_to_markdown(cells: list[list[str]]) -> str:
    if not cells:
        return ""
    max_cols = max(len(row) for row in cells)
    rows = [row + [""] * (max_cols - len(row)) for row in cells]
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend(
        "| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |"
        for row in body
    )
    return "\n".join(lines)


def escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def normalize_bbox(value) -> tuple[float, float, float, float]:
    rect = fitz.Rect(value)
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def table_asset_id(
    doc_id: str,
    page_no: int,
    table_index: int,
    bbox: tuple[float, float, float, float],
) -> str:
    bbox_digest = hashlib.sha256(",".join(f"{coordinate:.2f}" for coordinate in bbox).encode("utf-8")).hexdigest()[:8]
    return make_asset_id(doc_id, page_no, AssetKind.TABLE, f"table:{table_index}:{bbox_digest}")


def render_table_asset(
    page,
    bbox: tuple[float, float, float, float],
    output_dir: Path | None,
    asset_id: str,
    page_no: int,
    zoom: float,
) -> Path | None:
    if output_dir is None:
        return None
    path = output_dir / f"{asset_id}_page_{page_no:04d}_table.png"
    if path.exists():
        return path
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        clip=fitz.Rect(*bbox),
        alpha=False,
    )
    pixmap.save(path)
    return path
