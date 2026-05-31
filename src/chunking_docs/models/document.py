from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AssetKind(StrEnum):
    PAGE_IMAGE = "page_image"
    FIGURE = "figure"
    TABLE = "table"
    MAP = "map"
    CHART = "chart"
    UNKNOWN = "unknown"


class ChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    MAP = "map"
    PAGE_SUMMARY = "page_summary"
    SECTION_SUMMARY = "section_summary"


class TextQuality(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    EMPTY = "empty"


class SourceDocument(BaseModel):
    doc_id: str
    title: str
    source_url: str | None = None
    local_path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageProfile(BaseModel):
    doc_id: str
    page_no: int
    width: float
    height: float
    char_count: int
    line_count: int
    text_block_count: int
    image_block_count: int
    embedded_image_count: int
    drawing_count: int
    text_quality: TextQuality
    sample: str = ""


class VisualAsset(BaseModel):
    asset_id: str
    doc_id: str
    page_no: int
    kind: AssetKind
    path: Path | None = None
    bbox: tuple[float, float, float, float] | None = None
    caption: str | None = None
    ocr_text: str | None = None
    vlm_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionPath(BaseModel):
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    issue: str | None = None

    def label(self) -> str:
        return " > ".join(part for part in [self.chapter, self.section, self.subsection, self.issue] if part)


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    kind: ChunkKind
    text: str
    section: SectionPath = Field(default_factory=SectionPath)
    asset_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphTriple(BaseModel):
    triple_id: str
    doc_id: str
    chunk_id: str
    subject: str
    predicate: str
    object: str
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
