from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from chunking_docs.analysis.pdf_profile import (
    page_profile_text_quality_metadata,
    text_quality_analysis,
    text_quality_metadata,
)
from chunking_docs.chunking.section_map import SectionRange, section_for_page
from chunking_docs.models import ChunkKind, DocumentChunk, PageProfile, TextQuality


def chunk_id(doc_id: str, page_start: int, page_end: int, kind: ChunkKind, index: int = 0) -> str:
    raw = f"{doc_id}:{page_start}:{page_end}:{kind}:{index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def clean_pdf_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def page_level_chunks(
    pdf_path: Path,
    doc_id: str,
    profiles: list[PageProfile],
    section_ranges: list[SectionRange] | None = None,
) -> list[DocumentChunk]:
    profile_map = {profile.page_no: profile for profile in profiles}
    chunks: list[DocumentChunk] = []

    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            page_no = index + 1
            profile = profile_map.get(page_no)
            text = clean_pdf_text(page.get_text("text") or "")
            if profile:
                quality_metadata = page_profile_text_quality_metadata(profile)
            else:
                quality_metadata = text_quality_metadata(text_quality_analysis(text))
            quality = quality_metadata["text_quality"]
            section = section_for_page(page_no, section_ranges)

            if quality == TextQuality.GOOD and text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id(doc_id, page_no, page_no, ChunkKind.TEXT),
                        doc_id=doc_id,
                        page_start=page_no,
                        page_end=page_no,
                        kind=ChunkKind.TEXT,
                        text=text,
                        section=section,
                        metadata={
                            **quality_metadata,
                            "section_label": section.label(),
                            "source": "pdf_text_layer",
                        },
                    )
                )
            else:
                reason = "empty text layer" if quality == TextQuality.EMPTY else "degraded text layer"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id(doc_id, page_no, page_no, ChunkKind.PAGE_SUMMARY),
                        doc_id=doc_id,
                        page_start=page_no,
                        page_end=page_no,
                        kind=ChunkKind.PAGE_SUMMARY,
                        text=f"[{reason}] OCR/VLM processing required for page {page_no}.",
                        section=section,
                        metadata={
                            **quality_metadata,
                            "section_label": section.label(),
                            "requires_ocr": True,
                            "requires_vlm": True,
                        },
                    )
                )
    return chunks
