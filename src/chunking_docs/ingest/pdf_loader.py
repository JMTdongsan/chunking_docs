from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from chunking_docs.models import SourceDocument


def stable_doc_id(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.name.encode("utf-8"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_source_document(path: Path, title: str | None = None, source_url: str | None = None):
    path = path.resolve()
    with fitz.open(path) as document:
        metadata = dict(document.metadata or {})
        page_count = document.page_count

    return SourceDocument(
        doc_id=stable_doc_id(path),
        title=title or metadata.get("title") or path.stem,
        source_url=source_url,
        local_path=path,
        metadata={**metadata, "page_count": page_count},
    )


def render_pages(
    pdf_path: Path,
    output_dir: Path,
    pages: list[int] | None = None,
    zoom: float = 2.0,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    with fitz.open(pdf_path) as document:
        page_numbers = pages or list(range(1, document.page_count + 1))
        matrix = fitz.Matrix(zoom, zoom)
        for page_no in page_numbers:
            page = document[page_no - 1]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            output_path = output_dir / f"page_{page_no:04d}.png"
            pixmap.save(output_path)
            rendered.append(output_path)
    return rendered
