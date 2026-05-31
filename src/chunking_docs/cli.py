from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer
from rich import print

from chunking_docs.analysis.pdf_profile import profile_pdf, summarize_profiles, write_profile_outputs
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.ingest.pdf_loader import load_source_document, render_pages

app = typer.Typer(help="Document chunking utilities.")


@app.command()
def download(url: str, output: Path):
    """Download a source document."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with output.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    print(f"Downloaded {output}")


@app.command()
def profile(pdf: Path, output_dir: Path = Path("outputs/profile")):
    """Profile PDF text and visual density."""
    document = load_source_document(pdf)
    profiles = profile_pdf(pdf, document.doc_id)
    write_profile_outputs(profiles, output_dir)
    print_json(summarize_profiles(profiles))


@app.command()
def render(pdf: Path, output_dir: Path = Path("outputs/renders"), pages: str = ""):
    """Render selected PDF pages to PNG."""
    page_numbers = [int(item) for item in pages.split(",") if item.strip()] or None
    rendered = render_pages(pdf, output_dir, pages=page_numbers)
    print(f"Rendered {len(rendered)} pages into {output_dir}")


@app.command()
def chunk(pdf: Path, output: Path = Path("outputs/chunks.jsonl")):
    """Create page-level starter chunks."""
    source = load_source_document(pdf)
    profiles = profile_pdf(pdf, source.doc_id)
    chunks = page_level_chunks(pdf, source.doc_id, profiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(chunk.model_dump_json() for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(chunks)} chunks to {output}")


def print_json(payload: dict):
    print(json.dumps(payload, ensure_ascii=False, indent=2))
