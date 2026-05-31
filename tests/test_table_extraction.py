import json
from pathlib import Path

import fitz
from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.ingest.pdf_loader import stable_doc_id
from chunking_docs.ingest.tables import extract_pdf_tables, table_to_markdown
from chunking_docs.io import read_jsonl
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.pipeline import build_processing_package


def test_table_to_markdown_escapes_cells():
    markdown = table_to_markdown([["Name", "Value"], ["Alpha|Beta", "10"]])

    assert markdown.splitlines()[0] == "| Name | Value |"
    assert "Alpha\\|Beta" in markdown


def test_extract_pdf_tables_builds_table_asset_and_chunk(tmp_path):
    pdf_path = make_table_pdf(tmp_path / "tables.pdf")
    doc_id = stable_doc_id(pdf_path)

    assets, chunks = extract_pdf_tables(
        pdf_path=pdf_path,
        doc_id=doc_id,
        output_dir=tmp_path / "assets",
    )

    assert len(assets) == 1
    assert len(chunks) == 1
    assert assets[0].kind == AssetKind.TABLE
    assert assets[0].path is not None and assets[0].path.exists()
    assert assets[0].metadata["source"] == "pdf_table_detection"
    assert assets[0].metadata["table_rows"] == 3
    assert chunks[0].kind == ChunkKind.TABLE
    assert chunks[0].asset_ids == [assets[0].asset_id]
    assert "| Alpha | 10 |" in chunks[0].text


def test_extract_tables_cli_updates_package(tmp_path):
    pdf_path = make_table_pdf(tmp_path / "tables.pdf")
    package_dir = tmp_path / "package"
    section_map = tmp_path / "sections.jsonl"
    section_map.write_text(
        json.dumps({"page_start": 1, "page_end": 1, "chapter": "Reference"}) + "\n",
        encoding="utf-8",
    )
    build_processing_package(pdf_path, package_dir, extract_tables=False)

    result = CliRunner().invoke(
        app,
        [
            "extract-tables",
            "--package-dir",
            str(package_dir),
            "--pdf",
            str(pdf_path),
            "--section-map",
            str(section_map),
        ],
    )

    assert result.exit_code == 0, result.output
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    triples = read_jsonl(package_dir / "triples.jsonl", GraphTriple)
    payload = json.loads((package_dir / "qdrant_collection.json").read_text(encoding="utf-8"))
    table_assets = [asset for asset in assets if asset.kind == AssetKind.TABLE]
    table_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.TABLE]

    assert len(table_assets) == 1
    assert len(table_chunks) == 1
    assert any(table_assets[0].asset_id in chunk.asset_ids for chunk in chunks)
    assert any(triple.chunk_id == table_chunks[0].chunk_id for triple in triples)
    assert payload["named_vectors"]["caption_dense"]["size"] == 384


def make_table_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    left = 50
    top = 50
    cell_width = 80
    cell_height = 30
    for row in range(4):
        y = top + row * cell_height
        page.draw_line((left, y), (left + 2 * cell_width, y), color=(0, 0, 0), width=1)
    for col in range(3):
        x = left + col * cell_width
        page.draw_line((x, top), (x, top + 3 * cell_height), color=(0, 0, 0), width=1)
    rows = [["Name", "Value"], ["Alpha", "10"], ["Beta", "20"]]
    for row_index, row in enumerate(rows):
        for col_index, text in enumerate(row):
            page.insert_text(
                (left + col_index * cell_width + 5, top + row_index * cell_height + 20),
                text,
                fontsize=10,
            )
    doc.save(path)
    doc.close()
    return path
