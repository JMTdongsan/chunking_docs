import json
import hashlib
from pathlib import Path

import fitz
from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.ingest.pdf_loader import stable_doc_id
from chunking_docs.ingest.tables import (
    extract_pdf_tables,
    table_text_quality,
    table_to_markdown,
    useful_table,
    visual_table_chunks_from_assets,
)
from chunking_docs.io import read_jsonl, write_jsonl
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.pipeline import build_processing_package


def test_table_to_markdown_escapes_cells():
    markdown = table_to_markdown([["Name", "Value"], ["Alpha|Beta", "10"]])

    assert markdown.splitlines()[0] == "| Name | Value |"
    assert "Alpha\\|Beta" in markdown


def test_useful_table_rejects_control_character_noise():
    cells = [["Name", "Value"], ["Alpha", "10"], ["\x11\x12\x13", "20"]]
    quality = table_text_quality(cells)

    assert quality["control_char_count"] == 3
    assert useful_table(cells, quality=quality) is False
    assert useful_table([["Name", "Value"], ["Alpha", "10"]]) is True


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
    assert assets[0].metadata["table_text_quality"]["control_char_count"] == 0
    assert chunks[0].kind == ChunkKind.TABLE
    assert chunks[0].asset_ids == [assets[0].asset_id]
    assert "| Alpha | 10 |" in chunks[0].text


def test_visual_table_chunks_from_assets_promotes_annotated_page_table_asset():
    asset = VisualAsset(
        asset_id="table-page",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.TABLE,
        caption="Table page",
        ocr_text="Name Value\nAlpha 10",
        metadata={
            "asset_scope": "page",
            "section_label": "Reference",
            "visual_elements": ["Alpha value table"],
        },
    )

    chunks = visual_table_chunks_from_assets([asset])

    assert len(chunks) == 1
    assert chunks[0].kind == ChunkKind.TABLE
    assert chunks[0].asset_ids == ["table-page"]
    assert chunks[0].metadata["source"] == "visual_table_asset"
    assert chunks[0].metadata["table_source"] == "ocr_vlm_asset"
    assert "Alpha 10" in chunks[0].text
    assert "Visual elements: Alpha value table" in chunks[0].text


def test_visual_table_chunks_from_assets_skips_unannotated_page_render():
    asset = VisualAsset(
        asset_id="table-page",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.TABLE,
        caption="Full page render for page 3",
        metadata={"asset_scope": "page"},
    )

    assert visual_table_chunks_from_assets([asset]) == []


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
    metadata = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))["metadata"]
    table_assets = [asset for asset in assets if asset.kind == AssetKind.TABLE]
    table_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.TABLE]

    assert len(table_assets) == 1
    assert len(table_chunks) == 1
    assert any(table_assets[0].asset_id in chunk.asset_ids for chunk in chunks)
    assert any(triple.chunk_id == table_chunks[0].chunk_id for triple in triples)
    assert not (package_dir / "qdrant_collection.json").exists()
    assert not (package_dir / "embedding_manifest.json").exists()
    assert metadata["table_count"] == 1
    assert metadata["table_asset_count"] == 1


def test_extract_tables_cli_promotes_visual_table_assets(tmp_path):
    pdf_path = make_table_pdf(tmp_path / "tables.pdf")
    package_dir = tmp_path / "package"
    build_processing_package(pdf_path, package_dir, extract_tables=False)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    visual_table = VisualAsset(
        asset_id="visual-table",
        doc_id=stable_doc_id(pdf_path),
        page_no=1,
        kind=AssetKind.TABLE,
        caption="Visual table",
        ocr_text="Category Amount\nAlpha 10",
        metadata={
            "asset_scope": "page",
            "section_label": "Reference",
            "visual_elements": ["Category amount table"],
        },
    )
    write_jsonl(package_dir / "assets.jsonl", [*assets, visual_table])

    result = CliRunner().invoke(
        app,
        [
            "extract-tables",
            "--package-dir",
            str(package_dir),
            "--pdf",
            str(pdf_path),
        ],
    )

    assert result.exit_code == 0, result.output
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    table_chunks = [
        chunk
        for chunk in chunks
        if chunk.kind == ChunkKind.TABLE and chunk.metadata.get("source") == "visual_table_asset"
    ]

    assert len(table_chunks) == 1
    assert table_chunks[0].asset_ids == ["visual-table"]
    assert "Alpha 10" in table_chunks[0].text
    metadata = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))["metadata"]
    assert metadata["table_count"] == 2
    assert metadata["table_asset_count"] == 2


def test_build_processing_package_records_reproducible_package_metadata(tmp_path):
    pdf_path = make_table_pdf(tmp_path / "tables.pdf")
    package_dir = tmp_path / "package"

    manifest = build_processing_package(
        pdf_path,
        package_dir,
        render_zoom=2.0,
        tokenizer_config=LexicalTokenizerConfig(strategy="word", deduplicate=True),
        extract_tables=False,
    )

    persisted = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.metadata

    assert persisted["metadata"] == metadata
    assert metadata["source_file"] == {
        "name": "tables.pdf",
        "bytes": pdf_path.stat().st_size,
        "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    }
    assert metadata["package_config"] == {
        "base_chunking_strategy": "page",
        "render_zoom": 2.0,
        "dry_run_embeddings": True,
        "section_map_count": 0,
        "extract_tables": False,
        "lexical_tokenizer": {
            "strategy": "word",
            "lowercase": True,
            "min_n": 2,
            "max_n": 4,
            "ngram_cjk_only": True,
            "deduplicate": True,
        },
    }
    assert metadata["table_count"] == 0
    assert metadata["table_asset_count"] == 0


def test_refresh_package_metadata_cli_repairs_legacy_manifest(tmp_path):
    pdf_path = make_table_pdf(tmp_path / "tables.pdf")
    package_dir = tmp_path / "package"
    build_processing_package(pdf_path, package_dir, extract_tables=False)
    payload = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["metadata"] = {"profile_summary": payload["metadata"]["profile_summary"]}
    (package_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "refresh-package-metadata",
            "--package-dir",
            str(package_dir),
            "--pdf",
            str(pdf_path),
            "--render-zoom",
            "2.0",
            "--embedding-mode",
            "external",
            "--extract-tables",
            "false",
        ],
    )

    assert result.exit_code == 0, result.output
    refreshed = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = refreshed["metadata"]
    assert metadata["source_file"]["name"] == "tables.pdf"
    assert metadata["source_file"]["sha256"] == hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    assert metadata["embedding_mode"] == "external"
    assert metadata["package_config"]["base_chunking_strategy"] == "page"
    assert metadata["package_config"]["render_zoom"] == 2.0
    assert metadata["package_config"]["dry_run_embeddings"] is False
    assert metadata["package_config"]["extract_tables"] is False
    assert metadata["package_config"]["lexical_tokenizer"] == LexicalTokenizerConfig().model_dump()
    assert metadata["table_count"] == 0
    assert metadata["table_asset_count"] == 0


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
