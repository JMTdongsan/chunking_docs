import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.casegen import generate_retrieval_case_skeleton
from chunking_docs.io import write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    TextQuality,
    VisualAsset,
)


def test_generate_retrieval_case_skeleton_targets_pages_assets_and_triples():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map",
        vlm_summary="Shows corridor links.",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="corridor",
        predicate="connects",
        object="station",
    )

    cases = generate_retrieval_case_skeleton([chunk], [asset], [triple])

    assert len(cases) == 3
    assert cases[0].expected_pages == [1]
    assert cases[0].expected_chunk_ids == ["chunk-1"]
    assert cases[1].expected_asset_ids == ["asset-1"]
    assert cases[2].expected_triple_ids == ["triple-1"]
    assert cases[2].graph_expand is True


def test_generate_retrieval_case_skeleton_can_emit_todo_cases():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.PAGE_SUMMARY,
        text="[empty text layer] OCR/VLM processing required for page 2.",
    )

    cases = generate_retrieval_case_skeleton([chunk], [], [], include_todo=True)

    assert cases[0].query == "TODO: write query for page 2"
    assert cases[0].expected_pages == [2]


def test_generate_retrieval_cases_cli_writes_jsonl(tmp_path):
    package_dir = write_case_package(tmp_path)
    output = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "generate-retrieval-cases",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--page-limit",
            "1",
            "--asset-limit",
            "1",
            "--triple-limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["expected_pages"] == [1]
    assert rows[1]["expected_asset_ids"] == ["asset-1"]
    assert rows[2]["expected_triple_ids"] == ["triple-1"]


def write_case_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    doc = SourceDocument(
        doc_id="doc",
        title="Reference Document",
        local_path=tmp_path / "reference.pdf",
    )
    profile = PageProfile(
        doc_id="doc",
        page_no=1,
        width=100,
        height=100,
        char_count=100,
        line_count=4,
        text_block_count=1,
        image_block_count=1,
        embedded_image_count=0,
        drawing_count=0,
        text_quality=TextQuality.GOOD,
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="corridor",
        predicate="connects",
        object="station",
    )
    manifest = ProcessingManifest(
        doc=doc,
        profiles=[profile],
        chunks=[chunk],
        assets=[asset],
        triples=[triple],
    )
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", [profile])
    write_jsonl(package_dir / "chunks.jsonl", [chunk])
    write_jsonl(package_dir / "assets.jsonl", [asset])
    write_jsonl(package_dir / "triples.jsonl", [triple])
    return package_dir
