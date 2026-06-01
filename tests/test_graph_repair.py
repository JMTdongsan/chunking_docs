from typer.testing import CliRunner

from chunking_docs.evaluation.audit import audit_package
from chunking_docs.graph.repair import remap_triples_to_available_chunks, repair_visual_derived_triples
from chunking_docs.io import write_jsonl
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset


def test_remap_triples_to_first_available_child_chunk():
    child = DocumentChunk(
        chunk_id="child-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="child",
        metadata={"parent_chunk_id": "parent-1"},
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="parent-1",
        subject="north district",
        predicate="uses_axis",
        object="river corridor",
    )

    remapped = remap_triples_to_available_chunks([triple], [child])

    assert remapped[0].chunk_id == "child-1"
    assert remapped[0].qualifiers["original_chunk_id"] == "parent-1"
    assert remapped[0].triple_id != "triple-1"


def test_remap_triples_to_asset_linked_chunk():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual context",
        source_refs=["asset:asset-1"],
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="vlm-annotation",
        subject="diagram",
        predicate="depicts",
        object="process",
        qualifiers={"asset_id": "asset-1"},
    )

    remapped = remap_triples_to_available_chunks([triple], [chunk])

    assert remapped[0].chunk_id == "chunk-1"
    assert remapped[0].qualifiers["original_chunk_id"] == "vlm-annotation"
    assert remapped[0].qualifiers["remapped_by_asset_provenance"] is True
    assert remapped[0].qualifiers["remapped_asset_id"] == "asset-1"


def test_repair_visual_derived_triples_adds_missing_asset_provenance():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="map context",
        source_refs=["asset:asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="north map",
        metadata={
            "entities": ["station"],
            "visual_elements": ["blue corridor"],
            "objects": [{"label": "legend", "bbox_region": "lower right"}],
            "visual_job_id": "job-1",
        },
    )

    repaired, report = repair_visual_derived_triples([asset], [chunk], [])

    assert report.added_triples == 3
    assert report.updated_triples == 0
    assert report.repaired_asset_count == 1
    assert {triple.predicate for triple in repaired} == {
        "mentions_entity",
        "contains_visual_element",
        "contains_object",
    }
    assert {triple.qualifiers["asset_id"] for triple in repaired} == {"asset-1"}
    assert all(triple.qualifiers["visual_derived_triple_repair"] is True for triple in repaired)
    assert all(triple.qualifiers["visual_job_id"] == "job-1" for triple in repaired)

    audit = audit_package([], [chunk], [asset], repaired, require_visual_derived_triples=True)
    assert "missing_visual_derived_triples" not in {issue.code for issue in audit.issues}


def test_repair_visual_derived_triples_merges_asset_id_into_existing_triple():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="map context",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="north map",
        metadata={"entities": ["station"]},
    )
    existing = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="north map",
        predicate="mentions_entity",
        object="station",
        qualifiers={"source": "visual_annotation"},
    )

    repaired, report = repair_visual_derived_triples([asset], [chunk], [existing])

    assert len(repaired) == 1
    assert report.added_triples == 0
    assert report.updated_triples == 1
    assert repaired[0].qualifiers["asset_id"] == "asset-1"

    audit = audit_package([], [chunk], [asset], repaired, require_visual_derived_triples=True)
    assert "missing_visual_derived_triples" not in {issue.code for issue in audit.issues}


def test_repair_visual_triples_cli_writes_repaired_output(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="map context",
        source_refs=["asset:asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="north map",
        metadata={"entities": ["station"]},
    )
    write_jsonl(package_dir / "chunks.jsonl", [chunk])
    write_jsonl(package_dir / "assets.jsonl", [asset])
    write_jsonl(package_dir / "triples.jsonl", [])

    from chunking_docs.cli import app

    result = CliRunner().invoke(app, ["repair-visual-triples", "--package-dir", str(package_dir)])

    assert result.exit_code == 0, result.output
    assert (package_dir / "triples.visual_repaired.jsonl").exists()
    assert '"added_triples": 1' in result.output
