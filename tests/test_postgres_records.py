import json
from pathlib import Path

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
from chunking_docs.storage.postgres_records import (
    asset_row,
    chunk_asset_link_rows,
    chunk_row,
    document_row,
    embedding_artifact_rows,
    page_row,
    triple_row,
)
from chunking_docs.storage.postgres_store import (
    EXPECTED_POSTGRES_INDEXES,
    EXPECTED_POSTGRES_SCHEMA,
    PostgresSchemaReport,
    check_postgres_schema_snapshot,
    manifest_rows,
    postgres_schema_sql,
)


def test_postgres_row_transforms_are_json_ready():
    document = SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf"))
    page = PageProfile(
        doc_id="doc",
        page_no=1,
        width=1,
        height=2,
        char_count=3,
        line_count=4,
        text_block_count=5,
        image_block_count=6,
        embedded_image_count=7,
        drawing_count=8,
        text_quality=TextQuality.DEGRADED,
    )
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="hello",
        asset_ids=["asset"],
        source_refs=["asset:source-asset"],
    )
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        path=Path("/tmp/assets/page.png"),
        bbox=(0, 1, 2, 3),
    )
    triple = GraphTriple(
        triple_id="triple",
        doc_id="doc",
        chunk_id="chunk",
        subject="a",
        predicate="b",
        object="c",
    )

    assert document_row(document)["doc_id"] == "doc"
    assert page_row(page)["profile"]["text_quality"] == "degraded"
    assert chunk_row(chunk)["metadata"]["asset_ids"] == ["asset", "source-asset"]
    assert chunk_row(chunk)["metadata"]["source_refs"] == ["asset:source-asset"]
    assert asset_row(asset, base_dir=Path("/tmp"))["path"] == "assets/page.png"
    assert chunk_asset_link_rows([chunk], valid_asset_ids={"asset", "source-asset"}) == [
        {
            "chunk_id": "chunk",
            "asset_id": "asset",
            "doc_id": "doc",
            "source": "asset_ids",
            "metadata": {
                "sources": ["asset_ids"],
                "source_refs": [],
                "visual_asset_unlinked": False,
                "chunking_strategy": None,
            },
        },
        {
            "chunk_id": "chunk",
            "asset_id": "source-asset",
            "doc_id": "doc",
            "source": "source_refs",
            "metadata": {
                "sources": ["source_refs"],
                "source_refs": ["asset:source-asset"],
                "visual_asset_unlinked": False,
                "chunking_strategy": None,
            },
        },
    ]
    assert triple_row(triple)["object"] == "c"
    assert embedding_artifact_rows("doc") == []


def test_chunk_asset_link_rows_mark_standalone_visual_chunks():
    chunk = DocumentChunk(
        chunk_id="visual",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.MAP,
        text="visual evidence",
        asset_ids=["asset"],
        source_refs=["asset:asset"],
        metadata={
            "chunking_strategy": "visual_asset_text",
            "visual_asset_unlinked": True,
        },
    )

    rows = chunk_asset_link_rows([chunk], valid_asset_ids={"asset"})

    assert rows[0]["source"] == "asset_ids+source_refs+standalone_visual_chunk"
    assert rows[0]["metadata"]["visual_asset_unlinked"] is True
    assert rows[0]["metadata"]["chunking_strategy"] == "visual_asset_text"


def test_manifest_rows_counts():
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        profiles=[],
        chunks=[],
        assets=[],
        triples=[],
    )

    rows = manifest_rows(manifest)

    assert rows["document"]["doc_id"] == "doc"
    assert rows["pages"] == []
    assert rows["chunk_asset_links"] == []
    assert rows["embedding_artifacts"] == []


def test_manifest_rows_include_normalized_chunk_asset_links():
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        profiles=[],
        chunks=[
            DocumentChunk(
                chunk_id="chunk",
                doc_id="doc",
                page_start=1,
                page_end=1,
                kind=ChunkKind.TEXT,
                text="visual evidence",
                asset_ids=["asset"],
                source_refs=["asset:missing"],
            )
        ],
        assets=[
            VisualAsset(
                asset_id="asset",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.MAP,
            )
        ],
        triples=[],
    )

    rows = manifest_rows(manifest)

    assert rows["chunk_asset_links"] == [
        {
            "chunk_id": "chunk",
            "asset_id": "asset",
            "doc_id": "doc",
            "source": "asset_ids",
            "metadata": {
                "sources": ["asset_ids"],
                "source_refs": [],
                "visual_asset_unlinked": False,
                "chunking_strategy": None,
            },
        }
    ]


def test_manifest_rows_remap_asset_backed_triples_to_existing_chunks():
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        profiles=[],
        chunks=[
            DocumentChunk(
                chunk_id="chunk",
                doc_id="doc",
                page_start=1,
                page_end=1,
                kind=ChunkKind.TEXT,
                text="visual evidence",
                source_refs=["asset:asset"],
            )
        ],
        assets=[
            VisualAsset(
                asset_id="asset",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.MAP,
            )
        ],
        triples=[
            GraphTriple(
                triple_id="visual-triple",
                doc_id="doc",
                chunk_id="vlm-annotation",
                subject="diagram",
                predicate="depicts",
                object="process",
                qualifiers={"asset_id": "asset"},
            )
        ],
    )

    rows = manifest_rows(manifest)

    assert rows["triples"][0]["chunk_id"] == "chunk"
    assert rows["triples"][0]["qualifiers"]["original_chunk_id"] == "vlm-annotation"
    assert rows["triples"][0]["qualifiers"]["remapped_by_asset_provenance"] is True
    assert rows["triples"][0]["qualifiers"]["remapped_asset_id"] == "asset"


def test_manifest_rows_includes_embedding_artifact_provenance(tmp_path):
    (tmp_path / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": "custom_documents",
                "vectors": {
                    "text_dense": {
                        "file": "qdrant_text_records.jsonl",
                        "record_count": 2,
                        "dimension": 1024,
                        "distance": "Cosine",
                        "note": "text model",
                        "exists": True,
                        "bytes": 1234,
                        "sha256": "a" * 64,
                    }
                },
                "payload_indexes": [{"field": "doc_id", "schema": "keyword"}],
            }
        ),
        encoding="utf-8",
    )
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        profiles=[],
        chunks=[],
        assets=[],
        triples=[],
    )

    rows = manifest_rows(manifest, base_dir=tmp_path)

    assert rows["embedding_artifacts"] == [
        {
            "doc_id": "doc",
            "vector_name": "text_dense",
            "collection": "custom_documents",
            "file": "qdrant_text_records.jsonl",
            "record_count": 2,
            "dimension": 1024,
            "distance": "Cosine",
            "note": "text model",
            "bytes": 1234,
            "sha256": "a" * 64,
            "metadata": {
                "exists": True,
                "manifest_file": "embedding_manifest.json",
                "payload_indexes": [{"field": "doc_id", "schema": "keyword"}],
            },
        }
    ]


def expected_schema_rows():
    rows = []
    for table, columns in EXPECTED_POSTGRES_SCHEMA.items():
        for column, data_type in columns.items():
            if data_type == "double precision[]":
                rows.append((table, column, "ARRAY", "_float8"))
            else:
                rows.append((table, column, data_type, data_type))
    return rows


def expected_index_rows():
    return [(name, table) for name, table in EXPECTED_POSTGRES_INDEXES.items()]


def test_postgres_schema_sql_contains_expected_tables_and_indexes():
    schema = postgres_schema_sql()

    for table in EXPECTED_POSTGRES_SCHEMA:
        assert f"create table if not exists {table}" in schema
    for index_name in EXPECTED_POSTGRES_INDEXES:
        assert f"create index if not exists {index_name}" in schema


def test_check_postgres_schema_snapshot_passes_expected_schema():
    report = check_postgres_schema_snapshot(
        expected_schema_rows(),
        extension_names=["plpgsql", "vector"],
        index_rows=expected_index_rows(),
    )

    assert report.passed is True
    assert report.missing_tables == []
    assert report.missing_indexes == {}
    assert report.failed_checks == []


def test_check_postgres_schema_snapshot_flags_missing_table_column_type_and_extension():
    rows = [
        ("documents", "doc_id", "text", "text"),
        ("documents", "metadata", "text", "text"),
    ]

    report = check_postgres_schema_snapshot(rows, extension_names=["plpgsql"])

    assert report.passed is False
    assert "vector" not in report.present_extensions
    assert report.missing_extensions == ["vector"]
    assert "required_extensions" in report.failed_checks
    assert "pages" in report.missing_tables
    assert report.missing_columns["documents"] == ["created_at", "local_path", "source_url", "title"]
    assert report.type_mismatches["documents"]["metadata"] == {
        "expected": "jsonb",
        "actual": "text",
    }
    assert "required_indexes" in report.failed_checks
    assert report.missing_indexes == EXPECTED_POSTGRES_INDEXES


def test_check_postgres_schema_snapshot_flags_missing_indexes():
    report = check_postgres_schema_snapshot(
        expected_schema_rows(),
        extension_names=["vector"],
        index_rows=[],
    )

    assert report.passed is False
    assert "required_indexes" in report.failed_checks
    assert report.missing_indexes == EXPECTED_POSTGRES_INDEXES


def test_postgres_schema_cli_writes_sql(tmp_path):
    output = tmp_path / "postgres_schema.sql"

    from chunking_docs.cli import app
    from typer.testing import CliRunner

    result = CliRunner().invoke(app, ["postgres-schema", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8") == postgres_schema_sql()


def test_postgres_check_schema_cli_writes_report(tmp_path, monkeypatch):
    output = tmp_path / "postgres_schema_contract.json"

    class FakeStore:
        def __init__(self, dsn):
            self.dsn = dsn

        def check_schema(self, require_pgvector=True):
            return PostgresSchemaReport(
                passed=True,
                required_extensions=["vector"] if require_pgvector else [],
                present_extensions=["vector"],
                required_tables=["documents"],
                present_tables=["documents"],
            )

    monkeypatch.setattr("chunking_docs.storage.postgres_store.PostgresDocumentStore", FakeStore)

    from chunking_docs.cli import app
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        app,
        [
            "postgres-check-schema",
            "postgresql://user:pass@localhost/db",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
