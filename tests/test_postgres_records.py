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
    chunk_lexical_token_rows,
    chunk_row,
    document_row,
    embedding_artifact_rows,
    embedding_record_rows,
    embedding_vector_summary_rows,
    page_row,
    triple_row,
    visual_object_rows,
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


def test_visual_object_rows_normalize_vlm_object_metadata():
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=2,
        kind=AssetKind.FIGURE,
        caption="system diagram",
        vlm_summary="A pump is shown near a gauge.",
        metadata={
            "page_type": "diagram",
            "objects": [
                {
                    "label": "pump",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "attributes": ["blue"],
                    "description": "main pump",
                    "confidence": "92%",
                }
            ],
        },
    )

    rows = visual_object_rows([asset])

    assert rows == [
        {
            "object_id": "asset:object:0",
            "doc_id": "doc",
            "asset_id": "asset",
            "page_no": 2,
            "kind": "figure",
            "object_index": 0,
            "label": "pump",
            "source_key": "objects",
            "visual_feature_type": None,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "bbox_region": "upper left",
            "attributes": ["blue", "main pump"],
            "description": "main pump",
            "location": None,
            "confidence": 0.92,
            "text": (
                "Object: pump\nAttributes: blue; main pump\nBbox region: upper left\n"
                "Source field: objects\nPage type: diagram\nCaption: system diagram\n"
                "VLM summary: A pump is shown near a gauge."
            ),
            "metadata": {
                "caption": "system diagram",
                "vlm_summary": "A pump is shown near a gauge.",
                "page_type": "diagram",
                "objects": [
                    {
                        "label": "pump",
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                        "attributes": ["blue"],
                        "description": "main pump",
                        "confidence": "92%",
                    }
                ],
                "record_kind": "visual_object",
            },
        }
    ]


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
    assert rows["chunk_lexical_tokens"] == []
    assert rows["visual_objects"] == []
    assert rows["chunk_asset_links"] == []
    assert rows["embedding_artifacts"] == []
    assert rows["embedding_records"] == []
    assert rows["embedding_vector_summaries"] == []


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


def test_manifest_rows_include_visual_object_rows():
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        profiles=[],
        chunks=[],
        assets=[
            VisualAsset(
                asset_id="asset",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.MAP,
                metadata={"detected_objects": [{"label": "legend", "bbox_region": "lower right"}]},
            )
        ],
        triples=[],
    )

    rows = manifest_rows(manifest)

    assert rows["visual_objects"][0]["object_id"] == "asset:object:0"
    assert rows["visual_objects"][0]["label"] == "legend"
    assert rows["visual_objects"][0]["bbox_region"] == "lower right"


def test_manifest_rows_include_visual_element_feature_rows():
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        profiles=[],
        chunks=[],
        assets=[
            VisualAsset(
                asset_id="asset",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.MAP,
                metadata={"visual_elements": ["station access corridor"]},
            )
        ],
        triples=[],
    )

    rows = manifest_rows(manifest)

    assert rows["visual_objects"][0]["label"] == "station access corridor"
    assert rows["visual_objects"][0]["source_key"] == "visual_elements"
    assert rows["visual_objects"][0]["visual_feature_type"] == "visual_element"


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
                "payload_indexes": [
                    {"field": "doc_id", "schema": "keyword"},
                    {"field": "page_no", "schema": "integer"},
                    {"field": "requires_vlm", "schema": "bool"},
                ],
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
                "payload_indexes": [
                    {"field": "doc_id", "schema": "keyword"},
                    {"field": "page_no", "schema": "integer"},
                    {"field": "requires_vlm", "schema": "bool"},
                ],
                "payload_index_fields": ["doc_id", "page_no", "requires_vlm"],
                "payload_index_schemas": {
                    "doc_id": "keyword",
                    "page_no": "integer",
                    "requires_vlm": "bool",
                },
            },
        }
    ]


def test_manifest_rows_includes_embedding_record_rows(tmp_path):
    (tmp_path / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": "custom_documents",
                "vectors": {
                    "text_dense": {
                        "file": "qdrant_text_records.jsonl",
                        "record_count": 1,
                        "dimension": 2,
                        "distance": "Cosine",
                        "exists": True,
                    },
                    "image_dense": {
                        "file": "qdrant_image_records.jsonl",
                        "record_count": 1,
                        "dimension": 2,
                        "distance": "Cosine",
                        "exists": True,
                    },
                    "object_dense": {
                        "file": "qdrant_object_records.jsonl",
                        "record_count": 1,
                        "dimension": 2,
                        "distance": "Cosine",
                        "exists": True,
                    },
                    "triple_dense": {
                        "file": "qdrant_triple_records.jsonl",
                        "record_count": 1,
                        "dimension": 2,
                        "distance": "Cosine",
                        "exists": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    records = {
        "qdrant_text_records.jsonl": {
            "point_id": "text-point",
            "chunk_id": "chunk",
            "doc_id": "doc",
            "vector_name": "text_dense",
            "vector": [0.1, 0.2],
            "payload": {"chunk_id": "chunk", "doc_id": "doc"},
        },
        "qdrant_image_records.jsonl": {
            "point_id": "image-point",
            "chunk_id": "asset",
            "doc_id": "doc",
            "vector_name": "image_dense",
            "vector": [0.3, 0.4],
            "payload": {"asset_id": "asset", "doc_id": "doc"},
        },
        "qdrant_object_records.jsonl": {
            "point_id": "object-point",
            "chunk_id": "asset",
            "doc_id": "doc",
            "vector_name": "object_dense",
            "vector": [0.5, 0.6],
            "payload": {"object_id": "asset:object:0", "asset_id": "asset", "doc_id": "doc"},
        },
        "qdrant_triple_records.jsonl": {
            "point_id": "triple-point",
            "chunk_id": "chunk",
            "doc_id": "doc",
            "vector_name": "triple_dense",
            "vector": [0.7, 0.8],
            "payload": {
                "record_kind": "graph_triple",
                "triple_id": "triple",
                "chunk_id": "chunk",
                "doc_id": "doc",
            },
        },
    }
    for filename, record in records.items():
        (tmp_path / filename).write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        profiles=[],
        chunks=[],
        assets=[],
        triples=[],
    )

    rows = manifest_rows(manifest, base_dir=tmp_path)
    direct_rows = embedding_record_rows("doc", tmp_path)

    assert rows["embedding_records"] == direct_rows
    assert {row["target_kind"] for row in rows["embedding_records"]} == {
        "asset",
        "chunk",
        "object",
        "triple",
    }
    by_point = {row["point_id"]: row for row in rows["embedding_records"]}
    assert by_point["text-point"]["target_id"] == "chunk"
    assert by_point["image-point"]["target_id"] == "asset"
    assert by_point["object-point"]["target_id"] == "asset:object:0"
    assert by_point["triple-point"]["target_id"] == "triple"
    assert by_point["text-point"]["dimension"] == 2
    assert by_point["text-point"]["metadata"] == {
        "collection": "custom_documents",
        "manifest_file": "embedding_manifest.json",
        "record_file": "qdrant_text_records.jsonl",
        "manifest_vector_name": "text_dense",
        "manifest_dimension": 2,
        "manifest_record_count": 1,
    }
    assert rows["embedding_vector_summaries"] == [
        {
            "doc_id": "doc",
            "vector_name": "image_dense",
            "target_kind": "asset",
            "record_count": 1,
            "target_count": 1,
            "dimension": 2,
            "dimension_min": 2,
            "dimension_max": 2,
            "metadata": {
                "dimension_consistent": True,
                "record_files": ["qdrant_image_records.jsonl"],
                "target_id_sample": ["asset"],
            },
        },
        {
            "doc_id": "doc",
            "vector_name": "object_dense",
            "target_kind": "object",
            "record_count": 1,
            "target_count": 1,
            "dimension": 2,
            "dimension_min": 2,
            "dimension_max": 2,
            "metadata": {
                "dimension_consistent": True,
                "record_files": ["qdrant_object_records.jsonl"],
                "target_id_sample": ["asset:object:0"],
            },
        },
        {
            "doc_id": "doc",
            "vector_name": "text_dense",
            "target_kind": "chunk",
            "record_count": 1,
            "target_count": 1,
            "dimension": 2,
            "dimension_min": 2,
            "dimension_max": 2,
            "metadata": {
                "dimension_consistent": True,
                "record_files": ["qdrant_text_records.jsonl"],
                "target_id_sample": ["chunk"],
            },
        },
        {
            "doc_id": "doc",
            "vector_name": "triple_dense",
            "target_kind": "triple",
            "record_count": 1,
            "target_count": 1,
            "dimension": 2,
            "dimension_min": 2,
            "dimension_max": 2,
            "metadata": {
                "dimension_consistent": True,
                "record_files": ["qdrant_triple_records.jsonl"],
                "target_id_sample": ["triple"],
            },
        },
    ]


def test_embedding_vector_summary_rows_flags_mixed_dimensions():
    rows = [
        {
            "doc_id": "doc",
            "vector_name": "text_dense",
            "target_kind": "chunk",
            "target_id": "chunk-a",
            "dimension": 2,
            "metadata": {"record_file": "qdrant_text_records.jsonl"},
        },
        {
            "doc_id": "doc",
            "vector_name": "text_dense",
            "target_kind": "chunk",
            "target_id": "chunk-b",
            "dimension": 3,
            "metadata": {"record_file": "qdrant_text_records.jsonl"},
        },
        {
            "doc_id": "doc",
            "vector_name": "text_dense",
            "target_kind": "chunk",
            "target_id": "chunk-b",
            "dimension": 3,
            "metadata": {"record_file": "qdrant_text_records.2.jsonl"},
        },
    ]

    assert embedding_vector_summary_rows(rows) == [
        {
            "doc_id": "doc",
            "vector_name": "text_dense",
            "target_kind": "chunk",
            "record_count": 3,
            "target_count": 2,
            "dimension": None,
            "dimension_min": 2,
            "dimension_max": 3,
            "metadata": {
                "dimension_consistent": False,
                "record_files": ["qdrant_text_records.2.jsonl", "qdrant_text_records.jsonl"],
                "target_id_sample": ["chunk-a", "chunk-b"],
            },
        }
    ]


def test_embedding_record_rows_ignores_paths_outside_package(tmp_path):
    outside_path = tmp_path.parent / "qdrant_outside_records.jsonl"
    outside_path.write_text(
        json.dumps(
            {
                "point_id": "outside",
                "chunk_id": "chunk",
                "doc_id": "doc",
                "vector_name": "text_dense",
                "vector": [0.1, 0.2],
                "payload": {"chunk_id": "chunk", "doc_id": "doc"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": "custom_documents",
                "vectors": {
                    "text_dense": {
                        "file": "../qdrant_outside_records.jsonl",
                        "record_count": 1,
                        "dimension": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert embedding_record_rows("doc", tmp_path) == []


def test_manifest_rows_includes_bm25_token_rows(tmp_path):
    (tmp_path / "bm25_tokens.json").write_text(
        json.dumps(
            {
                "tokenizer": {
                    "strategy": "mixed",
                    "ngram_min": 2,
                    "ngram_max": 4,
                    "cjk_only": True,
                },
                "chunks": [
                    {
                        "chunk_id": "chunk",
                        "text_char_count": 12,
                        "tokens": ["alpha", "beta", "", " beta "],
                    },
                    {
                        "chunk_id": "stale",
                        "text_char_count": 5,
                        "tokens": ["ignored"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
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
                text="alpha beta",
            )
        ],
        assets=[],
        triples=[],
    )

    rows = manifest_rows(manifest, base_dir=tmp_path)

    assert rows["chunk_lexical_tokens"] == [
        {
            "chunk_id": "chunk",
            "doc_id": "doc",
            "tokenizer": {
                "strategy": "mixed",
                "ngram_min": 2,
                "ngram_max": 4,
                "cjk_only": True,
            },
            "text_char_count": 12,
            "token_count": 3,
            "tokens": ["alpha", "beta", "beta"],
            "metadata": {
                "manifest_file": "bm25_tokens.json",
                "manifest_chunk_count": 2,
            },
        }
    ]
    assert chunk_lexical_token_rows("doc", manifest.chunks, package_dir=tmp_path)[0][
        "token_count"
    ] == 3


def expected_schema_rows():
    rows = []
    for table, columns in EXPECTED_POSTGRES_SCHEMA.items():
        for column, data_type in columns.items():
            if data_type == "double precision[]":
                rows.append((table, column, "ARRAY", "_float8"))
            elif data_type == "text[]":
                rows.append((table, column, "ARRAY", "_text"))
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
