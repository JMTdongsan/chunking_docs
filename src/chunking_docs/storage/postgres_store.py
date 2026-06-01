from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from chunking_docs.graph.repair import remap_triples_to_available_chunks
from chunking_docs.models import ProcessingManifest
from chunking_docs.storage.postgres_records import (
    asset_row,
    chunk_asset_link_rows,
    chunk_lexical_token_rows,
    chunk_row,
    document_row,
    embedding_artifact_rows,
    page_row,
    triple_row,
)

SCHEMA_PATH = Path(__file__).with_name("postgres_schema.sql")


EXPECTED_POSTGRES_SCHEMA = {
    "documents": {
        "doc_id": "text",
        "title": "text",
        "source_url": "text",
        "local_path": "text",
        "metadata": "jsonb",
        "created_at": "timestamp with time zone",
    },
    "pages": {
        "doc_id": "text",
        "page_no": "integer",
        "width": "double precision",
        "height": "double precision",
        "text_quality": "text",
        "profile": "jsonb",
    },
    "chunks": {
        "chunk_id": "text",
        "doc_id": "text",
        "page_start": "integer",
        "page_end": "integer",
        "kind": "text",
        "section": "jsonb",
        "text": "text",
        "metadata": "jsonb",
    },
    "chunk_lexical_tokens": {
        "chunk_id": "text",
        "doc_id": "text",
        "tokenizer": "jsonb",
        "text_char_count": "integer",
        "token_count": "integer",
        "tokens": "text[]",
        "metadata": "jsonb",
    },
    "assets": {
        "asset_id": "text",
        "doc_id": "text",
        "page_no": "integer",
        "kind": "text",
        "path": "text",
        "bbox": "double precision[]",
        "caption": "text",
        "ocr_text": "text",
        "vlm_summary": "text",
        "metadata": "jsonb",
    },
    "chunk_asset_links": {
        "chunk_id": "text",
        "asset_id": "text",
        "doc_id": "text",
        "source": "text",
        "metadata": "jsonb",
    },
    "triples": {
        "triple_id": "text",
        "doc_id": "text",
        "chunk_id": "text",
        "subject": "text",
        "predicate": "text",
        "object": "text",
        "qualifiers": "jsonb",
        "confidence": "double precision",
    },
    "embedding_artifacts": {
        "doc_id": "text",
        "vector_name": "text",
        "collection": "text",
        "file": "text",
        "record_count": "integer",
        "dimension": "integer",
        "distance": "text",
        "note": "text",
        "bytes": "bigint",
        "sha256": "text",
        "metadata": "jsonb",
    },
}

EXPECTED_POSTGRES_INDEXES = {
    "assets_doc_page_idx": "assets",
    "chunk_lexical_tokens_doc_idx": "chunk_lexical_tokens",
    "chunk_lexical_tokens_tokens_idx": "chunk_lexical_tokens",
    "chunk_asset_links_asset_idx": "chunk_asset_links",
    "chunk_asset_links_doc_idx": "chunk_asset_links",
    "chunks_doc_page_idx": "chunks",
    "chunks_text_bm25_idx": "chunks",
    "embedding_artifacts_collection_idx": "embedding_artifacts",
    "triples_spo_idx": "triples",
}


class PostgresSchemaCheck(BaseModel):
    name: str
    passed: bool
    severity: str = "error"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PostgresSchemaReport(BaseModel):
    passed: bool
    required_extensions: list[str] = Field(default_factory=list)
    present_extensions: list[str] = Field(default_factory=list)
    missing_extensions: list[str] = Field(default_factory=list)
    required_tables: list[str] = Field(default_factory=list)
    present_tables: list[str] = Field(default_factory=list)
    missing_tables: list[str] = Field(default_factory=list)
    missing_columns: dict[str, list[str]] = Field(default_factory=dict)
    type_mismatches: dict[str, dict[str, dict[str, str | None]]] = Field(default_factory=dict)
    required_indexes: dict[str, str] = Field(default_factory=dict)
    present_indexes: dict[str, str] = Field(default_factory=dict)
    missing_indexes: dict[str, str] = Field(default_factory=dict)
    checks: list[PostgresSchemaCheck] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)


class PostgresDocumentStore:
    """Optional PostgreSQL writer for document/chunk/asset/triple metadata."""

    def __init__(self, dsn: str):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install chunking-docs[postgres] to use PostgresDocumentStore") from exc

        self.psycopg = psycopg
        self.dsn = dsn

    def apply_schema(self) -> None:
        schema = postgres_schema_sql()
        with self.psycopg.connect(self.dsn) as connection:
            connection.execute(schema)

    def check_schema(self, require_pgvector: bool = True) -> PostgresSchemaReport:
        table_names = sorted(EXPECTED_POSTGRES_SCHEMA)
        placeholders = ", ".join(["%s"] * len(table_names))
        index_names = sorted(EXPECTED_POSTGRES_INDEXES)
        index_placeholders = ", ".join(["%s"] * len(index_names))
        columns_query = f"""
            select table_name, column_name, data_type, udt_name
            from information_schema.columns
            where table_schema = 'public'
              and table_name in ({placeholders})
        """
        indexes_query = f"""
            select indexname, tablename
            from pg_indexes
            where schemaname = 'public'
              and indexname in ({index_placeholders})
        """
        with self.psycopg.connect(self.dsn) as connection:
            extension_rows = connection.execute("select extname from pg_extension").fetchall()
            column_rows = connection.execute(columns_query, tuple(table_names)).fetchall()
            index_rows = connection.execute(indexes_query, tuple(index_names)).fetchall()
        return check_postgres_schema_snapshot(
            column_rows=column_rows,
            index_rows=index_rows,
            extension_names=[row_value(row, 0, "extname") for row in extension_rows],
            require_pgvector=require_pgvector,
        )

    def upsert_manifest(self, manifest: ProcessingManifest, base_dir: Path | None = None) -> dict[str, int]:
        rows = manifest_rows(manifest, base_dir=base_dir)
        with self.psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                upsert_documents(cursor, [rows["document"]])
                upsert_pages(cursor, rows["pages"])
                upsert_chunks(cursor, rows["chunks"])
                upsert_chunk_lexical_tokens(cursor, rows["chunk_lexical_tokens"])
                upsert_assets(cursor, rows["assets"])
                upsert_chunk_asset_links(cursor, rows["chunk_asset_links"])
                upsert_triples(cursor, rows["triples"])
                upsert_embedding_artifacts(cursor, rows["embedding_artifacts"])
        return {
            "documents": 1,
            "pages": len(rows["pages"]),
            "chunks": len(rows["chunks"]),
            "chunk_lexical_tokens": len(rows["chunk_lexical_tokens"]),
            "assets": len(rows["assets"]),
            "chunk_asset_links": len(rows["chunk_asset_links"]),
            "triples": len(rows["triples"]),
            "embedding_artifacts": len(rows["embedding_artifacts"]),
        }


def manifest_rows(manifest: ProcessingManifest, base_dir: Path | None = None) -> dict[str, Any]:
    triples = remap_triples_to_available_chunks(manifest.triples, manifest.chunks)
    return {
        "document": document_row(manifest.doc),
        "pages": [page_row(profile) for profile in manifest.profiles],
        "chunks": [chunk_row(chunk) for chunk in manifest.chunks],
        "chunk_lexical_tokens": chunk_lexical_token_rows(
            manifest.doc.doc_id,
            manifest.chunks,
            package_dir=base_dir,
        ),
        "assets": [asset_row(asset, base_dir=base_dir) for asset in manifest.assets],
        "chunk_asset_links": chunk_asset_link_rows(
            manifest.chunks,
            valid_asset_ids={asset.asset_id for asset in manifest.assets},
        ),
        "triples": [triple_row(triple) for triple in triples],
        "embedding_artifacts": embedding_artifact_rows(manifest.doc.doc_id, package_dir=base_dir),
    }


def postgres_schema_sql() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def check_postgres_schema_snapshot(
    column_rows: Iterable,
    extension_names: Iterable[str],
    index_rows: Iterable | None = None,
    required_schema: dict[str, dict[str, str]] | None = None,
    required_indexes: dict[str, str] | None = None,
    require_pgvector: bool = True,
    require_indexes: bool = True,
) -> PostgresSchemaReport:
    if required_schema is None:
        required_schema = EXPECTED_POSTGRES_SCHEMA
    if required_indexes is None:
        required_indexes = EXPECTED_POSTGRES_INDEXES
    present_extensions = sorted(str(name) for name in extension_names if name)
    required_extensions = ["vector"] if require_pgvector else []
    actual_schema: dict[str, dict[str, str]] = {}
    for row in column_rows:
        table = str(row_value(row, 0, "table_name"))
        column = str(row_value(row, 1, "column_name"))
        data_type = str(row_value(row, 2, "data_type"))
        udt_name = row_value(row, 3, "udt_name")
        actual_schema.setdefault(table, {})[column] = normalize_postgres_type(data_type, udt_name)

    required_tables = sorted(required_schema)
    present_tables = sorted(set(actual_schema))
    missing_tables = sorted(set(required_tables) - set(present_tables))
    missing_columns = {
        table: sorted(set(columns) - set(actual_schema.get(table, {})))
        for table, columns in sorted(required_schema.items())
        if table not in missing_tables and set(columns) - set(actual_schema.get(table, {}))
    }
    type_mismatches: dict[str, dict[str, dict[str, str | None]]] = {}
    for table, columns in sorted(required_schema.items()):
        if table in missing_tables:
            continue
        for column, expected_type in sorted(columns.items()):
            actual_type = actual_schema.get(table, {}).get(column)
            if actual_type is not None and actual_type != expected_type:
                type_mismatches.setdefault(table, {})[column] = {
                    "expected": expected_type,
                    "actual": actual_type,
                }

    required_index_map = dict(sorted(required_indexes.items())) if require_indexes else {}
    present_index_map: dict[str, str] = {}
    for row in index_rows or []:
        index_name = row_value(row, 0, "indexname")
        table_name = row_value(row, 1, "tablename")
        if index_name and table_name:
            present_index_map[str(index_name)] = str(table_name)
    missing_index_map = {
        name: table for name, table in required_index_map.items() if name not in present_index_map
    }

    missing_extensions = sorted(set(required_extensions) - set(present_extensions))
    checks = [
        PostgresSchemaCheck(
            name="required_extensions",
            passed=not missing_extensions,
            message="Required PostgreSQL extensions are installed.",
            metadata={"missing": missing_extensions},
        ),
        PostgresSchemaCheck(
            name="required_tables",
            passed=not missing_tables,
            message="Required PostgreSQL tables exist.",
            metadata={"missing": missing_tables},
        ),
        PostgresSchemaCheck(
            name="required_columns",
            passed=not missing_columns,
            message="Required PostgreSQL columns exist.",
            metadata={"missing": missing_columns},
        ),
        PostgresSchemaCheck(
            name="column_types",
            passed=not type_mismatches,
            message="PostgreSQL column types match the package row contract.",
            metadata={"mismatches": type_mismatches},
        ),
        PostgresSchemaCheck(
            name="required_indexes",
            passed=not missing_index_map,
            message="Required PostgreSQL indexes exist.",
            metadata={"missing": missing_index_map},
        ),
    ]
    failed_checks = [check.name for check in checks if not check.passed and check.severity == "error"]
    return PostgresSchemaReport(
        passed=not failed_checks,
        required_extensions=required_extensions,
        present_extensions=present_extensions,
        missing_extensions=missing_extensions,
        required_tables=required_tables,
        present_tables=present_tables,
        missing_tables=missing_tables,
        missing_columns=missing_columns,
        type_mismatches=type_mismatches,
        required_indexes=required_index_map,
        present_indexes=dict(sorted(present_index_map.items())),
        missing_indexes=missing_index_map,
        checks=checks,
        failed_checks=failed_checks,
    )


def normalize_postgres_type(data_type: str, udt_name: Any = None) -> str:
    if data_type == "ARRAY" and udt_name == "_float8":
        return "double precision[]"
    if data_type == "ARRAY" and udt_name == "_text":
        return "text[]"
    if data_type == "USER-DEFINED" and udt_name:
        return str(udt_name)
    return data_type


def row_value(row, index: int, key: str):
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[index]
    except (IndexError, TypeError):
        return getattr(row, key, None)


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def upsert_documents(cursor, rows: Iterable[dict[str, Any]]) -> None:
    cursor.executemany(
        """
        insert into documents (doc_id, title, source_url, local_path, metadata)
        values (%(doc_id)s, %(title)s, %(source_url)s, %(local_path)s, %(metadata)s::jsonb)
        on conflict (doc_id) do update set
            title = excluded.title,
            source_url = excluded.source_url,
            local_path = excluded.local_path,
            metadata = excluded.metadata
        """,
        [with_json(row, "metadata") for row in rows],
    )


def upsert_pages(cursor, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cursor.executemany(
        """
        insert into pages (doc_id, page_no, width, height, text_quality, profile)
        values (%(doc_id)s, %(page_no)s, %(width)s, %(height)s, %(text_quality)s, %(profile)s::jsonb)
        on conflict (doc_id, page_no) do update set
            width = excluded.width,
            height = excluded.height,
            text_quality = excluded.text_quality,
            profile = excluded.profile
        """,
        [with_json(row, "profile") for row in rows],
    )


def upsert_chunks(cursor, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cursor.executemany(
        """
        insert into chunks (chunk_id, doc_id, page_start, page_end, kind, section, text, metadata)
        values (
            %(chunk_id)s, %(doc_id)s, %(page_start)s, %(page_end)s, %(kind)s,
            %(section)s::jsonb, %(text)s, %(metadata)s::jsonb
        )
        on conflict (chunk_id) do update set
            page_start = excluded.page_start,
            page_end = excluded.page_end,
            kind = excluded.kind,
            section = excluded.section,
            text = excluded.text,
            metadata = excluded.metadata
        """,
        [with_json(with_json(row, "section"), "metadata") for row in rows],
    )


def upsert_chunk_lexical_tokens(cursor, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cursor.executemany(
        """
        insert into chunk_lexical_tokens (
            chunk_id, doc_id, tokenizer, text_char_count, token_count, tokens, metadata
        )
        values (
            %(chunk_id)s, %(doc_id)s, %(tokenizer)s::jsonb, %(text_char_count)s,
            %(token_count)s, %(tokens)s, %(metadata)s::jsonb
        )
        on conflict (chunk_id) do update set
            doc_id = excluded.doc_id,
            tokenizer = excluded.tokenizer,
            text_char_count = excluded.text_char_count,
            token_count = excluded.token_count,
            tokens = excluded.tokens,
            metadata = excluded.metadata
        """,
        [with_json(with_json(row, "tokenizer"), "metadata") for row in rows],
    )


def upsert_assets(cursor, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cursor.executemany(
        """
        insert into assets (asset_id, doc_id, page_no, kind, path, bbox, caption, ocr_text, vlm_summary, metadata)
        values (
            %(asset_id)s, %(doc_id)s, %(page_no)s, %(kind)s, %(path)s, %(bbox)s,
            %(caption)s, %(ocr_text)s, %(vlm_summary)s, %(metadata)s::jsonb
        )
        on conflict (asset_id) do update set
            page_no = excluded.page_no,
            kind = excluded.kind,
            path = excluded.path,
            bbox = excluded.bbox,
            caption = excluded.caption,
            ocr_text = excluded.ocr_text,
            vlm_summary = excluded.vlm_summary,
            metadata = excluded.metadata
        """,
        [with_json(row, "metadata") for row in rows],
    )


def upsert_chunk_asset_links(cursor, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cursor.executemany(
        """
        insert into chunk_asset_links (chunk_id, asset_id, doc_id, source, metadata)
        values (
            %(chunk_id)s, %(asset_id)s, %(doc_id)s, %(source)s, %(metadata)s::jsonb
        )
        on conflict (chunk_id, asset_id) do update set
            source = excluded.source,
            metadata = excluded.metadata
        """,
        [with_json(row, "metadata") for row in rows],
    )


def upsert_triples(cursor, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cursor.executemany(
        """
        insert into triples (triple_id, doc_id, chunk_id, subject, predicate, object, qualifiers, confidence)
        values (
            %(triple_id)s, %(doc_id)s, %(chunk_id)s, %(subject)s, %(predicate)s,
            %(object)s, %(qualifiers)s::jsonb, %(confidence)s
        )
        on conflict (triple_id) do update set
            subject = excluded.subject,
            predicate = excluded.predicate,
            object = excluded.object,
            qualifiers = excluded.qualifiers,
            confidence = excluded.confidence
        """,
        [with_json(row, "qualifiers") for row in rows],
    )


def upsert_embedding_artifacts(cursor, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cursor.executemany(
        """
        insert into embedding_artifacts (
            doc_id, vector_name, collection, file, record_count, dimension,
            distance, note, bytes, sha256, metadata
        )
        values (
            %(doc_id)s, %(vector_name)s, %(collection)s, %(file)s, %(record_count)s,
            %(dimension)s, %(distance)s, %(note)s, %(bytes)s, %(sha256)s,
            %(metadata)s::jsonb
        )
        on conflict (doc_id, vector_name) do update set
            collection = excluded.collection,
            file = excluded.file,
            record_count = excluded.record_count,
            dimension = excluded.dimension,
            distance = excluded.distance,
            note = excluded.note,
            bytes = excluded.bytes,
            sha256 = excluded.sha256,
            metadata = excluded.metadata
        """,
        [with_json(row, "metadata") for row in rows],
    )


def with_json(row: dict[str, Any], key: str) -> dict[str, Any]:
    copied = dict(row)
    copied[key] = json_dump(copied[key])
    return copied
