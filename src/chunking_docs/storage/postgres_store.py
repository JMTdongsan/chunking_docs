from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from chunking_docs.models import ProcessingManifest
from chunking_docs.storage.postgres_records import (
    asset_row,
    chunk_row,
    document_row,
    embedding_artifact_rows,
    page_row,
    triple_row,
)

SCHEMA_PATH = Path(__file__).with_name("postgres_schema.sql")


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
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.psycopg.connect(self.dsn) as connection:
            connection.execute(schema)

    def upsert_manifest(self, manifest: ProcessingManifest, base_dir: Path | None = None) -> dict[str, int]:
        rows = manifest_rows(manifest, base_dir=base_dir)
        with self.psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                upsert_documents(cursor, [rows["document"]])
                upsert_pages(cursor, rows["pages"])
                upsert_chunks(cursor, rows["chunks"])
                upsert_assets(cursor, rows["assets"])
                upsert_triples(cursor, rows["triples"])
                upsert_embedding_artifacts(cursor, rows["embedding_artifacts"])
        return {
            "documents": 1,
            "pages": len(rows["pages"]),
            "chunks": len(rows["chunks"]),
            "assets": len(rows["assets"]),
            "triples": len(rows["triples"]),
            "embedding_artifacts": len(rows["embedding_artifacts"]),
        }


def manifest_rows(manifest: ProcessingManifest, base_dir: Path | None = None) -> dict[str, Any]:
    return {
        "document": document_row(manifest.doc),
        "pages": [page_row(profile) for profile in manifest.profiles],
        "chunks": [chunk_row(chunk) for chunk in manifest.chunks],
        "assets": [asset_row(asset, base_dir=base_dir) for asset in manifest.assets],
        "triples": [triple_row(triple) for triple in manifest.triples],
        "embedding_artifacts": embedding_artifact_rows(manifest.doc.doc_id, package_dir=base_dir),
    }


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
