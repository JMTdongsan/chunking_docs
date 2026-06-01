from __future__ import annotations

import warnings
from typing import Any
from typing import Iterable

from pydantic import BaseModel, Field

from chunking_docs.storage.qdrant_config import default_payload_schema
from chunking_docs.storage.qdrant_config import normalize_payload_schema
from chunking_docs.storage.qdrant_config import qdrant_payload_index_schemas
from chunking_docs.storage.records import EmbeddingRecord, UpsertResult, VectorSearchHit


class QdrantCollectionContractCheck(BaseModel):
    name: str
    passed: bool
    severity: str = "error"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QdrantCollectionContractReport(BaseModel):
    collection: str
    exists: bool
    passed: bool
    expected_vectors: dict[str, int] = Field(default_factory=dict)
    actual_vectors: dict[str, int] = Field(default_factory=dict)
    missing_vectors: list[str] = Field(default_factory=list)
    extra_vectors: list[str] = Field(default_factory=list)
    mismatched_vectors: dict[str, dict[str, int | None]] = Field(default_factory=dict)
    expected_payload_indexes: list[str] = Field(default_factory=list)
    actual_payload_indexes: list[str] = Field(default_factory=list)
    missing_payload_indexes: list[str] = Field(default_factory=list)
    expected_payload_index_schemas: dict[str, str] = Field(default_factory=dict)
    actual_payload_index_schemas: dict[str, str] = Field(default_factory=dict)
    mismatched_payload_indexes: dict[str, dict[str, str | None]] = Field(default_factory=dict)
    checks: list[QdrantCollectionContractCheck] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)


class QdrantChunkStore:
    """Thin Qdrant adapter kept optional so the core library runs without Qdrant."""

    def __init__(
        self,
        collection_name: str,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        path: str | None = None,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance,
                FieldCondition,
                Filter,
                MatchAny,
                MatchValue,
                PayloadSchemaType,
                PointStruct,
                Range,
                VectorParams,
            )
        except ImportError as exc:
            raise RuntimeError("Install chunking-docs[qdrant] to use QdrantChunkStore") from exc

        if path:
            self.client = QdrantClient(path=path)
        elif location:
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(url=url or "http://localhost:6333", api_key=api_key)
        self.collection_name = collection_name
        self._point_struct = PointStruct
        self._distance = Distance
        self._vector_params = VectorParams
        self._filter = Filter
        self._field_condition = FieldCondition
        self._match_any = MatchAny
        self._match_value = MatchValue
        self._range = Range
        self._payload_schema_type = PayloadSchemaType

    def ensure_collection(
        self,
        named_vectors: dict[str, int],
        payload_indexes: list[str | dict[str, str]] | None = None,
    ) -> None:
        collections = self.client.get_collections().collections
        exists = any(collection.name == self.collection_name for collection in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    name: self._vector_params(size=size, distance=self._distance.COSINE)
                    for name, size in named_vectors.items()
                },
            )
        self.ensure_payload_indexes(payload_indexes or [])

    def check_collection_contract(
        self,
        named_vectors: dict[str, int],
        payload_indexes: list[str | dict[str, str]] | None = None,
        allow_missing: bool = False,
    ) -> QdrantCollectionContractReport:
        exists = self.collection_exists()
        expected_payload_indexes = sorted(
            field_name
            for field_name, _ in [self._normalize_payload_index(index) for index in payload_indexes or []]
            if field_name
        )
        expected_payload_index_schemas = qdrant_payload_index_schemas(payload_indexes or [])
        if not exists:
            checks = [
                QdrantCollectionContractCheck(
                    name="collection_exists",
                    passed=allow_missing,
                    message="Qdrant collection exists or missing collection is explicitly allowed.",
                    metadata={"allow_missing": allow_missing},
                )
            ]
            failed_checks = [check.name for check in checks if not check.passed]
            return QdrantCollectionContractReport(
                collection=self.collection_name,
                exists=False,
                passed=not failed_checks,
                expected_vectors=dict(sorted(named_vectors.items())),
                expected_payload_indexes=expected_payload_indexes,
                expected_payload_index_schemas=dict(sorted(expected_payload_index_schemas.items())),
                checks=checks,
                failed_checks=failed_checks,
            )

        info = self.client.get_collection(collection_name=self.collection_name)
        actual_vectors = collection_vector_sizes(info)
        actual_payload_index_schemas = collection_payload_index_schemas(info)
        actual_payload_indexes = sorted(actual_payload_index_schemas)
        missing_vectors = sorted(set(named_vectors) - set(actual_vectors))
        extra_vectors = sorted(set(actual_vectors) - set(named_vectors))
        mismatched_vectors = {
            name: {"expected": expected_size, "actual": actual_vectors.get(name)}
            for name, expected_size in sorted(named_vectors.items())
            if name in actual_vectors and actual_vectors.get(name) != expected_size
        }
        missing_payload_indexes = sorted(set(expected_payload_indexes) - set(actual_payload_indexes))
        mismatched_payload_indexes = {
            field_name: {
                "expected": expected_schema,
                "actual": actual_payload_index_schemas.get(field_name),
            }
            for field_name, expected_schema in sorted(expected_payload_index_schemas.items())
            if field_name in actual_payload_index_schemas
            and actual_payload_index_schemas.get(field_name) is not None
            and actual_payload_index_schemas.get(field_name) != expected_schema
        }
        checks = [
            QdrantCollectionContractCheck(
                name="missing_vectors",
                passed=not missing_vectors,
                message="All expected named vectors exist in the Qdrant collection.",
                metadata={"vectors": missing_vectors},
            ),
            QdrantCollectionContractCheck(
                name="vector_size_mismatch",
                passed=not mismatched_vectors,
                message="Qdrant named vector sizes match the package contract.",
                metadata={"vectors": mismatched_vectors},
            ),
            QdrantCollectionContractCheck(
                name="missing_payload_indexes",
                passed=not missing_payload_indexes,
                message="All expected payload indexes exist in the Qdrant collection.",
                metadata={"fields": missing_payload_indexes},
            ),
            QdrantCollectionContractCheck(
                name="payload_index_schema_mismatch",
                passed=not mismatched_payload_indexes,
                message="Qdrant payload index schemas match the package contract.",
                metadata={"fields": mismatched_payload_indexes},
            ),
        ]
        failed_checks = [check.name for check in checks if not check.passed and check.severity == "error"]
        return QdrantCollectionContractReport(
            collection=self.collection_name,
            exists=True,
            passed=not failed_checks,
            expected_vectors=dict(sorted(named_vectors.items())),
            actual_vectors=dict(sorted(actual_vectors.items())),
            missing_vectors=missing_vectors,
            extra_vectors=extra_vectors,
            mismatched_vectors=mismatched_vectors,
            expected_payload_indexes=expected_payload_indexes,
            actual_payload_indexes=actual_payload_indexes,
            missing_payload_indexes=missing_payload_indexes,
            expected_payload_index_schemas=dict(sorted(expected_payload_index_schemas.items())),
            actual_payload_index_schemas=dict(sorted(actual_payload_index_schemas.items())),
            mismatched_payload_indexes=mismatched_payload_indexes,
            checks=checks,
            failed_checks=failed_checks,
        )

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(collection.name == self.collection_name for collection in collections)

    def ensure_payload_indexes(self, payload_indexes: list[str | dict[str, str]]) -> list[str]:
        created = []
        existing = self._existing_payload_indexes()
        for index in payload_indexes:
            field_name, schema = self._normalize_payload_index(index)
            if not field_name or field_name in existing:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
            created.append(field_name)
        return created

    def _existing_payload_indexes(self) -> set[str]:
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            return set()
        payload_schema = getattr(info, "payload_schema", {}) or {}
        return set(payload_schema.keys())

    def _normalize_payload_index(self, index: str | dict[str, str]):
        if isinstance(index, str):
            field_name = index
            schema_name = default_payload_schema(index)
        else:
            field_name = str(index.get("field") or index.get("field_name") or "").strip()
            schema_name = str(index.get("schema") or default_payload_schema(field_name)).strip().lower()
        return field_name, getattr(self._payload_schema_type, schema_name.upper())

    def upsert(self, records: Iterable[EmbeddingRecord]) -> UpsertResult:
        points = [
            self._point_struct(
                id=record.point_id,
                vector={record.vector_name: record.vector},
                payload={
                    "chunk_id": record.chunk_id,
                    "doc_id": record.doc_id,
                    **record.payload,
                },
            )
            for record in records
        ]
        if not points:
            return UpsertResult(collection=self.collection_name, count=0)

        result = self.client.upsert(collection_name=self.collection_name, points=points)
        return UpsertResult(
            collection=self.collection_name,
            count=len(points),
            detail={"operation_id": getattr(result, "operation_id", None)},
        )

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def query_vector(
        self,
        vector: list[float],
        vector_name: str = "text_dense",
        top_k: int = 10,
        must_payload: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchHit]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            using=vector_name,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
            query_filter=self._payload_filter(must_payload or {}),
        )
        return [
            VectorSearchHit(
                point_id=str(point.id),
                score=float(point.score),
                vector_name=vector_name,
                chunk_id=(point.payload or {}).get("chunk_id"),
                doc_id=(point.payload or {}).get("doc_id"),
                payload=point.payload or {},
            )
            for point in response.points
        ]

    def _payload_filter(self, must_payload: dict[str, Any]):
        if not must_payload:
            return None
        return self._filter(
            must=[self._payload_condition(key, value) for key, value in must_payload.items()]
        )

    def _payload_condition(self, key: str, value: Any):
        if isinstance(value, dict):
            if "any" in value:
                return self._field_condition(key=key, match=self._match_any(any=value["any"]))
            range_kwargs = {
                bound: value[bound]
                for bound in ("gt", "gte", "lt", "lte")
                if bound in value and value[bound] is not None
            }
            if range_kwargs:
                return self._field_condition(key=key, range=self._range(**range_kwargs))
            if "match" in value:
                return self._field_condition(key=key, match=self._match_value(value=value["match"]))
        if isinstance(value, (list, tuple, set)):
            return self._field_condition(key=key, match=self._match_any(any=list(value)))
        return self._field_condition(key=key, match=self._match_value(value=value))


def collection_vector_sizes(info) -> dict[str, int]:
    vectors = nested_get(info, "config", "params", "vectors")
    if isinstance(vectors, dict):
        return {
            str(name): int(size)
            for name, config in vectors.items()
            if (size := nested_get(config, "size")) is not None
        }
    size = nested_get(vectors, "size")
    return {"default": int(size)} if size is not None else {}


def collection_payload_indexes(info) -> set[str]:
    return set(collection_payload_index_schemas(info))


def collection_payload_index_schemas(info) -> dict[str, str | None]:
    payload_schema = nested_get(info, "payload_schema") or {}
    if not isinstance(payload_schema, dict):
        return {}
    return {
        str(field_name): payload_schema_name(schema)
        for field_name, schema in payload_schema.items()
    }


def payload_schema_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return normalize_payload_schema(value)
    for key_path in [
        ("data_type",),
        ("schema",),
        ("type",),
        ("params", "type"),
        ("params", "data_type"),
    ]:
        schema_value = nested_get(value, *key_path)
        if schema_value is not None:
            return normalize_payload_schema(str(schema_value))
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return normalize_payload_schema(name)
    return normalize_payload_schema(str(value))


def nested_get(value, *keys):
    current = value
    for key in keys:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
            continue
        current = getattr(current, key, None)
    return current
