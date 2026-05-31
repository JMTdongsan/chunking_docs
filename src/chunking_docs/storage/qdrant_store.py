from __future__ import annotations

from typing import Iterable

from chunking_docs.storage.records import EmbeddingRecord, UpsertResult


class QdrantChunkStore:
    """Thin Qdrant adapter kept optional so the core library runs without Qdrant."""

    def __init__(self, url: str, collection_name: str, api_key: str | None = None):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise RuntimeError("Install chunking-docs[qdrant] to use QdrantChunkStore") from exc

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self._point_struct = PointStruct

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
