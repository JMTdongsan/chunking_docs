from __future__ import annotations

from typing import Iterable

from chunking_docs.storage.records import EmbeddingRecord, UpsertResult


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
            from qdrant_client.models import Distance, PointStruct, VectorParams
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

    def ensure_collection(self, named_vectors: dict[str, int]) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                name: self._vector_params(size=size, distance=self._distance.COSINE)
                for name, size in named_vectors.items()
            },
        )

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
