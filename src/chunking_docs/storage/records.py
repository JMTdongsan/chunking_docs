from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EmbeddingRecord(BaseModel):
    point_id: str
    chunk_id: str
    doc_id: str
    vector_name: str
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class UpsertResult(BaseModel):
    collection: str
    count: int
    detail: dict[str, Any] = Field(default_factory=dict)


class VectorSearchHit(BaseModel):
    point_id: str
    score: float
    vector_name: str
    chunk_id: str | None = None
    doc_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
