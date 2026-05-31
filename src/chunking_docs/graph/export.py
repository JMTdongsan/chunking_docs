from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.models import DocumentChunk, GraphTriple


class GraphNode(BaseModel):
    node_id: str
    label: str
    node_type: str = "entity"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    edge_id: str
    source: str
    target: str
    predicate: str
    doc_id: str
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def export_graph(
    triples: list[GraphTriple],
    chunks: list[DocumentChunk] | None = None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks or []}
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    for triple in triples:
        subject_id = node_id(triple.subject)
        object_id = node_id(triple.object)
        nodes.setdefault(subject_id, GraphNode(node_id=subject_id, label=triple.subject))
        nodes.setdefault(object_id, GraphNode(node_id=object_id, label=triple.object))
        chunk = chunk_map.get(triple.chunk_id)
        metadata = {
            "qualifiers": triple.qualifiers,
            "confidence": triple.confidence,
        }
        if chunk is not None:
            metadata.update(
                {
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section.model_dump(mode="json"),
                }
            )
        edges.append(
            GraphEdge(
                edge_id=edge_id(triple),
                source=subject_id,
                target=object_id,
                predicate=triple.predicate,
                doc_id=triple.doc_id,
                chunk_id=triple.chunk_id,
                metadata=metadata,
            )
        )

    return list(nodes.values()), edges


def related_terms(triples: list[GraphTriple], query: str, limit: int = 12) -> list[str]:
    query = query.lower()
    scores: dict[str, int] = defaultdict(int)
    for triple in triples:
        subject = triple.subject.lower()
        object_ = triple.object.lower()
        predicate = triple.predicate.lower()
        if subject in query or any(token in subject for token in query.split()):
            scores[triple.object] += 2
            scores[triple.predicate] += 1
        if object_ in query or any(token in object_ for token in query.split()):
            scores[triple.subject] += 2
            scores[triple.predicate] += 1
        if predicate in query:
            scores[triple.subject] += 1
            scores[triple.object] += 1
    return [term for term, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def node_id(label: str) -> str:
    return hashlib.sha256(label.strip().lower().encode("utf-8")).hexdigest()[:24]


def edge_id(triple: GraphTriple) -> str:
    return hashlib.sha256(
        f"{triple.triple_id}:{triple.subject}:{triple.predicate}:{triple.object}".encode("utf-8")
    ).hexdigest()[:24]
