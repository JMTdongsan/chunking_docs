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


class GraphExportSummary(BaseModel):
    node_count: int
    edge_count: int
    connected_component_count: int
    largest_component_node_count: int
    isolated_node_count: int
    max_degree: int
    mean_degree: float
    predicate_counts: dict[str, int] = Field(default_factory=dict)
    doc_counts: dict[str, int] = Field(default_factory=dict)
    top_nodes: list[dict[str, Any]] = Field(default_factory=list)


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
            "triple_id": triple.triple_id,
            "subject": triple.subject,
            "object": triple.object,
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

    exported_nodes = list(nodes.values())
    annotate_node_metadata(exported_nodes, edges)
    return exported_nodes, edges


def summarize_graph(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    top_n: int = 10,
) -> GraphExportSummary:
    degree = {node.node_id: int(node.metadata.get("degree") or 0) for node in nodes}
    predicate_counts: dict[str, int] = defaultdict(int)
    doc_counts: dict[str, int] = defaultdict(int)
    for edge in edges:
        predicate_counts[edge.predicate] += 1
        doc_counts[edge.doc_id] += 1
    components = connected_components(nodes, edges)
    largest_component_node_count = max((len(component) for component in components), default=0)
    isolated_node_count = sum(1 for value in degree.values() if value == 0)
    mean_degree = round(sum(degree.values()) / len(nodes), 6) if nodes else 0.0
    top_nodes = [
        {
            "node_id": node.node_id,
            "label": node.label,
            "degree": int(node.metadata.get("degree") or 0),
            "in_degree": int(node.metadata.get("in_degree") or 0),
            "out_degree": int(node.metadata.get("out_degree") or 0),
            "doc_ids": node.metadata.get("doc_ids", []),
            "chunk_ids": node.metadata.get("chunk_ids", []),
        }
        for node in sorted(
            nodes,
            key=lambda item: (
                -int(item.metadata.get("degree") or 0),
                item.label.casefold(),
                item.node_id,
            ),
        )[:top_n]
    ]
    return GraphExportSummary(
        node_count=len(nodes),
        edge_count=len(edges),
        connected_component_count=len(components),
        largest_component_node_count=largest_component_node_count,
        isolated_node_count=isolated_node_count,
        max_degree=max(degree.values(), default=0),
        mean_degree=mean_degree,
        predicate_counts=dict(sorted(predicate_counts.items())),
        doc_counts=dict(sorted(doc_counts.items())),
        top_nodes=top_nodes,
    )


def annotate_node_metadata(nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
    stats = {
        node.node_id: {
            "degree": 0,
            "in_degree": 0,
            "out_degree": 0,
            "doc_ids": set(),
            "chunk_ids": set(),
            "predicates_as_subject": set(),
            "predicates_as_object": set(),
        }
        for node in nodes
    }
    for edge in edges:
        source_stats = stats.setdefault(edge.source, empty_node_stats())
        target_stats = stats.setdefault(edge.target, empty_node_stats())
        source_stats["degree"] += 1
        source_stats["out_degree"] += 1
        source_stats["doc_ids"].add(edge.doc_id)
        source_stats["chunk_ids"].add(edge.chunk_id)
        source_stats["predicates_as_subject"].add(edge.predicate)
        target_stats["degree"] += 1
        target_stats["in_degree"] += 1
        target_stats["doc_ids"].add(edge.doc_id)
        target_stats["chunk_ids"].add(edge.chunk_id)
        target_stats["predicates_as_object"].add(edge.predicate)

    for node in nodes:
        node_stats = stats.get(node.node_id, empty_node_stats())
        node.metadata.update(
            {
                "degree": node_stats["degree"],
                "in_degree": node_stats["in_degree"],
                "out_degree": node_stats["out_degree"],
                "doc_ids": sorted(node_stats["doc_ids"]),
                "chunk_ids": sorted(node_stats["chunk_ids"]),
                "predicates_as_subject": sorted(node_stats["predicates_as_subject"]),
                "predicates_as_object": sorted(node_stats["predicates_as_object"]),
            }
        )


def empty_node_stats() -> dict[str, Any]:
    return {
        "degree": 0,
        "in_degree": 0,
        "out_degree": 0,
        "doc_ids": set(),
        "chunk_ids": set(),
        "predicates_as_subject": set(),
        "predicates_as_object": set(),
    }


def connected_components(nodes: list[GraphNode], edges: list[GraphEdge]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {node.node_id: set() for node in nodes}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
        adjacency.setdefault(edge.target, set()).add(edge.source)
    components = []
    seen = set()
    for node_id in sorted(adjacency):
        if node_id in seen:
            continue
        stack = [node_id]
        component = set()
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(component)
    return components


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
