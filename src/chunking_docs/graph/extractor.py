from __future__ import annotations

import hashlib
from typing import Protocol

from chunking_docs.models import DocumentChunk, GraphTriple


class TripleExtractor(Protocol):
    def extract(self, chunk: DocumentChunk) -> list[GraphTriple]:
        """Extract graph triples from a chunk."""


class NullTripleExtractor:
    def extract(self, chunk: DocumentChunk) -> list[GraphTriple]:
        return []


def make_triple_id(chunk_id: str, subject: str, predicate: str, object_: str) -> str:
    raw = f"{chunk_id}:{subject}:{predicate}:{object_}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def triples_from_vlm_json(
    chunk: DocumentChunk,
    triples: list[dict],
    qualifiers: dict | None = None,
) -> list[GraphTriple]:
    results: list[GraphTriple] = []
    base_qualifiers = qualifiers or {}
    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("predicate", "")).strip()
        object_ = str(triple.get("object", "")).strip()
        if not subject or not predicate or not object_:
            continue
        triple_qualifiers = {
            k: v for k, v in triple.items() if k not in {"subject", "predicate", "object", "confidence"}
        }
        results.append(
            GraphTriple(
                triple_id=make_triple_id(chunk.chunk_id, subject, predicate, object_),
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                subject=subject,
                predicate=predicate,
                object=object_,
                qualifiers={**triple_qualifiers, **base_qualifiers},
                confidence=coerce_confidence(triple.get("confidence")),
            )
        )
    return results


def coerce_confidence(value) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))
