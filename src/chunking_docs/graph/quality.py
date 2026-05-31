from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.graph.extractor import make_triple_id
from chunking_docs.models import DocumentChunk, GraphTriple

_QUOTE_CHARS = "\"'`“”‘’"
_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_UNDERSCORE_RE = re.compile(r"_+")


class GraphTripleIssue(BaseModel):
    severity: str
    code: str
    message: str
    triple_id: str | None = None
    chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphTripleQualityReport(BaseModel):
    triple_count: int
    normalized_count: int
    duplicate_count: int
    empty_field_count: int
    orphan_count: int
    invalid_confidence_count: int
    predicate_counts: dict[str, int]
    issues: list[GraphTripleIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def normalize_entity_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.strip().strip(_QUOTE_CHARS)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def normalize_predicate(value: str) -> str:
    normalized = normalize_entity_label(value).casefold()
    normalized = _NON_WORD_RE.sub("_", normalized)
    normalized = _UNDERSCORE_RE.sub("_", normalized).strip("_")
    return normalized


def normalize_graph_triple(triple: GraphTriple) -> GraphTriple:
    subject = normalize_entity_label(triple.subject)
    predicate = normalize_predicate(triple.predicate)
    object_ = normalize_entity_label(triple.object)
    triple_id = make_triple_id(triple.chunk_id, subject, predicate, object_)
    update: dict[str, Any] = {
        "triple_id": triple_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
    }
    changed = (
        triple.triple_id != triple_id
        or triple.subject != subject
        or triple.predicate != predicate
        or triple.object != object_
    )
    if changed:
        qualifiers = dict(triple.qualifiers)
        if triple.triple_id != triple_id:
            qualifiers.setdefault("original_triple_id", triple.triple_id)
        qualifiers["normalized"] = True
        update["qualifiers"] = qualifiers
    return triple.model_copy(update=update)


def normalize_graph_triples(
    triples: list[GraphTriple],
    dedupe: bool = True,
) -> list[GraphTriple]:
    normalized = [
        triple
        for triple in (normalize_graph_triple(triple) for triple in triples)
        if graph_triple_has_required_fields(triple)
    ]
    if not dedupe:
        return normalized

    by_key: dict[tuple[str, str, str, str, str], GraphTriple] = {}
    duplicate_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for triple in normalized:
        key = semantic_key(triple)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = triple
            continue
        duplicate_counts[key] += 1
        if confidence_value(triple) > confidence_value(existing):
            by_key[key] = triple

    results: list[GraphTriple] = []
    for key, triple in by_key.items():
        duplicate_count = duplicate_counts.get(key, 0)
        if duplicate_count:
            qualifiers = {
                **triple.qualifiers,
                "deduped_duplicate_count": duplicate_count,
            }
            triple = triple.model_copy(update={"qualifiers": qualifiers})
        results.append(triple)
    return results


def graph_triple_has_required_fields(triple: GraphTriple) -> bool:
    return bool(triple.subject and triple.predicate and triple.object)


def audit_graph_triples(
    triples: list[GraphTriple],
    chunks: list[DocumentChunk] | None = None,
    max_issues: int = 200,
) -> GraphTripleQualityReport:
    chunk_ids = {chunk.chunk_id for chunk in chunks or []}
    check_orphans = chunks is not None
    normalized = [normalize_graph_triple(triple) for triple in triples]
    predicate_counts = Counter(triple.predicate for triple in normalized if triple.predicate)
    semantic_counts = Counter(semantic_key(triple) for triple in normalized)
    duplicate_keys = {key for key, count in semantic_counts.items() if count > 1}

    issues: list[GraphTripleIssue] = []
    empty_field_count = 0
    orphan_count = 0
    invalid_confidence_count = 0

    for triple, normalized_triple in zip(triples, normalized, strict=True):
        missing_fields = [
            field
            for field, value in [
                ("subject", normalized_triple.subject),
                ("predicate", normalized_triple.predicate),
                ("object", normalized_triple.object),
            ]
            if not value
        ]
        if missing_fields:
            empty_field_count += 1
            append_issue(
                issues,
                max_issues,
                GraphTripleIssue(
                    severity="error",
                    code="empty_triple_field",
                    message="A graph triple has an empty subject, predicate, or object after normalization.",
                    triple_id=triple.triple_id,
                    chunk_id=triple.chunk_id,
                    metadata={"fields": missing_fields},
                ),
            )

        if check_orphans and triple.chunk_id not in chunk_ids:
            orphan_count += 1
            append_issue(
                issues,
                max_issues,
                GraphTripleIssue(
                    severity="error",
                    code="orphan_triple_chunk",
                    message="A graph triple points to a chunk ID that is not present in chunks.jsonl.",
                    triple_id=triple.triple_id,
                    chunk_id=triple.chunk_id,
                ),
            )

        if triple.confidence is not None and not 0.0 <= triple.confidence <= 1.0:
            invalid_confidence_count += 1
            append_issue(
                issues,
                max_issues,
                GraphTripleIssue(
                    severity="warning",
                    code="invalid_triple_confidence",
                    message="A graph triple confidence value is outside the 0.0 to 1.0 range.",
                    triple_id=triple.triple_id,
                    chunk_id=triple.chunk_id,
                    metadata={"confidence": triple.confidence},
                ),
            )

        if semantic_key(normalized_triple) in duplicate_keys:
            append_issue(
                issues,
                max_issues,
                GraphTripleIssue(
                    severity="warning",
                    code="duplicate_graph_triple",
                    message="A graph triple has the same normalized subject, predicate, and object as another triple in the same chunk.",
                    triple_id=triple.triple_id,
                    chunk_id=triple.chunk_id,
                    metadata={
                        "subject": normalized_triple.subject,
                        "predicate": normalized_triple.predicate,
                        "object": normalized_triple.object,
                    },
                ),
            )

    normalized_count = sum(1 for original, new in zip(triples, normalized, strict=True) if original != new)
    duplicate_count = sum(count - 1 for count in semantic_counts.values() if count > 1)
    return GraphTripleQualityReport(
        triple_count=len(triples),
        normalized_count=normalized_count,
        duplicate_count=duplicate_count,
        empty_field_count=empty_field_count,
        orphan_count=orphan_count,
        invalid_confidence_count=invalid_confidence_count,
        predicate_counts=dict(sorted(predicate_counts.items())),
        issues=issues,
    )


def semantic_key(triple: GraphTriple) -> tuple[str, str, str, str, str]:
    return (
        triple.doc_id,
        triple.chunk_id,
        normalize_entity_label(triple.subject).casefold(),
        normalize_predicate(triple.predicate),
        normalize_entity_label(triple.object).casefold(),
    )


def confidence_value(triple: GraphTriple) -> float:
    return triple.confidence if triple.confidence is not None else -1.0


def append_issue(
    issues: list[GraphTripleIssue],
    max_issues: int,
    issue: GraphTripleIssue,
) -> None:
    if len(issues) < max_issues:
        issues.append(issue)
