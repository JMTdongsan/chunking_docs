from __future__ import annotations

from chunking_docs.graph.extractor import make_triple_id
from chunking_docs.models import DocumentChunk, GraphTriple


def section_triples(chunks: list[DocumentChunk]) -> list[GraphTriple]:
    triples: dict[str, GraphTriple] = {}
    for chunk in chunks:
        if chunk.section.chapter:
            add_triple(
                triples,
                chunk,
                subject="2030 서울도시기본계획",
                predicate="includes_chapter",
                object_=chunk.section.chapter,
            )
        if chunk.section.issue:
            add_triple(
                triples,
                chunk,
                subject=chunk.section.chapter or "2030 서울도시기본계획",
                predicate="includes_issue",
                object_=chunk.section.issue,
            )
        label = chunk.section.label()
        if label:
            add_triple(
                triples,
                chunk,
                subject=chunk.chunk_id,
                predicate="belongs_to_section",
                object_=label,
            )
    return list(triples.values())


def add_triple(
    triples: dict[str, GraphTriple],
    chunk: DocumentChunk,
    subject: str,
    predicate: str,
    object_: str,
) -> None:
    triple_id = make_triple_id(chunk.chunk_id, subject, predicate, object_)
    triples[triple_id] = GraphTriple(
        triple_id=triple_id,
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        subject=subject,
        predicate=predicate,
        object=object_,
        qualifiers={
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "source": "section_map",
        },
        confidence=0.75,
    )
