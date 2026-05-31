from __future__ import annotations

from chunking_docs.graph.extractor import make_triple_id
from chunking_docs.models import DocumentChunk, GraphTriple


def remap_triples_to_available_chunks(
    triples: list[GraphTriple],
    chunks: list[DocumentChunk],
) -> list[GraphTriple]:
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    parent_to_first_child: dict[str, str] = {}
    for chunk in chunks:
        parent_id = chunk.metadata.get("parent_chunk_id")
        if parent_id and parent_id not in parent_to_first_child:
            parent_to_first_child[parent_id] = chunk.chunk_id

    remapped: list[GraphTriple] = []
    for triple in triples:
        if triple.chunk_id in chunk_ids:
            remapped.append(triple)
            continue
        replacement = parent_to_first_child.get(triple.chunk_id)
        if replacement is None:
            remapped.append(triple)
            continue
        remapped.append(
            triple.model_copy(
                update={
                    "triple_id": make_triple_id(
                        replacement,
                        triple.subject,
                        triple.predicate,
                        triple.object,
                    ),
                    "chunk_id": replacement,
                    "qualifiers": {
                        **triple.qualifiers,
                        "original_chunk_id": triple.chunk_id,
                        "remapped_to_subchunk": True,
                    },
                }
            )
        )
    return dedupe_triples(remapped)


def dedupe_triples(triples: list[GraphTriple]) -> list[GraphTriple]:
    by_id = {triple.triple_id: triple for triple in triples}
    return list(by_id.values())
