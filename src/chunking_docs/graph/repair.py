from __future__ import annotations

from chunking_docs.graph.extractor import make_triple_id
from chunking_docs.graph.provenance import chunk_ids_by_asset_id, triple_asset_ids
from chunking_docs.models import DocumentChunk, GraphTriple


def remap_triples_to_available_chunks(
    triples: list[GraphTriple],
    chunks: list[DocumentChunk],
) -> list[GraphTriple]:
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    alias_to_first_chunk: dict[str, str] = {}
    alias_source: dict[str, str] = {}
    for chunk in chunks:
        for key in ("source_chunk_id", "parent_chunk_id"):
            alias = chunk.metadata.get(key)
            if isinstance(alias, str) and alias and alias not in alias_to_first_chunk:
                alias_to_first_chunk[alias] = chunk.chunk_id
                alias_source[alias] = key
    chunks_by_asset = chunk_ids_by_asset_id(chunks)

    remapped: list[GraphTriple] = []
    for triple in triples:
        if triple.chunk_id in chunk_ids:
            remapped.append(triple)
            continue
        replacement = alias_to_first_chunk.get(triple.chunk_id)
        remap_source = alias_source.get(triple.chunk_id)
        remap_asset_id = None
        if replacement is None:
            for asset_id in sorted(triple_asset_ids(triple)):
                linked_chunks = chunks_by_asset.get(asset_id, [])
                if linked_chunks:
                    replacement = linked_chunks[0]
                    remap_source = "asset"
                    remap_asset_id = asset_id
                    break
        if replacement is None:
            remapped.append(triple)
            continue
        qualifiers = {
            **triple.qualifiers,
            "original_chunk_id": triple.chunk_id,
        }
        if remap_source == "parent_chunk_id":
            qualifiers["remapped_to_subchunk"] = True
        elif remap_source == "source_chunk_id":
            qualifiers["remapped_from_source_chunk"] = True
        elif remap_source == "asset":
            qualifiers["remapped_by_asset_provenance"] = True
            qualifiers["remapped_asset_id"] = remap_asset_id
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
                    "qualifiers": qualifiers,
                }
            )
        )
    return dedupe_triples(remapped)


def dedupe_triples(triples: list[GraphTriple]) -> list[GraphTriple]:
    by_id = {triple.triple_id: triple for triple in triples}
    return list(by_id.values())
