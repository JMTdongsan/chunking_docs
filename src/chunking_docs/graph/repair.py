from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chunking_docs.graph.extractor import make_triple_id
from chunking_docs.graph.provenance import chunk_ids_by_asset_id, ordered_unique, triple_asset_ids
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.vision.vlm_output import visual_triples_from_payload


@dataclass(frozen=True)
class VisualDerivedTripleRepairReport:
    input_triples: int
    output_triples: int
    added_triples: int
    updated_triples: int
    repaired_asset_count: int
    skipped_asset_count: int
    skipped_asset_ids: list[str]


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


def repair_visual_derived_triples(
    assets: list[VisualAsset],
    chunks: list[DocumentChunk],
    triples: list[GraphTriple],
) -> tuple[list[GraphTriple], VisualDerivedTripleRepairReport]:
    """Add or update triples for structured VLM metadata already stored on assets."""

    chunks_by_asset = chunk_ids_by_asset_id(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    existing_keys_by_asset = triple_keys_by_asset_id(triples)
    triple_index = {
        semantic_triple_key(triple.doc_id, triple.chunk_id, triple.subject, triple.predicate, triple.object): index
        for index, triple in enumerate(triples)
    }

    repaired = list(triples)
    added = 0
    updated = 0
    repaired_asset_ids: set[str] = set()
    skipped_asset_ids: list[str] = []

    for asset in assets:
        chunk_id = preferred_chunk_id_for_asset(asset, chunks_by_asset, chunks_by_id)
        derived_triples = [
            triple
            for triple in visual_triples_from_payload(visual_triple_payload(asset))
            if triple.get("derived_from_vlm_field")
        ]
        if not derived_triples:
            continue
        if chunk_id is None:
            skipped_asset_ids.append(asset.asset_id)
            continue

        existing_asset_keys = existing_keys_by_asset.get(asset.asset_id, set())
        for triple_payload in derived_triples:
            subject = str(triple_payload.get("subject", "")).strip()
            predicate = str(triple_payload.get("predicate", "")).strip()
            object_ = str(triple_payload.get("object", "")).strip()
            if not subject or not predicate or not object_:
                continue
            expected_key = normalized_graph_triple_key(subject, predicate, object_)
            if expected_key in existing_asset_keys:
                continue

            semantic_key = semantic_triple_key(asset.doc_id, chunk_id, subject, predicate, object_)
            existing_index = triple_index.get(semantic_key)
            if existing_index is not None:
                current = repaired[existing_index]
                merged = current.model_copy(
                    update={
                        "qualifiers": merge_visual_asset_provenance(
                            current.qualifiers,
                            asset,
                            source_payload=triple_payload,
                            repaired=True,
                        )
                    }
                )
                repaired[existing_index] = merged
                updated += 1
            else:
                new_triple = GraphTriple(
                    triple_id=make_triple_id(chunk_id, subject, predicate, object_),
                    doc_id=asset.doc_id,
                    chunk_id=chunk_id,
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    qualifiers=merge_visual_asset_provenance(
                        {},
                        asset,
                        source_payload=triple_payload,
                        repaired=True,
                    ),
                    confidence=coerce_confidence(triple_payload.get("confidence")),
                )
                triple_index[semantic_key] = len(repaired)
                repaired.append(new_triple)
                added += 1
            existing_asset_keys.add(expected_key)
            repaired_asset_ids.add(asset.asset_id)

    return repaired, VisualDerivedTripleRepairReport(
        input_triples=len(triples),
        output_triples=len(repaired),
        added_triples=added,
        updated_triples=updated,
        repaired_asset_count=len(repaired_asset_ids),
        skipped_asset_count=len(skipped_asset_ids),
        skipped_asset_ids=skipped_asset_ids[:50],
    )


def preferred_chunk_id_for_asset(
    asset: VisualAsset,
    chunks_by_asset: dict[str, list[str]],
    chunks_by_id: dict[str, DocumentChunk],
) -> str | None:
    linked = chunks_by_asset.get(asset.asset_id, [])
    if linked:
        return linked[0]
    for chunk in chunks_by_id.values():
        if chunk.doc_id == asset.doc_id and chunk.page_start <= asset.page_no <= chunk.page_end:
            return chunk.chunk_id
    return None


def visual_triple_payload(asset: VisualAsset) -> dict[str, Any]:
    payload = dict(asset.metadata)
    if asset.caption and not any(payload.get(key) for key in ("title", "caption", "name")):
        payload["caption"] = asset.caption
    return payload


def triple_keys_by_asset_id(triples: list[GraphTriple]) -> dict[str, set[tuple[str, str, str]]]:
    indexed: dict[str, set[tuple[str, str, str]]] = {}
    for triple in triples:
        key = normalized_graph_triple_key(triple.subject, triple.predicate, triple.object)
        if key == ("", "", ""):
            continue
        for asset_id in sorted(triple_asset_ids(triple)):
            indexed.setdefault(asset_id, set()).add(key)
    return indexed


def merge_visual_asset_provenance(
    qualifiers: dict[str, Any],
    asset: VisualAsset,
    source_payload: dict[str, Any],
    repaired: bool = False,
) -> dict[str, Any]:
    merged = dict(qualifiers)
    current_asset_ids = sorted(triple_asset_ids(GraphTriple(
        triple_id="",
        doc_id=asset.doc_id,
        chunk_id="",
        subject="s",
        predicate="p",
        object="o",
        qualifiers=merged,
    )))
    asset_ids = ordered_unique([*current_asset_ids, asset.asset_id])
    if "asset_id" not in merged:
        merged["asset_id"] = asset.asset_id
    elif merged.get("asset_id") != asset.asset_id:
        merged["asset_ids"] = asset_ids
    elif len(asset_ids) > 1:
        merged["asset_ids"] = asset_ids

    merged.setdefault("source", "visual_annotation")
    merged.setdefault("page_no", asset.page_no)
    merged.setdefault("asset_kind", str(asset.kind))
    for key in [
        "source_field",
        "source_key",
        "derived_from_vlm_field",
        "attributes",
        "bbox",
        "bbox_region",
        "description",
        "location",
    ]:
        value = source_payload.get(key)
        if value not in (None, "", []):
            merged.setdefault(key, value)
    for key in [
        "annotation_source",
        "visual_job_id",
        "vlm_prompt_name",
        "vlm_prompt_schema_version",
        "vlm_prompt_sha256",
    ]:
        value = asset.metadata.get(key)
        if value:
            merged.setdefault(key, value)
    if repaired:
        merged["visual_derived_triple_repair"] = True
    return merged


def normalized_graph_triple_key(subject: Any, predicate: Any, object_: Any) -> tuple[str, str, str]:
    return (
        str(subject or "").strip().casefold(),
        str(predicate or "").strip().casefold(),
        str(object_ or "").strip().casefold(),
    )


def semantic_triple_key(
    doc_id: str,
    chunk_id: str,
    subject: Any,
    predicate: Any,
    object_: Any,
) -> tuple[str, str, str, str, str]:
    subject_key, predicate_key, object_key = normalized_graph_triple_key(subject, predicate, object_)
    return doc_id, chunk_id, subject_key, predicate_key, object_key


def coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))
