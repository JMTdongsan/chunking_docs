from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.graph.extractor import triples_from_vlm_json
from chunking_docs.graph.provenance import chunk_asset_ids
from chunking_docs.graph.quality import normalize_graph_triples
from chunking_docs.models import AssetKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.vision.annotate import merge_asset_annotations_into_chunks
from chunking_docs.vision.vlm_output import visual_triples_from_payload


class AssetAnnotation(BaseModel):
    asset_id: str | None = None
    page_no: int | None = None
    kind: AssetKind | None = None
    caption: str | None = None
    ocr_text: str | None = None
    vlm_summary: str | None = None
    triples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def apply_asset_annotations(
    assets: list[VisualAsset],
    chunks: list[DocumentChunk],
    annotations: list[AssetAnnotation],
    existing_triples: list[GraphTriple] | None = None,
) -> tuple[list[VisualAsset], list[DocumentChunk], list[GraphTriple]]:
    updated_assets = assets
    triples = list(existing_triples or [])

    for annotation in annotations:
        updated_assets = apply_one_annotation(updated_assets, annotation)
        matched_assets = matching_assets_for_annotation(updated_assets, annotation)
        annotation_triples = triples_for_annotation(annotation, matched_assets)
        if annotation_triples:
            chunk = chunk_for_annotation(chunks, updated_assets, annotation)
            if chunk is not None:
                triples.extend(
                    triples_from_vlm_json(
                        chunk,
                        annotation_triples,
                        qualifiers=triple_provenance(annotation, matched_assets),
                    )
                )

    updated_chunks = merge_asset_annotations_into_chunks(chunks, updated_assets)
    triples = normalize_graph_triples(triples)
    return updated_assets, updated_chunks, triples


def apply_one_annotation(
    assets: list[VisualAsset],
    annotation: AssetAnnotation,
) -> list[VisualAsset]:
    updated = []
    for asset in assets:
        if not annotation_matches(asset, annotation):
            updated.append(asset)
            continue
        metadata = {
            **asset.metadata,
            **annotation.metadata,
            "annotation_source": annotation.metadata.get("annotation_source", "manual"),
        }
        update = {
            "metadata": metadata,
        }
        if annotation.kind is not None:
            update["kind"] = annotation.kind
        if annotation.caption is not None:
            update["caption"] = annotation.caption
        if annotation.ocr_text is not None:
            update["ocr_text"] = annotation.ocr_text
        if annotation.vlm_summary is not None:
            update["vlm_summary"] = annotation.vlm_summary
        updated.append(asset.model_copy(update=update))
    return updated


def annotation_matches(asset: VisualAsset, annotation: AssetAnnotation) -> bool:
    if annotation.asset_id is not None:
        return asset.asset_id == annotation.asset_id
    if annotation.page_no is not None:
        return asset.page_no == annotation.page_no
    return False


def chunk_for_annotation(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    annotation: AssetAnnotation,
) -> DocumentChunk | None:
    if annotation.asset_id is not None:
        for chunk in chunks:
            if annotation.asset_id in chunk_asset_ids(chunk):
                return chunk
    if annotation.page_no is not None:
        for chunk in chunks:
            if chunk.page_start <= annotation.page_no <= chunk.page_end:
                return chunk
    return None


def matching_assets_for_annotation(
    assets: list[VisualAsset],
    annotation: AssetAnnotation,
) -> list[VisualAsset]:
    return [asset for asset in assets if annotation_matches(asset, annotation)]


def triples_for_annotation(
    annotation: AssetAnnotation,
    matched_assets: list[VisualAsset],
) -> list[dict[str, Any]]:
    payload = annotation_visual_payload(annotation, matched_assets)
    return visual_triples_from_payload(payload)


def annotation_visual_payload(
    annotation: AssetAnnotation,
    matched_assets: list[VisualAsset],
) -> dict[str, Any]:
    payload = dict(annotation.metadata)
    if annotation.triples:
        payload["triples"] = annotation.triples
    if any(payload.get(key) for key in ("title", "caption", "name")):
        return payload
    if annotation.caption:
        payload["caption"] = annotation.caption
        return payload
    if len(matched_assets) == 1 and matched_assets[0].caption:
        payload["caption"] = matched_assets[0].caption
    return payload


def triple_provenance(
    annotation: AssetAnnotation,
    matched_assets: list[VisualAsset] | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {"source": "visual_annotation"}
    matched_assets = matched_assets or []
    if annotation.asset_id is not None:
        provenance["asset_id"] = annotation.asset_id
    elif len(matched_assets) == 1:
        provenance["asset_id"] = matched_assets[0].asset_id
    elif matched_assets:
        provenance["asset_ids"] = sorted(asset.asset_id for asset in matched_assets)
    if annotation.page_no is not None:
        provenance["page_no"] = annotation.page_no
    if annotation.kind is not None:
        provenance["asset_kind"] = str(annotation.kind)
    elif len(matched_assets) == 1:
        provenance["asset_kind"] = str(matched_assets[0].kind)
    for key in [
        "annotation_source",
        "visual_job_id",
        "vlm_prompt_name",
        "vlm_prompt_schema_version",
        "vlm_prompt_sha256",
    ]:
        value = annotation.metadata.get(key)
        if value:
            provenance[key] = value
    return provenance
