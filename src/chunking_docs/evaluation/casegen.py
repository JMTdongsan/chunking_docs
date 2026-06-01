from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from chunking_docs.embeddings.records import asset_text
from chunking_docs.evaluation.retrieval import CASE_GROUP_METADATA_KEYS, RetrievalCase
from chunking_docs.graph.provenance import (
    chunk_asset_ids,
    chunk_id_alias_map,
    chunk_ids_by_asset_id,
    triple_asset_ids,
    triple_resolved_chunk_ids,
)
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.vision.spatial import bbox_region_from_bbox

_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_RE = re.compile(r"[|#*_`>\[\]()]")
_GENERATED_VISUAL_BLOCK_PREFIXES = ("[vlm page ", "[visual asset page ")
_NON_VISUAL_BLOCK_PREFIXES = ("[ocr page ",)
_TERM_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_./%+-]{2,}|"
    r"[0-9]+(?:[.,][0-9]+)*(?:[%A-Za-z가-힣㎡²³]*)?|"
    r"[가-힣]{2,}|"
    r"[\u4e00-\u9fff]{2,}"
)
_ID_LIKE_RE = re.compile(r"(?=.*[0-9])(?=.*[a-f])[a-f0-9]{2,}", re.IGNORECASE)
_PLACEHOLDER_PATTERNS = [
    "OCR/VLM processing required",
    "Full page render for page",
]
_STOP_TERMS = {
    "about",
    "above",
    "appendix",
    "belongs",
    "belong",
    "chapter",
    "contents",
    "defines",
    "document",
    "example",
    "figure",
    "includes",
    "introduction",
    "overview",
    "page",
    "range",
    "reference",
    "report",
    "section",
    "summary",
    "table",
    "본문",
    "보고서",
    "부록",
    "요약",
    "개요",
    "그림",
    "목차",
    "자료",
    "참고",
    "페이지",
}
VISUAL_OBJECT_METADATA_KEYS = ("objects", "detected_objects", "visual_objects", "detections", "regions", "areas")
VISUAL_FEATURE_METADATA_KEYS = ("visual_elements",)

QueryMode = Literal["snippet", "salient_terms", "question"]
SelectionStrategy = Literal["document_order", "salience"]


@dataclass(frozen=True)
class QueryDraft:
    query: str
    score: float = 0.0
    terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseCandidate:
    case: RetrievalCase
    score: float
    order: tuple


def generate_retrieval_case_skeleton(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    page_limit: int = 20,
    asset_limit: int = 20,
    triple_limit: int = 20,
    include_pages: bool = True,
    include_assets: bool = True,
    include_triples: bool = True,
    include_todo: bool = False,
    query_max_chars: int = 120,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    min_query_terms: int = 3,
    max_query_terms: int = 8,
    dedupe_queries: bool = True,
    visual_probe_limit: int = 0,
    image_probe_limit: int = 0,
    object_probe_limit: int = 0,
    object_probe_visual_only: bool = True,
) -> list[RetrievalCase]:
    validate_casegen_options(query_mode, selection_strategy, min_query_terms, max_query_terms)
    corpus_texts = case_source_texts(chunks, assets, triples)
    term_df = term_document_frequencies(corpus_texts)
    cases: list[RetrievalCase] = []
    if include_pages:
        cases.extend(
            page_cases(
                chunks,
                limit=page_limit,
                include_todo=include_todo,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
            )
        )
    if include_assets:
        cases.extend(
            asset_cases(
                assets,
                limit=asset_limit,
                include_todo=include_todo,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
            )
        )
    if visual_probe_limit > 0:
        cases.extend(
            visual_lexical_probe_cases(
                chunks,
                assets,
                limit=visual_probe_limit,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
            )
        )
    if image_probe_limit > 0:
        cases.extend(
            visual_image_probe_cases(
                chunks,
                assets,
                limit=image_probe_limit,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
            )
        )
    if object_probe_limit > 0:
        cases.extend(
            visual_object_probe_cases(
                chunks,
                assets,
                limit=object_probe_limit,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
                visual_only=object_probe_visual_only,
            )
        )
    if include_triples:
        cases.extend(
            triple_cases(
                triples,
                chunks=chunks,
                limit=triple_limit,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
            )
        )
    if dedupe_queries:
        cases = dedupe_cases_by_query(cases)
    return cases


def page_cases(
    chunks: list[DocumentChunk],
    limit: int,
    include_todo: bool,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> list[RetrievalCase]:
    candidates: list[CaseCandidate] = []
    for chunk in sorted(chunks, key=lambda item: (item.page_start, item.page_end, item.chunk_id)):
        draft = query_from_text(
            chunk.text,
            max_chars=query_max_chars,
            mode=query_mode,
            term_df=term_df,
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        query = draft.query
        if not query and include_todo:
            query = f"TODO: write query for page {chunk.page_start}"
        if not query:
            continue
        case = with_case_metadata(
            RetrievalCase(
                query=query,
                expected_pages=[chunk.page_start],
                expected_chunk_ids=[chunk.chunk_id],
            ),
            case_source="page",
            query_mode=query_mode,
            draft=draft,
        )
        candidates.append(
            CaseCandidate(
                case=case,
                score=draft.score,
                order=(chunk.page_start, chunk.page_end, chunk.chunk_id),
            )
        )
    return select_candidates(dedupe_page_candidates(candidates, selection_strategy), limit, selection_strategy)


def asset_cases(
    assets: list[VisualAsset],
    limit: int,
    include_todo: bool,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> list[RetrievalCase]:
    candidates: list[CaseCandidate] = []
    for asset in sorted(assets, key=lambda item: (asset_priority(item), item.page_no, item.asset_id)):
        draft = query_from_text(
            asset_text(asset),
            max_chars=query_max_chars,
            mode=query_mode,
            term_df=term_df,
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        query = draft.query
        if not query and include_todo:
            query = f"TODO: write query for {asset.kind} asset on page {asset.page_no}"
        if not query:
            continue
        case = with_case_metadata(
            RetrievalCase(
                query=query,
                expected_pages=[asset.page_no],
                expected_asset_ids=[asset.asset_id],
            ),
            case_source="asset",
            query_mode=query_mode,
            draft=draft,
        )
        candidates.append(
            CaseCandidate(
                case=case,
                score=draft.score,
                order=(asset_priority(asset), asset.page_no, asset.asset_id),
            )
        )
    return select_candidates(merge_case_candidates_by_query(candidates), limit, selection_strategy)


def visual_lexical_probe_cases(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    limit: int,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> list[RetrievalCase]:
    candidates: list[CaseCandidate] = []
    chunks_by_asset = chunks_by_asset_id(chunks)
    for asset in sorted(assets, key=lambda item: (asset_priority(item), item.page_no, item.asset_id)):
        linked_chunks = chunks_by_asset.get(asset.asset_id, [])
        if not linked_chunks:
            continue
        draft = visual_probe_query_from_asset(
            asset,
            linked_chunks,
            max_chars=query_max_chars,
            mode=query_mode,
            term_df=term_df or {},
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        if not draft.query:
            continue
        case = with_case_metadata(
            RetrievalCase(
                query=draft.query,
                expected_pages=[asset.page_no],
                expected_asset_ids=[asset.asset_id],
            ),
            case_source="visual_lexical_probe",
            query_mode=query_mode,
            draft=draft,
        )
        case = case.model_copy(
            update={
                "metadata": {
                    **case.metadata,
                    "case_family": "visual",
                    "evidence_family": "visual_text",
                    "modality": "visual_text",
                    "linked_chunk_ids": [chunk.chunk_id for chunk in linked_chunks],
                }
            }
        )
        candidates.append(
            CaseCandidate(
                case=case,
                score=draft.score,
                order=(asset_priority(asset), asset.page_no, asset.asset_id),
            )
        )
    return select_candidates(merge_case_candidates_by_query(candidates), limit, selection_strategy)


def visual_probe_query_from_asset(
    asset: VisualAsset,
    linked_chunks: list[DocumentChunk],
    max_chars: int,
    mode: QueryMode,
    term_df: dict[str, int],
    document_count: int,
    min_query_terms: int,
    max_query_terms: int,
) -> QueryDraft:
    visual_text = meaningful_query_text(asset_text(asset))
    if not visual_text:
        return QueryDraft(query="")
    linked_term_keys = linked_non_visual_term_keys(linked_chunks)
    scored_terms = [
        (term_score(key, term_df=term_df, document_count=document_count), position, term, key)
        for position, term, key in extracted_terms(visual_text)
        if key not in linked_term_keys
    ]
    if mode in {"salient_terms", "question"}:
        selected = select_salient_terms(
            scored_terms,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
    else:
        selected = first_distinct_terms(
            scored_terms,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
    if len(selected) < min_query_terms:
        return QueryDraft(query="")
    selected_by_position = sorted(selected, key=lambda item: item[1])
    query_terms = [term for _, _, term, _ in selected_by_position]
    return QueryDraft(
        query=terms_to_query(query_terms, mode=mode, max_chars=max_chars),
        score=sum(score for score, _, _, _ in selected),
        terms=tuple(query_terms),
    )


def visual_image_probe_cases(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    limit: int,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> list[RetrievalCase]:
    candidates: list[CaseCandidate] = []
    chunks_by_asset = chunks_by_asset_id(chunks)
    for asset in sorted(assets, key=lambda item: (asset_priority(item), item.page_no, item.asset_id)):
        if asset.path is None:
            continue
        draft = query_from_text(
            asset_text(asset),
            max_chars=query_max_chars,
            mode=query_mode,
            term_df=term_df,
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        if not draft.query:
            continue
        linked_chunks = chunks_by_asset.get(asset.asset_id, [])
        case = with_case_metadata(
            RetrievalCase(
                query=draft.query,
                expected_pages=[asset.page_no],
                expected_asset_ids=[asset.asset_id],
            ),
            case_source="visual_image_probe",
            query_mode=query_mode,
            draft=draft,
        )
        case = case.model_copy(
            update={
                "metadata": {
                    **case.metadata,
                    "case_family": "visual",
                    "evidence_family": "visual_image",
                    "modality": "image",
                    "target_vector": "image_dense",
                    "asset_kind": str(asset.kind),
                    "linked_chunk_ids": [chunk.chunk_id for chunk in linked_chunks],
                }
            }
        )
        candidates.append(
            CaseCandidate(
                case=case,
                score=draft.score,
                order=(asset_priority(asset), asset.page_no, asset.asset_id),
            )
        )
    return select_candidates(merge_case_candidates_by_query(candidates), limit, selection_strategy)


def visual_object_probe_cases(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    limit: int,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
    visual_only: bool = True,
) -> list[RetrievalCase]:
    candidates: list[CaseCandidate] = []
    chunks_by_asset = chunks_by_asset_id(chunks)
    for asset in sorted(assets, key=lambda item: (asset_priority(item), item.page_no, item.asset_id)):
        linked_chunks = chunks_by_asset.get(asset.asset_id, [])
        if not linked_chunks:
            continue
        for object_index, visual_object in enumerate(asset_visual_objects(asset), start=1):
            draft = visual_object_probe_query_from_object(
                visual_object,
                linked_chunks,
                max_chars=query_max_chars,
                mode=query_mode,
                term_df=term_df or {},
                document_count=document_count,
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
                visual_only=visual_only,
            )
            if not draft.query:
                continue
            label = object_label(visual_object)
            case = with_case_metadata(
                RetrievalCase(
                    query=draft.query,
                    expected_pages=[asset.page_no],
                    expected_asset_ids=[asset.asset_id],
                ),
                case_source="visual_object_probe",
                query_mode=query_mode,
                draft=draft,
            )
            case = case.model_copy(
                update={
                    "metadata": {
                        **case.metadata,
                        "case_family": "visual",
                        "evidence_family": "visual_object",
                        "modality": "vision_object",
                        "linked_chunk_ids": [chunk.chunk_id for chunk in linked_chunks],
                        "object_label": label,
                        "object_source_key": visual_object.get("source_key"),
                        "object_visual_feature_type": visual_object.get("visual_feature_type"),
                        "object_has_bbox": bool(visual_object.get("bbox")),
                        "object_probe_visual_only": visual_only,
                    }
                }
            )
            candidates.append(
                CaseCandidate(
                    case=case,
                    score=draft.score,
                    order=(asset_priority(asset), asset.page_no, asset.asset_id, object_index),
                )
            )
    return select_candidates(merge_case_candidates_by_query(candidates), limit, selection_strategy)


def visual_object_probe_query_from_object(
    visual_object: dict[str, object],
    linked_chunks: list[DocumentChunk],
    max_chars: int,
    mode: QueryMode,
    term_df: dict[str, int],
    document_count: int,
    min_query_terms: int,
    max_query_terms: int,
    visual_only: bool = True,
) -> QueryDraft:
    object_text = meaningful_query_text(visual_object_query_text(visual_object))
    if not object_text:
        return QueryDraft(query="")
    if not visual_only:
        return query_from_text(
            object_text,
            max_chars=max_chars,
            mode=mode,
            term_df=term_df,
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
    linked_term_keys = linked_non_visual_term_keys(linked_chunks)
    scored_terms = [
        (term_score(key, term_df=term_df, document_count=document_count), position, term, key)
        for position, term, key in extracted_terms(object_text)
        if key not in linked_term_keys
    ]
    selected = (
        select_salient_terms(
            scored_terms,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        if mode in {"salient_terms", "question"}
        else first_distinct_terms(
            scored_terms,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
    )
    if len(selected) < min_query_terms:
        return QueryDraft(query="")
    selected_by_position = sorted(selected, key=lambda item: item[1])
    query_terms = [term for _, _, term, _ in selected_by_position]
    return QueryDraft(
        query=terms_to_query(query_terms, mode=mode, max_chars=max_chars),
        score=sum(score for score, _, _, _ in selected),
        terms=tuple(query_terms),
    )


def asset_visual_objects(asset: VisualAsset) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for key in VISUAL_OBJECT_METADATA_KEYS:
        value = asset.metadata.get(key)
        if isinstance(value, list):
            for item in value:
                normalized = normalize_visual_object(item, source_key=key)
                if normalized:
                    objects.append(normalized)
        elif isinstance(value, dict):
            normalized = normalize_visual_object(value, source_key=key)
            if normalized:
                objects.append(normalized)
            else:
                for label, details in value.items():
                    normalized = normalize_visual_object_mapping(label, details, source_key=key)
                    if normalized:
                        objects.append(normalized)
        else:
            normalized = normalize_visual_object(value, source_key=key)
            if normalized:
                objects.append(normalized)
    for key in VISUAL_FEATURE_METADATA_KEYS:
        value = asset.metadata.get(key)
        if isinstance(value, list):
            values = value
        elif value:
            values = [value]
        else:
            values = []
        for item in values:
            normalized = normalize_visual_object(item, source_key=key)
            if normalized:
                normalized["visual_feature_type"] = key.removesuffix("s")
                objects.append(normalized)
    return dedupe_visual_objects(objects)


def normalize_visual_object(value: object, source_key: str) -> dict[str, object] | None:
    if isinstance(value, str):
        label = value.strip()
        return {"label": label, "source_key": source_key} if label else None
    if not isinstance(value, dict):
        return None
    label = first_object_string(value, ["label", "name", "title", "object", "type", "category"])
    if not label:
        return None
    normalized: dict[str, object] = {"label": label, "source_key": source_key}
    attributes = object_text_items(
        value.get("attributes") or value.get("features") or value.get("descriptors")
    )
    description = first_object_string(value, ["description", "summary", "text"])
    if description and description not in attributes:
        attributes.append(description)
    location = first_object_string(value, ["location", "position", "region"])
    if location:
        normalized["location"] = location
    if attributes:
        normalized["attributes"] = attributes[:6]
    bbox = value.get("bbox") or value.get("box") or value.get("bounding_box") or value.get("boundingBox")
    if bbox:
        normalized["bbox"] = bbox
    bbox_region = first_object_string(value, ["bbox_region", "spatial_region"])
    if not bbox_region and bbox:
        bbox_region = bbox_region_from_bbox(bbox) or ""
    if bbox_region:
        normalized["bbox_region"] = bbox_region
    return normalized


def normalize_visual_object_mapping(label: object, details: object, source_key: str) -> dict[str, object] | None:
    label_text = str(label).strip()
    if not label_text:
        return None
    if isinstance(details, dict):
        return normalize_visual_object({**details, "label": label_text}, source_key=source_key)
    if isinstance(details, str):
        return {
            "label": label_text,
            "description": details.strip(),
            "source_key": source_key,
        }
    return {"label": label_text, "source_key": source_key}


def dedupe_visual_objects(objects: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    seen = set()
    for item in objects:
        key = (
            str(item.get("label") or "").casefold(),
            str(item.get("location") or "").casefold(),
            str(item.get("bbox_region") or "").casefold(),
            str(item.get("visual_feature_type") or "").casefold(),
            " ".join(object_text_items(item.get("attributes"))).casefold(),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return selected


def visual_object_query_text(visual_object: dict[str, object]) -> str:
    parts = [object_label(visual_object)]
    parts.extend(object_text_items(visual_object.get("attributes")))
    description = object_text_value(visual_object.get("description"))
    if description:
        parts.append(description)
    location = object_text_value(visual_object.get("location"))
    if location:
        parts.append(location)
    bbox_region = object_text_value(visual_object.get("bbox_region"))
    if bbox_region and not location:
        parts.append(bbox_region)
    return " ".join(part for part in parts if part)


def object_label(visual_object: dict[str, object]) -> str:
    return object_text_value(visual_object.get("label"))


def first_object_string(payload: dict, keys: list[str]) -> str:
    for key in keys:
        value = object_text_value(payload.get(key))
        if value:
            return value
    return ""


def object_text_items(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, dict):
                text = first_object_string(item, ["label", "name", "title", "description", "summary", "text"])
                if text:
                    values.append(text)
            else:
                values.append(str(item))
    else:
        values = [str(value)]
    return dedupe_text_values(values)


def object_text_value(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def dedupe_text_values(values: list[str]) -> list[str]:
    selected = []
    seen = set()
    for value in values:
        text = object_text_value(value)
        key = text.casefold()
        if not text or key in seen:
            continue
        selected.append(text)
        seen.add(key)
    return selected


def first_distinct_terms(
    scored_terms: list[tuple[float, int, str, str]],
    min_query_terms: int,
    max_query_terms: int,
) -> list[tuple[float, int, str, str]]:
    selected = []
    selected_keys: set[str] = set()
    for item in sorted(scored_terms, key=lambda item: item[1]):
        _, _, _, key = item
        if len(selected) >= max_query_terms:
            break
        if any(key in selected_key or selected_key in key for selected_key in selected_keys):
            continue
        selected.append(item)
        selected_keys.add(key)
    if len(selected) < min_query_terms:
        return []
    return selected


def chunks_by_asset_id(chunks: list[DocumentChunk]) -> dict[str, list[DocumentChunk]]:
    indexed: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        for asset_id in chunk_asset_ids(chunk):
            indexed.setdefault(asset_id, []).append(chunk)
    return indexed


def linked_non_visual_term_keys(chunks: list[DocumentChunk]) -> set[str]:
    keys: set[str] = set()
    for chunk in chunks:
        text = non_visual_chunk_text(chunk)
        keys.update(key for _, _, key in extracted_terms(meaningful_query_text(text)))
    return keys


def non_visual_chunk_text(chunk: DocumentChunk) -> str:
    if chunk.metadata.get("chunking_strategy") == "visual_asset_text":
        return ""
    kept_lines: list[str] = []
    skipping_generated_visual = False
    for line in chunk.text.splitlines():
        stripped = line.strip().casefold()
        if stripped.startswith("visual context:") or stripped.startswith(_GENERATED_VISUAL_BLOCK_PREFIXES):
            skipping_generated_visual = True
            continue
        if skipping_generated_visual and stripped.startswith(_NON_VISUAL_BLOCK_PREFIXES):
            skipping_generated_visual = False
        if not skipping_generated_visual:
            kept_lines.append(line)
    return "\n".join(kept_lines)


def triple_cases(
    triples: list[GraphTriple],
    chunks: list[DocumentChunk],
    limit: int,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> list[RetrievalCase]:
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    chunk_id_by_alias = chunk_id_alias_map(chunks)
    chunk_ids_by_asset = chunk_ids_by_asset_id(chunks)
    candidates: list[CaseCandidate] = []
    for triple in sorted(triples, key=lambda item: (item.chunk_id, item.triple_id)):
        draft = query_from_text(
            triple_query_text(triple),
            max_chars=query_max_chars,
            mode=query_mode,
            term_df=term_df,
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        query = draft.query
        if not query:
            continue
        expected_chunk_ids = triple_resolved_chunk_ids(
            triple,
            chunk_ids,
            chunk_id_by_alias,
            chunk_ids_by_asset,
        )
        case = with_case_metadata(
            RetrievalCase(
                query=query,
                expected_chunk_ids=expected_chunk_ids or [triple.chunk_id],
                expected_asset_ids=sorted(triple_asset_ids(triple)),
                expected_triple_ids=[triple.triple_id],
                graph_expand=True,
            ),
            case_source="triple",
            query_mode=query_mode,
            draft=draft,
        )
        candidates.append(
            CaseCandidate(
                case=case,
                score=draft.score,
                order=(triple.chunk_id, triple.triple_id),
            )
        )
    return select_candidates(merge_case_candidates_by_query(candidates), limit, selection_strategy)


def query_from_text(
    text: str,
    max_chars: int = 120,
    mode: QueryMode = "snippet",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> QueryDraft:
    normalized = meaningful_query_text(text)
    if not normalized:
        return QueryDraft(query="")
    if mode in {"salient_terms", "question"}:
        return salient_terms_query(
            normalized,
            term_df=term_df or {},
            document_count=document_count,
            max_chars=max_chars,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
            question=mode == "question",
        )
    return QueryDraft(query=trim_query(normalized, max_chars=max_chars))


def normalize_query_text(text: str) -> str:
    text = _MARKDOWN_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_placeholder_text(text: str) -> bool:
    return not meaningful_query_text(text) and any(pattern in text for pattern in _PLACEHOLDER_PATTERNS)


def meaningful_query_text(text: str) -> str:
    lines = []
    for line in text.splitlines() or [text]:
        if any(pattern in line for pattern in _PLACEHOLDER_PATTERNS):
            continue
        lines.append(line)
    return normalize_query_text(" ".join(lines))


def salient_terms_query(
    text: str,
    term_df: dict[str, int],
    document_count: int,
    max_chars: int,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
    question: bool = False,
) -> QueryDraft:
    terms = extracted_terms(text)
    if not terms:
        return QueryDraft(query="")
    scored_terms = [
        (term_score(key, term_df=term_df, document_count=document_count), position, term, key)
        for position, term, key in terms
    ]
    selected = select_salient_terms(scored_terms, min_query_terms=min_query_terms, max_query_terms=max_query_terms)
    if len(selected) < min_query_terms:
        return QueryDraft(query="")
    selected_by_position = sorted(selected, key=lambda item: item[1])
    query_terms = [term for _, _, term, _ in selected_by_position]
    query = terms_to_query(
        query_terms,
        mode="question" if question else "salient_terms",
        max_chars=max_chars,
    )
    return QueryDraft(
        query=query,
        score=sum(score for score, _, _, _ in selected),
        terms=tuple(query_terms),
    )


def terms_to_query(
    terms: list[str],
    mode: QueryMode,
    max_chars: int,
) -> str:
    term_text = " ".join(terms)
    if mode != "question":
        return trim_query(term_text, max_chars=max_chars)
    return trim_query(question_from_terms(term_text), max_chars=max_chars)


def question_from_terms(term_text: str) -> str:
    if contains_hangul(term_text):
        return f"어떤 근거가 {term_text} 내용을 설명하는가"
    return f"Which evidence explains {term_text}?"


def contains_hangul(text: str) -> bool:
    return any("\uac00" <= character <= "\ud7a3" for character in text)


def extracted_terms(text: str) -> list[tuple[int, str, str]]:
    terms = []
    seen_keys = set()
    for match in _TERM_RE.finditer(text):
        term = match.group(0).strip(".,;:!?/\\-+")
        key = normalize_term(term)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        terms.append((match.start(), term, key))
    return terms


def normalize_term(term: str) -> str:
    key = term.lower().strip()
    if len(key) < 2 or key in _STOP_TERMS:
        return ""
    if is_identifier_like(key):
        return ""
    if key.isdigit() and len(key) != 4:
        return ""
    return key


def is_identifier_like(value: str) -> bool:
    return bool(_ID_LIKE_RE.fullmatch(value)) or (
        len(value) >= 12 and any(character.isdigit() for character in value)
    )


def term_document_frequencies(texts: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        normalized = meaningful_query_text(text)
        if not normalized:
            continue
        counts.update({key for _, _, key in extracted_terms(normalized)})
    return dict(counts)


def term_score(term: str, term_df: dict[str, int], document_count: int) -> float:
    df = max(1, term_df.get(term, 1))
    idf = math.log((document_count + 1) / df) + 1 if document_count else 1.0
    length_score = min(len(term), 16) / 8
    digit_penalty = 0.75 if term.replace(".", "").replace(",", "").isdigit() else 1.0
    return idf * length_score * digit_penalty


def select_salient_terms(
    scored_terms: list[tuple[float, int, str, str]],
    min_query_terms: int,
    max_query_terms: int,
) -> list[tuple[float, int, str, str]]:
    selected: list[tuple[float, int, str, str]] = []
    selected_keys: set[str] = set()
    for score, position, term, key in sorted(scored_terms, key=lambda item: (-item[0], item[1])):
        if len(selected) >= max_query_terms:
            break
        if any(key in selected_key or selected_key in key for selected_key in selected_keys):
            continue
        selected.append((score, position, term, key))
        selected_keys.add(key)
    if len(selected) >= min_query_terms:
        return selected
    for score, position, term, key in sorted(scored_terms, key=lambda item: item[1]):
        if len(selected) >= min_query_terms:
            break
        if key in selected_keys:
            continue
        selected.append((score, position, term, key))
        selected_keys.add(key)
    return selected


def trim_query(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() or text[:max_chars].strip()


def case_source_texts(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
) -> list[str]:
    texts = [chunk.text for chunk in chunks]
    texts.extend(asset_text(asset) for asset in assets)
    texts.extend(triple_query_text(triple) for triple in triples)
    return [text for text in texts if meaningful_query_text(text)]


def triple_query_text(triple: GraphTriple) -> str:
    components = [triple.subject, triple.predicate.replace("_", " "), triple.object]
    return " ".join(component for component in components if meaningful_component(component))


def meaningful_component(value: str) -> bool:
    normalized = normalize_query_text(value)
    if not normalized:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", normalized)
    if compact and is_identifier_like(compact):
        return False
    terms = [key for _, _, key in extracted_terms(normalized)]
    return any(term for term in terms if term not in _STOP_TERMS)


def select_candidates(
    candidates: list[CaseCandidate],
    limit: int,
    strategy: SelectionStrategy,
) -> list[RetrievalCase]:
    ordered = (
        sorted(candidates, key=lambda item: (-item.score, item.order))
        if strategy == "salience"
        else sorted(candidates, key=lambda item: item.order)
    )
    return [candidate.case for candidate in ordered[:limit]]


def merge_case_candidates_by_query(candidates: list[CaseCandidate]) -> list[CaseCandidate]:
    merged: dict[str, CaseCandidate] = {}
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = normalize_query_text(candidate.case.query).lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            counts[key] = 1
            continue
        counts[key] += 1
        merged_case = merge_retrieval_cases(existing.case, candidate.case, counts[key])
        merged[key] = CaseCandidate(
            case=merged_case,
            score=max(existing.score, candidate.score),
            order=min(existing.order, candidate.order),
        )
    return list(merged.values())


def merge_retrieval_cases(
    left: RetrievalCase,
    right: RetrievalCase,
    merged_count: int,
) -> RetrievalCase:
    metadata = merge_case_metadata(left.metadata, right.metadata)
    metadata["merged_case_count"] = merged_count
    return left.model_copy(
        update={
            "expected_pages": merge_stable_values(left.expected_pages, right.expected_pages),
            "expected_chunk_ids": merge_stable_values(left.expected_chunk_ids, right.expected_chunk_ids),
            "expected_asset_ids": merge_stable_values(left.expected_asset_ids, right.expected_asset_ids),
            "expected_triple_ids": merge_stable_values(left.expected_triple_ids, right.expected_triple_ids),
            "graph_expand": left.graph_expand or right.graph_expand,
            "metadata": metadata,
        }
    )


def merge_case_metadata(
    left: dict,
    right: dict,
) -> dict:
    metadata = dict(left)
    for key, value in right.items():
        if key == "merged_case_count":
            continue
        if key not in metadata:
            metadata[key] = value
            continue
        if key in CASE_GROUP_METADATA_KEYS or key in {
            "linked_chunk_ids",
            "query_terms",
        }:
            metadata[key] = merge_stable_values(
                metadata_values(metadata[key]),
                metadata_values(value),
            )
        elif key == "selection_score":
            metadata[key] = max(float(metadata[key]), float(value))
        elif key == "object_probe_visual_only":
            metadata[key] = bool(metadata[key]) and bool(value)
    return metadata


def metadata_values(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def merge_stable_values(left: list, right: list) -> list:
    seen = set()
    merged = []
    for value in [*left, *right]:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def dedupe_page_candidates(
    candidates: list[CaseCandidate],
    strategy: SelectionStrategy,
) -> list[CaseCandidate]:
    ordered = (
        sorted(candidates, key=lambda item: (-item.score, item.order))
        if strategy == "salience"
        else sorted(candidates, key=lambda item: item.order)
    )
    seen_pages = set()
    selected = []
    for candidate in ordered:
        page = candidate.case.expected_pages[0] if candidate.case.expected_pages else None
        if page in seen_pages:
            continue
        seen_pages.add(page)
        selected.append(candidate)
    return selected


def with_case_metadata(
    case: RetrievalCase,
    case_source: str,
    query_mode: QueryMode,
    draft: QueryDraft,
) -> RetrievalCase:
    metadata = {
        **case.metadata,
        "case_source": case_source,
        "query_mode": query_mode,
    }
    if draft.terms:
        metadata["query_terms"] = list(draft.terms)
        metadata["selection_score"] = round(draft.score, 6)
    return case.model_copy(update={"metadata": metadata})


def dedupe_cases_by_query(cases: list[RetrievalCase]) -> list[RetrievalCase]:
    selected_by_query: dict[str, RetrievalCase] = {}
    counts: dict[str, int] = {}
    for case in cases:
        key = normalize_query_text(case.query).lower()
        existing = selected_by_query.get(key)
        if existing is None:
            selected_by_query[key] = case
            counts[key] = 1
            continue
        counts[key] += 1
        selected_by_query[key] = merge_retrieval_cases(existing, case, counts[key])
    return list(selected_by_query.values())


def validate_casegen_options(
    query_mode: str,
    selection_strategy: str,
    min_query_terms: int,
    max_query_terms: int,
) -> None:
    if query_mode not in {"snippet", "salient_terms", "question"}:
        raise ValueError("query_mode must be one of: snippet, salient_terms, question")
    if selection_strategy not in {"document_order", "salience"}:
        raise ValueError("selection_strategy must be one of: document_order, salience")
    if min_query_terms < 1:
        raise ValueError("min_query_terms must be at least 1")
    if max_query_terms < min_query_terms:
        raise ValueError("max_query_terms must be greater than or equal to min_query_terms")


def asset_priority(asset: VisualAsset) -> int:
    return {
        "map": 0,
        "table": 1,
        "chart": 2,
        "figure": 3,
        "page_image": 4,
    }.get(str(asset.kind), 5)
