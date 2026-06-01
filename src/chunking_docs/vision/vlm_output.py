from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class ParsedVLMOutput(BaseModel):
    caption: str | None = None
    summary: str
    triples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


DERIVED_VISUAL_TRIPLE_LIMIT = 12
VISUAL_OBJECT_KEYS = ["objects", "detected_objects", "visual_objects", "detections", "regions", "areas"]


def parse_vlm_output(text: str) -> ParsedVLMOutput:
    payload = extract_json_payload(text)
    repaired = False
    if payload is None:
        payload = repair_partial_json_object(text)
        repaired = payload is not None
        if payload is None:
            return ParsedVLMOutput(
                summary=text.strip(),
                metadata={"vlm_parse_status": "raw_text"},
            )
    if isinstance(payload, list):
        return ParsedVLMOutput(
            summary=json.dumps(payload, ensure_ascii=False),
            triples=normalize_triples(payload),
            metadata={"vlm_parse_status": "json_list"},
        )
    if not isinstance(payload, dict):
        return ParsedVLMOutput(
            summary=str(payload),
            metadata={"vlm_parse_status": "json_scalar"},
        )

    caption = first_string(payload, ["title", "caption", "name"])
    triples = visual_triples_from_payload(payload)
    metadata = object_metadata(payload, repaired=repaired)
    return ParsedVLMOutput(
        caption=caption,
        summary=summary_from_payload(payload),
        triples=triples,
        metadata=metadata,
    )


def extract_json_payload(text: str) -> Any | None:
    stripped = text.strip()
    for candidate in json_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def json_candidates(text: str) -> list[str]:
    candidates = []
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    if text.startswith("{") or text.startswith("["):
        candidates.append(text)
    object_candidate = bounded_json(text, "{", "}")
    if object_candidate:
        candidates.append(object_candidate)
    array_candidate = bounded_json(text, "[", "]")
    if array_candidate:
        candidates.append(array_candidate)
    return candidates


def bounded_json(text: str, start_char: str, end_char: str) -> str | None:
    start = text.find(start_char)
    end = text.rfind(end_char)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def repair_partial_json_object(text: str) -> dict[str, Any] | None:
    stripped = strip_markdown_json_fence(text)
    if "{" not in stripped:
        return None

    payload: dict[str, Any] = {}
    for key in ["page_type", "title", "summary"]:
        value = extract_json_string_field(stripped, key)
        if value:
            payload[key] = value
    for key, limit in [("key_points", 5), ("visual_elements", 6), ("entities", 8)]:
        values = extract_json_string_array_field(stripped, key, limit=limit)
        if values:
            payload[key] = values

    return payload if has_structured_fields(payload) else None


def strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^\s*```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```\s*$", "", stripped)
    return stripped.strip()


def extract_json_string_field(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if not match:
        return None
    return decode_json_string(match.group(1))


def extract_json_string_array_field(text: str, key: str, limit: int) -> list[str]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if not match:
        return []
    open_index = match.end() - 1
    close_index = matching_bracket_index(text, open_index)
    end_index = close_index if close_index is not None else len(text)
    values = [
        value
        for value in (decode_json_string(item) for item in re.findall(r'"((?:\\.|[^"\\])*)"', text[match.end() : end_index]))
        if value
    ]
    return dedupe_preserve_order(values, limit=limit)


def matching_bracket_index(text: str, open_index: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def decode_json_string(value: str) -> str | None:
    try:
        decoded = json.loads(f'"{value}"')
    except json.JSONDecodeError:
        decoded = value
    decoded = str(decoded).strip()
    return decoded or None


def dedupe_preserve_order(values: list[str], limit: int) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= limit:
            break
    return deduped


def has_structured_fields(payload: dict[str, Any]) -> bool:
    return any(payload.get(key) for key in ["title", "summary", "key_points", "visual_elements", "entities"])


def summary_from_payload(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    title = first_string(payload, ["title", "caption", "name"])
    if title:
        parts.append(title)
    for key in ["summary", "vlm_summary", "description"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            break
    parts.extend(normalize_text_items(payload.get("key_points"), limit=5))
    parts.extend(normalize_text_items(payload.get("visual_elements"), limit=6))
    parts.extend(normalize_text_items(payload.get("entities"), limit=8))
    parts.extend(object_summary_lines(visual_objects_from_payload(payload)))
    parts.extend(triple_summary_lines(normalize_triples(payload.get("triples") or payload.get("relationships") or [])))
    return "\n".join(parts).strip() or json.dumps(payload, ensure_ascii=False)


def first_string(payload: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_text_items(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        return dedupe_preserve_order([value], limit=limit)
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if isinstance(item, str):
            items.append(item)
        elif isinstance(item, dict):
            label = first_string(item, ["label", "name", "title", "entity", "object", "type", "category"])
            description = first_string(item, ["description", "summary", "text"])
            if label and description and description != label:
                items.append(f"{label}: {description}")
            elif label:
                items.append(label)
            elif description:
                items.append(description)
        else:
            item_text = str(item).strip()
            if item_text:
                items.append(item_text)
    return dedupe_preserve_order(items, limit=limit)


def visual_objects_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    objects = []
    for key in VISUAL_OBJECT_KEYS:
        objects.extend(normalize_visual_objects(payload.get(key), limit=8, source_key=key))
    return dedupe_visual_objects(objects, limit=8)


def normalize_visual_objects(value: Any, limit: int, source_key: str | None = None) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict):
        if first_string(value, ["label", "name", "title", "object", "type", "category"]):
            value = [value]
        else:
            value = object_mapping_items(value)
    if not isinstance(value, list):
        return []

    objects: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            label = item.strip()
            if label:
                normalized = {"label": label}
                if source_key:
                    normalized["source_key"] = source_key
                objects.append(normalized)
        elif isinstance(item, dict):
            label = first_string(item, ["label", "name", "title", "object", "type", "category"])
            if not label:
                continue
            normalized: dict[str, Any] = {"label": label}
            if source_key:
                normalized["source_key"] = source_key
            attributes = normalize_text_items(
                item.get("attributes") or item.get("features") or item.get("descriptors"),
                limit=6,
            )
            description = first_string(item, ["description", "summary", "text"])
            if description and description not in attributes:
                attributes.append(description)
                normalized["description"] = description
            if attributes:
                normalized["attributes"] = attributes[:6]
            bbox = normalize_bbox(first_present(item, ["bbox", "box", "bounding_box", "boundingBox", "bounds"]))
            if bbox is not None:
                normalized["bbox"] = bbox
            location = normalize_location(first_present(item, ["location", "position", "region"]))
            if location:
                normalized["location"] = location
            confidence = normalize_confidence(first_present(item, ["confidence", "score"]))
            if confidence is not None:
                normalized["confidence"] = confidence
            objects.append(normalized)
        if len(objects) >= limit:
            break
    return objects


def first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def object_mapping_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for label, details in value.items():
        label_text = str(label).strip()
        if not label_text:
            continue
        if isinstance(details, dict):
            item = {**details}
            item.setdefault("label", label_text)
        elif isinstance(details, list) and len(details) == 4:
            item = {"label": label_text, "bbox": details}
        elif isinstance(details, str):
            item = {"label": label_text, "description": details}
        else:
            item = {"label": label_text}
        items.append(item)
    return items


def dedupe_visual_objects(objects: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for item in objects:
        key = (
            str(item.get("label", "")).casefold(),
            tuple(str(value).casefold() for value in item.get("attributes", [])),
            str(item.get("description", "")).casefold(),
            str(item.get("location", "")).casefold(),
            tuple(item.get("bbox", [])),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def normalize_bbox(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        value = bbox_values_from_mapping(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, str):
        value = [part.strip() for part in re.split(r"[, ]+", value) if part.strip()]
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [round(float(item), 6) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_values_from_mapping(value: dict[str, Any]) -> list[Any] | None:
    for keys in [
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ]:
        if all(key in value for key in keys):
            return [value[key] for key in keys]
    if all(key in value for key in ["x", "y", "width", "height"]):
        try:
            x = float(value["x"])
            y = float(value["y"])
            width = float(value["width"])
            height = float(value["height"])
        except (TypeError, ValueError):
            return None
        return [x, y, x + width, y + height]
    return None


def normalize_location(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = str(item).strip()
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "; ".join(parts) or None
    text = str(value).strip()
    return text or None


def normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            value = value[:-1].strip()
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def normalize_triples(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    triples = []
    for item in value:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        object_ = str(item.get("object", item.get("object_", ""))).strip()
        if not subject or not predicate or not object_:
            continue
        triples.append(
            {
                **{key: val for key, val in item.items() if key not in {"object_"}},
                "subject": subject,
                "predicate": predicate,
                "object": object_,
            }
        )
    return triples


def visual_triples_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = normalize_triples(payload.get("triples") or payload.get("relationships") or [])
    return merge_triples(explicit, derived_visual_triples(payload, explicit))


def derived_visual_triples(payload: dict[str, Any], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subject = first_string(payload, ["title", "caption", "name"]) or "visual_asset"
    triples = []
    seen = {triple_key(triple) for triple in existing}

    for entity in normalize_text_items(payload.get("entities"), limit=8):
        add_derived_triple(
            triples,
            seen,
            subject=subject,
            predicate="mentions_entity",
            object_=entity,
            source_field="entities",
            confidence=0.62,
        )
    for element in normalize_text_items(payload.get("visual_elements"), limit=6):
        add_derived_triple(
            triples,
            seen,
            subject=subject,
            predicate="contains_visual_element",
            object_=element,
            source_field="visual_elements",
            confidence=0.58,
        )
    for visual_object in visual_objects_from_payload(payload):
        add_derived_triple(
            triples,
            seen,
            subject=subject,
            predicate="contains_object",
            object_=str(visual_object["label"]),
            source_field="objects",
            confidence=float(visual_object.get("confidence", 0.6)),
            extra={
                key: value
                for key, value in visual_object.items()
                if key in {"attributes", "bbox", "description", "location", "source_key"}
            },
        )
    return triples


def add_derived_triple(
    triples: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    subject: str,
    predicate: str,
    object_: str,
    source_field: str,
    confidence: float,
    extra: dict[str, Any] | None = None,
) -> None:
    if len(triples) >= DERIVED_VISUAL_TRIPLE_LIMIT:
        return
    object_text = object_.strip()
    if not object_text:
        return
    triple = {
        "subject": subject,
        "predicate": predicate,
        "object": object_text,
        "source_field": source_field,
        "derived_from_vlm_field": True,
        "confidence": normalize_confidence(confidence) or confidence,
    }
    if extra:
        triple.update(extra)
    key = triple_key(triple)
    if key in seen:
        return
    seen.add(key)
    triples.append(triple)


def merge_triples(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for triple in [*left, *right]:
        key = triple_key(triple)
        if key in seen:
            continue
        seen.add(key)
        merged.append(triple)
    return merged


def triple_key(triple: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(triple.get("subject", "")).casefold(),
        str(triple.get("predicate", "")).casefold(),
        str(triple.get("object", "")).casefold(),
    )


def triple_summary_lines(triples: list[dict[str, Any]]) -> list[str]:
    lines = []
    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("predicate", "")).strip()
        object_ = str(triple.get("object", "")).strip()
        if subject and predicate and object_:
            lines.append(f"{subject} {predicate} {object_}")
    return lines


def object_summary_lines(objects: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in objects:
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        attributes = [str(value).strip() for value in item.get("attributes", []) if str(value).strip()]
        location = str(item.get("location", "")).strip()
        if location and location not in attributes:
            attributes.append(location)
        if attributes:
            lines.append(f"{label}: {', '.join(attributes)}")
        else:
            lines.append(label)
    return lines


def object_metadata(payload: dict[str, Any], repaired: bool = False) -> dict[str, Any]:
    metadata = {
        "vlm_parse_status": "json_repaired" if repaired else "json_object",
    }
    page_type = first_string(payload, ["page_type"])
    if page_type:
        metadata["page_type"] = page_type
    entities = normalize_text_items(payload.get("entities"), limit=8)
    if entities:
        metadata["entities"] = entities
    visual_elements = normalize_text_items(payload.get("visual_elements"), limit=6)
    if visual_elements:
        metadata["visual_elements"] = visual_elements
    objects = visual_objects_from_payload(payload)
    if objects:
        metadata["objects"] = objects
    metadata["entity_count"] = len(entities)
    metadata["visual_element_count"] = len(visual_elements)
    metadata["object_count"] = len(objects)
    metadata["object_bbox_count"] = sum(1 for item in objects if item.get("bbox"))
    metadata["explicit_triple_count"] = len(
        normalize_triples(payload.get("triples") or payload.get("relationships") or [])
    )
    metadata["derived_triple_count"] = len(
        [triple for triple in visual_triples_from_payload(payload) if triple.get("derived_from_vlm_field")]
    )
    if repaired:
        metadata["vlm_parse_repaired"] = True
    return metadata
