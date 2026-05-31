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
    triples = normalize_triples(payload.get("triples") or payload.get("relationships") or [])
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
    key_points = payload.get("key_points")
    if isinstance(key_points, list):
        parts.extend(str(item).strip() for item in key_points if str(item).strip())
    visual_elements = payload.get("visual_elements")
    if isinstance(visual_elements, str) and visual_elements.strip():
        parts.append(visual_elements.strip())
    elif isinstance(visual_elements, list):
        parts.extend(str(item).strip() for item in visual_elements if str(item).strip())
    entities = payload.get("entities")
    if isinstance(entities, str) and entities.strip():
        parts.append(entities.strip())
    elif isinstance(entities, list):
        parts.extend(str(item).strip() for item in entities if str(item).strip())
    parts.extend(triple_summary_lines(normalize_triples(payload.get("triples") or payload.get("relationships") or [])))
    return "\n".join(parts).strip() or json.dumps(payload, ensure_ascii=False)


def first_string(payload: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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


def triple_summary_lines(triples: list[dict[str, Any]]) -> list[str]:
    lines = []
    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("predicate", "")).strip()
        object_ = str(triple.get("object", "")).strip()
        if subject and predicate and object_:
            lines.append(f"{subject} {predicate} {object_}")
    return lines


def selected_metadata(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def object_metadata(payload: dict[str, Any], repaired: bool = False) -> dict[str, Any]:
    metadata = {
        "vlm_parse_status": "json_repaired" if repaired else "json_object",
        **selected_metadata(payload, ["page_type", "entities", "visual_elements"]),
    }
    if repaired:
        metadata["vlm_parse_repaired"] = True
    return metadata
