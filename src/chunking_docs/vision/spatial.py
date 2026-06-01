from __future__ import annotations

import re
from typing import Any


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


def bbox_region_from_bbox(value: Any) -> str | None:
    bbox = normalize_bbox(value)
    if bbox is None or not normalized_bbox_bounds(bbox):
        return None
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    horizontal = axis_region(x_center, low="left", middle="center", high="right")
    vertical = axis_region(y_center, low="upper", middle="middle", high="lower")
    if horizontal == "center" and vertical == "middle":
        return "center"
    return f"{vertical} {horizontal}"


def normalized_bbox_bounds(bbox: list[float]) -> bool:
    x0, y0, x1, y1 = bbox
    return all(0.0 <= value <= 1.0 for value in bbox) and x0 <= x1 and y0 <= y1


def axis_region(value: float, low: str, middle: str, high: str) -> str:
    if value < 1 / 3:
        return low
    if value > 2 / 3:
        return high
    return middle


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
