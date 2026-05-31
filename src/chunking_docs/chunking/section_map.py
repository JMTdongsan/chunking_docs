from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunking_docs.models import SectionPath


@dataclass(frozen=True)
class SectionRange:
    page_start: int
    page_end: int
    section: SectionPath


def section_for_page(page_no: int, ranges: list[SectionRange] | None = None) -> SectionPath:
    ranges = ranges or []
    matches = [item for item in ranges if item.page_start <= page_no <= item.page_end]
    if not matches:
        return SectionPath()
    return max(matches, key=lambda item: item.page_start).section


def load_section_ranges(path: Path | None) -> list[SectionRange]:
    if path is None:
        return []
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["sections"] if isinstance(payload, dict) and "sections" in payload else payload
    return [section_range_from_mapping(record) for record in records]


def section_range_from_mapping(record: dict[str, Any]) -> SectionRange:
    section_payload = record.get("section")
    if section_payload is None:
        section_payload = {
            key: record.get(key)
            for key in ["chapter", "section", "subsection", "issue"]
            if record.get(key) is not None
        }
    return SectionRange(
        page_start=int(record["page_start"]),
        page_end=int(record["page_end"]),
        section=SectionPath.model_validate(section_payload),
    )
