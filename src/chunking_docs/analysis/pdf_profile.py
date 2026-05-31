from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import fitz

from chunking_docs.models import PageProfile, TextQuality


CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
HANGUL_RE = re.compile(r"[가-힣]")


def classify_text_quality(text: str) -> TextQuality:
    stripped = text.strip()
    if not stripped:
        return TextQuality.EMPTY

    control_count = len(CONTROL_CHAR_RE.findall(stripped))
    hangul_count = len(HANGUL_RE.findall(stripped))
    visible_count = max(len(stripped), 1)

    control_ratio = control_count / visible_count
    hangul_ratio = hangul_count / visible_count
    if control_ratio > 0.02 or hangul_ratio < 0.05:
        return TextQuality.DEGRADED
    return TextQuality.GOOD


def profile_pdf(pdf_path: Path, doc_id: str) -> list[PageProfile]:
    profiles: list[PageProfile] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            blocks = page.get_text("dict").get("blocks", [])
            text_blocks = sum(1 for block in blocks if block.get("type") == 0)
            image_blocks = sum(1 for block in blocks if block.get("type") == 1)
            profiles.append(
                PageProfile(
                    doc_id=doc_id,
                    page_no=index + 1,
                    width=round(page.rect.width, 2),
                    height=round(page.rect.height, 2),
                    char_count=len(text.strip()),
                    line_count=len([line for line in text.splitlines() if line.strip()]),
                    text_block_count=text_blocks,
                    image_block_count=image_blocks,
                    embedded_image_count=len(page.get_images(full=True)),
                    drawing_count=len(page.get_drawings()),
                    text_quality=classify_text_quality(text),
                    sample=" ".join(text.split())[:240],
                )
            )
    return profiles


def summarize_profiles(profiles: list[PageProfile]) -> dict:
    chars = [profile.char_count for profile in profiles]
    return {
        "page_count": len(profiles),
        "chars_total": sum(chars),
        "chars_mean": round(statistics.mean(chars), 1) if chars else 0,
        "chars_median": statistics.median(chars) if chars else 0,
        "good_text_pages": [p.page_no for p in profiles if p.text_quality == TextQuality.GOOD],
        "degraded_text_pages": [p.page_no for p in profiles if p.text_quality == TextQuality.DEGRADED],
        "empty_text_pages": [p.page_no for p in profiles if p.text_quality == TextQuality.EMPTY],
        "visual_heavy_pages": [
            p.page_no
            for p in profiles
            if p.embedded_image_count + p.image_block_count >= 2 or p.drawing_count >= 20
        ],
    }


def write_profile_outputs(profiles: list[PageProfile], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "page_profiles.jsonl").write_text(
        "\n".join(profile.model_dump_json() for profile in profiles),
        encoding="utf-8",
    )
    (output_dir / "profile_summary.json").write_text(
        json.dumps(summarize_profiles(profiles), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
