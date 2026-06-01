from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import fitz

from chunking_docs.models import PageProfile, TextQuality


CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
MIN_READABLE_CHAR_RATIO = 0.2
MIN_READABLE_TEXT_LENGTH = 40


def classify_text_quality(text: str) -> TextQuality:
    return text_quality_analysis(text)["text_quality"]


def text_quality_analysis(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {
            "text_quality": TextQuality.EMPTY,
            "control_char_count": 0,
            "control_char_ratio": 0.0,
            "letter_or_number_ratio": 0.0,
            "cjk_char_ratio": 0.0,
            "text_quality_reasons": ["empty_text"],
        }

    control_count = sum(1 for char in stripped if is_control_character(char))
    cjk_count = len(CJK_RE.findall(stripped))
    letter_or_number_count = sum(1 for char in stripped if char.isalpha() or char.isdigit())
    visible_count = max(len(stripped), 1)

    control_ratio = control_count / visible_count
    letter_or_number_ratio = letter_or_number_count / visible_count
    reasons = []
    if control_ratio > 0.02:
        reasons.append("high_control_char_ratio")
    if len(stripped) >= MIN_READABLE_TEXT_LENGTH and letter_or_number_ratio < MIN_READABLE_CHAR_RATIO:
        reasons.append("low_letter_or_number_ratio")
    return {
        "text_quality": TextQuality.DEGRADED if reasons else TextQuality.GOOD,
        "control_char_count": control_count,
        "control_char_ratio": round(control_ratio, 6),
        "letter_or_number_ratio": round(letter_or_number_ratio, 6),
        "cjk_char_ratio": round(cjk_count / visible_count, 6),
        "text_quality_reasons": reasons,
    }


def is_control_character(char: str) -> bool:
    return unicodedata.category(char).startswith("C") and not char.isspace()


def profile_pdf(pdf_path: Path, doc_id: str) -> list[PageProfile]:
    profiles: list[PageProfile] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            quality = text_quality_analysis(text)
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
                    text_quality=quality["text_quality"],
                    control_char_count=quality["control_char_count"],
                    control_char_ratio=quality["control_char_ratio"],
                    letter_or_number_ratio=quality["letter_or_number_ratio"],
                    cjk_char_ratio=quality["cjk_char_ratio"],
                    text_quality_reasons=quality["text_quality_reasons"],
                    sample=" ".join(text.split())[:240],
                )
            )
    return profiles


def summarize_profiles(profiles: list[PageProfile]) -> dict:
    chars = [profile.char_count for profile in profiles]
    control_ratios = [profile.control_char_ratio for profile in profiles]
    letter_or_number_ratios = [profile.letter_or_number_ratio for profile in profiles]
    cjk_ratios = [profile.cjk_char_ratio for profile in profiles]
    reason_counts = Counter(reason for profile in profiles for reason in profile.text_quality_reasons)
    return {
        "page_count": len(profiles),
        "chars_total": sum(chars),
        "chars_mean": round(statistics.mean(chars), 1) if chars else 0,
        "chars_median": statistics.median(chars) if chars else 0,
        "control_char_ratio_mean": round(statistics.mean(control_ratios), 6) if control_ratios else 0,
        "letter_or_number_ratio_mean": (
            round(statistics.mean(letter_or_number_ratios), 6) if letter_or_number_ratios else 0
        ),
        "cjk_char_ratio_mean": round(statistics.mean(cjk_ratios), 6) if cjk_ratios else 0,
        "text_quality_reason_counts": dict(sorted(reason_counts.items())),
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
