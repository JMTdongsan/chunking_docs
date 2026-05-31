from __future__ import annotations

from dataclasses import dataclass

from chunking_docs.models import SectionPath


@dataclass(frozen=True)
class SectionRange:
    page_start: int
    page_end: int
    section: SectionPath


SEOUL_PLAN_SECTION_RANGES = [
    SectionRange(1, 16, SectionPath(chapter="제1장 2030 서울플랜의 개요")),
    SectionRange(17, 32, SectionPath(chapter="제2장 2030 서울의 미래상")),
    SectionRange(33, 134, SectionPath(chapter="제3장 핵심이슈별 계획")),
    SectionRange(
        37,
        61,
        SectionPath(
            chapter="제3장 핵심이슈별 계획",
            issue="핵심이슈 1 차별없이 더불어 사는 사람중심 도시",
        ),
    ),
    SectionRange(
        62,
        77,
        SectionPath(
            chapter="제3장 핵심이슈별 계획",
            issue="핵심이슈 2 일자리와 활력이 넘치는 글로벌 상생도시",
        ),
    ),
    SectionRange(
        78,
        92,
        SectionPath(
            chapter="제3장 핵심이슈별 계획",
            issue="핵심이슈 3 역사가 살아있는 즐거운 문화도시",
        ),
    ),
    SectionRange(
        93,
        111,
        SectionPath(
            chapter="제3장 핵심이슈별 계획",
            issue="핵심이슈 4 생명이 살아 숨 쉬는 안심도시",
        ),
    ),
    SectionRange(
        112,
        134,
        SectionPath(
            chapter="제3장 핵심이슈별 계획",
            issue="핵심이슈 5 주거가 안정되고 이동이 편한 주민 공동체 도시",
        ),
    ),
    SectionRange(135, 166, SectionPath(chapter="제4장 공간구조 및 토지이용계획")),
    SectionRange(167, 194, SectionPath(chapter="제5장 생활권계획")),
    SectionRange(195, 206, SectionPath(chapter="제6장 계획의 실현")),
    SectionRange(207, 225, SectionPath(chapter="부록 계획단계별 참여진")),
]


def section_for_page(page_no: int, ranges: list[SectionRange] | None = None) -> SectionPath:
    ranges = ranges or SEOUL_PLAN_SECTION_RANGES
    matches = [item for item in ranges if item.page_start <= page_no <= item.page_end]
    if not matches:
        return SectionPath()
    return max(matches, key=lambda item: item.page_start).section
