from chunking_docs.chunking.section_map import (
    SectionRange,
    load_section_ranges,
    section_for_page,
)
from chunking_docs.models import SectionPath


def test_section_for_page_uses_provided_ranges():
    ranges = [
        SectionRange(1, 10, SectionPath(chapter="Chapter A")),
        SectionRange(4, 6, SectionPath(chapter="Chapter A", section="Nested Section")),
    ]

    section = section_for_page(5, ranges)

    assert section.chapter == "Chapter A"
    assert section.section == "Nested Section"


def test_section_for_page_is_empty_without_ranges():
    section = section_for_page(5)

    assert section.label() == ""


def test_load_section_ranges_from_jsonl(tmp_path):
    section_map = tmp_path / "sections.jsonl"
    section_map.write_text(
        '{"page_start":1,"page_end":3,"chapter":"Intro"}\n'
        '{"page_start":4,"page_end":9,"section":{"chapter":"Body","issue":"Policy A"}}\n',
        encoding="utf-8",
    )

    ranges = load_section_ranges(section_map)

    assert section_for_page(2, ranges).chapter == "Intro"
    assert section_for_page(5, ranges).issue == "Policy A"
