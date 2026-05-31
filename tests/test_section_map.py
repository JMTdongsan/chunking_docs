from chunking_docs.chunking.section_map import section_for_page


def test_section_for_core_issue_page():
    section = section_for_page(100)

    assert section.chapter == "제3장 핵심이슈별 계획"
    assert section.issue == "핵심이슈 4 생명이 살아 숨 쉬는 안심도시"


def test_section_for_spatial_plan_page():
    section = section_for_page(150)

    assert section.chapter == "제4장 공간구조 및 토지이용계획"
