import fitz

from chunking_docs.analysis.pdf_profile import classify_text_quality, text_quality_analysis
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.models import PageProfile, TextQuality


def test_classify_good_korean_text():
    text = "도시계획 문서는 지역의 미래상과 교통 전략을 제시한다."

    assert classify_text_quality(text) == TextQuality.GOOD


def test_classify_good_english_text_without_language_assumption():
    text = "The document describes transit corridors, station areas, and housing policy."

    assert classify_text_quality(text) == TextQuality.GOOD


def test_classify_degraded_text():
    text = "\x03\x04\x05 abc def 123"

    assert classify_text_quality(text) == TextQuality.DEGRADED
    analysis = text_quality_analysis(text)
    assert analysis["control_char_count"] == 3
    assert analysis["control_char_ratio"] > 0.02
    assert analysis["text_quality_reasons"] == ["high_control_char_ratio"]


def test_classify_empty_text():
    assert classify_text_quality("   ") == TextQuality.EMPTY


def test_page_level_chunks_preserve_text_quality_metadata(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page(width=100, height=100)
    doc.save(pdf_path)
    doc.close()
    profile = PageProfile(
        doc_id="doc",
        page_no=1,
        width=100,
        height=100,
        char_count=12,
        line_count=1,
        text_block_count=1,
        image_block_count=0,
        embedded_image_count=0,
        drawing_count=0,
        text_quality=TextQuality.DEGRADED,
        control_char_count=4,
        control_char_ratio=0.333333,
        letter_or_number_ratio=0.25,
        cjk_char_ratio=0.0,
        text_quality_reasons=["high_control_char_ratio"],
    )

    chunks = page_level_chunks(pdf_path, "doc", [profile])

    assert chunks[0].metadata["text_quality"] == TextQuality.DEGRADED
    assert chunks[0].metadata["control_char_count"] == 4
    assert chunks[0].metadata["control_char_ratio"] == 0.333333
    assert chunks[0].metadata["text_quality_reasons"] == ["high_control_char_ratio"]
    assert chunks[0].metadata["requires_ocr"] is True
