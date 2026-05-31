from chunking_docs.analysis.pdf_profile import classify_text_quality
from chunking_docs.models import TextQuality


def test_classify_good_korean_text():
    text = "도시계획 문서는 지역의 미래상과 교통 전략을 제시한다."

    assert classify_text_quality(text) == TextQuality.GOOD


def test_classify_degraded_text():
    text = "\x03\x04\x05 abc def 123"

    assert classify_text_quality(text) == TextQuality.DEGRADED


def test_classify_empty_text():
    assert classify_text_quality("   ") == TextQuality.EMPTY
