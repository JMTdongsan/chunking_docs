from chunking_docs.chunking.semantic_splitter import (
    hard_split,
    overlap_tail,
    semantic_subchunks,
    split_text,
)
from chunking_docs.models import ChunkKind, DocumentChunk, SectionPath


def test_split_text_keeps_overlap_for_long_blocks():
    text = "\n\n".join([f"{index}. " + ("technical policy " * 35) for index in range(8)])

    chunks = split_text(text, max_chars=500, overlap_chars=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 560 for chunk in chunks)


def test_hard_split_prefers_sentence_boundaries():
    text = (
        "The first sentence describes the document background. "
        "The second sentence lists processing criteria. "
        "The third sentence explains validation. "
        "The fourth sentence summarizes reuse."
    )

    chunks = hard_split(text, max_chars=95, overlap_chars=0)

    assert len(chunks) > 1
    assert chunks[0].endswith(".")
    assert chunks[1].startswith("The second sentence")
    assert chunks[-1].startswith("The fourth sentence")


def test_hard_split_preserves_word_boundaries_without_punctuation():
    text = "alpha beta gamma " * 16

    chunks = hard_split(text, max_chars=55, overlap_chars=0)

    assert len(chunks) > 1
    assert all(token in {"alpha", "beta", "gamma"} for chunk in chunks for token in chunk.split())


def test_overlap_tail_preserves_whole_tokens():
    assert overlap_tail("alpha beta gamma delta", overlap_chars=10) == "gamma delta"


def test_split_text_prefers_korean_sentence_boundaries():
    text = (
        "첫 문장은 문서의 배경을 설명한다. "
        "둘째 문장은 처리 기준을 설명한다. "
        "셋째 문장은 검증 방법을 설명한다. "
        "넷째 문장은 결과 활용을 설명한다."
    )

    chunks = split_text(text, max_chars=50, overlap_chars=0)

    assert len(chunks) > 1
    assert chunks[0].endswith(".")
    assert chunks[1].startswith("셋째 문장은")


def test_semantic_subchunks_preserve_metadata_and_parent():
    chunk = DocumentChunk(
        chunk_id="parent-1",
        doc_id="doc",
        page_start=10,
        page_end=10,
        kind=ChunkKind.PAGE_SUMMARY,
        text="\n\n".join([f"Section {index} " + ("transit corridor " * 60) for index in range(1, 4)]),
        section=SectionPath(chapter="Chapter 4 Mobility Strategy"),
        metadata={"source": "test"},
    )

    chunks = semantic_subchunks([chunk], max_chars=450, overlap_chars=40)

    assert len(chunks) > 1
    assert chunks[0].kind == ChunkKind.TEXT
    assert chunks[0].metadata["parent_chunk_id"] == "parent-1"
    assert chunks[0].section.chapter == "Chapter 4 Mobility Strategy"
