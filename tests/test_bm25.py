from chunking_docs.embeddings.bm25 import BM25LexicalIndex
from chunking_docs.models import ChunkKind, DocumentChunk


def test_bm25_uses_lexical_overlap_when_idf_is_zero():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="동북권 발전구상 중랑천",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="인구구조 변화",
        ),
    ]

    results = BM25LexicalIndex(chunks).search("동북권 중랑천", top_k=2)

    assert len(results) == 1
    assert results[0][0].chunk_id == "a"
    assert results[0][1] > 0
