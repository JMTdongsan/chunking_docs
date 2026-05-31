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
            text="north district river corridor",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="population structure change",
        ),
    ]

    results = BM25LexicalIndex(chunks).search("north river", top_k=2)

    assert len(results) == 1
    assert results[0][0].chunk_id == "a"
    assert results[0][1] > 0
