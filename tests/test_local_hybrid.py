from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from chunking_docs.models import ChunkKind, DocumentChunk, GraphTriple
from chunking_docs.retrieval.local_hybrid import LocalHybridSearcher


def test_local_hybrid_search_returns_bm25_and_dense_sources():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="동북권 발전구상 중랑천 상계 창동",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="인구구조 변화 고령인구 1인 가구",
        ),
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64))
    hits = searcher.search("동북권 중랑천", top_k=1)

    assert hits[0].chunk.chunk_id == "a"
    assert "bm25" in hits[0].sources
    assert "dense" in hits[0].sources


def test_local_hybrid_search_omits_zero_score_noise():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="동북권 발전구상",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="[degraded text layer] OCR required",
        ),
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64))
    hits = searcher.search("동북권", top_k=10)

    assert [hit.chunk.chunk_id for hit in hits] == ["a"]


def test_local_hybrid_graph_expansion_can_recover_related_chunk():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=188,
            page_end=188,
            kind=ChunkKind.TEXT,
            text="중랑천 수변축 동부간선도로 축",
        )
    ]
    triples = [
        GraphTriple(
            triple_id="t",
            doc_id="doc",
            chunk_id="a",
            subject="동북권",
            predicate="uses_axis",
            object="중랑천 수변축",
        )
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64), triples=triples)
    hits = searcher.search("동북권", top_k=1, graph_expand=True)

    assert hits[0].chunk.chunk_id == "a"
    assert "graph" in hits[0].sources
