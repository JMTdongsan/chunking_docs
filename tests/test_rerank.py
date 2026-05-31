from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from chunking_docs.models import ChunkKind, DocumentChunk
from chunking_docs.retrieval.local_hybrid import HybridSearchHit, LocalHybridSearcher
from chunking_docs.retrieval.rerank import LexicalOverlapReranker, rerank_hits


def test_lexical_overlap_reranker_scores_query_coverage():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="river corridor station hub",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="population forecast",
        ),
    ]
    reranker = LexicalOverlapReranker()

    scores = reranker.score("river station", chunks)

    assert scores[0] > scores[1]


def test_rerank_hits_updates_order_score_and_source():
    weak = DocumentChunk(
        chunk_id="weak",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="unrelated",
    )
    strong = DocumentChunk(
        chunk_id="strong",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="river station corridor",
    )
    hits = [
        HybridSearchHit(chunk=weak, score=10.0, sources=["dense"]),
        HybridSearchHit(chunk=strong, score=0.1, sources=["bm25"]),
    ]

    reranked = rerank_hits("river station", hits, LexicalOverlapReranker(), top_k=2)

    assert [hit.chunk.chunk_id for hit in reranked] == ["strong", "weak"]
    assert "rerank:lexical" in reranked[0].sources
    assert reranked[0].score > reranked[1].score


def test_local_hybrid_search_can_rerank_fused_candidates():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="river station corridor",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="river station unrelated unrelated unrelated",
        ),
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64))
    hits = searcher.search(
        "river station corridor",
        top_k=2,
        use_dense=False,
        use_bm25=True,
        reranker=LexicalOverlapReranker(),
        rerank_top_k=2,
    )

    assert hits[0].chunk.chunk_id == "a"
    assert "rerank:lexical" in hits[0].sources
