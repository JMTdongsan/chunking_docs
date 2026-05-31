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
            text="north district river corridor station hub",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="population aging single household trend",
        ),
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64))
    hits = searcher.search("north river", top_k=1)

    assert hits[0].chunk.chunk_id == "a"
    assert "bm25" in hits[0].sources
    assert "dense" in hits[0].sources


def test_local_hybrid_can_disable_retrieval_components():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="capital budget transit corridor",
        )
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64))
    bm25_hits = searcher.search("capital budget", top_k=1, use_dense=False, use_bm25=True)
    disabled_hits = searcher.search("capital budget", top_k=1, use_dense=False, use_bm25=False)

    assert bm25_hits[0].chunk.chunk_id == "a"
    assert bm25_hits[0].sources == ["bm25"]
    assert disabled_hits == []


def test_local_hybrid_search_omits_zero_score_noise():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="north district development concept",
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
    hits = searcher.search("north district", top_k=10)

    assert [hit.chunk.chunk_id for hit in hits] == ["a"]


def test_local_hybrid_graph_expansion_can_recover_related_chunk():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=188,
            page_end=188,
            kind=ChunkKind.TEXT,
            text="riverfront axis expressway corridor",
        )
    ]
    triples = [
        GraphTriple(
            triple_id="t",
            doc_id="doc",
            chunk_id="a",
            subject="north district",
            predicate="uses_axis",
            object="riverfront axis",
        )
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64), triples=triples)
    hits = searcher.search("north district", top_k=1, graph_expand=True)

    assert hits[0].chunk.chunk_id == "a"
    assert "graph" in hits[0].sources


def test_local_hybrid_graph_hits_resolve_source_chunk_alias():
    chunks = [
        DocumentChunk(
            chunk_id="parent",
            doc_id="doc",
            page_start=5,
            page_end=5,
            kind=ChunkKind.PAGE_SUMMARY,
            text="summary",
            metadata={"source_chunk_id": "source-a"},
        )
    ]
    triples = [
        GraphTriple(
            triple_id="t",
            doc_id="doc",
            chunk_id="source-a",
            subject="north district",
            predicate="uses_axis",
            object="riverfront axis",
        )
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64), triples=triples)
    hits = searcher.search("north district", top_k=1, graph_expand=True, use_dense=False, use_bm25=False)

    assert hits[0].chunk.chunk_id == "parent"
    assert hits[0].sources == ["graph"]


def test_local_hybrid_graph_hits_prioritize_exact_triple_components():
    chunks = [
        DocumentChunk(
            chunk_id="generic",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="generic center system note",
        ),
        DocumentChunk(
            chunk_id="target",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="specific center system evidence",
        ),
    ]
    triples = [
        GraphTriple(
            triple_id="generic-triple",
            doc_id="doc",
            chunk_id="generic",
            subject="center system",
            predicate="core",
            object="generic hub",
        ),
        GraphTriple(
            triple_id="target-triple",
            doc_id="doc",
            chunk_id="target",
            subject="center system",
            predicate="core",
            object="specific hub",
        ),
    ]

    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(embedding_dim=64), triples=triples)
    hits = searcher.search(
        "center system core specific hub",
        top_k=1,
        graph_expand=True,
        use_dense=False,
        use_bm25=False,
    )

    assert hits[0].chunk.chunk_id == "target"


def test_local_hybrid_can_collapse_hierarchical_child_to_parent():
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=7,
        page_end=7,
        kind=ChunkKind.PAGE_SUMMARY,
        text="page summary",
        metadata={"retrieval_role": "parent"},
    )
    child = DocumentChunk(
        chunk_id="child",
        doc_id="doc",
        page_start=7,
        page_end=7,
        kind=ChunkKind.TEXT,
        text="capital program station access evidence",
        metadata={
            "retrieval_role": "child",
            "hierarchical_parent_chunk_id": "parent",
        },
    )

    searcher = LocalHybridSearcher([parent, child], HashingTextEmbedder(embedding_dim=64))
    hits = searcher.search("station access", top_k=1, collapse_hierarchical=True)

    assert hits[0].chunk.chunk_id == "parent"
    assert [chunk.chunk_id for chunk in hits[0].evidence_chunks] == ["child"]
