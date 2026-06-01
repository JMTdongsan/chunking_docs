from chunking_docs.embeddings.bm25 import BM25LexicalIndex, chunk_lexical_texts
from chunking_docs.embeddings.records import asset_text
from chunking_docs.embeddings.tokenizers import LexicalTokenizer, LexicalTokenizerConfig
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, VisualAsset


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


def test_bm25_uses_lexical_overlap_when_rank_score_is_non_positive():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="north river corridor diagram",
        )
    ]

    results = BM25LexicalIndex(chunks).search("north river corridor diagram", top_k=1)

    assert results[0][0].chunk_id == "a"
    assert results[0][1] == 1.0


def test_bm25_mixed_tokenizer_matches_cjk_compound_terms():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="도시재생계획",
        )
    ]

    word_results = BM25LexicalIndex(
        chunks,
        tokenizer_config=LexicalTokenizerConfig(strategy="word"),
    ).search("재생 계획")
    mixed_results = BM25LexicalIndex(
        chunks,
        tokenizer_config=LexicalTokenizerConfig(strategy="mixed", min_n=2, max_n=3),
    ).search("재생 계획")

    assert word_results == []
    assert mixed_results[0][0].chunk_id == "a"


def test_mixed_tokenizer_preserves_term_frequency_by_default():
    tokens = LexicalTokenizer(
        LexicalTokenizerConfig(strategy="mixed", min_n=2, max_n=2)
    ).tokenize("도시재생 도시재생")

    assert tokens.count("도시재생") == 2
    assert tokens.count("도시") == 2


def test_mixed_tokenizer_can_deduplicate_for_compact_manifests():
    tokens = LexicalTokenizer(
        LexicalTokenizerConfig(strategy="mixed", min_n=2, max_n=2, deduplicate=True)
    ).tokenize("도시재생 도시재생")

    assert tokens.count("도시재생") == 1
    assert tokens.count("도시") == 1


def test_bm25_ranking_uses_repeated_term_frequency():
    chunks = [
        DocumentChunk(
            chunk_id="once",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="target corridor",
        ),
        DocumentChunk(
            chunk_id="repeated",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="target target target corridor",
        ),
        DocumentChunk(
            chunk_id="other-1",
            doc_id="doc",
            page_start=3,
            page_end=3,
            kind=ChunkKind.TEXT,
            text="population trend",
        ),
        DocumentChunk(
            chunk_id="other-2",
            doc_id="doc",
            page_start=4,
            page_end=4,
            kind=ChunkKind.TEXT,
            text="housing supply",
        ),
        DocumentChunk(
            chunk_id="other-3",
            doc_id="doc",
            page_start=5,
            page_end=5,
            kind=ChunkKind.TEXT,
            text="transport network",
        ),
    ]

    results = BM25LexicalIndex(
        chunks,
        tokenizer_config=LexicalTokenizerConfig(strategy="word"),
    ).search("target", top_k=2)

    assert [chunk.chunk_id for chunk, _ in results] == ["repeated", "once"]


def test_bm25_can_index_linked_visual_asset_text():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="base text",
            asset_ids=["asset-1"],
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
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            caption="north river corridor diagram",
        )
    ]

    results = BM25LexicalIndex(chunks, texts=chunk_lexical_texts(chunks, assets)).search(
        "north river",
        top_k=2,
    )

    assert results[0][0].chunk_id == "a"
    assert "north river corridor diagram" in chunk_lexical_texts(chunks, assets)[0]


def test_bm25_can_index_source_ref_visual_asset_text():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="base text",
            source_refs=["asset:asset-1"],
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
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.FIGURE,
            caption="transfer hub diagram",
        )
    ]
    indexed_texts = chunk_lexical_texts(chunks, assets)

    results = BM25LexicalIndex(chunks, texts=indexed_texts).search("transfer hub", top_k=2)

    assert results[0][0].chunk_id == "a"
    assert "transfer hub diagram" in indexed_texts[0]


def test_bm25_indexes_structured_visual_metadata_text():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="base text",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            metadata={
                "entities": ["transfer hub"],
                "visual_elements": ["blue route arrow"],
                "objects": [
                    {
                        "label": "station marker",
                        "attributes": ["red circle", "north edge"],
                    }
                ],
            },
        )
    ]

    indexed_text = chunk_lexical_texts(chunks, assets)[0]
    results = BM25LexicalIndex(chunks, texts=[indexed_text]).search("station marker", top_k=1)

    assert results[0][0].chunk_id == "a"
    assert "Entities: transfer hub" in indexed_text
    assert "Visual elements: blue route arrow" in indexed_text
    assert "Objects: station marker: red circle, north edge" in asset_text(assets[0])


def test_bm25_indexes_visual_object_metadata_aliases():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="base text",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.FIGURE,
            metadata={
                "detections": {
                    "signal marker": {
                        "description": "green square",
                        "location": "lower left",
                    }
                },
                "regions": [{"label": "transfer deck", "attributes": ["blue platform"]}],
                "areas": ["pedestrian access zone"],
            },
        )
    ]

    indexed_text = chunk_lexical_texts(chunks, assets)[0]
    results = BM25LexicalIndex(chunks, texts=[indexed_text]).search("transfer deck", top_k=1)

    assert results[0][0].chunk_id == "a"
    assert "signal marker: green square, lower left" in asset_text(assets[0])
    assert "transfer deck: blue platform" in indexed_text
    assert "pedestrian access zone" in indexed_text


def test_bm25_manifest_records_tokenizer_config(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="도시재생계획",
        )
    ]
    output = tmp_path / "bm25_tokens.json"

    BM25LexicalIndex(
        chunks,
        tokenizer_config=LexicalTokenizerConfig(strategy="mixed", min_n=2, max_n=3),
    ).dump_manifest(output)

    assert '"strategy": "mixed"' in output.read_text(encoding="utf-8")
    assert '"min_n": 2' in output.read_text(encoding="utf-8")
    assert '"deduplicate": false' in output.read_text(encoding="utf-8")
