from chunking_docs.embeddings.bm25 import BM25LexicalIndex, chunk_lexical_texts
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
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
