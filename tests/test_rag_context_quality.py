import pytest

from chunking_docs.evaluation.context_quality import evaluate_rag_contexts
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.retrieval.context import (
    RAGContextAsset,
    RAGContextBundle,
    RAGContextChunk,
    RAGContextTriple,
)


def test_evaluate_rag_contexts_scores_final_context_targets():
    case = RetrievalCase(
        query="station access map",
        expected_pages=[4],
        expected_chunk_ids=["source-chunk"],
        expected_asset_ids=["asset-2"],
        expected_triple_ids=["triple-2"],
        excluded_asset_ids=["asset-x"],
        metadata={"case_source": "visual_object_probe"},
    )
    bundle = RAGContextBundle(
        query="station access map",
        chunks=[
            RAGContextChunk(
                chunk_id="chunk-1",
                doc_id="doc",
                page_start=4,
                page_end=4,
                kind="text",
                text="station access evidence",
                metadata={
                    "source_chunk_id": "source-chunk",
                    "retrieved_asset_ids": ["asset-2"],
                    "retrieved_triple_ids": ["triple-2"],
                },
            )
        ],
        assets=[
            RAGContextAsset(
                asset_id="asset-2",
                page_no=4,
                kind="map",
                caption="station access map",
            )
        ],
        triples=[
            RAGContextTriple(
                triple_id="triple-2",
                chunk_id="chunk-1",
                subject="station",
                predicate="connects_to",
                object="corridor",
            )
        ],
    )

    evaluation = evaluate_rag_contexts([case], [bundle], latencies_ms=[12.0])

    assert evaluation.passed_count == 1
    assert evaluation.target_coverage == 1.0
    assert evaluation.excluded_target_hit_rate == 0.0
    assert evaluation.mean_latency_ms == 12.0
    assert evaluation.target_metrics["asset"].coverage == 1.0
    assert evaluation.target_metrics["triple"].coverage == 1.0
    assert (
        evaluation.case_group_metrics["case_source"]["visual_object_probe"].target_coverage
        == 1.0
    )


def test_evaluate_rag_contexts_fails_on_missing_or_excluded_targets():
    case = RetrievalCase(
        query="wrong visual evidence",
        expected_asset_ids=["asset-needed"],
        excluded_pages=[9],
    )
    bundle = RAGContextBundle(
        query="wrong visual evidence",
        chunks=[
            RAGContextChunk(
                chunk_id="chunk-1",
                doc_id="doc",
                page_start=9,
                page_end=9,
                kind="text",
                text="wrong page evidence",
            )
        ],
    )

    evaluation = evaluate_rag_contexts([case], [bundle])

    assert evaluation.passed_count == 0
    assert evaluation.failed_queries == ["wrong visual evidence"]
    assert evaluation.target_coverage == 0.0
    assert evaluation.excluded_target_hit_rate == 1.0
    assert evaluation.results[0].target_key_matches == {"asset:asset-needed": False}
    assert evaluation.results[0].excluded_target_key_matches == {"page:9": True}


def test_evaluate_rag_contexts_requires_aligned_lengths():
    case = RetrievalCase(query="query", expected_pages=[1])

    with pytest.raises(ValueError, match="bundles"):
        evaluate_rag_contexts([case], [])

    with pytest.raises(ValueError, match="latencies_ms"):
        evaluate_rag_contexts(
            [case],
            [
                RAGContextBundle(
                    query="query",
                    chunks=[],
                )
            ],
            latencies_ms=[],
        )
