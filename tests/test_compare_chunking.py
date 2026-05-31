from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.compare import compare_chunking_reports
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.models import ChunkKind, DocumentChunk


def test_compare_chunking_reports_ranks_by_retrieval_then_quality():
    weak_chunks = [
        DocumentChunk(
            chunk_id="weak",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="unrelated",
        )
    ]
    strong_chunks = [
        DocumentChunk(
            chunk_id="strong",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="river corridor station hub",
        )
    ]
    cases = [RetrievalCase(query="river corridor", expected_pages=[1])]
    reports = {
        "weak": evaluate_chunking_quality(weak_chunks, [], [], [], retrieval_cases=cases),
        "strong": evaluate_chunking_quality(strong_chunks, [], [], [], retrieval_cases=cases),
    }

    comparison = compare_chunking_reports(reports)

    assert comparison.rows[0].name == "strong"
    assert comparison.best_by_retrieval == "strong"
    assert comparison.fastest_by_mean_latency in {"weak", "strong"}
    assert comparison.rows[0].retrieval_recall_at_k == 1.0
    assert comparison.rows[0].retrieval_mean_latency_ms is not None
    assert comparison.rows[-1].failed_queries == ["river corridor"]
