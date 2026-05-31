from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from chunking_docs.io import read_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple
from chunking_docs.retrieval.local_hybrid import LocalHybridSearcher


class RetrievalCase(BaseModel):
    query: str
    expected_pages: list[int] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    graph_expand: bool = False


class RetrievalCaseResult(BaseModel):
    query: str
    passed: bool
    top_pages: list[int]
    top_chunk_ids: list[str]
    expected_pages: list[int]
    expected_chunk_ids: list[str]


class RetrievalEvaluation(BaseModel):
    case_count: int
    passed_count: int
    hit_rate: float
    results: list[RetrievalCaseResult]


def evaluate_retrieval(
    chunks: list[DocumentChunk],
    triples: list[GraphTriple],
    cases: list[RetrievalCase],
    top_k: int = 5,
) -> RetrievalEvaluation:
    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(), triples=triples)
    results: list[RetrievalCaseResult] = []
    for case in cases:
        hits = searcher.search(case.query, top_k=top_k, graph_expand=case.graph_expand)
        top_pages = [hit.chunk.page_start for hit in hits]
        top_chunk_ids = [hit.chunk.chunk_id for hit in hits]
        page_hit = bool(set(case.expected_pages).intersection(top_pages))
        chunk_hit = bool(set(case.expected_chunk_ids).intersection(top_chunk_ids))
        expected_any = bool(case.expected_pages or case.expected_chunk_ids)
        passed = (page_hit or chunk_hit) if expected_any else bool(hits)
        results.append(
            RetrievalCaseResult(
                query=case.query,
                passed=passed,
                top_pages=top_pages,
                top_chunk_ids=top_chunk_ids,
                expected_pages=case.expected_pages,
                expected_chunk_ids=case.expected_chunk_ids,
            )
        )
    passed_count = sum(1 for result in results if result.passed)
    return RetrievalEvaluation(
        case_count=len(cases),
        passed_count=passed_count,
        hit_rate=passed_count / len(cases) if cases else 0.0,
        results=results,
    )


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    return read_jsonl(path, RetrievalCase)
