from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
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
    latency_ms: float = 0.0
    latency_samples_ms: list[float] = Field(default_factory=list)
    top_pages: list[int]
    top_page_ranges: list[tuple[int, int]] = Field(default_factory=list)
    top_chunk_ids: list[str]
    top_evidence_chunk_ids: list[list[str]] = Field(default_factory=list)
    top_sources: list[list[str]] = Field(default_factory=list)
    expected_pages: list[int]
    expected_chunk_ids: list[str]
    matched_rank: int | None = None
    matched_chunk_id: str | None = None
    matched_page: int | None = None
    reciprocal_rank: float = 0.0


class RetrievalEvaluation(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    case_count: int
    expected_case_count: int
    passed_count: int
    failed_count: int
    hit_rate: float
    recall_at_k: float
    mrr: float
    top_k: int
    repeat: int = 1
    index_build_ms: float = 0.0
    total_query_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    failed_queries: list[str]
    results: list[RetrievalCaseResult]


def evaluate_retrieval(
    chunks: list[DocumentChunk],
    triples: list[GraphTriple],
    cases: list[RetrievalCase],
    top_k: int = 5,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = False,
    graph_expand_override: bool | None = None,
    use_dense: bool = True,
    use_bm25: bool = True,
    use_graph: bool | None = None,
    repeat: int = 1,
) -> RetrievalEvaluation:
    repeat = max(1, repeat)
    index_start = perf_counter()
    searcher = LocalHybridSearcher(
        chunks,
        HashingTextEmbedder(),
        triples=triples,
        tokenizer_config=tokenizer_config,
    )
    index_build_ms = elapsed_ms(index_start)
    return evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: searcher.search(
            case.query,
            top_k=top_k,
            graph_expand=graph_expand,
            collapse_hierarchical=collapse_hierarchical,
            use_dense=use_dense,
            use_bm25=use_bm25,
            use_graph=use_graph,
        ),
        top_k=top_k,
        repeat=repeat,
        index_build_ms=index_build_ms,
        graph_expand_override=graph_expand_override,
    )


def evaluate_search_results(
    cases: list[RetrievalCase],
    search_fn: Callable[[RetrievalCase, bool], list],
    top_k: int = 5,
    repeat: int = 1,
    index_build_ms: float = 0.0,
    graph_expand_override: bool | None = None,
) -> RetrievalEvaluation:
    repeat = max(1, repeat)
    results: list[RetrievalCaseResult] = []
    latency_samples: list[float] = []
    for case in cases:
        graph_expand = case.graph_expand if graph_expand_override is None else graph_expand_override
        hits = []
        case_latencies = []
        for index in range(repeat):
            query_start = perf_counter()
            search_hits = search_fn(case, graph_expand)
            case_latencies.append(elapsed_ms(query_start))
            if index == 0:
                hits = search_hits
        latency_samples.extend(case_latencies)
        hits_with_chunks = [hit for hit in hits if hit_chunk(hit) is not None]
        top_pages = [hit_chunk(hit).page_start for hit in hits_with_chunks]
        top_page_ranges = [
            (hit_chunk(hit).page_start, hit_chunk(hit).page_end) for hit in hits_with_chunks
        ]
        top_chunk_ids = [hit_chunk(hit).chunk_id for hit in hits_with_chunks]
        top_evidence_chunk_ids = [
            [chunk.chunk_id for chunk in hit_evidence_chunks(hit)] for hit in hits_with_chunks
        ]
        top_sources = [hit_sources(hit) for hit in hits_with_chunks]
        expected_any = bool(case.expected_pages or case.expected_chunk_ids)
        match = first_relevant_hit(hits_with_chunks, case)
        passed = match is not None if expected_any else bool(hits_with_chunks)
        matched_rank = match.rank if match else (1 if not expected_any and hits_with_chunks else None)
        results.append(
            RetrievalCaseResult(
                query=case.query,
                passed=passed,
                latency_ms=sum(case_latencies) / len(case_latencies) if case_latencies else 0.0,
                latency_samples_ms=case_latencies,
                top_pages=top_pages,
                top_page_ranges=top_page_ranges,
                top_chunk_ids=top_chunk_ids,
                top_evidence_chunk_ids=top_evidence_chunk_ids,
                top_sources=top_sources,
                expected_pages=case.expected_pages,
                expected_chunk_ids=case.expected_chunk_ids,
                matched_rank=matched_rank,
                matched_chunk_id=match.chunk_id if match else None,
                matched_page=match.page if match else None,
                reciprocal_rank=(1.0 / matched_rank) if matched_rank else 0.0,
            )
        )
    passed_count = sum(1 for result in results if result.passed)
    expected_results = [
        result for result, case in zip(results, cases) if case.expected_pages or case.expected_chunk_ids
    ]
    expected_passed = sum(1 for result in expected_results if result.passed)
    return RetrievalEvaluation(
        case_count=len(cases),
        expected_case_count=len(expected_results),
        passed_count=passed_count,
        failed_count=len(cases) - passed_count,
        hit_rate=passed_count / len(cases) if cases else 0.0,
        recall_at_k=expected_passed / len(expected_results) if expected_results else 0.0,
        mrr=sum(result.reciprocal_rank for result in expected_results) / len(expected_results)
        if expected_results
        else 0.0,
        top_k=top_k,
        repeat=repeat,
        index_build_ms=index_build_ms,
        total_query_latency_ms=sum(latency_samples),
        mean_latency_ms=sum(latency_samples) / len(latency_samples) if latency_samples else 0.0,
        p95_latency_ms=percentile_latency(latency_samples, 0.95),
        failed_queries=[result.query for result in results if not result.passed],
        results=results,
    )


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    return read_jsonl(path, RetrievalCase)


class RelevantHit(BaseModel):
    rank: int
    chunk_id: str
    page: int | None = None


def first_relevant_hit(hits, case: RetrievalCase) -> RelevantHit | None:
    expected_chunk_ids = set(case.expected_chunk_ids)
    expected_pages = set(case.expected_pages)
    for index, hit in enumerate(hits):
        rank = index + 1
        chunk = hit_chunk(hit)
        if chunk is None:
            continue
        if chunk.chunk_id in expected_chunk_ids:
            return RelevantHit(rank=rank, chunk_id=chunk.chunk_id)
        for evidence_chunk in hit_evidence_chunks(hit):
            if evidence_chunk.chunk_id in expected_chunk_ids:
                return RelevantHit(rank=rank, chunk_id=evidence_chunk.chunk_id)
        matched_page = first_page_match(chunk.page_start, chunk.page_end, expected_pages)
        if matched_page is not None:
            return RelevantHit(rank=rank, chunk_id=chunk.chunk_id, page=matched_page)
    return None


def first_page_match(page_start: int, page_end: int, expected_pages: set[int]) -> int | None:
    for page in sorted(expected_pages):
        if page_start <= page <= page_end:
            return page
    return None


def hit_chunk(hit):
    return getattr(hit, "chunk", None)


def hit_sources(hit) -> list[str]:
    return list(getattr(hit, "sources", []))


def hit_evidence_chunks(hit) -> list[DocumentChunk]:
    return [chunk for chunk in getattr(hit, "evidence_chunks", []) if chunk is not None]


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def percentile_latency(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
