from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.embeddings.interfaces import HashingTextEmbedder
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.graph.provenance import chunk_asset_ids, string_values, triple_asset_ids
from chunking_docs.io import read_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval.local_hybrid import LocalHybridSearcher
from chunking_docs.retrieval.rerank import Reranker


class RetrievalCase(BaseModel):
    query: str
    expected_pages: list[int] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_asset_ids: list[str] = Field(default_factory=list)
    expected_triple_ids: list[str] = Field(default_factory=list)
    excluded_pages: list[int] = Field(default_factory=list)
    excluded_chunk_ids: list[str] = Field(default_factory=list)
    excluded_asset_ids: list[str] = Field(default_factory=list)
    excluded_triple_ids: list[str] = Field(default_factory=list)
    graph_expand: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCaseResult(BaseModel):
    query: str
    case_metadata: dict[str, Any] = Field(default_factory=dict)
    passed: bool
    latency_ms: float = 0.0
    latency_samples_ms: list[float] = Field(default_factory=list)
    result_consistent: bool = True
    distinct_result_sets: int = 1
    top_pages: list[int]
    top_page_ranges: list[tuple[int, int]] = Field(default_factory=list)
    top_chunk_ids: list[str]
    top_asset_ids: list[list[str]] = Field(default_factory=list)
    top_triple_ids: list[list[str]] = Field(default_factory=list)
    top_evidence_chunk_ids: list[list[str]] = Field(default_factory=list)
    top_sources: list[list[str]] = Field(default_factory=list)
    top_chunking_strategies: list[list[str]] = Field(default_factory=list)
    top_retrieval_roles: list[list[str]] = Field(default_factory=list)
    top_matched_targets: list[list[str]] = Field(default_factory=list)
    expected_pages: list[int]
    expected_chunk_ids: list[str]
    expected_asset_ids: list[str] = Field(default_factory=list)
    expected_triple_ids: list[str] = Field(default_factory=list)
    excluded_pages: list[int] = Field(default_factory=list)
    excluded_chunk_ids: list[str] = Field(default_factory=list)
    excluded_asset_ids: list[str] = Field(default_factory=list)
    excluded_triple_ids: list[str] = Field(default_factory=list)
    expected_target_count: int = 0
    matched_target_count: int = 0
    target_coverage_at_k: float = 0.0
    target_ndcg_at_k: float = 0.0
    relevant_hit_count: int = 0
    precision_at_k: float = 0.0
    excluded_target_count: int = 0
    excluded_matched_target_count: int = 0
    excluded_hit_count: int = 0
    excluded_target_hit_rate: float = 0.0
    matched_rank: int | None = None
    matched_chunk_id: str | None = None
    matched_asset_id: str | None = None
    matched_triple_id: str | None = None
    matched_page: int | None = None
    reciprocal_rank: float = 0.0
    target_matches: dict[str, bool] = Field(default_factory=dict)
    target_matched_ranks: dict[str, int] = Field(default_factory=dict)
    target_key_matched_ranks: dict[str, int] = Field(default_factory=dict)
    target_reciprocal_ranks: dict[str, float] = Field(default_factory=dict)
    top_excluded_targets: list[list[str]] = Field(default_factory=list)
    excluded_target_matched_ranks: dict[str, int] = Field(default_factory=dict)


class RetrievalTargetMetric(BaseModel):
    expected_count: int = 0
    passed_count: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    target_count: int = 0
    matched_target_count: int = 0
    coverage_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    failed_queries: list[str] = Field(default_factory=list)


class RetrievalSourceMetric(BaseModel):
    query_count: int = 0
    relevant_query_count: int = 0
    excluded_query_count: int = 0
    hit_count: int = 0
    relevant_hit_count: int = 0
    excluded_hit_count: int = 0
    expected_target_count: int = 0
    matched_target_count: int = 0
    excluded_target_count: int = 0
    excluded_matched_target_count: int = 0
    precision_at_hits: float = 0.0
    excluded_precision_at_hits: float = 0.0
    target_coverage_at_k: float = 0.0
    excluded_target_hit_rate: float = 0.0
    mean_relevant_rank: float = 0.0


class RetrievalCaseGroupMetric(BaseModel):
    case_count: int = 0
    expected_case_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    target_count: int = 0
    matched_target_count: int = 0
    target_coverage_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    precision_at_k: float = 0.0
    mean_latency_ms: float = 0.0
    failed_queries: list[str] = Field(default_factory=list)


class RetrievalEvaluation(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    case_count: int
    expected_case_count: int
    passed_count: int
    failed_count: int
    hit_rate: float
    recall_at_k: float
    mrr: float
    target_coverage_at_k: float = 0.0
    mean_target_ndcg_at_k: float = 0.0
    mean_precision_at_k: float = 0.0
    excluded_query_count: int = 0
    excluded_hit_query_count: int = 0
    excluded_query_hit_rate: float = 0.0
    excluded_target_count: int = 0
    excluded_matched_target_count: int = 0
    excluded_target_hit_rate: float = 0.0
    top_k: int
    repeat: int = 1
    unstable_result_count: int = 0
    result_stability_rate: float = 1.0
    index_build_ms: float = 0.0
    total_query_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    target_metrics: dict[str, RetrievalTargetMetric] = Field(default_factory=dict)
    source_metrics: dict[str, RetrievalSourceMetric] = Field(default_factory=dict)
    source_family_metrics: dict[str, RetrievalSourceMetric] = Field(default_factory=dict)
    chunk_strategy_metrics: dict[str, RetrievalSourceMetric] = Field(default_factory=dict)
    retrieval_role_metrics: dict[str, RetrievalSourceMetric] = Field(default_factory=dict)
    case_group_metrics: dict[str, dict[str, RetrievalCaseGroupMetric]] = Field(default_factory=dict)
    failed_queries: list[str]
    results: list[RetrievalCaseResult]


CASE_GROUP_METADATA_KEYS = (
    "case_source",
    "query_mode",
    "case_family",
    "difficulty",
    "modality",
    "evidence_family",
)

SearchResultSignature = tuple[
    tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    ...,
]


def evaluate_retrieval(
    chunks: list[DocumentChunk],
    triples: list[GraphTriple],
    cases: list[RetrievalCase],
    assets: list[VisualAsset] | None = None,
    top_k: int = 5,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = False,
    graph_expand_override: bool | None = None,
    use_dense: bool = True,
    use_bm25: bool = True,
    use_graph: bool | None = None,
    repeat: int = 1,
    fusion_weights: dict[str, float] | None = None,
    reranker: Reranker | None = None,
    rerank_top_k: int | None = None,
) -> RetrievalEvaluation:
    repeat = max(1, repeat)
    index_start = perf_counter()
    searcher = LocalHybridSearcher(
        chunks,
        HashingTextEmbedder(),
        triples=triples,
        tokenizer_config=tokenizer_config,
        assets=assets,
    )
    index_build_ms = elapsed_ms(index_start)
    evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: searcher.search(
            case.query,
            top_k=top_k,
            graph_expand=graph_expand,
            collapse_hierarchical=collapse_hierarchical,
            use_dense=use_dense,
            use_bm25=use_bm25,
            use_graph=use_graph,
            fusion_weights=fusion_weights,
            reranker=reranker,
            rerank_top_k=rerank_top_k,
        ),
        top_k=top_k,
        repeat=repeat,
        index_build_ms=index_build_ms,
        graph_expand_override=graph_expand_override,
        triples=triples,
    )
    if fusion_weights:
        evaluation.metadata["fusion_weights"] = fusion_weights
    if reranker is not None:
        evaluation.metadata["reranker"] = reranker.source
        evaluation.metadata["rerank_top_k"] = rerank_top_k or top_k
    return evaluation


def evaluate_search_results(
    cases: list[RetrievalCase],
    search_fn: Callable[[RetrievalCase, bool], list],
    top_k: int = 5,
    repeat: int = 1,
    index_build_ms: float = 0.0,
    graph_expand_override: bool | None = None,
    triples: list[GraphTriple] | None = None,
) -> RetrievalEvaluation:
    repeat = max(1, repeat)
    indexed_triples = triples or []
    triples_by_chunk = index_triples_by_chunk(indexed_triples)
    triples_by_asset = index_triples_by_asset(indexed_triples)
    results: list[RetrievalCaseResult] = []
    latency_samples: list[float] = []
    for case in cases:
        graph_expand = case.graph_expand if graph_expand_override is None else graph_expand_override
        hits = []
        case_latencies = []
        result_signatures = []
        for index in range(repeat):
            query_start = perf_counter()
            search_hits = search_fn(case, graph_expand)
            case_latencies.append(elapsed_ms(query_start))
            result_signatures.append(search_result_signature(search_hits, top_k))
            if index == 0:
                hits = search_hits
        latency_samples.extend(case_latencies)
        distinct_result_sets = len(set(result_signatures)) if result_signatures else 0
        hits_with_chunks = [hit for hit in hits if hit_chunk(hit) is not None]
        hits_with_chunks = hits_with_chunks[:top_k] if top_k > 0 else []
        top_pages = [hit_chunk(hit).page_start for hit in hits_with_chunks]
        top_page_ranges = [
            (hit_chunk(hit).page_start, hit_chunk(hit).page_end) for hit in hits_with_chunks
        ]
        top_chunk_ids = [hit_chunk(hit).chunk_id for hit in hits_with_chunks]
        top_asset_ids = [hit_asset_ids(hit) for hit in hits_with_chunks]
        top_triple_ids = [
            hit_triple_ids(hit, triples_by_chunk, triples_by_asset) for hit in hits_with_chunks
        ]
        top_evidence_chunk_ids = [
            [chunk.chunk_id for chunk in hit_evidence_chunks(hit)] for hit in hits_with_chunks
        ]
        top_sources = [hit_sources(hit) for hit in hits_with_chunks]
        top_chunking_strategies = [hit_chunking_strategies(hit) for hit in hits_with_chunks]
        top_retrieval_roles = [hit_retrieval_roles(hit) for hit in hits_with_chunks]
        expected_targets = expected_target_keys(case)
        excluded_targets = excluded_target_keys(case)
        seen_targets: set[str] = set()
        seen_excluded_targets: set[str] = set()
        top_matched_targets: list[list[str]] = []
        top_excluded_targets: list[list[str]] = []
        relevant_hit_count = 0
        excluded_hit_count = 0
        for hit in hits_with_chunks:
            matched_targets = hit_target_keys(
                hit,
                case,
                triples_by_chunk=triples_by_chunk,
                triples_by_asset=triples_by_asset,
            )
            excluded_matched_targets = hit_excluded_target_keys(
                hit,
                case,
                triples_by_chunk=triples_by_chunk,
                triples_by_asset=triples_by_asset,
            )
            if matched_targets:
                relevant_hit_count += 1
            if excluded_matched_targets:
                excluded_hit_count += 1
            seen_targets.update(matched_targets)
            seen_excluded_targets.update(excluded_matched_targets)
            top_matched_targets.append(sorted_target_keys(matched_targets))
            top_excluded_targets.append(sorted_target_keys(excluded_matched_targets))
        matched_target_count = len(seen_targets)
        excluded_matched_target_count = len(seen_excluded_targets)
        target_key_matched_ranks = matched_target_key_ranks(top_matched_targets)
        excluded_target_matched_ranks = matched_target_key_ranks(top_excluded_targets)
        expected_any = bool(
            case.expected_pages
            or case.expected_chunk_ids
            or case.expected_asset_ids
            or case.expected_triple_ids
        )
        excluded_any = bool(
            case.excluded_pages
            or case.excluded_chunk_ids
            or case.excluded_asset_ids
            or case.excluded_triple_ids
        )
        match = first_relevant_hit(
            hits_with_chunks,
            case,
            triples_by_chunk=triples_by_chunk,
            triples_by_asset=triples_by_asset,
        )
        target_hits = target_relevant_hits(
            hits_with_chunks,
            case,
            triples_by_chunk=triples_by_chunk,
            triples_by_asset=triples_by_asset,
        )
        target_matched_ranks = {
            target: target_match.rank
            for target, target_match in target_hits.items()
            if target_match is not None
        }
        if expected_any:
            passed = match is not None and excluded_hit_count == 0
        elif excluded_any:
            passed = excluded_hit_count == 0
        else:
            passed = bool(hits_with_chunks)
        matched_rank = match.rank if match else (1 if not expected_any and hits_with_chunks else None)
        results.append(
            RetrievalCaseResult(
                query=case.query,
                case_metadata=case.metadata,
                passed=passed,
                latency_ms=sum(case_latencies) / len(case_latencies) if case_latencies else 0.0,
                latency_samples_ms=case_latencies,
                result_consistent=distinct_result_sets <= 1,
                distinct_result_sets=distinct_result_sets,
                top_pages=top_pages,
                top_page_ranges=top_page_ranges,
                top_chunk_ids=top_chunk_ids,
                top_asset_ids=top_asset_ids,
                top_triple_ids=top_triple_ids,
                top_evidence_chunk_ids=top_evidence_chunk_ids,
                top_sources=top_sources,
                top_chunking_strategies=top_chunking_strategies,
                top_retrieval_roles=top_retrieval_roles,
                top_matched_targets=top_matched_targets,
                expected_pages=case.expected_pages,
                expected_chunk_ids=case.expected_chunk_ids,
                expected_asset_ids=case.expected_asset_ids,
                expected_triple_ids=case.expected_triple_ids,
                excluded_pages=case.excluded_pages,
                excluded_chunk_ids=case.excluded_chunk_ids,
                excluded_asset_ids=case.excluded_asset_ids,
                excluded_triple_ids=case.excluded_triple_ids,
                expected_target_count=len(expected_targets),
                matched_target_count=matched_target_count,
                target_coverage_at_k=matched_target_count / len(expected_targets)
                if expected_targets
                else 0.0,
                target_ndcg_at_k=target_ndcg_score(expected_targets, target_key_matched_ranks),
                relevant_hit_count=relevant_hit_count,
                precision_at_k=relevant_hit_count / top_k if top_k > 0 else 0.0,
                excluded_target_count=len(excluded_targets),
                excluded_matched_target_count=excluded_matched_target_count,
                excluded_hit_count=excluded_hit_count,
                excluded_target_hit_rate=excluded_matched_target_count / len(excluded_targets)
                if excluded_targets
                else 0.0,
                matched_rank=matched_rank,
                matched_chunk_id=match.chunk_id if match else None,
                matched_asset_id=match.asset_id if match else None,
                matched_triple_id=match.triple_id if match else None,
                matched_page=match.page if match else None,
                reciprocal_rank=(1.0 / matched_rank) if matched_rank else 0.0,
                target_matches={
                    target: target_match is not None for target, target_match in target_hits.items()
                },
                target_matched_ranks=target_matched_ranks,
                target_key_matched_ranks=target_key_matched_ranks,
                target_reciprocal_ranks={
                    target: 1.0 / rank for target, rank in target_matched_ranks.items()
                },
                top_excluded_targets=top_excluded_targets,
                excluded_target_matched_ranks=excluded_target_matched_ranks,
            )
        )
    passed_count = sum(1 for result in results if result.passed)
    expected_results = [
        result
        for result, case in zip(results, cases)
        if case.expected_pages or case.expected_chunk_ids or case.expected_asset_ids or case.expected_triple_ids
    ]
    expected_passed = sum(1 for result in expected_results if result.matched_rank is not None)
    excluded_results = [result for result in results if result.excluded_target_count > 0]
    excluded_target_count = sum(result.excluded_target_count for result in excluded_results)
    excluded_matched_target_count = sum(
        result.excluded_matched_target_count for result in excluded_results
    )
    excluded_hit_query_count = sum(1 for result in excluded_results if result.excluded_hit_count > 0)
    unstable_result_count = sum(1 for result in results if not result.result_consistent)
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
        target_coverage_at_k=sum(result.matched_target_count for result in expected_results)
        / sum(result.expected_target_count for result in expected_results)
        if sum(result.expected_target_count for result in expected_results)
        else 0.0,
        mean_target_ndcg_at_k=sum(result.target_ndcg_at_k for result in expected_results)
        / len(expected_results)
        if expected_results
        else 0.0,
        mean_precision_at_k=sum(result.precision_at_k for result in expected_results)
        / len(expected_results)
        if expected_results
        else 0.0,
        excluded_query_count=len(excluded_results),
        excluded_hit_query_count=excluded_hit_query_count,
        excluded_query_hit_rate=excluded_hit_query_count / len(excluded_results)
        if excluded_results
        else 0.0,
        excluded_target_count=excluded_target_count,
        excluded_matched_target_count=excluded_matched_target_count,
        excluded_target_hit_rate=excluded_matched_target_count / excluded_target_count
        if excluded_target_count
        else 0.0,
        top_k=top_k,
        repeat=repeat,
        unstable_result_count=unstable_result_count,
        result_stability_rate=(len(results) - unstable_result_count) / len(results)
        if results
        else 1.0,
        index_build_ms=index_build_ms,
        total_query_latency_ms=sum(latency_samples),
        mean_latency_ms=sum(latency_samples) / len(latency_samples) if latency_samples else 0.0,
        p95_latency_ms=percentile_latency(latency_samples, 0.95),
        target_metrics=target_metrics(results),
        source_metrics=source_metrics(results),
        source_family_metrics=source_metrics(results, family=True),
        chunk_strategy_metrics=hit_group_metrics(results, "top_chunking_strategies"),
        retrieval_role_metrics=hit_group_metrics(results, "top_retrieval_roles"),
        case_group_metrics=case_group_metrics(results, cases),
        failed_queries=[result.query for result in results if not result.passed],
        results=results,
    )


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    return read_jsonl(path, RetrievalCase)


class RelevantHit(BaseModel):
    rank: int
    chunk_id: str
    asset_id: str | None = None
    triple_id: str | None = None
    page: int | None = None


def first_relevant_hit(
    hits,
    case: RetrievalCase,
    triples_by_chunk: dict[str, list[GraphTriple]] | None = None,
    triples_by_asset: dict[str, list[GraphTriple]] | None = None,
) -> RelevantHit | None:
    expected_chunk_ids = set(case.expected_chunk_ids)
    expected_asset_ids = set(case.expected_asset_ids)
    expected_triple_ids = set(case.expected_triple_ids)
    expected_pages = set(case.expected_pages)
    triples_by_chunk = triples_by_chunk or {}
    triples_by_asset = triples_by_asset or {}
    for index, hit in enumerate(hits):
        rank = index + 1
        chunk = hit_chunk(hit)
        if chunk is None:
            continue
        matched_chunk_id = first_matching_chunk_alias(chunk, expected_chunk_ids)
        if matched_chunk_id is not None:
            return RelevantHit(rank=rank, chunk_id=matched_chunk_id)
        for evidence_chunk in hit_evidence_chunks(hit):
            matched_chunk_id = first_matching_chunk_alias(evidence_chunk, expected_chunk_ids)
            if matched_chunk_id is not None:
                return RelevantHit(rank=rank, chunk_id=matched_chunk_id)
        for asset_id in hit_asset_ids(hit):
            if asset_id in expected_asset_ids:
                return RelevantHit(rank=rank, chunk_id=chunk.chunk_id, asset_id=asset_id)
        for triple_id in hit_triple_ids(hit, triples_by_chunk, triples_by_asset):
            if triple_id in expected_triple_ids:
                return RelevantHit(rank=rank, chunk_id=chunk.chunk_id, triple_id=triple_id)
        matched_page = first_page_match(chunk.page_start, chunk.page_end, expected_pages)
        if matched_page is not None:
            return RelevantHit(rank=rank, chunk_id=chunk.chunk_id, page=matched_page)
    return None


def target_relevant_hits(
    hits,
    case: RetrievalCase,
    triples_by_chunk: dict[str, list[GraphTriple]] | None = None,
    triples_by_asset: dict[str, list[GraphTriple]] | None = None,
) -> dict[str, RelevantHit | None]:
    targets: dict[str, RelevantHit | None] = {}
    triples_by_chunk = triples_by_chunk or {}
    triples_by_asset = triples_by_asset or {}
    if case.expected_pages:
        targets["page"] = first_page_hit(hits, set(case.expected_pages))
    if case.expected_chunk_ids:
        targets["chunk"] = first_chunk_hit(hits, set(case.expected_chunk_ids))
    if case.expected_asset_ids:
        targets["asset"] = first_asset_hit(hits, set(case.expected_asset_ids))
    if case.expected_triple_ids:
        targets["triple"] = first_triple_hit(
            hits,
            set(case.expected_triple_ids),
            triples_by_chunk=triples_by_chunk,
            triples_by_asset=triples_by_asset,
        )
    return targets


def first_page_hit(hits, expected_pages: set[int]) -> RelevantHit | None:
    for index, hit in enumerate(hits):
        rank = index + 1
        chunk = hit_chunk(hit)
        if chunk is None:
            continue
        matched_page = first_page_match(chunk.page_start, chunk.page_end, expected_pages)
        if matched_page is not None:
            return RelevantHit(rank=rank, chunk_id=chunk.chunk_id, page=matched_page)
    return None


def first_chunk_hit(hits, expected_chunk_ids: set[str]) -> RelevantHit | None:
    for index, hit in enumerate(hits):
        rank = index + 1
        chunk = hit_chunk(hit)
        if chunk is None:
            continue
        matched_chunk_id = first_matching_chunk_alias(chunk, expected_chunk_ids)
        if matched_chunk_id is not None:
            return RelevantHit(rank=rank, chunk_id=matched_chunk_id)
        for evidence_chunk in hit_evidence_chunks(hit):
            matched_chunk_id = first_matching_chunk_alias(evidence_chunk, expected_chunk_ids)
            if matched_chunk_id is not None:
                return RelevantHit(rank=rank, chunk_id=matched_chunk_id)
    return None


def first_asset_hit(hits, expected_asset_ids: set[str]) -> RelevantHit | None:
    for index, hit in enumerate(hits):
        rank = index + 1
        chunk = hit_chunk(hit)
        if chunk is None:
            continue
        for asset_id in hit_asset_ids(hit):
            if asset_id in expected_asset_ids:
                return RelevantHit(rank=rank, chunk_id=chunk.chunk_id, asset_id=asset_id)
    return None


def first_triple_hit(
    hits,
    expected_triple_ids: set[str],
    triples_by_chunk: dict[str, list[GraphTriple]],
    triples_by_asset: dict[str, list[GraphTriple]] | None = None,
) -> RelevantHit | None:
    triples_by_asset = triples_by_asset or {}
    for index, hit in enumerate(hits):
        rank = index + 1
        chunk = hit_chunk(hit)
        if chunk is None:
            continue
        for triple_id in hit_triple_ids(hit, triples_by_chunk, triples_by_asset):
            if triple_id in expected_triple_ids:
                return RelevantHit(rank=rank, chunk_id=chunk.chunk_id, triple_id=triple_id)
    return None


def first_page_match(page_start: int, page_end: int, expected_pages: set[int]) -> int | None:
    for page in sorted(expected_pages):
        if page_start <= page <= page_end:
            return page
    return None


def expected_target_keys(case: RetrievalCase) -> set[str]:
    keys = {f"page:{page}" for page in case.expected_pages}
    keys.update(f"chunk:{chunk_id}" for chunk_id in case.expected_chunk_ids)
    keys.update(f"asset:{asset_id}" for asset_id in case.expected_asset_ids)
    keys.update(f"triple:{triple_id}" for triple_id in case.expected_triple_ids)
    return keys


def excluded_target_keys(case: RetrievalCase) -> set[str]:
    keys = {f"page:{page}" for page in case.excluded_pages}
    keys.update(f"chunk:{chunk_id}" for chunk_id in case.excluded_chunk_ids)
    keys.update(f"asset:{asset_id}" for asset_id in case.excluded_asset_ids)
    keys.update(f"triple:{triple_id}" for triple_id in case.excluded_triple_ids)
    return keys


def hit_target_keys(
    hit,
    case: RetrievalCase,
    triples_by_chunk: dict[str, list[GraphTriple]],
    triples_by_asset: dict[str, list[GraphTriple]] | None = None,
) -> set[str]:
    return hit_selected_target_keys(
        hit,
        expected_pages=set(case.expected_pages),
        expected_chunk_ids=set(case.expected_chunk_ids),
        expected_asset_ids=set(case.expected_asset_ids),
        expected_triple_ids=set(case.expected_triple_ids),
        triples_by_chunk=triples_by_chunk,
        triples_by_asset=triples_by_asset,
    )


def hit_excluded_target_keys(
    hit,
    case: RetrievalCase,
    triples_by_chunk: dict[str, list[GraphTriple]],
    triples_by_asset: dict[str, list[GraphTriple]] | None = None,
) -> set[str]:
    return hit_selected_target_keys(
        hit,
        expected_pages=set(case.excluded_pages),
        expected_chunk_ids=set(case.excluded_chunk_ids),
        expected_asset_ids=set(case.excluded_asset_ids),
        expected_triple_ids=set(case.excluded_triple_ids),
        triples_by_chunk=triples_by_chunk,
        triples_by_asset=triples_by_asset,
    )


def hit_selected_target_keys(
    hit,
    expected_pages: set[int],
    expected_chunk_ids: set[str],
    expected_asset_ids: set[str],
    expected_triple_ids: set[str],
    triples_by_chunk: dict[str, list[GraphTriple]],
    triples_by_asset: dict[str, list[GraphTriple]] | None = None,
) -> set[str]:
    chunk = hit_chunk(hit)
    if chunk is None:
        return set()

    keys: set[str] = set()

    candidate_chunks = [chunk, *hit_evidence_chunks(hit)]
    for candidate_chunk in candidate_chunks:
        for page in expected_pages:
            if candidate_chunk.page_start <= page <= candidate_chunk.page_end:
                keys.add(f"page:{page}")
        matched_chunk_id = first_matching_chunk_alias(candidate_chunk, expected_chunk_ids)
        if matched_chunk_id is not None:
            keys.add(f"chunk:{matched_chunk_id}")

    for asset_id in hit_asset_ids(hit):
        if asset_id in expected_asset_ids:
            keys.add(f"asset:{asset_id}")
    for triple_id in hit_triple_ids(hit, triples_by_chunk, triples_by_asset or {}):
        if triple_id in expected_triple_ids:
            keys.add(f"triple:{triple_id}")
    return keys


def sorted_target_keys(keys: set[str]) -> list[str]:
    order = {"page": 0, "chunk": 1, "asset": 2, "triple": 3}
    return sorted(keys, key=lambda key: (order.get(key.split(":", 1)[0], 99), key))


def search_result_signature(hits, top_k: int) -> SearchResultSignature:
    hits_with_chunks = [hit for hit in hits if hit_chunk(hit) is not None]
    limited_hits = hits_with_chunks[:top_k] if top_k > 0 else []
    return tuple(
        (
            hit_chunk(hit).chunk_id,
            tuple(hit_sources(hit)),
            tuple(hit_asset_ids(hit)),
            tuple(chunk.chunk_id for chunk in hit_evidence_chunks(hit)),
        )
        for hit in limited_hits
    )


def hit_chunk(hit):
    return getattr(hit, "chunk", None)


def hit_sources(hit) -> list[str]:
    return list(getattr(hit, "sources", []))


def hit_chunking_strategies(hit) -> list[str]:
    return hit_chunk_metadata_labels(hit, "chunking_strategy")


def hit_retrieval_roles(hit) -> list[str]:
    return hit_chunk_metadata_labels(hit, "retrieval_role")


def hit_chunk_metadata_labels(hit, metadata_key: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for chunk in [hit_chunk(hit), *hit_evidence_chunks(hit)]:
        if chunk is None:
            continue
        label = normalized_metric_label(chunk.metadata.get(metadata_key))
        if label:
            add_metric_label(labels, seen, label)
    return labels


def normalized_metric_label(value: Any) -> str:
    if value is None:
        return "unspecified"
    label = str(value).strip().lower()
    return label or "unspecified"


def add_metric_label(labels: list[str], seen: set[str], label: str) -> None:
    if label in seen:
        return
    seen.add(label)
    labels.append(label)


def hit_asset_ids(hit) -> list[str]:
    seen: set[str] = set()
    asset_ids: list[str] = []
    chunk = hit_chunk(hit)
    for asset_id in chunk_asset_ids(chunk) if chunk is not None else []:
        add_asset_id(asset_ids, seen, asset_id)
    for evidence_chunk in hit_evidence_chunks(hit):
        for asset_id in chunk_asset_ids(evidence_chunk):
            add_asset_id(asset_ids, seen, asset_id)
    for payload in getattr(hit, "payloads", []):
        if isinstance(payload, dict):
            add_asset_id(asset_ids, seen, payload.get("asset_id"))
    return asset_ids


def hit_triple_ids(
    hit,
    triples_by_chunk: dict[str, list[GraphTriple]],
    triples_by_asset: dict[str, list[GraphTriple]] | None = None,
) -> list[str]:
    triples_by_asset = triples_by_asset or {}
    seen: set[str] = set()
    triple_ids: list[str] = []
    for payload in getattr(hit, "payloads", []):
        if isinstance(payload, dict):
            add_triple_id_value(triple_ids, seen, payload.get("triple_id"))
            add_triple_id_value(triple_ids, seen, payload.get("triple_ids"))
    chunk = hit_chunk(hit)
    if chunk is not None:
        add_triple_ids_for_chunk(triple_ids, seen, chunk, triples_by_chunk)
    for evidence_chunk in hit_evidence_chunks(hit):
        add_triple_ids_for_chunk(triple_ids, seen, evidence_chunk, triples_by_chunk)
    for asset_id in hit_asset_ids(hit):
        add_triple_ids(triple_ids, seen, triples_by_asset.get(asset_id, []))
    return triple_ids


def add_triple_ids_for_chunk(
    triple_ids: list[str],
    seen: set[str],
    chunk: DocumentChunk,
    triples_by_chunk: dict[str, list[GraphTriple]],
) -> None:
    for chunk_id in chunk_alias_ids(chunk):
        add_triple_ids(triple_ids, seen, triples_by_chunk.get(chunk_id, []))


def add_triple_ids(triple_ids: list[str], seen: set[str], triples: list[GraphTriple]):
    for triple in triples:
        if triple.triple_id not in seen:
            seen.add(triple.triple_id)
            triple_ids.append(triple.triple_id)


def add_triple_id_value(triple_ids: list[str], seen: set[str], value: Any) -> None:
    for triple_id in string_values(value):
        if triple_id not in seen:
            seen.add(triple_id)
            triple_ids.append(triple_id)


def index_triples_by_chunk(triples: list[GraphTriple]) -> dict[str, list[GraphTriple]]:
    triples_by_chunk: dict[str, list[GraphTriple]] = {}
    for triple in triples:
        triples_by_chunk.setdefault(triple.chunk_id, []).append(triple)
    return triples_by_chunk


def index_triples_by_asset(triples: list[GraphTriple]) -> dict[str, list[GraphTriple]]:
    triples_by_asset: dict[str, list[GraphTriple]] = {}
    for triple in triples:
        for asset_id in sorted(triple_asset_ids(triple)):
            triples_by_asset.setdefault(asset_id, []).append(triple)
    return triples_by_asset


def add_asset_id(asset_ids: list[str], seen: set[str], asset_id: str | list[str] | None):
    for value in string_values(asset_id):
        if value not in seen:
            seen.add(value)
            asset_ids.append(value)


def hit_evidence_chunks(hit) -> list[DocumentChunk]:
    return [chunk for chunk in getattr(hit, "evidence_chunks", []) if chunk is not None]


def first_matching_chunk_alias(chunk: DocumentChunk, expected_chunk_ids: set[str]) -> str | None:
    for chunk_id in chunk_alias_ids(chunk):
        if chunk_id in expected_chunk_ids:
            return chunk_id
    return None


def chunk_alias_ids(chunk: DocumentChunk) -> list[str]:
    aliases = [chunk.chunk_id]
    for key in ("source_chunk_id", "parent_chunk_id"):
        value = chunk.metadata.get(key)
        if isinstance(value, str):
            aliases.append(value)
    return stable_ordered_values(aliases)


def stable_ordered_values(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def target_metrics(results: list[RetrievalCaseResult]) -> dict[str, RetrievalTargetMetric]:
    metrics = {}
    for target in ["page", "chunk", "asset", "triple"]:
        target_results = [result for result in results if target in result.target_matches]
        if not target_results:
            continue
        passed_count = sum(1 for result in target_results if result.target_matches.get(target))
        target_count = sum(len(expected_result_target_keys(result, target)) for result in target_results)
        matched_target_count = sum(
            len(matched_result_target_keys(result, target)) for result in target_results
        )
        metrics[target] = RetrievalTargetMetric(
            expected_count=len(target_results),
            passed_count=passed_count,
            recall_at_k=passed_count / len(target_results),
            mrr=sum(result.target_reciprocal_ranks.get(target, 0.0) for result in target_results)
            / len(target_results),
            target_count=target_count,
            matched_target_count=matched_target_count,
            coverage_at_k=matched_target_count / target_count if target_count else 0.0,
            ndcg_at_k=sum(result_target_ndcg_score(result, target) for result in target_results)
            / len(target_results),
            failed_queries=[
                result.query for result in target_results if not result.target_matches.get(target)
            ],
        )
    return metrics


def source_metrics(
    results: list[RetrievalCaseResult],
    family: bool = False,
) -> dict[str, RetrievalSourceMetric]:
    total_expected_targets = sum(result.expected_target_count for result in results)
    total_excluded_targets = sum(result.excluded_target_count for result in results)
    accumulators: dict[str, dict[str, Any]] = {}

    for result in results:
        sources_seen_in_query: set[str] = set()
        relevant_sources_seen_in_query: set[str] = set()
        excluded_sources_seen_in_query: set[str] = set()
        best_rank_by_source: dict[str, int] = {}
        matched_targets_by_source: dict[str, set[str]] = {}
        excluded_targets_by_source: dict[str, set[str]] = {}

        for index, raw_sources in enumerate(result.top_sources):
            rank = index + 1
            matched_targets = set(result.top_matched_targets[index]) if index < len(result.top_matched_targets) else set()
            excluded_targets = (
                set(result.top_excluded_targets[index])
                if index < len(result.top_excluded_targets)
                else set()
            )
            for source in metric_source_keys(raw_sources, family=family):
                accumulator = accumulators.setdefault(
                    source,
                    {
                        "query_count": 0,
                        "relevant_query_count": 0,
                        "excluded_query_count": 0,
                        "hit_count": 0,
                        "relevant_hit_count": 0,
                        "excluded_hit_count": 0,
                        "matched_target_count": 0,
                        "excluded_matched_target_count": 0,
                        "rank_sum": 0,
                        "rank_count": 0,
                    },
                )
                sources_seen_in_query.add(source)
                accumulator["hit_count"] += 1
                if matched_targets:
                    accumulator["relevant_hit_count"] += 1
                    relevant_sources_seen_in_query.add(source)
                    best_rank_by_source[source] = min(best_rank_by_source.get(source, rank), rank)
                    matched_targets_by_source.setdefault(source, set()).update(matched_targets)
                if excluded_targets:
                    accumulator["excluded_hit_count"] += 1
                    excluded_sources_seen_in_query.add(source)
                    excluded_targets_by_source.setdefault(source, set()).update(excluded_targets)

        for source in sources_seen_in_query:
            accumulators[source]["query_count"] += 1
        for source in relevant_sources_seen_in_query:
            accumulators[source]["relevant_query_count"] += 1
            accumulators[source]["rank_sum"] += best_rank_by_source[source]
            accumulators[source]["rank_count"] += 1
        for source in excluded_sources_seen_in_query:
            accumulators[source]["excluded_query_count"] += 1
        for source, matched_targets in matched_targets_by_source.items():
            accumulators[source]["matched_target_count"] += len(matched_targets)
        for source, excluded_targets in excluded_targets_by_source.items():
            accumulators[source]["excluded_matched_target_count"] += len(excluded_targets)

    return {
        source: RetrievalSourceMetric(
            query_count=values["query_count"],
            relevant_query_count=values["relevant_query_count"],
            excluded_query_count=values["excluded_query_count"],
            hit_count=values["hit_count"],
            relevant_hit_count=values["relevant_hit_count"],
            excluded_hit_count=values["excluded_hit_count"],
            expected_target_count=total_expected_targets,
            matched_target_count=values["matched_target_count"],
            excluded_target_count=total_excluded_targets,
            excluded_matched_target_count=values["excluded_matched_target_count"],
            precision_at_hits=values["relevant_hit_count"] / values["hit_count"]
            if values["hit_count"]
            else 0.0,
            excluded_precision_at_hits=values["excluded_hit_count"] / values["hit_count"]
            if values["hit_count"]
            else 0.0,
            target_coverage_at_k=values["matched_target_count"] / total_expected_targets
            if total_expected_targets
            else 0.0,
            excluded_target_hit_rate=values["excluded_matched_target_count"]
            / total_excluded_targets
            if total_excluded_targets
            else 0.0,
            mean_relevant_rank=values["rank_sum"] / values["rank_count"]
            if values["rank_count"]
            else 0.0,
        )
        for source, values in sorted(accumulators.items())
    }


def hit_group_metrics(
    results: list[RetrievalCaseResult],
    group_attr: str,
) -> dict[str, RetrievalSourceMetric]:
    total_expected_targets = sum(result.expected_target_count for result in results)
    total_excluded_targets = sum(result.excluded_target_count for result in results)
    accumulators: dict[str, dict[str, Any]] = {}

    for result in results:
        groups_by_hit = getattr(result, group_attr, [])
        groups_seen_in_query: set[str] = set()
        relevant_groups_seen_in_query: set[str] = set()
        excluded_groups_seen_in_query: set[str] = set()
        best_rank_by_group: dict[str, int] = {}
        matched_targets_by_group: dict[str, set[str]] = {}
        excluded_targets_by_group: dict[str, set[str]] = {}

        for index, groups in enumerate(groups_by_hit):
            rank = index + 1
            matched_targets = set(result.top_matched_targets[index]) if index < len(result.top_matched_targets) else set()
            excluded_targets = (
                set(result.top_excluded_targets[index])
                if index < len(result.top_excluded_targets)
                else set()
            )
            for group in groups:
                accumulator = accumulators.setdefault(
                    group,
                    {
                        "query_count": 0,
                        "relevant_query_count": 0,
                        "excluded_query_count": 0,
                        "hit_count": 0,
                        "relevant_hit_count": 0,
                        "excluded_hit_count": 0,
                        "matched_target_count": 0,
                        "excluded_matched_target_count": 0,
                        "rank_sum": 0,
                        "rank_count": 0,
                    },
                )
                groups_seen_in_query.add(group)
                accumulator["hit_count"] += 1
                if matched_targets:
                    accumulator["relevant_hit_count"] += 1
                    relevant_groups_seen_in_query.add(group)
                    best_rank_by_group[group] = min(best_rank_by_group.get(group, rank), rank)
                    matched_targets_by_group.setdefault(group, set()).update(matched_targets)
                if excluded_targets:
                    accumulator["excluded_hit_count"] += 1
                    excluded_groups_seen_in_query.add(group)
                    excluded_targets_by_group.setdefault(group, set()).update(excluded_targets)

        for group in groups_seen_in_query:
            accumulators[group]["query_count"] += 1
        for group in relevant_groups_seen_in_query:
            accumulators[group]["relevant_query_count"] += 1
            accumulators[group]["rank_sum"] += best_rank_by_group[group]
            accumulators[group]["rank_count"] += 1
        for group in excluded_groups_seen_in_query:
            accumulators[group]["excluded_query_count"] += 1
        for group, matched_targets in matched_targets_by_group.items():
            accumulators[group]["matched_target_count"] += len(matched_targets)
        for group, excluded_targets in excluded_targets_by_group.items():
            accumulators[group]["excluded_matched_target_count"] += len(excluded_targets)

    return {
        group: RetrievalSourceMetric(
            query_count=values["query_count"],
            relevant_query_count=values["relevant_query_count"],
            excluded_query_count=values["excluded_query_count"],
            hit_count=values["hit_count"],
            relevant_hit_count=values["relevant_hit_count"],
            excluded_hit_count=values["excluded_hit_count"],
            expected_target_count=total_expected_targets,
            matched_target_count=values["matched_target_count"],
            excluded_target_count=total_excluded_targets,
            excluded_matched_target_count=values["excluded_matched_target_count"],
            precision_at_hits=values["relevant_hit_count"] / values["hit_count"]
            if values["hit_count"]
            else 0.0,
            excluded_precision_at_hits=values["excluded_hit_count"] / values["hit_count"]
            if values["hit_count"]
            else 0.0,
            target_coverage_at_k=values["matched_target_count"] / total_expected_targets
            if total_expected_targets
            else 0.0,
            excluded_target_hit_rate=values["excluded_matched_target_count"]
            / total_excluded_targets
            if total_excluded_targets
            else 0.0,
            mean_relevant_rank=values["rank_sum"] / values["rank_count"]
            if values["rank_count"]
            else 0.0,
        )
        for group, values in sorted(accumulators.items())
    }


def case_group_metrics(
    results: list[RetrievalCaseResult],
    cases: list[RetrievalCase],
) -> dict[str, dict[str, RetrievalCaseGroupMetric]]:
    groups: dict[str, dict[str, list[RetrievalCaseResult]]] = {}
    for result, case in zip(results, cases):
        for group_name, group_value in case_group_labels(case):
            groups.setdefault(group_name, {}).setdefault(group_value, []).append(result)
    return {
        group_name: {
            group_value: summarize_case_group(group_results)
            for group_value, group_results in sorted(values.items())
        }
        for group_name, values in sorted(groups.items())
    }


def case_group_labels(case: RetrievalCase) -> list[tuple[str, str]]:
    labels = []
    for key in CASE_GROUP_METADATA_KEYS:
        value = case.metadata.get(key)
        for label in metadata_group_values(value):
            labels.append((normalized_metric_label(key), normalized_metric_label(label)))
    labels.append(("graph_expand", "true" if case.graph_expand else "false"))
    return labels


def metadata_group_values(value: Any) -> list[str]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, (str, int, float)):
                text = str(item).strip()
                if text:
                    values.append(text)
        return stable_ordered_values(values)
    return []


def summarize_case_group(results: list[RetrievalCaseResult]) -> RetrievalCaseGroupMetric:
    expected_results = [result for result in results if result.expected_target_count > 0]
    target_count = sum(result.expected_target_count for result in expected_results)
    return RetrievalCaseGroupMetric(
        case_count=len(results),
        expected_case_count=len(expected_results),
        passed_count=sum(1 for result in results if result.passed),
        failed_count=sum(1 for result in results if not result.passed),
        recall_at_k=sum(1 for result in expected_results if result.matched_rank is not None)
        / len(expected_results)
        if expected_results
        else 0.0,
        mrr=sum(result.reciprocal_rank for result in expected_results) / len(expected_results)
        if expected_results
        else 0.0,
        target_count=target_count,
        matched_target_count=sum(result.matched_target_count for result in expected_results),
        target_coverage_at_k=sum(result.matched_target_count for result in expected_results)
        / target_count
        if target_count
        else 0.0,
        ndcg_at_k=sum(result.target_ndcg_at_k for result in expected_results) / len(expected_results)
        if expected_results
        else 0.0,
        precision_at_k=sum(result.precision_at_k for result in expected_results) / len(expected_results)
        if expected_results
        else 0.0,
        mean_latency_ms=sum(result.latency_ms for result in results) / len(results) if results else 0.0,
        failed_queries=[result.query for result in results if not result.passed],
    )


def metric_source_keys(sources: list[str], family: bool = False) -> set[str]:
    if family:
        return {source_family(source) for source in sources}
    return set(sources)


def source_family(source: str) -> str:
    normalized = source.strip().lower()
    if "triple_dense" in normalized:
        return "graph"
    if "caption_dense" in normalized or "object_dense" in normalized or "image_dense" in normalized:
        return "visual"
    if normalized == "dense" or "text_dense" in normalized:
        return "dense_text"
    if normalized.startswith("rerank:"):
        return "reranker"
    if normalized == "bm25" or "lexical" in normalized:
        return "lexical"
    if normalized == "graph":
        return "graph"
    return normalized.split(":", 1)[0] if normalized else "unknown"


def expected_result_target_keys(result: RetrievalCaseResult, target: str) -> set[str]:
    if target == "page":
        return {f"page:{page}" for page in result.expected_pages}
    if target == "chunk":
        return {f"chunk:{chunk_id}" for chunk_id in result.expected_chunk_ids}
    if target == "asset":
        return {f"asset:{asset_id}" for asset_id in result.expected_asset_ids}
    if target == "triple":
        return {f"triple:{triple_id}" for triple_id in result.expected_triple_ids}
    return set()


def matched_result_target_keys(result: RetrievalCaseResult, target: str) -> set[str]:
    prefix = f"{target}:"
    return {
        target_key
        for matched_targets in result.top_matched_targets
        for target_key in matched_targets
        if target_key.startswith(prefix)
    }


def result_target_ndcg_score(result: RetrievalCaseResult, target: str) -> float:
    expected_keys = expected_result_target_keys(result, target)
    if not expected_keys:
        return 0.0
    prefix = f"{target}:"
    ranks = {
        target_key: rank
        for target_key, rank in result.target_key_matched_ranks.items()
        if target_key.startswith(prefix)
    }
    return target_ndcg_score(expected_keys, ranks)


def matched_target_key_ranks(top_matched_targets: list[list[str]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for index, matched_targets in enumerate(top_matched_targets):
        rank = index + 1
        for target in matched_targets:
            ranks.setdefault(target, rank)
    return ranks


def target_ndcg_score(expected_targets: set[str], matched_ranks: dict[str, int]) -> float:
    if not expected_targets:
        return 0.0
    discounted_gain = sum(
        1.0 / math.log2(matched_ranks[target] + 1)
        for target in expected_targets
        if target in matched_ranks
    )
    return discounted_gain / len(expected_targets)


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
