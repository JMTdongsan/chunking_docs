from __future__ import annotations

from pydantic import BaseModel, Field

from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.compare import (
    PAIRWISE_BOOTSTRAP_SAMPLES,
    PAIRWISE_CONFIDENCE_LEVEL,
    bootstrap_mean_interval,
    case_mean_target_rank,
    compare_case_results,
    first_relevant_rank,
    mean,
    results_by_query,
    stable_seed,
)
from chunking_docs.evaluation.gate import (
    RetrievalGateCheck,
    case_group_metric_key,
    case_group_target_coverage_checks,
    chunk_strategy_metric_key,
    maximum_check,
    minimum_check,
    retrieval_case_group_metrics,
    retrieval_chunk_strategy_metrics,
    retrieval_rank_metrics,
    retrieval_role_metric_key,
    retrieval_role_metrics_payload,
    retrieval_source_metrics,
    retrieval_source_family_metrics,
    retrieval_target_metrics,
    source_excluded_target_hit_rate_checks,
    source_family_excluded_target_hit_rate_checks,
    source_metric_key,
    source_target_coverage_checks,
    source_family_metric_key,
    source_family_target_coverage_checks,
    target_type_coverage_checks,
    target_type_metric_key,
)
from chunking_docs.evaluation.retrieval import RetrievalCase, RetrievalEvaluation, evaluate_retrieval
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset


class RetrievalAblationMode(BaseModel):
    name: str
    use_dense: bool = True
    use_bm25: bool = True
    use_graph: bool = False
    graph_expand: bool = False
    include_asset_text: bool = True


class RetrievalAblationRow(BaseModel):
    mode: RetrievalAblationMode
    evaluation: RetrievalEvaluation


class AblationBestModeMetric(BaseModel):
    mode: str | None = None
    value: float | None = None


class AblationPairwiseComparison(BaseModel):
    candidate: str
    baseline: str
    shared_query_count: int
    candidate_win_count: int = 0
    baseline_win_count: int = 0
    tie_count: int = 0
    candidate_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    mean_reciprocal_rank_delta: float = 0.0
    mean_target_coverage_delta: float = 0.0
    mean_target_ndcg_delta: float = 0.0
    mean_precision_delta: float = 0.0
    mean_first_relevant_rank_delta: float | None = None
    mean_target_rank_delta: float | None = None
    mean_latency_delta_ms: float | None = None
    bootstrap_samples: int = 0
    confidence_level: float = 0.95
    reciprocal_rank_delta_ci_low: float | None = None
    reciprocal_rank_delta_ci_high: float | None = None
    target_coverage_delta_ci_low: float | None = None
    target_coverage_delta_ci_high: float | None = None
    target_ndcg_delta_ci_low: float | None = None
    target_ndcg_delta_ci_high: float | None = None
    precision_delta_ci_low: float | None = None
    precision_delta_ci_high: float | None = None
    first_relevant_rank_delta_ci_low: float | None = None
    first_relevant_rank_delta_ci_high: float | None = None
    target_rank_delta_ci_low: float | None = None
    target_rank_delta_ci_high: float | None = None
    latency_delta_ci_low_ms: float | None = None
    latency_delta_ci_high_ms: float | None = None


class RetrievalAblationReport(BaseModel):
    rows: list[RetrievalAblationRow]
    best_by_recall: str | None
    best_by_target_coverage: str | None
    best_by_target_ndcg: str | None
    best_by_mrr: str | None
    fastest_by_mean_latency: str | None
    case_group_best_modes: dict[
        str,
        dict[str, dict[str, AblationBestModeMetric]],
    ] = Field(default_factory=dict)
    pairwise: list[AblationPairwiseComparison] = Field(default_factory=list)


class RetrievalAblationGateReport(BaseModel):
    passed: bool
    mode: str
    baseline_mode: str | None = None
    metrics: dict[str, float]
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    chunk_strategy_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    retrieval_role_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    case_group_metrics: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    baseline_target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_source_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_chunk_strategy_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_retrieval_role_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_case_group_metrics: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict
    )
    pairwise_metrics: dict[str, float | None] = Field(default_factory=dict)
    best_by_recall: str | None = None
    best_by_target_coverage: str | None = None
    best_by_target_ndcg: str | None = None
    best_by_mrr: str | None = None
    fastest_by_mean_latency: str | None = None
    case_group_best_modes: dict[
        str,
        dict[str, dict[str, AblationBestModeMetric]],
    ] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    checks: list[RetrievalGateCheck] = Field(default_factory=list)


class QdrantVectorAblationMode(BaseModel):
    name: str
    vector_names: list[str] = Field(default_factory=list)
    graph_expand: bool = False


class QdrantVectorAblationRow(BaseModel):
    mode: QdrantVectorAblationMode
    evaluation: RetrievalEvaluation


class QdrantVectorAblationReport(BaseModel):
    rows: list[QdrantVectorAblationRow]
    best_by_recall: str | None
    best_by_target_coverage: str | None
    best_by_target_ndcg: str | None
    best_by_mrr: str | None
    fastest_by_mean_latency: str | None
    case_group_best_modes: dict[
        str,
        dict[str, dict[str, AblationBestModeMetric]],
    ] = Field(default_factory=dict)
    pairwise: list[AblationPairwiseComparison] = Field(default_factory=list)


class QdrantVectorAblationGateReport(BaseModel):
    passed: bool
    mode: str
    baseline_mode: str | None = None
    vector_names: list[str] = Field(default_factory=list)
    graph_expand: bool = False
    metrics: dict[str, float]
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    chunk_strategy_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    retrieval_role_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    case_group_metrics: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    pairwise_metrics: dict[str, float | None] = Field(default_factory=dict)
    best_by_recall: str | None = None
    best_by_target_coverage: str | None = None
    best_by_target_ndcg: str | None = None
    best_by_mrr: str | None = None
    fastest_by_mean_latency: str | None = None
    case_group_best_modes: dict[
        str,
        dict[str, dict[str, AblationBestModeMetric]],
    ] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    checks: list[RetrievalGateCheck] = Field(default_factory=list)


DEFAULT_ABLATION_MODES = {
    "dense": RetrievalAblationMode(name="dense", use_dense=True, use_bm25=False),
    "bm25": RetrievalAblationMode(name="bm25", use_dense=False, use_bm25=True),
    "bm25_text": RetrievalAblationMode(
        name="bm25_text",
        use_dense=False,
        use_bm25=True,
        include_asset_text=False,
    ),
    "bm25_visual": RetrievalAblationMode(
        name="bm25_visual",
        use_dense=False,
        use_bm25=True,
        include_asset_text=True,
    ),
    "hybrid": RetrievalAblationMode(name="hybrid", use_dense=True, use_bm25=True),
    "hybrid_text": RetrievalAblationMode(
        name="hybrid_text",
        use_dense=True,
        use_bm25=True,
        include_asset_text=False,
    ),
    "hybrid_visual": RetrievalAblationMode(
        name="hybrid_visual",
        use_dense=True,
        use_bm25=True,
        include_asset_text=True,
    ),
    "graph": RetrievalAblationMode(
        name="graph",
        use_dense=False,
        use_bm25=False,
        use_graph=True,
    ),
    "hybrid_graph": RetrievalAblationMode(
        name="hybrid_graph",
        use_dense=True,
        use_bm25=True,
        use_graph=True,
        graph_expand=True,
    ),
}


DEFAULT_QDRANT_VECTOR_ABLATION_MODES = {
    "text": QdrantVectorAblationMode(name="text", vector_names=["text_dense"]),
    "caption": QdrantVectorAblationMode(name="caption", vector_names=["caption_dense"]),
    "object": QdrantVectorAblationMode(name="object", vector_names=["object_dense"]),
    "image": QdrantVectorAblationMode(name="image", vector_names=["image_dense"]),
    "triple": QdrantVectorAblationMode(name="triple", vector_names=["triple_dense"]),
    "text_caption": QdrantVectorAblationMode(
        name="text_caption",
        vector_names=["text_dense", "caption_dense"],
    ),
    "text_object": QdrantVectorAblationMode(
        name="text_object",
        vector_names=["text_dense", "object_dense"],
    ),
    "caption_object": QdrantVectorAblationMode(
        name="caption_object",
        vector_names=["caption_dense", "object_dense"],
    ),
    "text_triple": QdrantVectorAblationMode(
        name="text_triple",
        vector_names=["text_dense", "triple_dense"],
    ),
    "text_image": QdrantVectorAblationMode(
        name="text_image",
        vector_names=["text_dense", "image_dense"],
    ),
    "caption_image": QdrantVectorAblationMode(
        name="caption_image",
        vector_names=["caption_dense", "image_dense"],
    ),
    "all": QdrantVectorAblationMode(
        name="all",
        vector_names=["text_dense", "caption_dense", "image_dense"],
    ),
    "all_with_object": QdrantVectorAblationMode(
        name="all_with_object",
        vector_names=["text_dense", "caption_dense", "object_dense", "image_dense"],
    ),
    "all_with_triple": QdrantVectorAblationMode(
        name="all_with_triple",
        vector_names=["text_dense", "caption_dense", "image_dense", "triple_dense"],
    ),
    "all_with_object_triple": QdrantVectorAblationMode(
        name="all_with_object_triple",
        vector_names=[
            "text_dense",
            "caption_dense",
            "object_dense",
            "image_dense",
            "triple_dense",
        ],
    ),
    "text_caption_graph": QdrantVectorAblationMode(
        name="text_caption_graph",
        vector_names=["text_dense", "caption_dense"],
        graph_expand=True,
    ),
    "text_object_graph": QdrantVectorAblationMode(
        name="text_object_graph",
        vector_names=["text_dense", "object_dense"],
        graph_expand=True,
    ),
    "text_triple_graph": QdrantVectorAblationMode(
        name="text_triple_graph",
        vector_names=["text_dense", "triple_dense"],
        graph_expand=True,
    ),
    "all_graph": QdrantVectorAblationMode(
        name="all_graph",
        vector_names=["text_dense", "caption_dense", "image_dense"],
        graph_expand=True,
    ),
    "all_with_object_graph": QdrantVectorAblationMode(
        name="all_with_object_graph",
        vector_names=["text_dense", "caption_dense", "object_dense", "image_dense"],
        graph_expand=True,
    ),
    "all_with_triple_graph": QdrantVectorAblationMode(
        name="all_with_triple_graph",
        vector_names=["text_dense", "caption_dense", "image_dense", "triple_dense"],
        graph_expand=True,
    ),
    "all_with_object_triple_graph": QdrantVectorAblationMode(
        name="all_with_object_triple_graph",
        vector_names=[
            "text_dense",
            "caption_dense",
            "object_dense",
            "image_dense",
            "triple_dense",
        ],
        graph_expand=True,
    ),
}


def evaluate_retrieval_ablation(
    chunks: list[DocumentChunk],
    triples: list[GraphTriple],
    cases: list[RetrievalCase],
    modes: list[RetrievalAblationMode] | None = None,
    top_k: int = 5,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = False,
    repeat: int = 1,
    fusion_weights: dict[str, float] | None = None,
    assets: list[VisualAsset] | None = None,
) -> RetrievalAblationReport:
    rows = [
        evaluate_retrieval_ablation_mode(
            mode,
            chunks=chunks,
            triples=triples,
            cases=cases,
            assets=assets,
            top_k=top_k,
            tokenizer_config=tokenizer_config,
            collapse_hierarchical=collapse_hierarchical,
            repeat=repeat,
            fusion_weights=fusion_weights,
        )
        for mode in (modes or list(DEFAULT_ABLATION_MODES.values()))
    ]
    rows.sort(
        key=lambda row: (
            row.evaluation.recall_at_k,
            row.evaluation.target_coverage_at_k,
            row.evaluation.mean_target_ndcg_at_k,
            row.evaluation.mrr,
            row.evaluation.hit_rate,
        ),
        reverse=True,
    )
    return RetrievalAblationReport(
        rows=rows,
        best_by_recall=rows[0].mode.name if rows else None,
        best_by_target_coverage=max(
            rows,
            key=lambda row: (row.evaluation.target_coverage_at_k, row.evaluation.recall_at_k),
        ).mode.name
        if rows
        else None,
        best_by_target_ndcg=max(
            rows,
            key=lambda row: (row.evaluation.mean_target_ndcg_at_k, row.evaluation.recall_at_k),
        ).mode.name
        if rows
        else None,
        best_by_mrr=max(rows, key=lambda row: row.evaluation.mrr).mode.name if rows else None,
        fastest_by_mean_latency=min(rows, key=lambda row: row.evaluation.mean_latency_ms).mode.name
        if rows
        else None,
        case_group_best_modes=case_group_best_modes(rows),
        pairwise=ablation_pairwise_comparisons(rows),
    )


def evaluate_retrieval_ablation_mode(
    mode: RetrievalAblationMode,
    chunks: list[DocumentChunk],
    triples: list[GraphTriple],
    cases: list[RetrievalCase],
    assets: list[VisualAsset] | None = None,
    top_k: int = 5,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = False,
    repeat: int = 1,
    fusion_weights: dict[str, float] | None = None,
) -> RetrievalAblationRow:
    evaluation = evaluate_retrieval(
        chunks=chunks,
        triples=triples,
        cases=cases,
        assets=assets if mode.include_asset_text else None,
        top_k=top_k,
        tokenizer_config=tokenizer_config,
        collapse_hierarchical=collapse_hierarchical,
        graph_expand_override=mode.graph_expand,
        use_dense=mode.use_dense,
        use_bm25=mode.use_bm25,
        use_graph=mode.use_graph,
        repeat=repeat,
        fusion_weights=fusion_weights,
    )
    evaluation.metadata["ablation_mode"] = mode.name
    evaluation.metadata["include_asset_text"] = mode.include_asset_text
    return RetrievalAblationRow(mode=mode, evaluation=evaluation)


def parse_ablation_modes(value: str) -> list[RetrievalAblationMode]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        return list(DEFAULT_ABLATION_MODES.values())
    unknown = sorted(set(names) - set(DEFAULT_ABLATION_MODES))
    if unknown:
        raise ValueError(f"Unsupported ablation modes: {', '.join(unknown)}")
    return [DEFAULT_ABLATION_MODES[name] for name in names]


def gate_retrieval_ablation(
    report: RetrievalAblationReport,
    mode: str,
    baseline_mode: str | None = None,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    max_excluded_target_hit_rate: float | None = None,
    max_excluded_query_hit_rate: float | None = None,
    max_excluded_hit_query_count: int | None = None,
    min_target_type_coverage: dict[str, float] | None = None,
    min_source_target_coverage: dict[str, float] | None = None,
    min_source_family_target_coverage: dict[str, float] | None = None,
    max_source_excluded_target_hit_rate: dict[str, float] | None = None,
    max_source_family_excluded_target_hit_rate: dict[str, float] | None = None,
    min_case_group_target_coverage: dict[str, float] | None = None,
    min_recall_lift: float | None = None,
    min_target_coverage_lift: float | None = None,
    min_target_ndcg_lift: float | None = None,
    min_mrr_lift: float | None = None,
    min_precision_lift: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
    min_pairwise_shared_queries: int | None = None,
    min_pairwise_win_rate: float | None = None,
    min_pairwise_target_coverage_lift: float | None = None,
    min_pairwise_target_ndcg_lift: float | None = None,
    min_pairwise_mrr_lift: float | None = None,
    min_pairwise_precision_lift: float | None = None,
    min_pairwise_target_coverage_ci_low: float | None = None,
    min_pairwise_target_ndcg_ci_low: float | None = None,
    min_pairwise_mrr_ci_low: float | None = None,
    min_pairwise_precision_ci_low: float | None = None,
    max_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_pairwise_mean_target_rank_delta: float | None = None,
    max_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_pairwise_target_rank_delta_ci_high: float | None = None,
    max_pairwise_mean_latency_delta_ms: float | None = None,
    require_best_by_recall: bool = False,
    require_best_by_target_coverage: bool = False,
    require_best_by_target_ndcg: bool = False,
    require_fastest_by_mean_latency: bool = False,
) -> RetrievalAblationGateReport:
    row = retrieval_ablation_row(report, mode)
    if row is None:
        raise ValueError(f"Retrieval ablation mode not found: {mode}")

    baseline_row = None
    if baseline_mode is not None:
        baseline_row = retrieval_ablation_row(report, baseline_mode)
        if baseline_row is None:
            raise ValueError(f"Baseline retrieval ablation mode not found: {baseline_mode}")
    elif requires_baseline(
        min_recall_lift,
        min_target_coverage_lift,
        min_target_ndcg_lift,
        min_mrr_lift,
        min_precision_lift,
        max_mean_latency_ratio,
        max_p95_latency_ratio,
        min_pairwise_shared_queries,
        min_pairwise_win_rate,
        min_pairwise_target_coverage_lift,
        min_pairwise_target_ndcg_lift,
        min_pairwise_mrr_lift,
        min_pairwise_precision_lift,
        min_pairwise_target_coverage_ci_low,
        min_pairwise_target_ndcg_ci_low,
        min_pairwise_mrr_ci_low,
        min_pairwise_precision_ci_low,
        max_pairwise_mean_first_relevant_rank_delta,
        max_pairwise_mean_target_rank_delta,
        max_pairwise_first_relevant_rank_delta_ci_high,
        max_pairwise_target_rank_delta_ci_high,
        max_pairwise_mean_latency_delta_ms,
    ):
        raise ValueError("A baseline mode is required for lift, latency-ratio, or pairwise checks.")

    target_metrics = retrieval_target_metrics(row.evaluation)
    source_metrics = retrieval_source_metrics(row.evaluation)
    source_family_metrics = retrieval_source_family_metrics(row.evaluation)
    chunk_strategy_metrics = retrieval_chunk_strategy_metrics(row.evaluation)
    retrieval_role_metrics = retrieval_role_metrics_payload(row.evaluation)
    case_group_metrics = retrieval_case_group_metrics(row.evaluation)
    metrics = qdrant_vector_ablation_metrics(
        row.evaluation,
        target_metrics,
        source_metrics,
        source_family_metrics,
        chunk_strategy_metrics,
        retrieval_role_metrics,
        case_group_metrics,
    )
    baseline_target_metrics = retrieval_target_metrics(baseline_row.evaluation) if baseline_row else {}
    baseline_source_metrics = retrieval_source_metrics(baseline_row.evaluation) if baseline_row else {}
    baseline_source_family_metrics = (
        retrieval_source_family_metrics(baseline_row.evaluation) if baseline_row else {}
    )
    baseline_chunk_strategy_metrics = (
        retrieval_chunk_strategy_metrics(baseline_row.evaluation) if baseline_row else {}
    )
    baseline_retrieval_role_metrics = (
        retrieval_role_metrics_payload(baseline_row.evaluation) if baseline_row else {}
    )
    baseline_case_group_metrics = (
        retrieval_case_group_metrics(baseline_row.evaluation) if baseline_row else {}
    )
    baseline_metrics = (
        qdrant_vector_ablation_metrics(
            baseline_row.evaluation,
            baseline_target_metrics,
            baseline_source_metrics,
            baseline_source_family_metrics,
            baseline_chunk_strategy_metrics,
            baseline_retrieval_role_metrics,
            baseline_case_group_metrics,
        )
        if baseline_row
        else {}
    )
    pairwise_metrics = ablation_pairwise_metrics(
        find_ablation_pairwise_comparison(
            report.pairwise,
            mode,
            baseline_mode,
        )
        if baseline_mode is not None
        else None
    )
    checks = [
        minimum_check("min_recall_at_k", "recall_at_k", metrics, min_recall_at_k),
        minimum_check(
            "min_target_coverage_at_k",
            "target_coverage_at_k",
            metrics,
            min_target_coverage_at_k,
        ),
        minimum_check(
            "min_target_ndcg_at_k",
            "mean_target_ndcg_at_k",
            metrics,
            min_target_ndcg_at_k,
        ),
        minimum_check("min_mrr", "mrr", metrics, min_mrr),
        minimum_check("min_precision_at_k", "mean_precision_at_k", metrics, min_precision_at_k),
    ]
    if max_failed_queries is not None:
        checks.append(
            maximum_check(
                "max_failed_queries",
                "failed_query_count",
                metrics,
                float(max_failed_queries),
            )
        )
    if max_mean_latency_ms is not None:
        checks.append(maximum_check("max_mean_latency_ms", "mean_latency_ms", metrics, max_mean_latency_ms))
    if max_p95_latency_ms is not None:
        checks.append(maximum_check("max_p95_latency_ms", "p95_latency_ms", metrics, max_p95_latency_ms))
    checks.extend(
        excluded_target_limit_checks(
            metrics,
            max_excluded_target_hit_rate=max_excluded_target_hit_rate,
            max_excluded_query_hit_rate=max_excluded_query_hit_rate,
            max_excluded_hit_query_count=max_excluded_hit_query_count,
        )
    )
    checks.extend(
        rank_limit_checks(
            metrics,
            max_mean_first_relevant_rank=max_mean_first_relevant_rank,
            max_p95_first_relevant_rank=max_p95_first_relevant_rank,
            max_mean_target_rank=max_mean_target_rank,
            max_p95_target_rank=max_p95_target_rank,
        )
    )
    checks.extend(target_type_coverage_checks(metrics, min_target_type_coverage or {}))
    checks.extend(source_target_coverage_checks(metrics, min_source_target_coverage or {}))
    checks.extend(
        source_family_target_coverage_checks(metrics, min_source_family_target_coverage or {})
    )
    checks.extend(
        source_excluded_target_hit_rate_checks(
            metrics,
            max_source_excluded_target_hit_rate or {},
        )
    )
    checks.extend(
        source_family_excluded_target_hit_rate_checks(
            metrics,
            max_source_family_excluded_target_hit_rate or {},
        )
    )
    checks.extend(case_group_target_coverage_checks(metrics, min_case_group_target_coverage or {}))
    if baseline_row is not None:
        checks.extend(
            baseline_lift_checks(
                metrics,
                baseline_metrics,
                {
                    "recall_at_k": min_recall_lift,
                    "target_coverage_at_k": min_target_coverage_lift,
                    "mean_target_ndcg_at_k": min_target_ndcg_lift,
                    "mrr": min_mrr_lift,
                    "mean_precision_at_k": min_precision_lift,
                },
            )
        )
        checks.extend(
            latency_ratio_checks(
                metrics,
                baseline_metrics,
                {
                    "mean_latency_ms": max_mean_latency_ratio,
                    "p95_latency_ms": max_p95_latency_ratio,
                },
            )
        )
        checks.extend(
            pairwise_threshold_checks(
                pairwise_metrics,
                min_pairwise_shared_queries=min_pairwise_shared_queries,
                min_pairwise_win_rate=min_pairwise_win_rate,
                min_pairwise_target_coverage_lift=min_pairwise_target_coverage_lift,
                min_pairwise_target_ndcg_lift=min_pairwise_target_ndcg_lift,
                min_pairwise_mrr_lift=min_pairwise_mrr_lift,
                min_pairwise_precision_lift=min_pairwise_precision_lift,
                min_pairwise_target_coverage_ci_low=min_pairwise_target_coverage_ci_low,
                min_pairwise_target_ndcg_ci_low=min_pairwise_target_ndcg_ci_low,
                min_pairwise_mrr_ci_low=min_pairwise_mrr_ci_low,
                min_pairwise_precision_ci_low=min_pairwise_precision_ci_low,
                max_pairwise_mean_first_relevant_rank_delta=(
                    max_pairwise_mean_first_relevant_rank_delta
                ),
                max_pairwise_mean_target_rank_delta=max_pairwise_mean_target_rank_delta,
                max_pairwise_first_relevant_rank_delta_ci_high=(
                    max_pairwise_first_relevant_rank_delta_ci_high
                ),
                max_pairwise_target_rank_delta_ci_high=(
                    max_pairwise_target_rank_delta_ci_high
                ),
                max_pairwise_mean_latency_delta_ms=max_pairwise_mean_latency_delta_ms,
            )
        )
    if require_best_by_recall:
        checks.append(best_mode_check("require_best_by_recall", mode, report.best_by_recall))
    if require_best_by_target_coverage:
        checks.append(
            best_mode_check(
                "require_best_by_target_coverage",
                mode,
                report.best_by_target_coverage,
            )
        )
    if require_best_by_target_ndcg:
        checks.append(
            best_mode_check(
                "require_best_by_target_ndcg",
                mode,
                report.best_by_target_ndcg,
            )
        )
    if require_fastest_by_mean_latency:
        checks.append(
            best_mode_check(
                "require_fastest_by_mean_latency",
                mode,
                report.fastest_by_mean_latency,
            )
        )

    failed_checks = [check.name for check in checks if not check.passed]
    return RetrievalAblationGateReport(
        passed=not failed_checks,
        mode=mode,
        baseline_mode=baseline_mode,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        target_metrics=target_metrics,
        source_metrics=source_metrics,
        source_family_metrics=source_family_metrics,
        chunk_strategy_metrics=chunk_strategy_metrics,
        retrieval_role_metrics=retrieval_role_metrics,
        case_group_metrics=case_group_metrics,
        baseline_target_metrics=baseline_target_metrics,
        baseline_source_metrics=baseline_source_metrics,
        baseline_source_family_metrics=baseline_source_family_metrics,
        baseline_chunk_strategy_metrics=baseline_chunk_strategy_metrics,
        baseline_retrieval_role_metrics=baseline_retrieval_role_metrics,
        baseline_case_group_metrics=baseline_case_group_metrics,
        pairwise_metrics=pairwise_metrics,
        best_by_recall=report.best_by_recall,
        best_by_target_coverage=report.best_by_target_coverage,
        best_by_target_ndcg=report.best_by_target_ndcg,
        best_by_mrr=report.best_by_mrr,
        fastest_by_mean_latency=report.fastest_by_mean_latency,
        case_group_best_modes=report.case_group_best_modes,
        failed_checks=failed_checks,
        checks=checks,
    )


def retrieval_ablation_row(
    report: RetrievalAblationReport,
    mode: str,
) -> RetrievalAblationRow | None:
    for row in report.rows:
        if row.mode.name == mode:
            return row
    return None


def requires_baseline(*thresholds: float | None) -> bool:
    return any(threshold is not None for threshold in thresholds)


def baseline_lift_checks(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    thresholds: dict[str, float | None],
) -> list[RetrievalGateCheck]:
    checks = []
    for metric, threshold in thresholds.items():
        if threshold is None:
            continue
        actual = metrics[metric]
        baseline = baseline_metrics[metric]
        delta = actual - baseline
        checks.append(
            RetrievalGateCheck(
                name=f"min_{metric}_lift",
                metric=metric,
                operator="actual-baseline>=",
                actual=actual,
                baseline=baseline,
                delta=delta,
                threshold=threshold,
                passed=delta >= threshold,
            )
        )
    return checks


def latency_ratio_checks(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    thresholds: dict[str, float | None],
) -> list[RetrievalGateCheck]:
    checks = []
    for metric, threshold in thresholds.items():
        if threshold is None:
            continue
        actual = metrics[metric]
        baseline = baseline_metrics[metric]
        ratio = safe_ratio(actual, baseline)
        checks.append(
            RetrievalGateCheck(
                name=f"max_{metric}_ratio",
                metric=metric,
                operator="actual/baseline<=",
                actual=actual,
                baseline=baseline,
                ratio=ratio,
                delta=actual - baseline,
                threshold=threshold,
                passed=ratio is not None and ratio <= threshold,
            )
        )
    return checks


def pairwise_threshold_checks(
    pairwise_metrics: dict[str, float | None],
    min_pairwise_shared_queries: int | None = None,
    min_pairwise_win_rate: float | None = None,
    min_pairwise_target_coverage_lift: float | None = None,
    min_pairwise_target_ndcg_lift: float | None = None,
    min_pairwise_mrr_lift: float | None = None,
    min_pairwise_precision_lift: float | None = None,
    min_pairwise_target_coverage_ci_low: float | None = None,
    min_pairwise_target_ndcg_ci_low: float | None = None,
    min_pairwise_mrr_ci_low: float | None = None,
    min_pairwise_precision_ci_low: float | None = None,
    max_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_pairwise_mean_target_rank_delta: float | None = None,
    max_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_pairwise_target_rank_delta_ci_high: float | None = None,
    max_pairwise_mean_latency_delta_ms: float | None = None,
) -> list[RetrievalGateCheck]:
    checks = [
        pairwise_minimum_check(
            "min_pairwise_shared_queries",
            "pairwise_shared_query_count",
            pairwise_metrics,
            float(min_pairwise_shared_queries)
            if min_pairwise_shared_queries is not None
            else None,
        ),
        pairwise_minimum_check(
            "min_pairwise_win_rate",
            "pairwise_candidate_win_rate",
            pairwise_metrics,
            min_pairwise_win_rate,
        ),
        pairwise_minimum_check(
            "min_pairwise_target_coverage_lift",
            "pairwise_mean_target_coverage_delta",
            pairwise_metrics,
            min_pairwise_target_coverage_lift,
        ),
        pairwise_minimum_check(
            "min_pairwise_target_ndcg_lift",
            "pairwise_mean_target_ndcg_delta",
            pairwise_metrics,
            min_pairwise_target_ndcg_lift,
        ),
        pairwise_minimum_check(
            "min_pairwise_mrr_lift",
            "pairwise_mean_reciprocal_rank_delta",
            pairwise_metrics,
            min_pairwise_mrr_lift,
        ),
        pairwise_minimum_check(
            "min_pairwise_precision_lift",
            "pairwise_mean_precision_delta",
            pairwise_metrics,
            min_pairwise_precision_lift,
        ),
        pairwise_minimum_check(
            "min_pairwise_target_coverage_ci_low",
            "pairwise_target_coverage_delta_ci_low",
            pairwise_metrics,
            min_pairwise_target_coverage_ci_low,
        ),
        pairwise_minimum_check(
            "min_pairwise_target_ndcg_ci_low",
            "pairwise_target_ndcg_delta_ci_low",
            pairwise_metrics,
            min_pairwise_target_ndcg_ci_low,
        ),
        pairwise_minimum_check(
            "min_pairwise_mrr_ci_low",
            "pairwise_reciprocal_rank_delta_ci_low",
            pairwise_metrics,
            min_pairwise_mrr_ci_low,
        ),
        pairwise_minimum_check(
            "min_pairwise_precision_ci_low",
            "pairwise_precision_delta_ci_low",
            pairwise_metrics,
            min_pairwise_precision_ci_low,
        ),
        pairwise_maximum_check(
            "max_pairwise_mean_first_relevant_rank_delta",
            "pairwise_mean_first_relevant_rank_delta",
            pairwise_metrics,
            max_pairwise_mean_first_relevant_rank_delta,
        ),
        pairwise_maximum_check(
            "max_pairwise_mean_target_rank_delta",
            "pairwise_mean_target_rank_delta",
            pairwise_metrics,
            max_pairwise_mean_target_rank_delta,
        ),
        pairwise_maximum_check(
            "max_pairwise_first_relevant_rank_delta_ci_high",
            "pairwise_first_relevant_rank_delta_ci_high",
            pairwise_metrics,
            max_pairwise_first_relevant_rank_delta_ci_high,
        ),
        pairwise_maximum_check(
            "max_pairwise_target_rank_delta_ci_high",
            "pairwise_target_rank_delta_ci_high",
            pairwise_metrics,
            max_pairwise_target_rank_delta_ci_high,
        ),
        pairwise_maximum_check(
            "max_pairwise_mean_latency_delta_ms",
            "pairwise_mean_latency_delta_ms",
            pairwise_metrics,
            max_pairwise_mean_latency_delta_ms,
        ),
    ]
    return [check for check in checks if check is not None]


def pairwise_minimum_check(
    name: str,
    metric: str,
    metrics: dict[str, float | None],
    threshold: float | None,
) -> RetrievalGateCheck | None:
    if threshold is None:
        return None
    actual = metrics.get(metric)
    actual_value = float(actual) if actual is not None else 0.0
    return RetrievalGateCheck(
        name=name,
        metric=metric,
        operator=">=",
        actual=actual_value,
        threshold=threshold,
        passed=actual is not None and actual_value >= threshold,
    )


def pairwise_maximum_check(
    name: str,
    metric: str,
    metrics: dict[str, float | None],
    threshold: float | None,
) -> RetrievalGateCheck | None:
    if threshold is None:
        return None
    actual = metrics.get(metric)
    actual_value = float(actual) if actual is not None else 0.0
    return RetrievalGateCheck(
        name=name,
        metric=metric,
        operator="<=",
        actual=actual_value,
        threshold=threshold,
        passed=actual is not None and actual_value <= threshold,
    )


def safe_ratio(actual: float, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    return actual / baseline


def parse_qdrant_vector_ablation_modes(value: str) -> list[QdrantVectorAblationMode]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        names = ["text", "caption", "text_caption", "text_caption_graph"]
    unknown = sorted(set(names) - set(DEFAULT_QDRANT_VECTOR_ABLATION_MODES))
    if unknown:
        raise ValueError(f"Unsupported Qdrant vector ablation modes: {', '.join(unknown)}")
    return [DEFAULT_QDRANT_VECTOR_ABLATION_MODES[name].model_copy(deep=True) for name in names]


def qdrant_vector_names_for_modes(modes: list[QdrantVectorAblationMode]) -> list[str]:
    seen: set[str] = set()
    vector_names: list[str] = []
    for mode in modes:
        for vector_name in mode.vector_names:
            if vector_name not in seen:
                seen.add(vector_name)
                vector_names.append(vector_name)
    return vector_names


def build_qdrant_vector_ablation_report(
    rows: list[QdrantVectorAblationRow],
) -> QdrantVectorAblationReport:
    rows = sorted(
        rows,
        key=lambda row: (
            row.evaluation.recall_at_k,
            row.evaluation.target_coverage_at_k,
            row.evaluation.mean_target_ndcg_at_k,
            row.evaluation.mrr,
            row.evaluation.hit_rate,
        ),
        reverse=True,
    )
    return QdrantVectorAblationReport(
        rows=rows,
        best_by_recall=rows[0].mode.name if rows else None,
        best_by_target_coverage=max(
            rows,
            key=lambda row: (row.evaluation.target_coverage_at_k, row.evaluation.recall_at_k),
        ).mode.name
        if rows
        else None,
        best_by_target_ndcg=max(
            rows,
            key=lambda row: (row.evaluation.mean_target_ndcg_at_k, row.evaluation.recall_at_k),
        ).mode.name
        if rows
        else None,
        best_by_mrr=max(rows, key=lambda row: row.evaluation.mrr).mode.name if rows else None,
        fastest_by_mean_latency=min(rows, key=lambda row: row.evaluation.mean_latency_ms).mode.name
        if rows
        else None,
        case_group_best_modes=case_group_best_modes(rows),
        pairwise=ablation_pairwise_comparisons(rows),
    )


def ablation_pairwise_comparisons(
    rows: list[RetrievalAblationRow] | list[QdrantVectorAblationRow],
) -> list[AblationPairwiseComparison]:
    comparisons = []
    for candidate_row in rows:
        for baseline_row in rows:
            if candidate_row.mode.name == baseline_row.mode.name:
                continue
            comparison = compare_ablation_rows_pairwise(candidate_row, baseline_row)
            if comparison is not None:
                comparisons.append(comparison)
    return comparisons


def compare_ablation_rows_pairwise(
    candidate_row: RetrievalAblationRow | QdrantVectorAblationRow,
    baseline_row: RetrievalAblationRow | QdrantVectorAblationRow,
) -> AblationPairwiseComparison | None:
    candidate_results = results_by_query(candidate_row.evaluation.results)
    baseline_results = results_by_query(baseline_row.evaluation.results)
    shared_queries = sorted(candidate_results.keys() & baseline_results.keys())
    if not shared_queries:
        return None

    candidate_wins = 0
    baseline_wins = 0
    ties = 0
    reciprocal_rank_deltas = []
    target_coverage_deltas = []
    target_ndcg_deltas = []
    precision_deltas = []
    first_relevant_rank_deltas = []
    target_rank_deltas = []
    latency_deltas = []
    missing_rank = float(
        max(candidate_row.evaluation.top_k, baseline_row.evaluation.top_k) + 1
    )
    for query in shared_queries:
        candidate_result = candidate_results[query]
        baseline_result = baseline_results[query]
        winner = compare_case_results(candidate_result, baseline_result)
        if winner > 0:
            candidate_wins += 1
        elif winner < 0:
            baseline_wins += 1
        else:
            ties += 1
        reciprocal_rank_deltas.append(
            candidate_result.reciprocal_rank - baseline_result.reciprocal_rank
        )
        target_coverage_deltas.append(
            candidate_result.target_coverage_at_k - baseline_result.target_coverage_at_k
        )
        target_ndcg_deltas.append(
            candidate_result.target_ndcg_at_k - baseline_result.target_ndcg_at_k
        )
        precision_deltas.append(candidate_result.precision_at_k - baseline_result.precision_at_k)
        first_relevant_rank_deltas.append(
            first_relevant_rank(candidate_result, missing_rank)
            - first_relevant_rank(baseline_result, missing_rank)
        )
        target_rank_deltas.append(
            case_mean_target_rank(candidate_result, missing_rank)
            - case_mean_target_rank(baseline_result, missing_rank)
        )
        latency_deltas.append(candidate_result.latency_ms - baseline_result.latency_ms)

    candidate_name = candidate_row.mode.name
    baseline_name = baseline_row.mode.name
    shared_count = len(shared_queries)
    reciprocal_rank_ci = bootstrap_mean_interval(
        reciprocal_rank_deltas,
        seed=stable_seed(candidate_name, baseline_name, "mrr"),
    )
    target_coverage_ci = bootstrap_mean_interval(
        target_coverage_deltas,
        seed=stable_seed(candidate_name, baseline_name, "coverage"),
    )
    target_ndcg_ci = bootstrap_mean_interval(
        target_ndcg_deltas,
        seed=stable_seed(candidate_name, baseline_name, "ndcg"),
    )
    precision_ci = bootstrap_mean_interval(
        precision_deltas,
        seed=stable_seed(candidate_name, baseline_name, "precision"),
    )
    first_rank_ci = bootstrap_mean_interval(
        first_relevant_rank_deltas,
        seed=stable_seed(candidate_name, baseline_name, "first-rank"),
    )
    target_rank_ci = bootstrap_mean_interval(
        target_rank_deltas,
        seed=stable_seed(candidate_name, baseline_name, "target-rank"),
    )
    latency_ci = bootstrap_mean_interval(
        latency_deltas,
        seed=stable_seed(candidate_name, baseline_name, "latency"),
    )
    return AblationPairwiseComparison(
        candidate=candidate_name,
        baseline=baseline_name,
        shared_query_count=shared_count,
        candidate_win_count=candidate_wins,
        baseline_win_count=baseline_wins,
        tie_count=ties,
        candidate_win_rate=candidate_wins / shared_count,
        baseline_win_rate=baseline_wins / shared_count,
        mean_reciprocal_rank_delta=mean(reciprocal_rank_deltas),
        mean_target_coverage_delta=mean(target_coverage_deltas),
        mean_target_ndcg_delta=mean(target_ndcg_deltas),
        mean_precision_delta=mean(precision_deltas),
        mean_first_relevant_rank_delta=mean(first_relevant_rank_deltas),
        mean_target_rank_delta=mean(target_rank_deltas),
        mean_latency_delta_ms=mean(latency_deltas),
        bootstrap_samples=PAIRWISE_BOOTSTRAP_SAMPLES,
        confidence_level=PAIRWISE_CONFIDENCE_LEVEL,
        reciprocal_rank_delta_ci_low=reciprocal_rank_ci[0],
        reciprocal_rank_delta_ci_high=reciprocal_rank_ci[1],
        target_coverage_delta_ci_low=target_coverage_ci[0],
        target_coverage_delta_ci_high=target_coverage_ci[1],
        target_ndcg_delta_ci_low=target_ndcg_ci[0],
        target_ndcg_delta_ci_high=target_ndcg_ci[1],
        precision_delta_ci_low=precision_ci[0],
        precision_delta_ci_high=precision_ci[1],
        first_relevant_rank_delta_ci_low=first_rank_ci[0],
        first_relevant_rank_delta_ci_high=first_rank_ci[1],
        target_rank_delta_ci_low=target_rank_ci[0],
        target_rank_delta_ci_high=target_rank_ci[1],
        latency_delta_ci_low_ms=latency_ci[0],
        latency_delta_ci_high_ms=latency_ci[1],
    )


def find_ablation_pairwise_comparison(
    pairwise: list[AblationPairwiseComparison],
    candidate: str,
    baseline: str | None,
) -> AblationPairwiseComparison | None:
    if baseline is None:
        return None
    for comparison in pairwise:
        if comparison.candidate == candidate and comparison.baseline == baseline:
            return comparison
    return None


def ablation_pairwise_metrics(
    pairwise: AblationPairwiseComparison | None,
) -> dict[str, float | None]:
    if pairwise is None:
        return {
            "pairwise_shared_query_count": None,
            "pairwise_candidate_win_count": None,
            "pairwise_baseline_win_count": None,
            "pairwise_tie_count": None,
            "pairwise_candidate_win_rate": None,
            "pairwise_baseline_win_rate": None,
            "pairwise_mean_reciprocal_rank_delta": None,
            "pairwise_mean_target_coverage_delta": None,
            "pairwise_mean_target_ndcg_delta": None,
            "pairwise_mean_precision_delta": None,
            "pairwise_mean_first_relevant_rank_delta": None,
            "pairwise_mean_target_rank_delta": None,
            "pairwise_mean_latency_delta_ms": None,
            "pairwise_bootstrap_samples": None,
            "pairwise_confidence_level": None,
            "pairwise_reciprocal_rank_delta_ci_low": None,
            "pairwise_reciprocal_rank_delta_ci_high": None,
            "pairwise_target_coverage_delta_ci_low": None,
            "pairwise_target_coverage_delta_ci_high": None,
            "pairwise_target_ndcg_delta_ci_low": None,
            "pairwise_target_ndcg_delta_ci_high": None,
            "pairwise_precision_delta_ci_low": None,
            "pairwise_precision_delta_ci_high": None,
            "pairwise_first_relevant_rank_delta_ci_low": None,
            "pairwise_first_relevant_rank_delta_ci_high": None,
            "pairwise_target_rank_delta_ci_low": None,
            "pairwise_target_rank_delta_ci_high": None,
            "pairwise_latency_delta_ci_low_ms": None,
            "pairwise_latency_delta_ci_high_ms": None,
        }
    return {
        "pairwise_shared_query_count": float(pairwise.shared_query_count),
        "pairwise_candidate_win_count": float(pairwise.candidate_win_count),
        "pairwise_baseline_win_count": float(pairwise.baseline_win_count),
        "pairwise_tie_count": float(pairwise.tie_count),
        "pairwise_candidate_win_rate": pairwise.candidate_win_rate,
        "pairwise_baseline_win_rate": pairwise.baseline_win_rate,
        "pairwise_mean_reciprocal_rank_delta": pairwise.mean_reciprocal_rank_delta,
        "pairwise_mean_target_coverage_delta": pairwise.mean_target_coverage_delta,
        "pairwise_mean_target_ndcg_delta": pairwise.mean_target_ndcg_delta,
        "pairwise_mean_precision_delta": pairwise.mean_precision_delta,
        "pairwise_mean_first_relevant_rank_delta": pairwise.mean_first_relevant_rank_delta,
        "pairwise_mean_target_rank_delta": pairwise.mean_target_rank_delta,
        "pairwise_mean_latency_delta_ms": pairwise.mean_latency_delta_ms,
        "pairwise_bootstrap_samples": float(pairwise.bootstrap_samples),
        "pairwise_confidence_level": pairwise.confidence_level,
        "pairwise_reciprocal_rank_delta_ci_low": pairwise.reciprocal_rank_delta_ci_low,
        "pairwise_reciprocal_rank_delta_ci_high": pairwise.reciprocal_rank_delta_ci_high,
        "pairwise_target_coverage_delta_ci_low": pairwise.target_coverage_delta_ci_low,
        "pairwise_target_coverage_delta_ci_high": pairwise.target_coverage_delta_ci_high,
        "pairwise_target_ndcg_delta_ci_low": pairwise.target_ndcg_delta_ci_low,
        "pairwise_target_ndcg_delta_ci_high": pairwise.target_ndcg_delta_ci_high,
        "pairwise_precision_delta_ci_low": pairwise.precision_delta_ci_low,
        "pairwise_precision_delta_ci_high": pairwise.precision_delta_ci_high,
        "pairwise_first_relevant_rank_delta_ci_low": (
            pairwise.first_relevant_rank_delta_ci_low
        ),
        "pairwise_first_relevant_rank_delta_ci_high": (
            pairwise.first_relevant_rank_delta_ci_high
        ),
        "pairwise_target_rank_delta_ci_low": pairwise.target_rank_delta_ci_low,
        "pairwise_target_rank_delta_ci_high": pairwise.target_rank_delta_ci_high,
        "pairwise_latency_delta_ci_low_ms": pairwise.latency_delta_ci_low_ms,
        "pairwise_latency_delta_ci_high_ms": pairwise.latency_delta_ci_high_ms,
    }


def case_group_best_modes(
    rows: list[RetrievalAblationRow] | list[QdrantVectorAblationRow],
) -> dict[str, dict[str, dict[str, AblationBestModeMetric]]]:
    groups = sorted(
        {
            (group_name, group_value)
            for row in rows
            for group_name, group_values in row.evaluation.case_group_metrics.items()
            for group_value in group_values
        }
    )
    return {
        group_name: {
            group_value: {
                "recall_at_k": best_case_group_mode(
                    rows,
                    group_name,
                    group_value,
                    "recall_at_k",
                ),
                "target_coverage_at_k": best_case_group_mode(
                    rows,
                    group_name,
                    group_value,
                    "target_coverage_at_k",
                ),
                "ndcg_at_k": best_case_group_mode(
                    rows,
                    group_name,
                    group_value,
                    "ndcg_at_k",
                ),
                "mrr": best_case_group_mode(rows, group_name, group_value, "mrr"),
                "precision_at_k": best_case_group_mode(
                    rows,
                    group_name,
                    group_value,
                    "precision_at_k",
                ),
                "fastest_mean_latency_ms": best_case_group_mode(
                    rows,
                    group_name,
                    group_value,
                    "mean_latency_ms",
                    prefer_lower=True,
                ),
            }
            for group_value in sorted(
                value for name, value in groups if name == group_name
            )
        }
        for group_name in sorted({name for name, _value in groups})
    }


def best_case_group_mode(
    rows: list[RetrievalAblationRow] | list[QdrantVectorAblationRow],
    group_name: str,
    group_value: str,
    metric_name: str,
    prefer_lower: bool = False,
) -> AblationBestModeMetric:
    scored_rows = []
    for row in rows:
        metric = row.evaluation.case_group_metrics.get(group_name, {}).get(group_value)
        if metric is None:
            continue
        value = float(getattr(metric, metric_name))
        scored_rows.append((case_group_mode_score(row, metric, metric_name, prefer_lower), row, value))
    if not scored_rows:
        return AblationBestModeMetric()
    _score, best_row, value = max(scored_rows, key=lambda item: item[0])
    return AblationBestModeMetric(mode=best_row.mode.name, value=value)


def case_group_mode_score(
    row: RetrievalAblationRow | QdrantVectorAblationRow,
    metric,
    metric_name: str,
    prefer_lower: bool,
) -> tuple[float, float, float, float, str]:
    metric_value = float(getattr(metric, metric_name, 0.0))
    primary = -metric_value if prefer_lower else metric_value
    return (
        primary,
        metric.target_coverage_at_k,
        metric.ndcg_at_k,
        metric.recall_at_k,
        row.mode.name,
    )


def gate_qdrant_vector_ablation(
    report: QdrantVectorAblationReport,
    mode: str,
    baseline_mode: str | None = None,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    max_excluded_target_hit_rate: float | None = None,
    max_excluded_query_hit_rate: float | None = None,
    max_excluded_hit_query_count: int | None = None,
    min_target_type_coverage: dict[str, float] | None = None,
    min_source_target_coverage: dict[str, float] | None = None,
    min_source_family_target_coverage: dict[str, float] | None = None,
    max_source_excluded_target_hit_rate: dict[str, float] | None = None,
    max_source_family_excluded_target_hit_rate: dict[str, float] | None = None,
    min_case_group_target_coverage: dict[str, float] | None = None,
    min_pairwise_shared_queries: int | None = None,
    min_pairwise_win_rate: float | None = None,
    min_pairwise_target_coverage_lift: float | None = None,
    min_pairwise_target_ndcg_lift: float | None = None,
    min_pairwise_mrr_lift: float | None = None,
    min_pairwise_precision_lift: float | None = None,
    min_pairwise_target_coverage_ci_low: float | None = None,
    min_pairwise_target_ndcg_ci_low: float | None = None,
    min_pairwise_mrr_ci_low: float | None = None,
    min_pairwise_precision_ci_low: float | None = None,
    max_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_pairwise_mean_target_rank_delta: float | None = None,
    max_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_pairwise_target_rank_delta_ci_high: float | None = None,
    max_pairwise_mean_latency_delta_ms: float | None = None,
    require_best_by_recall: bool = False,
    require_best_by_target_coverage: bool = False,
    require_best_by_target_ndcg: bool = False,
    require_fastest_by_mean_latency: bool = False,
) -> QdrantVectorAblationGateReport:
    row = qdrant_vector_ablation_row(report, mode)
    if row is None:
        raise ValueError(f"Qdrant vector ablation mode not found: {mode}")

    baseline_row = None
    if baseline_mode is not None:
        baseline_row = qdrant_vector_ablation_row(report, baseline_mode)
        if baseline_row is None:
            raise ValueError(f"Baseline Qdrant vector ablation mode not found: {baseline_mode}")
    elif requires_baseline(
        min_pairwise_shared_queries,
        min_pairwise_win_rate,
        min_pairwise_target_coverage_lift,
        min_pairwise_target_ndcg_lift,
        min_pairwise_mrr_lift,
        min_pairwise_precision_lift,
        min_pairwise_target_coverage_ci_low,
        min_pairwise_target_ndcg_ci_low,
        min_pairwise_mrr_ci_low,
        min_pairwise_precision_ci_low,
        max_pairwise_mean_first_relevant_rank_delta,
        max_pairwise_mean_target_rank_delta,
        max_pairwise_first_relevant_rank_delta_ci_high,
        max_pairwise_target_rank_delta_ci_high,
        max_pairwise_mean_latency_delta_ms,
    ):
        raise ValueError("A baseline mode is required for pairwise checks.")

    target_metrics = retrieval_target_metrics(row.evaluation)
    source_metrics = retrieval_source_metrics(row.evaluation)
    source_family_metrics = retrieval_source_family_metrics(row.evaluation)
    chunk_strategy_metrics = retrieval_chunk_strategy_metrics(row.evaluation)
    retrieval_role_metrics = retrieval_role_metrics_payload(row.evaluation)
    case_group_metrics = retrieval_case_group_metrics(row.evaluation)
    metrics = qdrant_vector_ablation_metrics(
        row.evaluation,
        target_metrics,
        source_metrics,
        source_family_metrics,
        chunk_strategy_metrics,
        retrieval_role_metrics,
        case_group_metrics,
    )
    baseline_metrics = {}
    if baseline_row is not None:
        baseline_metrics = qdrant_vector_ablation_metrics(
            baseline_row.evaluation,
            retrieval_target_metrics(baseline_row.evaluation),
            retrieval_source_metrics(baseline_row.evaluation),
            retrieval_source_family_metrics(baseline_row.evaluation),
            retrieval_chunk_strategy_metrics(baseline_row.evaluation),
            retrieval_role_metrics_payload(baseline_row.evaluation),
            retrieval_case_group_metrics(baseline_row.evaluation),
        )
    pairwise_metrics = ablation_pairwise_metrics(
        find_ablation_pairwise_comparison(report.pairwise, mode, baseline_mode)
    )
    checks = [
        minimum_check("min_recall_at_k", "recall_at_k", metrics, min_recall_at_k),
        minimum_check(
            "min_target_coverage_at_k",
            "target_coverage_at_k",
            metrics,
            min_target_coverage_at_k,
        ),
        minimum_check(
            "min_target_ndcg_at_k",
            "mean_target_ndcg_at_k",
            metrics,
            min_target_ndcg_at_k,
        ),
        minimum_check("min_mrr", "mrr", metrics, min_mrr),
        minimum_check("min_precision_at_k", "mean_precision_at_k", metrics, min_precision_at_k),
    ]
    if max_failed_queries is not None:
        checks.append(
            maximum_check(
                "max_failed_queries",
                "failed_query_count",
                metrics,
                float(max_failed_queries),
            )
        )
    if max_mean_latency_ms is not None:
        checks.append(
            maximum_check(
                "max_mean_latency_ms",
                "mean_latency_ms",
                metrics,
                max_mean_latency_ms,
            )
        )
    if max_p95_latency_ms is not None:
        checks.append(
            maximum_check(
                "max_p95_latency_ms",
                "p95_latency_ms",
                metrics,
                max_p95_latency_ms,
            )
        )
    checks.extend(
        excluded_target_limit_checks(
            metrics,
            max_excluded_target_hit_rate=max_excluded_target_hit_rate,
            max_excluded_query_hit_rate=max_excluded_query_hit_rate,
            max_excluded_hit_query_count=max_excluded_hit_query_count,
        )
    )
    checks.extend(
        rank_limit_checks(
            metrics,
            max_mean_first_relevant_rank=max_mean_first_relevant_rank,
            max_p95_first_relevant_rank=max_p95_first_relevant_rank,
            max_mean_target_rank=max_mean_target_rank,
            max_p95_target_rank=max_p95_target_rank,
        )
    )
    checks.extend(
        target_type_coverage_checks(
            metrics,
            min_target_type_coverage or {},
        )
    )
    checks.extend(source_target_coverage_checks(metrics, min_source_target_coverage or {}))
    checks.extend(
        source_family_target_coverage_checks(
            metrics,
            min_source_family_target_coverage or {},
        )
    )
    checks.extend(
        source_excluded_target_hit_rate_checks(
            metrics,
            max_source_excluded_target_hit_rate or {},
        )
    )
    checks.extend(
        source_family_excluded_target_hit_rate_checks(
            metrics,
            max_source_family_excluded_target_hit_rate or {},
        )
    )
    checks.extend(case_group_target_coverage_checks(metrics, min_case_group_target_coverage or {}))
    if baseline_row is not None:
        checks.extend(
            pairwise_threshold_checks(
                pairwise_metrics,
                min_pairwise_shared_queries=min_pairwise_shared_queries,
                min_pairwise_win_rate=min_pairwise_win_rate,
                min_pairwise_target_coverage_lift=min_pairwise_target_coverage_lift,
                min_pairwise_target_ndcg_lift=min_pairwise_target_ndcg_lift,
                min_pairwise_mrr_lift=min_pairwise_mrr_lift,
                min_pairwise_precision_lift=min_pairwise_precision_lift,
                min_pairwise_target_coverage_ci_low=min_pairwise_target_coverage_ci_low,
                min_pairwise_target_ndcg_ci_low=min_pairwise_target_ndcg_ci_low,
                min_pairwise_mrr_ci_low=min_pairwise_mrr_ci_low,
                min_pairwise_precision_ci_low=min_pairwise_precision_ci_low,
                max_pairwise_mean_first_relevant_rank_delta=(
                    max_pairwise_mean_first_relevant_rank_delta
                ),
                max_pairwise_mean_target_rank_delta=max_pairwise_mean_target_rank_delta,
                max_pairwise_first_relevant_rank_delta_ci_high=(
                    max_pairwise_first_relevant_rank_delta_ci_high
                ),
                max_pairwise_target_rank_delta_ci_high=(
                    max_pairwise_target_rank_delta_ci_high
                ),
                max_pairwise_mean_latency_delta_ms=max_pairwise_mean_latency_delta_ms,
            )
        )
    if require_best_by_recall:
        checks.append(best_mode_check("require_best_by_recall", mode, report.best_by_recall))
    if require_best_by_target_coverage:
        checks.append(
            best_mode_check(
                "require_best_by_target_coverage",
                mode,
                report.best_by_target_coverage,
            )
        )
    if require_best_by_target_ndcg:
        checks.append(
            best_mode_check(
                "require_best_by_target_ndcg",
                mode,
                report.best_by_target_ndcg,
            )
        )
    if require_fastest_by_mean_latency:
        checks.append(
            best_mode_check(
                "require_fastest_by_mean_latency",
                mode,
                report.fastest_by_mean_latency,
            )
        )

    failed_checks = [check.name for check in checks if not check.passed]
    return QdrantVectorAblationGateReport(
        passed=not failed_checks,
        mode=mode,
        baseline_mode=baseline_mode,
        vector_names=row.mode.vector_names,
        graph_expand=row.mode.graph_expand,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        target_metrics=target_metrics,
        source_metrics=source_metrics,
        source_family_metrics=source_family_metrics,
        chunk_strategy_metrics=chunk_strategy_metrics,
        retrieval_role_metrics=retrieval_role_metrics,
        case_group_metrics=case_group_metrics,
        pairwise_metrics=pairwise_metrics,
        best_by_recall=report.best_by_recall,
        best_by_target_coverage=report.best_by_target_coverage,
        best_by_target_ndcg=report.best_by_target_ndcg,
        best_by_mrr=report.best_by_mrr,
        fastest_by_mean_latency=report.fastest_by_mean_latency,
        case_group_best_modes=report.case_group_best_modes,
        failed_checks=failed_checks,
        checks=checks,
    )


def qdrant_vector_ablation_row(
    report: QdrantVectorAblationReport,
    mode: str,
) -> QdrantVectorAblationRow | None:
    for row in report.rows:
        if row.mode.name == mode:
            return row
    return None


def qdrant_vector_ablation_metrics(
    evaluation: RetrievalEvaluation,
    target_metrics: dict[str, dict[str, float]] | None = None,
    source_metrics: dict[str, dict[str, float]] | None = None,
    source_family_metrics: dict[str, dict[str, float]] | None = None,
    chunk_strategy_metrics: dict[str, dict[str, float]] | None = None,
    retrieval_role_metrics: dict[str, dict[str, float]] | None = None,
    case_group_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, float]:
    metrics = {
        "hit_rate": evaluation.hit_rate,
        "recall_at_k": evaluation.recall_at_k,
        "mrr": evaluation.mrr,
        "target_coverage_at_k": evaluation.target_coverage_at_k,
        "mean_target_ndcg_at_k": evaluation.mean_target_ndcg_at_k,
        "mean_precision_at_k": evaluation.mean_precision_at_k,
        "mean_latency_ms": evaluation.mean_latency_ms,
        "p95_latency_ms": evaluation.p95_latency_ms,
        "failed_query_count": float(evaluation.failed_count),
        "excluded_query_count": float(evaluation.excluded_query_count),
        "excluded_hit_query_count": float(evaluation.excluded_hit_query_count),
        "excluded_query_hit_rate": evaluation.excluded_query_hit_rate,
        "excluded_target_count": float(evaluation.excluded_target_count),
        "excluded_matched_target_count": float(evaluation.excluded_matched_target_count),
        "excluded_target_hit_rate": evaluation.excluded_target_hit_rate,
    }
    metrics.update(retrieval_rank_metrics(evaluation))
    for source, source_metric_values in (source_metrics or {}).items():
        for key, value in source_metric_values.items():
            metrics[source_metric_key(source, key)] = value
    for target_type, target_type_metrics in (target_metrics or {}).items():
        for key, value in target_type_metrics.items():
            metrics[target_type_metric_key(target_type, key)] = value
    for family, family_metrics in (source_family_metrics or {}).items():
        for key, value in family_metrics.items():
            metrics[source_family_metric_key(family, key)] = value
    for strategy, strategy_metrics in (chunk_strategy_metrics or {}).items():
        for key, value in strategy_metrics.items():
            metrics[chunk_strategy_metric_key(strategy, key)] = value
    for role, role_metrics in (retrieval_role_metrics or {}).items():
        for key, value in role_metrics.items():
            metrics[retrieval_role_metric_key(role, key)] = value
    for group_name, group_values in (case_group_metrics or {}).items():
        for group_value, group_metrics in group_values.items():
            for key, value in group_metrics.items():
                metrics[case_group_metric_key(group_name, group_value, key)] = value
    return metrics


def excluded_target_limit_checks(
    metrics: dict[str, float],
    max_excluded_target_hit_rate: float | None = None,
    max_excluded_query_hit_rate: float | None = None,
    max_excluded_hit_query_count: int | None = None,
) -> list[RetrievalGateCheck]:
    checks = []
    if max_excluded_target_hit_rate is not None:
        checks.append(
            maximum_check(
                "max_excluded_target_hit_rate",
                "excluded_target_hit_rate",
                metrics,
                max_excluded_target_hit_rate,
            )
        )
    if max_excluded_query_hit_rate is not None:
        checks.append(
            maximum_check(
                "max_excluded_query_hit_rate",
                "excluded_query_hit_rate",
                metrics,
                max_excluded_query_hit_rate,
            )
        )
    if max_excluded_hit_query_count is not None:
        checks.append(
            maximum_check(
                "max_excluded_hit_query_count",
                "excluded_hit_query_count",
                metrics,
                float(max_excluded_hit_query_count),
            )
        )
    return checks


def rank_limit_checks(
    metrics: dict[str, float],
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
) -> list[RetrievalGateCheck]:
    checks = []
    if max_mean_first_relevant_rank is not None:
        checks.append(
            maximum_check(
                "max_mean_first_relevant_rank",
                "mean_first_relevant_rank",
                metrics,
                max_mean_first_relevant_rank,
            )
        )
    if max_p95_first_relevant_rank is not None:
        checks.append(
            maximum_check(
                "max_p95_first_relevant_rank",
                "p95_first_relevant_rank",
                metrics,
                max_p95_first_relevant_rank,
            )
        )
    if max_mean_target_rank is not None:
        checks.append(maximum_check("max_mean_target_rank", "mean_target_rank", metrics, max_mean_target_rank))
    if max_p95_target_rank is not None:
        checks.append(maximum_check("max_p95_target_rank", "p95_target_rank", metrics, max_p95_target_rank))
    return checks


def best_mode_check(name: str, mode: str, actual_best: str | None) -> RetrievalGateCheck:
    actual = 1.0 if actual_best == mode else 0.0
    return RetrievalGateCheck(
        name=name,
        metric=name,
        operator="==",
        actual=actual,
        threshold=1.0,
        passed=actual_best == mode,
    )
