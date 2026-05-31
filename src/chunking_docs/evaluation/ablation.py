from __future__ import annotations

from pydantic import BaseModel, Field

from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.gate import (
    RetrievalGateCheck,
    maximum_check,
    minimum_check,
    retrieval_source_family_metrics,
    retrieval_target_metrics,
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


class RetrievalAblationReport(BaseModel):
    rows: list[RetrievalAblationRow]
    best_by_recall: str | None
    best_by_target_coverage: str | None
    best_by_target_ndcg: str | None
    best_by_mrr: str | None
    fastest_by_mean_latency: str | None


class RetrievalAblationGateReport(BaseModel):
    passed: bool
    mode: str
    baseline_mode: str | None = None
    metrics: dict[str, float]
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    best_by_recall: str | None = None
    best_by_target_coverage: str | None = None
    best_by_target_ndcg: str | None = None
    best_by_mrr: str | None = None
    fastest_by_mean_latency: str | None = None
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


class QdrantVectorAblationGateReport(BaseModel):
    passed: bool
    mode: str
    vector_names: list[str] = Field(default_factory=list)
    graph_expand: bool = False
    metrics: dict[str, float]
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    best_by_recall: str | None = None
    best_by_target_coverage: str | None = None
    best_by_target_ndcg: str | None = None
    best_by_mrr: str | None = None
    fastest_by_mean_latency: str | None = None
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
    "image": QdrantVectorAblationMode(name="image", vector_names=["image_dense"]),
    "text_caption": QdrantVectorAblationMode(
        name="text_caption",
        vector_names=["text_dense", "caption_dense"],
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
    "text_caption_graph": QdrantVectorAblationMode(
        name="text_caption_graph",
        vector_names=["text_dense", "caption_dense"],
        graph_expand=True,
    ),
    "all_graph": QdrantVectorAblationMode(
        name="all_graph",
        vector_names=["text_dense", "caption_dense", "image_dense"],
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
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    min_target_type_coverage: dict[str, float] | None = None,
    min_source_family_target_coverage: dict[str, float] | None = None,
    min_recall_lift: float | None = None,
    min_target_coverage_lift: float | None = None,
    min_target_ndcg_lift: float | None = None,
    min_mrr_lift: float | None = None,
    min_precision_lift: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
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
    ):
        raise ValueError("A baseline mode is required for lift or latency-ratio checks.")

    target_metrics = retrieval_target_metrics(row.evaluation)
    source_family_metrics = retrieval_source_family_metrics(row.evaluation)
    metrics = qdrant_vector_ablation_metrics(row.evaluation, target_metrics, source_family_metrics)
    baseline_target_metrics = retrieval_target_metrics(baseline_row.evaluation) if baseline_row else {}
    baseline_source_family_metrics = (
        retrieval_source_family_metrics(baseline_row.evaluation) if baseline_row else {}
    )
    baseline_metrics = (
        qdrant_vector_ablation_metrics(
            baseline_row.evaluation,
            baseline_target_metrics,
            baseline_source_family_metrics,
        )
        if baseline_row
        else {}
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
    checks.extend(target_type_coverage_checks(metrics, min_target_type_coverage or {}))
    checks.extend(
        source_family_target_coverage_checks(metrics, min_source_family_target_coverage or {})
    )
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
        source_family_metrics=source_family_metrics,
        baseline_target_metrics=baseline_target_metrics,
        baseline_source_family_metrics=baseline_source_family_metrics,
        best_by_recall=report.best_by_recall,
        best_by_target_coverage=report.best_by_target_coverage,
        best_by_target_ndcg=report.best_by_target_ndcg,
        best_by_mrr=report.best_by_mrr,
        fastest_by_mean_latency=report.fastest_by_mean_latency,
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
    )


def gate_qdrant_vector_ablation(
    report: QdrantVectorAblationReport,
    mode: str,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    min_target_type_coverage: dict[str, float] | None = None,
    min_source_family_target_coverage: dict[str, float] | None = None,
    require_best_by_recall: bool = False,
    require_best_by_target_coverage: bool = False,
    require_best_by_target_ndcg: bool = False,
    require_fastest_by_mean_latency: bool = False,
) -> QdrantVectorAblationGateReport:
    row = qdrant_vector_ablation_row(report, mode)
    if row is None:
        raise ValueError(f"Qdrant vector ablation mode not found: {mode}")

    target_metrics = retrieval_target_metrics(row.evaluation)
    source_family_metrics = retrieval_source_family_metrics(row.evaluation)
    metrics = qdrant_vector_ablation_metrics(row.evaluation, target_metrics, source_family_metrics)
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
        target_type_coverage_checks(
            metrics,
            min_target_type_coverage or {},
        )
    )
    checks.extend(
        source_family_target_coverage_checks(
            metrics,
            min_source_family_target_coverage or {},
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
        vector_names=row.mode.vector_names,
        graph_expand=row.mode.graph_expand,
        metrics=metrics,
        target_metrics=target_metrics,
        source_family_metrics=source_family_metrics,
        best_by_recall=report.best_by_recall,
        best_by_target_coverage=report.best_by_target_coverage,
        best_by_target_ndcg=report.best_by_target_ndcg,
        best_by_mrr=report.best_by_mrr,
        fastest_by_mean_latency=report.fastest_by_mean_latency,
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
    source_family_metrics: dict[str, dict[str, float]] | None = None,
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
    }
    for target_type, target_type_metrics in (target_metrics or {}).items():
        for key, value in target_type_metrics.items():
            metrics[target_type_metric_key(target_type, key)] = value
    for family, family_metrics in (source_family_metrics or {}).items():
        for key, value in family_metrics.items():
            metrics[source_family_metric_key(family, key)] = value
    return metrics


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
