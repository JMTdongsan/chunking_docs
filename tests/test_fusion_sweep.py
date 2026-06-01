from chunking_docs.evaluation.fusion_sweep import (
    QdrantFusionSweepCandidate,
    build_fusion_weight_grid,
    build_qdrant_fusion_sweep_report,
    fusion_weight_candidate_name,
)
from chunking_docs.evaluation.retrieval import RetrievalCaseGroupMetric, RetrievalEvaluation


def test_build_fusion_weight_grid_combines_fixed_and_grid_values():
    candidates = build_fusion_weight_grid(
        {
            "qdrant:caption_dense": [0.5, 1.0],
            "qdrant:image_dense": [0.0, 0.25],
        },
        fixed_weights={"bm25": 1.2},
        include_fixed_candidate=True,
    )

    assert candidates[0] == {"bm25": 1.2}
    assert {
        tuple(sorted(candidate.items()))
        for candidate in candidates
    } == {
        (("bm25", 1.2),),
        (("bm25", 1.2), ("qdrant:caption_dense", 0.5), ("qdrant:image_dense", 0.0)),
        (("bm25", 1.2), ("qdrant:caption_dense", 0.5), ("qdrant:image_dense", 0.25)),
        (("bm25", 1.2), ("qdrant:caption_dense", 1.0), ("qdrant:image_dense", 0.0)),
        (("bm25", 1.2), ("qdrant:caption_dense", 1.0), ("qdrant:image_dense", 0.25)),
    }


def test_fusion_weight_candidate_name_is_stable_and_path_safe():
    name = fusion_weight_candidate_name(
        {
            "qdrant:caption_dense": 1.25,
            "graph": 0.5,
        }
    )

    assert name == "graph_0p5__qdrant_caption_dense_1p25"


def test_build_qdrant_fusion_sweep_report_recommends_best_eligible_candidate():
    weak = QdrantFusionSweepCandidate(
        name="weak",
        fusion_weights={"qdrant:image_dense": 1.0},
        evaluation=evaluation(
            recall=0.7,
            coverage=0.72,
            ndcg=0.6,
            mrr=0.55,
            precision=0.14,
            failed=3,
            latency=20.0,
        ),
    )
    strong = QdrantFusionSweepCandidate(
        name="strong",
        fusion_weights={"bm25": 1.2, "qdrant:caption_dense": 1.1},
        evaluation=evaluation(
            recall=0.98,
            coverage=0.99,
            ndcg=0.94,
            mrr=0.91,
            precision=0.2,
            failed=1,
            latency=35.0,
        ),
    )
    fast = QdrantFusionSweepCandidate(
        name="fast",
        fusion_weights={"qdrant:image_dense": 0.5},
        evaluation=evaluation(
            recall=0.96,
            coverage=0.96,
            ndcg=0.7,
            mrr=0.68,
            precision=0.19,
            failed=0,
            latency=12.0,
        ),
    )

    report = build_qdrant_fusion_sweep_report(
        [weak, strong, fast],
        vector_names=["text_dense", "caption_dense", "image_dense"],
        min_recall_at_k=0.95,
        min_target_coverage_at_k=0.95,
        min_target_ndcg_at_k=0.65,
        max_failed_queries=1,
    )

    assert report.recommended == "strong"
    assert report.best_by_recall == "strong"
    assert report.fastest_by_mean_latency == "fast"
    assert report.eligible_count == 2
    assert report.candidates[0].rank == 1
    assert report.candidates[0].name == "strong"
    assert report.candidates[-1].name == "weak"
    assert report.candidates[-1].eligibility_failures == [
        "min_recall_at_k",
        "min_target_coverage_at_k",
        "min_target_ndcg_at_k",
        "max_failed_queries",
    ]


def test_build_qdrant_fusion_sweep_report_penalizes_excluded_hits():
    clean = QdrantFusionSweepCandidate(
        name="clean",
        fusion_weights={"bm25": 1.0},
        evaluation=evaluation(
            recall=0.95,
            coverage=0.95,
            ndcg=0.9,
            mrr=0.85,
            precision=0.2,
            failed=0,
            latency=30.0,
        ),
    )
    leaky = QdrantFusionSweepCandidate(
        name="leaky",
        fusion_weights={"qdrant:image_dense": 2.0},
        evaluation=evaluation(
            recall=1.0,
            coverage=1.0,
            ndcg=0.98,
            mrr=0.95,
            precision=0.2,
            failed=1,
            latency=20.0,
            excluded_query_count=2,
            excluded_hit_query_count=1,
            excluded_target_count=2,
            excluded_matched_target_count=1,
        ),
    )

    report = build_qdrant_fusion_sweep_report(
        [clean, leaky],
        vector_names=["text_dense", "image_dense"],
        max_excluded_target_hit_rate=0.0,
        max_excluded_query_hit_rate=0.0,
        max_excluded_hit_query_count=0,
    )

    assert report.recommended == "clean"
    assert report.eligible_count == 1
    clean_row = next(candidate for candidate in report.candidates if candidate.name == "clean")
    leaky_row = next(candidate for candidate in report.candidates if candidate.name == "leaky")
    assert leaky_row.eligibility_failures == [
        "max_excluded_target_hit_rate",
        "max_excluded_query_hit_rate",
        "max_excluded_hit_query_count",
    ]
    assert leaky_row.selection_score < clean_row.selection_score


def test_build_qdrant_fusion_sweep_report_recommends_case_group_candidates():
    balanced = QdrantFusionSweepCandidate(
        name="balanced",
        fusion_weights={"bm25": 1.0, "qdrant:caption_dense": 1.0},
        evaluation=evaluation(
            recall=0.95,
            coverage=0.95,
            ndcg=0.9,
            mrr=0.86,
            precision=0.2,
            failed=0,
            latency=30.0,
            visual_object_group=case_group_metric(
                coverage=0.5,
                ndcg=0.48,
                mrr=0.5,
            ),
        ),
    )
    object_weighted = QdrantFusionSweepCandidate(
        name="object_weighted",
        fusion_weights={"bm25": 1.0, "qdrant:object_dense": 1.4},
        evaluation=evaluation(
            recall=0.9,
            coverage=0.9,
            ndcg=0.86,
            mrr=0.82,
            precision=0.2,
            failed=0,
            latency=35.0,
            visual_object_group=case_group_metric(
                coverage=0.9,
                ndcg=0.88,
                mrr=0.84,
            ),
        ),
    )
    ineligible_specialist = QdrantFusionSweepCandidate(
        name="ineligible_specialist",
        fusion_weights={"qdrant:object_dense": 2.0},
        evaluation=evaluation(
            recall=0.5,
            coverage=0.5,
            ndcg=0.5,
            mrr=0.5,
            precision=0.1,
            failed=5,
            latency=10.0,
            visual_object_group=case_group_metric(
                coverage=1.0,
                ndcg=1.0,
                mrr=1.0,
            ),
        ),
    )

    report = build_qdrant_fusion_sweep_report(
        [balanced, object_weighted, ineligible_specialist],
        vector_names=["text_dense", "caption_dense", "object_dense"],
        min_recall_at_k=0.8,
        min_target_coverage_at_k=0.8,
        max_failed_queries=1,
    )

    group = report.case_group_recommendations["case_source"]["visual_object_probe"]
    assert report.recommended == "balanced"
    assert group.recommended == "object_weighted"
    assert group.recommended_from_globally_eligible is True
    assert group.best_by_target_coverage == "ineligible_specialist"
    assert group.eligible_count == 2
    assert [candidate.name for candidate in group.top_candidates[:2]] == [
        "object_weighted",
        "balanced",
    ]
    assert group.top_candidates[0].fusion_weights == {"bm25": 1.0, "qdrant:object_dense": 1.4}


def evaluation(
    recall: float,
    coverage: float,
    ndcg: float,
    mrr: float,
    precision: float,
    failed: int,
    latency: float,
    visual_object_group: RetrievalCaseGroupMetric | None = None,
    excluded_query_count: int = 0,
    excluded_hit_query_count: int = 0,
    excluded_target_count: int = 0,
    excluded_matched_target_count: int = 0,
) -> RetrievalEvaluation:
    case_group_metrics = {}
    if visual_object_group is not None:
        case_group_metrics = {"case_source": {"visual_object_probe": visual_object_group}}
    return RetrievalEvaluation(
        case_count=10,
        expected_case_count=10,
        passed_count=10 - failed,
        failed_count=failed,
        hit_rate=recall,
        recall_at_k=recall,
        mrr=mrr,
        target_coverage_at_k=coverage,
        mean_target_ndcg_at_k=ndcg,
        mean_precision_at_k=precision,
        excluded_query_count=excluded_query_count,
        excluded_hit_query_count=excluded_hit_query_count,
        excluded_query_hit_rate=excluded_hit_query_count / excluded_query_count
        if excluded_query_count
        else 0.0,
        excluded_target_count=excluded_target_count,
        excluded_matched_target_count=excluded_matched_target_count,
        excluded_target_hit_rate=excluded_matched_target_count / excluded_target_count
        if excluded_target_count
        else 0.0,
        top_k=5,
        total_query_latency_ms=latency * 10,
        mean_latency_ms=latency,
        p95_latency_ms=latency,
        failed_queries=[f"failed-{index}" for index in range(failed)],
        case_group_metrics=case_group_metrics,
        results=[],
    )


def case_group_metric(
    coverage: float,
    ndcg: float,
    mrr: float,
    recall: float = 1.0,
    precision: float = 0.2,
    failed: int = 0,
    latency: float = 20.0,
) -> RetrievalCaseGroupMetric:
    return RetrievalCaseGroupMetric(
        case_count=5,
        expected_case_count=5,
        passed_count=5 - failed,
        failed_count=failed,
        recall_at_k=recall,
        mrr=mrr,
        target_count=10,
        matched_target_count=int(round(coverage * 10)),
        target_coverage_at_k=coverage,
        ndcg_at_k=ndcg,
        precision_at_k=precision,
        mean_latency_ms=latency,
        failed_queries=[f"group-failed-{index}" for index in range(failed)],
    )
