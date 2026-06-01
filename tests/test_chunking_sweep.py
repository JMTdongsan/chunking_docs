import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.chunking_quality import ChunkingQualityReport, NumericSummary
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.evaluation.retrieval import RetrievalEvaluation
from chunking_docs.evaluation.sweep import (
    ChunkingSweepCandidate,
    build_sweep_selection,
    dominates,
    run_chunking_sweep,
)
from chunking_docs.io import write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    TextQuality,
    VisualAsset,
)


def test_run_chunking_sweep_writes_candidates_and_comparison(tmp_path):
    manifest = make_manifest(tmp_path)
    output_dir = tmp_path / "sweep"

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["semantic", "hierarchical"],
        max_chars_values=[140],
        overlap_chars_values=[20],
        min_chars=40,
        parent_max_chars_values=[90],
        visual_context_chars_values=[120],
        retrieval_cases=[RetrievalCase(query="capital investment table", expected_pages=[1])],
        output_dir=output_dir,
    )

    assert len(report.candidates) == 2
    assert report.comparison.best_by_retrieval is not None
    assert report.selection.recommended is not None
    assert report.selection.ranking[0].score >= report.selection.ranking[-1].score
    assert report.selection.ranking[0].metrics["mean_target_rank"] is not None
    assert report.selection.ranking[0].metrics["target_rank_efficiency"] is not None
    assert report.selection.ranking[0].metrics["result_stability_rate"] == 1.0
    assert report.selection.ranking[0].metrics["unstable_result_count"] == 0.0
    assert report.selection.ranking[0].metrics["total_chunk_chars"] is not None
    assert report.selection.ranking[0].metrics["embedding_text_kchars"] is not None
    assert report.selection.ranking[0].metrics["retrieval_score"] is not None
    assert report.selection.ranking[0].metrics["retrieval_score_per_embedding_kchar"] is not None
    assert report.selection.ranking[0].metrics["target_coverage_per_embedding_kchar"] is not None
    assert report.selection.ranking[0].metrics["target_ndcg_per_embedding_kchar"] is not None
    assert report.selection.ranking[0].metrics["p95_latency_ms"] is not None
    assert report.selection.ranking[0].metrics["retrieval_score_per_mean_latency_ms"] is not None
    assert report.selection.ranking[0].metrics["target_coverage_per_p95_latency_ms"] is not None
    assert report.selection.ranking[0].metrics["visual_text_part_coverage_ratio"] is not None
    assert report.selection.eligible_count == 2
    assert report.selection.rejected_count == 0
    assert all(row.eligible for row in report.selection.ranking)
    assert set(report.selection.pareto_front)
    assert all(candidate.chunks_file for candidate in report.candidates)
    assert any(candidate.strategy == "hierarchical" for candidate in report.candidates)
    assert any(output_dir.glob("chunks.hierarchical-*.jsonl"))


def test_multimodal_sweep_varies_visual_context_chars(tmp_path):
    manifest = make_manifest(tmp_path)

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["multimodal"],
        max_chars_values=[200],
        overlap_chars_values=[20],
        min_chars=40,
        visual_context_chars_values=[0, 120],
    )

    assert len(report.candidates) == 2
    assert {candidate.config["visual_context_chars"] for candidate in report.candidates} == {0, 120}
    assert any("visual120" in candidate.name for candidate in report.candidates)


def test_object_aware_sweep_tracks_visual_object_chunk_cost(tmp_path):
    manifest = make_manifest(tmp_path)
    assets = [
        asset.model_copy(
            update={
                "metadata": {
                    "objects": [
                        {"label": "station marker"},
                        {"label": "route line"},
                    ]
                }
            }
        )
        for asset in manifest.assets
    ]

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["object_aware"],
        max_chars_values=[200],
        overlap_chars_values=[20],
        min_chars=40,
        visual_context_chars_values=[120],
        selection_constraints={"max_visual_object_chunk_count": 1},
    )

    assert len(report.candidates) == 1
    assert report.candidates[0].strategy == "object_aware"
    assert report.candidates[0].report.visual_object_chunk_count == 2
    assert report.selection.ranking[0].metrics["visual_object_chunk_count"] == 2.0
    assert report.selection.ranking[0].failed_constraints == ["max_visual_object_chunk_count"]
    assert report.selection.recommended is None


def test_sweep_selection_constraints_filter_recommendation(tmp_path):
    manifest = make_manifest(tmp_path)

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["semantic", "hierarchical"],
        max_chars_values=[140],
        overlap_chars_values=[20],
        min_chars=40,
        parent_max_chars_values=[90],
        visual_context_chars_values=[120],
        retrieval_cases=[RetrievalCase(query="capital investment table", expected_pages=[1])],
        selection_constraints={"max_chunk_count": 2},
    )

    assert report.selection.constraints == {"max_chunk_count": 2.0}
    assert report.selection.recommended == "semantic-max140-ov20-min40"
    assert report.selection.eligible_count == 1
    assert report.selection.rejected_count == 1
    assert report.selection.ranking[0].eligible is True
    assert report.selection.ranking[0].name == report.selection.recommended
    rejected = next(row for row in report.selection.ranking if not row.eligible)
    assert rejected.failed_constraints == ["max_chunk_count"]
    assert report.candidates[0].name == report.selection.recommended


def test_sweep_selection_constraints_filter_embedding_text_budget(tmp_path):
    manifest = make_manifest(tmp_path)

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["semantic", "hierarchical"],
        max_chars_values=[140],
        overlap_chars_values=[20],
        min_chars=40,
        parent_max_chars_values=[90],
        visual_context_chars_values=[120],
        retrieval_cases=[RetrievalCase(query="capital investment table", expected_pages=[1])],
        selection_constraints={"max_total_chunk_chars": 250},
    )

    assert report.selection.constraints == {"max_total_chunk_chars": 250.0}
    assert report.selection.recommended == "semantic-max140-ov20-min40"
    assert report.selection.eligible_count == 1
    assert report.selection.rejected_count == 1
    assert report.selection.ranking[0].metrics["total_chunk_chars"] == 194.0
    rejected = next(row for row in report.selection.ranking if not row.eligible)
    assert rejected.failed_constraints == ["max_total_chunk_chars"]
    assert rejected.metrics["total_chunk_chars"] == 495.0


def test_sweep_selection_constraints_filter_retrieval_value_per_embedding_cost(tmp_path):
    manifest = make_manifest(tmp_path)

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["semantic", "hierarchical"],
        max_chars_values=[140],
        overlap_chars_values=[20],
        min_chars=40,
        parent_max_chars_values=[90],
        visual_context_chars_values=[120],
        retrieval_cases=[RetrievalCase(query="capital investment table", expected_pages=[1])],
        selection_constraints={"min_target_coverage_per_embedding_kchar": 3.0},
    )

    assert report.selection.constraints == {
        "min_target_coverage_per_embedding_kchar": 3.0,
    }
    assert report.selection.recommended == "semantic-max140-ov20-min40"
    assert report.selection.eligible_count == 1
    assert report.selection.rejected_count == 1
    top = report.selection.ranking[0]
    rejected = next(row for row in report.selection.ranking if not row.eligible)
    assert top.metrics["target_coverage_per_embedding_kchar"] > 3.0
    assert rejected.failed_constraints == ["min_target_coverage_per_embedding_kchar"]
    assert rejected.metrics["target_coverage_per_embedding_kchar"] < 3.0


def test_sweep_selection_constraints_filter_retrieval_value_per_latency():
    fast = sweep_candidate(
        "fast",
        mean_latency_ms=10.0,
        p95_latency_ms=12.0,
        retrieval_score=0.84,
    )
    slow = sweep_candidate(
        "slow",
        mean_latency_ms=35.0,
        p95_latency_ms=45.0,
        retrieval_score=0.84,
    )

    selection = build_sweep_selection(
        [fast, slow],
        selection_constraints={
            "max_p95_latency_ms": 20.0,
            "min_retrieval_score_per_p95_latency_ms": 0.05,
        },
    )

    assert selection.recommended == "fast"
    assert selection.eligible_count == 1
    assert selection.rejected_count == 1
    top = selection.ranking[0]
    rejected = next(row for row in selection.ranking if not row.eligible)
    assert round(top.metrics["retrieval_score_per_p95_latency_ms"] or 0.0, 4) == 0.07
    assert rejected.failed_constraints == [
        "min_retrieval_score_per_p95_latency_ms",
        "max_p95_latency_ms",
    ]


def test_sweep_selection_constraints_can_require_visual_text_part_coverage(tmp_path):
    manifest = make_manifest(tmp_path)

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["semantic", "multimodal"],
        max_chars_values=[140],
        overlap_chars_values=[20],
        min_chars=40,
        visual_context_chars_values=[120],
        selection_constraints={"min_visual_text_part_coverage_ratio": 1.0},
    )

    assert report.selection.constraints == {
        "min_visual_text_part_coverage_ratio": 1.0,
    }
    assert report.selection.recommended == "multimodal-max140-ov20-min40-visual120"
    assert report.selection.eligible_count == 1
    assert report.selection.rejected_count == 1
    assert report.selection.ranking[0].metrics["visual_text_part_coverage_ratio"] == 1.0
    rejected = next(row for row in report.selection.ranking if not row.eligible)
    assert rejected.name == "semantic-max140-ov20-min40"
    assert rejected.failed_constraints == ["min_visual_text_part_coverage_ratio"]
    assert rejected.metrics["visual_text_part_coverage_ratio"] == 0.0


def test_sweep_selection_reports_no_recommendation_when_constraints_all_fail(tmp_path):
    manifest = make_manifest(tmp_path)

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["semantic", "hierarchical"],
        max_chars_values=[140],
        overlap_chars_values=[20],
        min_chars=40,
        parent_max_chars_values=[90],
        visual_context_chars_values=[120],
        retrieval_cases=[RetrievalCase(query="capital investment table", expected_pages=[1])],
        selection_constraints={"min_target_coverage_at_k": 1.01},
    )

    assert report.selection.recommended is None
    assert report.selection.eligible_count == 0
    assert report.selection.rejected_count == 2
    assert all(
        row.failed_constraints == ["min_target_coverage_at_k"]
        for row in report.selection.ranking
    )


def test_sweep_selection_constraints_can_require_visual_case_coverage(tmp_path):
    manifest = make_manifest(tmp_path)

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["semantic", "hierarchical"],
        max_chars_values=[140],
        overlap_chars_values=[20],
        min_chars=40,
        parent_max_chars_values=[90],
        visual_context_chars_values=[120],
        retrieval_cases=[
            RetrievalCase(
                query="capital investment table",
                expected_asset_ids=["asset-1"],
                metadata={"case_source": "visual_object_probe"},
            )
        ],
        selection_constraints={
            "min_target_type_coverage:asset": 1.0,
            "min_case_group_target_coverage:case_source:visual_object_probe": 1.0,
        },
    )

    assert report.selection.eligible_count == 2
    assert report.selection.rejected_count == 0
    assert report.selection.constraints == {
        "min_target_type_coverage:asset": 1.0,
        "min_case_group_target_coverage:case_source:visual_object_probe": 1.0,
    }
    top_metrics = report.selection.ranking[0].metrics
    assert top_metrics["target_type.asset.coverage_at_k"] == 1.0
    assert (
        top_metrics["case_group.case_source.visual_object_probe.target_coverage_at_k"]
        == 1.0
    )


def test_sweep_selection_rejects_missing_target_type_coverage(tmp_path):
    manifest = make_manifest(tmp_path)

    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=["semantic", "hierarchical"],
        max_chars_values=[140],
        overlap_chars_values=[20],
        min_chars=40,
        parent_max_chars_values=[90],
        visual_context_chars_values=[120],
        retrieval_cases=[RetrievalCase(query="capital investment table", expected_pages=[1])],
        selection_constraints={"min_target_type_coverage:asset": 0.1},
    )

    assert report.selection.recommended is None
    assert report.selection.eligible_count == 0
    assert report.selection.rejected_count == 2
    assert all(
        row.failed_constraints == ["min_target_type_coverage:asset"]
        for row in report.selection.ranking
    )


def test_sweep_pareto_dominance_accounts_for_quality_and_cost():
    stronger = {
        "retrieval_recall_at_k": 0.9,
        "target_coverage_at_k": 0.8,
        "target_ndcg_at_k": 0.7,
        "precision_at_k": 0.6,
        "quality_score": 0.85,
        "visual_text_coverage_ratio": 1.0,
        "visual_text_part_coverage_ratio": 1.0,
        "target_rank_efficiency": 1.0,
        "mean_target_rank": 1.0,
        "p95_target_rank": 1.0,
        "mean_latency_ms": 10.0,
        "p95_latency_ms": 12.0,
        "chunk_count": 20.0,
        "total_chunk_chars": 1200.0,
        "mean_chunk_chars": 60.0,
        "p95_chunk_chars": 80.0,
        "embedding_text_kchars": 1.2,
        "standalone_visual_chunk_count": 0.0,
    }
    weaker = {
        "retrieval_recall_at_k": 0.8,
        "target_coverage_at_k": 0.7,
        "target_ndcg_at_k": 0.6,
        "precision_at_k": 0.5,
        "quality_score": 0.8,
        "visual_text_coverage_ratio": 1.0,
        "visual_text_part_coverage_ratio": 1.0,
        "target_rank_efficiency": 1 / 3,
        "mean_target_rank": 3.0,
        "p95_target_rank": 3.0,
        "mean_latency_ms": 12.0,
        "p95_latency_ms": 16.0,
        "chunk_count": 25.0,
        "total_chunk_chars": 1800.0,
        "mean_chunk_chars": 72.0,
        "p95_chunk_chars": 100.0,
        "embedding_text_kchars": 1.8,
        "standalone_visual_chunk_count": 1.0,
    }
    tradeoff = {**stronger, "target_coverage_at_k": 0.82, "mean_latency_ms": 14.0}
    embedding_text_tradeoff = {
        **stronger,
        "target_coverage_at_k": 0.82,
        "total_chunk_chars": 2200.0,
        "mean_chunk_chars": 110.0,
        "p95_chunk_chars": 180.0,
        "embedding_text_kchars": 2.2,
    }
    leaner_same_quality = {
        **stronger,
        "total_chunk_chars": 900.0,
        "mean_chunk_chars": 45.0,
        "p95_chunk_chars": 70.0,
        "embedding_text_kchars": 0.9,
    }

    assert dominates(stronger, weaker)
    assert not dominates(weaker, stronger)
    assert not dominates(stronger, tradeoff)
    assert not dominates(embedding_text_tradeoff, stronger)
    assert dominates(leaner_same_quality, stronger)


def test_sweep_chunking_cli_writes_report(tmp_path):
    package_dir = write_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "sweep_report.json"
    candidates_dir = tmp_path / "candidates"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="capital investment table",
                expected_asset_ids=["asset-1"],
                metadata={"case_source": "visual_object_probe"},
            )
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "sweep-chunking",
            "--package-dir",
            str(package_dir),
            "--strategies",
            "semantic,hierarchical",
            "--max-chars",
            "140",
            "--overlap-chars",
            "20",
            "--min-chars",
            "40",
            "--parent-max-chars",
            "90",
            "--visual-context-chars",
            "120",
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
            "--candidates-dir",
            str(candidates_dir),
            "--selection-max-total-chunk-chars",
            "250",
            "--selection-min-target-coverage-per-embedding-kchar",
            "3.0",
            "--selection-min-retrieval-score-per-p95-latency-ms",
            "0.0",
            "--selection-max-p95-latency-ms",
            "1000",
            "--selection-min-target-type-coverage",
            "asset=1.0",
            "--selection-min-case-group-target-coverage",
            "case_source:visual_object_probe=1.0",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparison"]["best_by_retrieval"] is not None
    assert payload["selection"]["recommended"] == "semantic-max140-ov20-min40"
    assert payload["selection"]["constraints"] == {
        "max_total_chunk_chars": 250.0,
        "min_target_coverage_per_embedding_kchar": 3.0,
        "min_retrieval_score_per_p95_latency_ms": 0.0,
        "max_p95_latency_ms": 1000.0,
        "min_target_type_coverage:asset": 1.0,
        "min_case_group_target_coverage:case_source:visual_object_probe": 1.0,
    }
    assert payload["selection"]["eligible_count"] == 1
    assert payload["selection"]["rejected_count"] == 1
    assert payload["selection"]["ranking"][0]["eligible"] is True
    assert payload["selection"]["ranking"][-1]["eligible"] is False
    assert payload["selection"]["ranking"][0]["metrics"]["total_chunk_chars"] == 194.0
    assert (
        payload["selection"]["ranking"][0]["metrics"][
            "target_coverage_per_embedding_kchar"
        ]
        > 3.0
    )
    assert payload["selection"]["ranking"][0]["metrics"]["p95_latency_ms"] is not None
    assert (
        payload["selection"]["ranking"][0]["metrics"][
            "retrieval_score_per_p95_latency_ms"
        ]
        is not None
    )
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["name"] == payload["selection"]["recommended"]
    assert any(candidates_dir.glob("chunks.semantic-*.jsonl"))


def test_sweep_chunking_cli_can_require_visual_text_part_coverage(tmp_path):
    package_dir = write_package(tmp_path)
    output_path = tmp_path / "sweep_report.json"

    result = CliRunner().invoke(
        app,
        [
            "sweep-chunking",
            "--package-dir",
            str(package_dir),
            "--strategies",
            "semantic,multimodal",
            "--max-chars",
            "140",
            "--overlap-chars",
            "20",
            "--min-chars",
            "40",
            "--visual-context-chars",
            "120",
            "--selection-min-visual-text-part-coverage-ratio",
            "1.0",
            "--output",
            str(output_path),
            "--no-write-candidates",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["selection"]["recommended"] == "multimodal-max140-ov20-min40-visual120"
    assert payload["selection"]["constraints"] == {
        "min_visual_text_part_coverage_ratio": 1.0,
    }
    assert payload["selection"]["eligible_count"] == 1
    rejected = next(row for row in payload["selection"]["ranking"] if not row["eligible"])
    assert rejected["failed_constraints"] == ["min_visual_text_part_coverage_ratio"]


def write_package(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    manifest = make_manifest(tmp_path)
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", manifest.profiles)
    write_jsonl(package_dir / "chunks.jsonl", manifest.chunks)
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)
    write_jsonl(package_dir / "triples.jsonl", manifest.triples)
    return package_dir


def sweep_candidate(
    name: str,
    mean_latency_ms: float,
    p95_latency_ms: float,
    retrieval_score: float,
) -> ChunkingSweepCandidate:
    retrieval = RetrievalEvaluation(
        case_count=1,
        expected_case_count=1,
        passed_count=1,
        failed_count=0,
        hit_rate=retrieval_score,
        recall_at_k=retrieval_score,
        mrr=retrieval_score,
        target_coverage_at_k=retrieval_score,
        mean_target_ndcg_at_k=retrieval_score,
        mean_precision_at_k=retrieval_score,
        top_k=5,
        total_query_latency_ms=mean_latency_ms,
        mean_latency_ms=mean_latency_ms,
        p95_latency_ms=p95_latency_ms,
        failed_queries=[],
        results=[],
    )
    report = ChunkingQualityReport(
        page_count=1,
        chunk_count=1,
        covered_page_count=1,
        page_coverage_ratio=1.0,
        char_count=NumericSummary(
            count=1,
            minimum=100,
            maximum=100,
            mean=100.0,
            p50=100.0,
            p95=100.0,
        ),
        empty_chunk_count=0,
        chunks_under_min_chars=0,
        chunks_over_max_chars=0,
        section_coverage_ratio=1.0,
        visual_asset_linkage_ratio=1.0,
        visual_annotation_ratio=1.0,
        retrieval=retrieval,
        quality_score=retrieval_score,
    )
    return ChunkingSweepCandidate(
        name=name,
        strategy="semantic",
        chunk_count=1,
        report=report,
    )


def make_manifest(tmp_path):
    doc = SourceDocument(
        doc_id="doc",
        title="Reference Document",
        local_path=tmp_path / "reference.pdf",
    )
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=100,
            height=100,
            char_count=240,
            line_count=8,
            text_block_count=2,
            image_block_count=1,
            embedded_image_count=0,
            drawing_count=1,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text=(
                "Capital program overview for station access. "
                "The investment table describes priority corridors and visual evidence. "
                "Benchmark retrieval should find this page."
            ),
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.TABLE,
            caption="capital investment table",
            vlm_summary="priority corridors and station access evidence",
        )
    ]
    return ProcessingManifest(doc=doc, profiles=profiles, chunks=chunks, assets=assets, triples=[])
