import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.experiment import build_experiment_report
from chunking_docs.evaluation.retrieval import RetrievalCase
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


def test_build_experiment_report_summarizes_artifacts_and_candidates(tmp_path):
    package_dir, manifest = write_minimal_package(tmp_path)
    cases = [RetrievalCase(query="retrieval benchmark", expected_pages=[1])]

    report = build_experiment_report(
        package_dir=package_dir,
        manifest=manifest,
        candidates={"current": package_dir / "chunks.jsonl"},
        retrieval_cases=cases,
        min_chars=10,
    )

    artifacts = {artifact.path: artifact for artifact in report.artifacts}
    assert report.package_counts == {"pages": 1, "chunks": 1, "assets": 1, "triples": 0}
    assert artifacts["chunks.jsonl"].record_count == 1
    assert artifacts["chunks.jsonl"].sha256 is not None
    assert artifacts["ingestion_readiness.final.json"].exists is True
    assert artifacts["retrieval_gate.final.json"].exists is True
    assert artifacts["qdrant_eval.final.json"].exists is True
    assert artifacts["qdrant_fusion_sweep.final.json"].exists is True
    assert artifacts["qdrant_retrieval_config.final.json"].exists is True
    assert artifacts["chunking_comparison_gate.final.json"].exists is True
    assert artifacts["chunking_sweep.final.json"].exists is True
    assert artifacts["graph_audit.final.json"].exists is True
    assert artifacts["visual_gate.final.json"].exists is True
    assert artifacts["visual_run_comparison.json"].exists is True
    assert artifacts["vlm_experiment_plan.json"].exists is True
    validations = {validation.path: validation for validation in report.validation_summaries}
    assert validations["ingestion_readiness.final.json"].passed is True
    assert validations["ingestion_readiness.final.json#retrieval_gate"].metrics["recall_at_k"] == 1.0
    assert validations["retrieval_gate.final.json"].metrics["recall_at_k"] == 1.0
    assert validations["retrieval_gate.final.json"].metrics["mrr"] == 1.0
    assert validations["retrieval_gate.final.json"].metrics["mean_target_rank"] == 1.0
    assert validations["retrieval_gate.final.json"].metrics["p95_latency_ms"] == 12.0
    assert validations["retrieval_gate.final.json"].metrics[
        "chunk_strategy.visual_asset_text.target_coverage_at_k"
    ] == 1.0
    assert validations["retrieval_gate.final.json"].metrics[
        "retrieval_role.child.target_coverage_at_k"
    ] == 1.0
    assert validations["retrieval_gate.final.json"].metrics[
        "case_group.case_source.visual_object_probe.target_coverage_at_k"
    ] == 1.0
    assert validations["qdrant_eval.final.json"].metrics["target_coverage_at_k"] == 1.0
    assert validations["qdrant_eval.final.json"].metrics["mean_latency_ms"] == 8.0
    assert validations["qdrant_eval.final.json"].metrics["case_count"] == 1.0
    assert validations["qdrant_eval.final.json"].metrics[
        "case_group.modality.vision_object.ndcg_at_k"
    ] == 0.8
    assert validations["qdrant_fusion_sweep.final.json"].passed is True
    assert validations["qdrant_fusion_sweep.final.json"].candidate == "caption_weighted"
    assert validations["qdrant_fusion_sweep.final.json"].metrics["candidate_count"] == 2.0
    assert validations["qdrant_fusion_sweep.final.json"].metrics["eligible_count"] == 1.0
    assert validations["qdrant_fusion_sweep.final.json"].metrics["selection_score"] == 4.2
    assert validations["qdrant_fusion_sweep.final.json"].metrics["target_coverage_at_k"] == 0.98
    assert validations["qdrant_fusion_sweep.final.json"].metrics["failed_query_count"] == 1.0
    assert (
        validations["qdrant_fusion_sweep.final.json"]
        .metrics["case_group_recommendation_count"]
        == 1.0
    )
    assert (
        validations["qdrant_fusion_sweep.final.json"]
        .metrics[
            "case_group_recommendation.case_source.visual_object_probe.top_candidate.target_coverage_at_k"
        ]
        == 0.9
    )
    assert (
        validations["qdrant_fusion_sweep.final.json"]
        .metrics[
            "case_group_recommendation.case_source.visual_object_probe.recommended_from_globally_eligible"
        ]
        == 1.0
    )
    assert validations["qdrant_retrieval_config.final.json"].passed is True
    assert validations["qdrant_retrieval_config.final.json"].candidate == "caption_weighted"
    assert validations["qdrant_retrieval_config.final.json"].metrics["vector_count"] == 2.0
    assert validations["qdrant_retrieval_config.final.json"].metrics["fusion_weight_count"] == 2.0
    assert validations["qdrant_retrieval_config.final.json"].metrics["top_k"] == 5.0
    assert validations["qdrant_retrieval_config.final.json"].metrics["candidate_rank"] == 1.0
    assert validations["qdrant_retrieval_config.final.json"].metrics["target_coverage_at_k"] == 0.98
    assert validations["qdrant_retrieval_config.final.json"].metrics[
        "case_group_selection.target_coverage_at_k"
    ] == 0.9
    assert validations["chunking_comparison_gate.final.json"].metrics[
        "case_group.case_source.visual_object_probe.target_coverage_at_k"
    ] == 0.9
    assert validations["chunking_comparison_gate.final.json"].metrics[
        "pairwise_target_ndcg_delta_ci_low"
    ] == 0.03
    assert validations["chunking_comparison_gate.final.json"].metrics[
        "pairwise_candidate_win_rate"
    ] == 0.7
    assert validations["chunking_sweep.final.json"].passed is True
    assert validations["chunking_sweep.final.json"].candidate == "semantic"
    assert validations["chunking_sweep.final.json"].metrics["candidate_count"] == 2.0
    assert validations["chunking_sweep.final.json"].metrics["eligible_count"] == 1.0
    assert validations["chunking_sweep.final.json"].metrics["rejected_count"] == 1.0
    assert validations["chunking_sweep.final.json"].metrics["selection_score"] == 0.82
    assert validations["chunking_sweep.final.json"].metrics["target_coverage_at_k"] == 0.9
    assert (
        validations["chunking_sweep.final.json"]
        .metrics["target_coverage_per_embedding_kchar"]
        == 4.5
    )
    assert (
        validations["chunking_sweep.final.json"]
        .metrics["retrieval_score_per_p95_latency_ms"]
        == 0.07
    )
    assert validations["graph_audit.final.json"].metrics["orphan_count"] == 0.0
    assert validations["visual_gate.final.json"].metrics["vlm_summary_coverage"] == 1.0
    assert validations["retrieval_diagnostics.final.json"].metrics["no_hit_count"] == 1.0
    assert (
        validations["retrieval_diagnostics.final.json"]
        .metrics["reason.no_expected_target_retrieved"]
        == 2.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"]
        .metrics["missing_target_type.triple"]
        == 1.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "case_group.case_source.visual_object_probe.reason.missing_asset"
        ]
        == 1.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "case_group.query_mode.salient_terms.missing_target_type.triple"
        ]
        == 1.0
    )
    assert validations["retrieval_diagnostics.final.json"].metrics["source_hit.bm25"] == 5.0
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "source_family_hit.visual"
        ]
        == 4.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "matched_source_hit.qdrant:image_dense"
        ]
        == 2.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "matched_source_family_hit.visual"
        ]
        == 2.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "excluded_source_hit.qdrant:image_dense"
        ]
        == 1.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "excluded_source_family_hit.visual"
        ]
        == 1.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "case_group.case_source.visual_object_probe.source_family_hit.visual"
        ]
        == 3.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "case_group.case_source.visual_object_probe.matched_source_family_hit.visual"
        ]
        == 2.0
    )
    assert (
        validations["retrieval_diagnostics.final.json"].metrics[
            "case_group.case_source.visual_object_probe.excluded_source_family_hit.visual"
        ]
        == 1.0
    )
    assert (
        validations["ingestion_readiness.final.json#visual_text_coverage"]
        .metrics["visual_text_coverage_ratio"]
        == 1.0
    )
    assert validations["visual_run_comparison.json"].passed is True
    assert validations["visual_run_comparison.json"].candidate == "structured"
    assert validations["visual_run_comparison.json"].metrics["run_count"] == 2.0
    assert validations["visual_run_comparison.json"].metrics["shared_job_count"] == 1.0
    assert validations["visual_run_comparison.json"].metrics["job_set_mismatch"] == 0.0
    assert validations["visual_run_comparison.json"].metrics["best_quality_score"] == 0.92
    assert validations["vlm_experiment_plan.json"].passed is True
    assert validations["vlm_experiment_plan.json"].metrics["profile_count"] == 2.0
    assert validations["vlm_experiment_plan.json"].metrics["recipe_count"] == 2.0
    assert validations["vlm_experiment_plan.json"].metrics["doctor_output_count"] == 2.0
    assert validations["vlm_experiment_plan.json"].metrics["selected_job_count"] == 3.0
    assert validations["vlm_experiment_plan.json"].metrics["operation.vlm.count"] == 3.0
    assert validations["vlm_experiment_plan.json"].metrics[
        "recipe.max_generation_tokens_upper_bound.max"
    ] == 1536.0
    assert validations["vlm_experiment_plan_gate.json"].passed is True
    assert validations["vlm_experiment_plan_gate.json"].metrics["profile_count"] == 2.0
    assert validations["vlm_experiment_plan_gate.json"].metrics[
        "existing_doctor_output_count"
    ] == 2.0
    assert validations["vlm_experiment_plan_gate.json"].metrics[
        "completed_result_profile_count"
    ] == 2.0
    assert report.qdrant_collection["collection"] == "document_chunks"
    assert report.bm25_tokenizer["strategy"] == "mixed"
    assert report.source_file == {"name": "reference.pdf", "bytes": 1234, "sha256": "abc"}
    assert report.package_config["base_chunking_strategy"] == "page"
    assert report.package_config["lexical_tokenizer"]["strategy"] == "mixed"
    assert report.candidate_files == {"current": str(package_dir / "chunks.jsonl")}
    assert report.comparison is not None
    assert report.comparison.best_by_retrieval == "current"
    assert report.comparison.rows[0].visual_text_coverage_ratio == 1.0
    assert report.comparison.rows[0].chunk_strategy_metrics["visual_asset_text"][
        "target_coverage_at_k"
    ] == 1.0
    assert report.comparison.rows[0].retrieval_role_metrics["child"][
        "target_coverage_at_k"
    ] == 1.0


def test_write_experiment_report_cli_writes_json(tmp_path):
    package_dir, _ = write_minimal_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "experiment_report.json"
    write_jsonl(cases_path, [RetrievalCase(query="retrieval benchmark", expected_pages=[1])])

    result = CliRunner().invoke(
        app,
        [
            "write-experiment-report",
            "--package-dir",
            str(package_dir),
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparison"]["best_by_retrieval"] == "current"
    assert payload["validation_summaries"][0]["passed"] is True
    assert any(
        summary["path"] == "ingestion_readiness.final.json#retrieval_gate"
        and summary["metrics"]["recall_at_k"] == 1.0
        for summary in payload["validation_summaries"]
    )
    assert payload["artifacts"][0]["sha256"]


def write_minimal_package(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
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
            char_count=128,
            line_count=6,
            text_block_count=1,
            image_block_count=1,
            embedded_image_count=0,
            drawing_count=2,
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
            text="retrieval benchmark visual evidence table policy",
            asset_ids=["asset-1"],
            metadata={
                "chunking_strategy": "visual_asset_text",
                "retrieval_role": "child",
            },
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.TABLE,
            caption="benchmark table",
            vlm_summary="visual evidence table",
        )
    ]
    manifest = ProcessingManifest(
        doc=doc,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=[],
        metadata={
            "profile_summary": {"page_count": 1, "degraded_page_count": 0},
            "source_file": {"name": "reference.pdf", "bytes": 1234, "sha256": "abc"},
            "package_config": {
                "base_chunking_strategy": "page",
                "render_zoom": 1.5,
                "dry_run_embeddings": True,
                "section_map_count": 0,
                "extract_tables": True,
                "lexical_tokenizer": {"strategy": "mixed"},
            },
        },
    )
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", profiles)
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", [])
    (package_dir / "bm25_tokens.json").write_text(
        json.dumps({"tokenizer": {"strategy": "mixed"}}, indent=2),
        encoding="utf-8",
    )
    (package_dir / "ingestion_readiness.final.json").write_text(
        json.dumps(
            {
                "passed": True,
                "failed_components": [],
                "components": [
                    {
                        "name": "retrieval_gate",
                        "passed": True,
                        "metadata": {
                            "failed_checks": [],
                            "metrics": {
                                "recall_at_k": 1.0,
                                "target_type.asset.coverage_at_k": 1.0,
                            },
                        },
                    },
                    {
                        "name": "visual_text_coverage",
                        "passed": True,
                        "metadata": {
                            "visual_text_asset_count": 1,
                            "visual_text_covered_asset_count": 1,
                            "visual_text_coverage_ratio": 1.0,
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "retrieval_gate.final.json").write_text(
        json.dumps(
            {
                "passed": True,
                "failed_checks": [],
                "metrics": {
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "mean_target_rank": 1.0,
                    "p95_target_rank": 1.0,
                    "p95_latency_ms": 12.0,
                    "target_type.asset.coverage_at_k": 1.0,
                    "chunk_strategy.visual_asset_text.target_coverage_at_k": 1.0,
                    "retrieval_role.child.target_coverage_at_k": 1.0,
                },
                "case_group_metrics": {
                    "case_source": {
                        "visual_object_probe": {
                            "target_coverage_at_k": 1.0,
                            "ndcg_at_k": 1.0,
                        }
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "qdrant_eval.final.json").write_text(
        json.dumps(
            {
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "target_coverage_at_k": 1.0,
                "mean_target_ndcg_at_k": 1.0,
                "mean_latency_ms": 8.0,
                "case_count": 1,
                "case_group_metrics": {
                    "modality": {
                        "vision_object": {
                            "target_coverage_at_k": 1.0,
                            "ndcg_at_k": 0.8,
                        }
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "qdrant_fusion_sweep.final.json").write_text(
        json.dumps(
            {
                "vector_names": ["text_dense", "caption_dense"],
                "graph_expand": False,
                "candidate_count": 2,
                "eligible_count": 1,
                "recommended": "caption_weighted",
                "best_by_target_ndcg": "caption_weighted",
                "case_group_recommendations": {
                    "case_source": {
                        "visual_object_probe": {
                            "group_name": "case_source",
                            "group_value": "visual_object_probe",
                            "candidate_count": 2,
                            "eligible_count": 1,
                            "recommended": "caption_weighted",
                            "recommended_from_globally_eligible": True,
                            "top_candidates": [
                                {
                                    "name": "caption_weighted",
                                    "fusion_weights": {
                                        "bm25": 1.2,
                                        "qdrant:caption_dense": 1.1,
                                    },
                                    "global_rank": 1,
                                    "globally_eligible": True,
                                    "selection_score": 4.2,
                                    "case_count": 3,
                                    "recall_at_k": 1.0,
                                    "target_coverage_at_k": 0.9,
                                    "ndcg_at_k": 0.88,
                                    "mrr": 0.75,
                                    "precision_at_k": 0.4,
                                    "mean_latency_ms": 34.0,
                                    "failed_query_count": 0,
                                }
                            ],
                        }
                    }
                },
                "candidates": [
                    {
                        "name": "caption_weighted",
                        "fusion_weights": {"bm25": 1.2, "qdrant:caption_dense": 1.1},
                        "selection_score": 4.2,
                        "eligible": True,
                        "eligibility_failures": [],
                        "rank": 1,
                        "evaluation": {
                            "recall_at_k": 0.97,
                            "target_coverage_at_k": 0.98,
                            "mean_target_ndcg_at_k": 0.93,
                            "mrr": 0.91,
                            "mean_precision_at_k": 0.2,
                            "mean_latency_ms": 34.0,
                            "p95_latency_ms": 40.0,
                            "failed_queries": ["miss"],
                        },
                    },
                    {
                        "name": "image_weighted",
                        "fusion_weights": {"bm25": 1.2, "qdrant:image_dense": 0.5},
                        "selection_score": 3.1,
                        "eligible": False,
                        "eligibility_failures": ["min_target_ndcg_at_k"],
                        "rank": 2,
                        "evaluation": {
                            "recall_at_k": 0.95,
                            "target_coverage_at_k": 0.95,
                            "mean_target_ndcg_at_k": 0.68,
                            "mrr": 0.7,
                            "mean_precision_at_k": 0.19,
                            "mean_latency_ms": 28.0,
                            "p95_latency_ms": 32.0,
                            "failed_queries": [],
                        },
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "qdrant_retrieval_config.final.json").write_text(
        json.dumps(
            {
                "backend": "qdrant_hybrid",
                "collection_name": "document_chunks",
                "vector_names": ["text_dense", "caption_dense"],
                "top_k": 5,
                "graph_expand": False,
                "collapse_hierarchical": False,
                "fusion_weights": {"bm25": 1.2, "qdrant:caption_dense": 1.1},
                "selection": {
                    "candidate": "caption_weighted",
                    "source": "global_recommended",
                    "candidate_rank": 1,
                    "candidate_eligible": True,
                    "eligibility_failures": [],
                    "metrics": {
                        "selection_score": 4.2,
                        "target_coverage_at_k": 0.98,
                        "mean_target_ndcg_at_k": 0.93,
                        "failed_query_count": 1.0,
                    },
                    "case_group_metrics": {
                        "target_coverage_at_k": 0.9,
                        "ndcg_at_k": 0.85,
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "retrieval_diagnostics.final.json").write_text(
        json.dumps(
            {
                "case_count": 3,
                "failed_count": 2,
                "partial_count": 1,
                "no_hit_count": 1,
                "low_precision_count": 1,
                "low_target_ndcg_count": 2,
                "reason_counts": {
                    "no_expected_target_retrieved": 2,
                    "missing_asset": 1,
                    "missing_triple": 1,
                },
                "missing_target_type_counts": {
                    "asset": 1,
                    "triple": 1,
                },
                "source_counts": {
                    "bm25": 5,
                    "qdrant:image_dense": 4,
                },
                "source_family_counts": {
                    "lexical": 5,
                    "visual": 4,
                },
                "matched_source_counts": {
                    "bm25": 3,
                    "qdrant:image_dense": 2,
                },
                "matched_source_family_counts": {
                    "lexical": 3,
                    "visual": 2,
                },
                "excluded_source_counts": {
                    "qdrant:image_dense": 1,
                },
                "excluded_source_family_counts": {
                    "visual": 1,
                },
                "source_counts_by_case_group": {
                    "case_source": {
                        "visual_object_probe": {
                            "bm25": 2,
                            "qdrant:image_dense": 3,
                        }
                    }
                },
                "source_family_counts_by_case_group": {
                    "case_source": {
                        "visual_object_probe": {
                            "lexical": 2,
                            "visual": 3,
                        }
                    }
                },
                "matched_source_counts_by_case_group": {
                    "case_source": {
                        "visual_object_probe": {
                            "bm25": 1,
                            "qdrant:image_dense": 2,
                        }
                    }
                },
                "matched_source_family_counts_by_case_group": {
                    "case_source": {
                        "visual_object_probe": {
                            "lexical": 1,
                            "visual": 2,
                        }
                    }
                },
                "excluded_source_counts_by_case_group": {
                    "case_source": {
                        "visual_object_probe": {
                            "qdrant:image_dense": 1,
                        }
                    }
                },
                "excluded_source_family_counts_by_case_group": {
                    "case_source": {
                        "visual_object_probe": {
                            "visual": 1,
                        }
                    }
                },
                "reason_counts_by_case_group": {
                    "case_source": {
                        "visual_object_probe": {
                            "missing_asset": 1,
                            "no_expected_target_retrieved": 1,
                        }
                    },
                    "query_mode": {
                        "salient_terms": {
                            "missing_triple": 1,
                        }
                    },
                },
                "missing_target_type_counts_by_case_group": {
                    "case_source": {
                        "visual_object_probe": {
                            "asset": 1,
                        }
                    },
                    "query_mode": {
                        "salient_terms": {
                            "triple": 1,
                        }
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "chunking_comparison_gate.final.json").write_text(
        json.dumps(
            {
                "passed": True,
                "failed_checks": [],
                "candidate": "multimodal",
                "metrics": {"retrieval_target_coverage_at_k": 0.9},
                "case_group_metrics": {
                    "case_source": {
                        "visual_object_probe": {
                            "target_coverage_at_k": 0.9,
                            "precision_at_k": 0.75,
                        }
                    }
                },
                "pairwise_metrics": {
                    "pairwise_candidate_win_rate": 0.7,
                    "pairwise_mean_target_ndcg_delta": 0.05,
                    "pairwise_target_ndcg_delta_ci_low": 0.03,
                    "pairwise_target_ndcg_delta_ci_high": 0.08,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "chunking_sweep.final.json").write_text(
        json.dumps(
            {
                "candidates": [{"name": "semantic"}, {"name": "hierarchical"}],
                "selection": {
                    "recommended": "semantic",
                    "eligible_count": 1,
                    "rejected_count": 1,
                    "pareto_front": ["semantic", "hierarchical"],
                    "eligible_pareto_front": ["semantic"],
                    "constraints": {"min_target_coverage_per_embedding_kchar": 3.0},
                    "ranking": [
                        {
                            "name": "semantic",
                            "score": 0.82,
                            "eligible": True,
                            "failed_constraints": [],
                            "metrics": {
                                "target_coverage_at_k": 0.9,
                                "target_ndcg_at_k": 0.8,
                                "embedding_text_kchars": 0.2,
                                "target_coverage_per_embedding_kchar": 4.5,
                                "target_ndcg_per_embedding_kchar": 4.0,
                                "retrieval_score_per_embedding_kchar": 4.125,
                                "p95_latency_ms": 12.0,
                                "retrieval_score_per_p95_latency_ms": 0.07,
                            },
                        },
                        {
                            "name": "hierarchical",
                            "score": 0.78,
                            "eligible": False,
                            "failed_constraints": [
                                "min_target_coverage_per_embedding_kchar"
                            ],
                            "metrics": {
                                "target_coverage_at_k": 0.9,
                                "embedding_text_kchars": 0.5,
                                "target_coverage_per_embedding_kchar": 1.8,
                            },
                        },
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "graph_audit.final.json").write_text(
        json.dumps({"triple_count": 0, "orphan_count": 0, "duplicate_count": 0}, indent=2),
        encoding="utf-8",
    )
    (package_dir / "visual_gate.final.json").write_text(
        json.dumps(
            {
                "passed": True,
                "failed_checks": [],
                "vlm_summary_coverage": 1.0,
                "ocr_text_coverage": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "visual_run_comparison.json").write_text(
        json.dumps(
            {
                "best_by_quality": "structured",
                "fastest_by_total_latency": "raw",
                "best_by_triple_density": "structured",
                "union_job_count": 1,
                "shared_job_count": 1,
                "job_set_mismatch": False,
                "rows": [
                    {
                        "name": "structured",
                        "quality_score": 0.92,
                        "triples_per_vlm_job": 1.0,
                        "total_mean_latency_ms": 42.0,
                    },
                    {
                        "name": "raw",
                        "quality_score": 0.55,
                        "triples_per_vlm_job": 0.0,
                        "total_mean_latency_ms": 18.0,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "vlm_experiment_plan.json").write_text(
        json.dumps(
            {
                "package_dir": str(package_dir),
                "jobs_file": str(package_dir / "visual_jobs.priority.jsonl"),
                "profiles": ["qwen2_5_vl_7b", "phi3_5_vision"],
                "limit": 3,
                "batch_size": 2,
                "job_summary": {
                    "jobs_file": str(package_dir / "visual_jobs.priority.jsonl"),
                    "exists": True,
                    "total_job_count": 30,
                    "selected_job_count": 3,
                    "selected_pending_job_count": 3,
                    "skipped_by_limit_count": 27,
                    "operation_counts": {"ocr": 3, "vlm": 3},
                    "asset_kind_counts": {"map": 3},
                },
                "batches": [
                    {"batch_id": "batch_001", "offset": 0, "limit": 2},
                    {"batch_id": "batch_002", "offset": 2, "limit": 1},
                ],
                "recipes": [
                    {
                        "name": "qwen2_5_vl_7b",
                        "profile": "qwen2_5_vl_7b",
                        "doctor_output": str(package_dir / "runtime_doctor.qwen2_5_vl_7b.json"),
                        "results_output": str(
                            package_dir / "visual_job_results.qwen2_5_vl_7b.jsonl"
                        ),
                        "annotations_output": str(
                            package_dir / "visual_annotations.qwen2_5_vl_7b.jsonl"
                        ),
                        "doctor_command": "chunking-docs doctor --output runtime_doctor.qwen2_5_vl_7b.json",
                        "command": "chunking-docs run-visual-jobs",
                        "metadata": {
                            "selected_vlm_job_count": 3,
                            "selected_ocr_job_count": 3,
                            "max_generation_tokens_upper_bound": 1536,
                            "min_gpu_memory_mib": 24576,
                            "batch_count": 2,
                        },
                    },
                    {
                        "name": "phi3_5_vision",
                        "profile": "phi3_5_vision",
                        "doctor_output": str(package_dir / "runtime_doctor.phi3_5_vision.json"),
                        "results_output": str(
                            package_dir / "visual_job_results.phi3_5_vision.jsonl"
                        ),
                        "annotations_output": str(
                            package_dir / "visual_annotations.phi3_5_vision.jsonl"
                        ),
                        "doctor_command": "chunking-docs doctor --output runtime_doctor.phi3_5_vision.json",
                        "command": "chunking-docs run-visual-jobs",
                        "metadata": {
                            "selected_vlm_job_count": 3,
                            "selected_ocr_job_count": 3,
                            "max_generation_tokens_upper_bound": 768,
                            "min_gpu_memory_mib": 12288,
                            "batch_count": 2,
                        },
                    },
                ],
                "compare_command": "chunking-docs compare-visual-runs --require-same-jobs",
                "batch_compare_commands": [
                    "chunking-docs compare-visual-runs --output visual_run_comparison.batch_001.json",
                    "chunking-docs compare-visual-runs --output visual_run_comparison.batch_002.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "vlm_experiment_plan_gate.json").write_text(
        json.dumps(
            {
                "plan_path": str(package_dir / "vlm_experiment_plan.json"),
                "passed": True,
                "failed_checks": [],
                "profile_count": 2,
                "recipe_count": 2,
                "doctor_output_count": 2,
                "existing_doctor_output_count": 2,
                "passed_doctor_output_count": 2,
                "results_output_count": 2,
                "existing_results_output_count": 2,
                "completed_result_profile_count": 2,
                "annotations_output_count": 2,
                "existing_annotations_output_count": 2,
                "union_job_count": 1,
                "shared_job_count": 1,
                "job_set_mismatch": False,
                "checks": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "qdrant_collection.json").write_text(
        json.dumps({"collection": "document_chunks", "named_vectors": {"text_dense": {"size": 384}}}),
        encoding="utf-8",
    )
    return package_dir, manifest
