import json

import pytest
from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.fusion_sweep import (
    QdrantFusionSweepCandidate,
    build_qdrant_fusion_sweep_report,
)
from chunking_docs.evaluation.retrieval import (
    RetrievalCaseGroupMetric,
    RetrievalCaseResult,
    RetrievalEvaluation,
)
from chunking_docs.evaluation.retrieval_config import (
    build_qdrant_retrieval_config_from_fusion_sweep,
)
from chunking_docs.models import ChunkKind, DocumentChunk
from chunking_docs.retrieval.qdrant_hybrid import QdrantHybridSearchHit


def test_qdrant_retrieval_config_exports_global_recommended_candidate():
    report = fusion_sweep_report()

    config = build_qdrant_retrieval_config_from_fusion_sweep(report)

    assert config.backend == "qdrant_hybrid"
    assert config.collection_name == "seoul_plan"
    assert config.package_dir == "outputs/package"
    assert config.bm25_tokens_path == "outputs/package/bm25_tokens.json"
    assert config.vector_names == ["text_dense", "caption_dense", "object_dense"]
    assert config.top_k == 7
    assert config.collapse_hierarchical is True
    assert config.fusion_weights == {"bm25": 1.0, "qdrant:caption_dense": 1.0}
    assert config.query_encoders["object_dense"] == "BAAI/bge-m3"
    assert config.lexical_tokenizer["strategy"] == "mixed"
    assert config.selection.source == "global_recommended"
    assert config.selection.candidate == "balanced"
    assert config.selection.metrics["target_coverage_at_k"] == pytest.approx(0.96)
    assert config.selection.pairwise_comparisons[0]["baseline"] == "object_weighted"
    assert config.selection.pairwise_comparisons[0]["candidate_win_rate"] == 1.0
    assert config.selection.pairwise_comparisons[0]["mean_target_coverage_delta"] == 1.0
    assert config.metadata["sweep_eligible_count"] == 2


def test_qdrant_retrieval_config_exports_case_group_recommendation():
    report = fusion_sweep_report()

    config = build_qdrant_retrieval_config_from_fusion_sweep(
        report,
        case_group="visual_object_probe",
        source_report="qdrant_fusion_sweep.json",
    )

    assert config.selection.source == "case_group_recommended"
    assert config.selection.source_report == "qdrant_fusion_sweep.json"
    assert config.selection.global_recommended == "balanced"
    assert config.selection.case_group == "case_source:visual_object_probe"
    assert config.selection.candidate == "object_weighted"
    assert config.selection.case_group_recommended_from_globally_eligible is True
    assert config.fusion_weights == {"bm25": 1.0, "qdrant:object_dense": 1.4}
    assert config.selection.case_group_metrics["target_coverage_at_k"] == pytest.approx(0.9)


def test_export_qdrant_retrieval_config_cli_writes_json(tmp_path):
    report_path = tmp_path / "qdrant_fusion_sweep.json"
    output_path = tmp_path / "qdrant_retrieval_config.json"
    report_path.write_text(
        fusion_sweep_report().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "export-qdrant-retrieval-config",
            str(report_path),
            "--output",
            str(output_path),
            "--case-group",
            "case_source:visual_object_probe",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "qdrant_hybrid"
    assert payload["selection"]["candidate"] == "object_weighted"
    assert payload["selection"]["pairwise_comparisons"][0]["baseline"] == "balanced"
    assert payload["fusion_weights"] == {"bm25": 1.0, "qdrant:object_dense": 1.4}


def test_eval_qdrant_retrieval_config_cli_uses_exported_settings(tmp_path, monkeypatch):
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())
    config = config.model_copy(
        update={
            "package_dir": str(tmp_path / "package"),
            "collection_name": "config_collection",
            "vector_names": ["text_dense"],
            "graph_expand": True,
            "fusion_weights": {"bm25": 1.2},
            "top_k": 3,
            "collapse_hierarchical": True,
            "lexical_tokenizer": {
                "strategy": "mixed",
                "min_n": 2,
                "max_n": 3,
                "ngram_cjk_only": False,
                "deduplicate": True,
            },
        }
    )
    config_path = tmp_path / "qdrant_retrieval_config.json"
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "eval.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    cases_path.write_text(
        json.dumps({"query": "redevelopment policy", "expected_pages": [1]}) + "\n",
        encoding="utf-8",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="redevelopment policy",
        metadata={"chunking_strategy": "semantic", "retrieval_role": "child"},
    )
    calls = {}

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            calls["search"] = kwargs
            return [
                QdrantHybridSearchHit(
                    item_id="chunk-1",
                    score=1.0,
                    sources=["bm25"],
                    chunk=chunk,
                )
            ]

    def fake_prepare_qdrant_hybrid_search(**kwargs):
        calls["prepare"] = kwargs
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": kwargs["collection"],
            "selected_vectors": ["text_dense"],
            "query_encoders": {"text_dense": "default_text"},
            "query_encoder_details": {"text_dense": {"backend": "hashing"}},
            "upserted": {"count": 1},
            "chunks": [chunk],
            "assets": [],
            "triples": [],
        }

    monkeypatch.setattr(
        "chunking_docs.cli.prepare_qdrant_hybrid_search",
        fake_prepare_qdrant_hybrid_search,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-retrieval-config",
            str(config_path),
            str(cases_path),
            "--output",
            str(output_path),
            "--repeat",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["prepare"]["package_dir"] == tmp_path / "package"
    assert calls["prepare"]["collection"] == "config_collection"
    assert calls["prepare"]["vector_names"] == "text_dense"
    assert calls["prepare"]["ngram_max"] == 3
    assert calls["prepare"]["ngram_cjk_only"] is False
    assert calls["prepare"]["deduplicate_tokens"] is True
    assert calls["search"]["top_k"] == 3
    assert calls["search"]["graph_expand"] is True
    assert calls["search"]["collapse_hierarchical"] is True
    assert calls["search"]["fusion_weights"] == {"bm25": 1.2}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["recall_at_k"] == 1.0
    assert payload["metadata"]["backend"] == "qdrant_hybrid_config"
    assert payload["metadata"]["config_selection"]["candidate"] == "balanced"


def test_qdrant_rag_context_config_cli_uses_exported_settings(tmp_path, monkeypatch):
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())
    config = config.model_copy(
        update={
            "package_dir": str(tmp_path / "package"),
            "collection_name": "config_collection",
            "vector_names": ["text_dense"],
            "graph_expand": True,
            "fusion_weights": {"bm25": 1.2},
            "top_k": 3,
            "collapse_hierarchical": True,
            "lexical_tokenizer": {
                "strategy": "mixed",
                "min_n": 2,
                "max_n": 3,
                "ngram_cjk_only": False,
                "deduplicate": True,
            },
        }
    )
    config_path = tmp_path / "qdrant_retrieval_config.json"
    output_path = tmp_path / "context.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="redevelopment policy",
        metadata={"chunking_strategy": "semantic", "retrieval_role": "child"},
    )
    calls = {}

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            calls["search"] = kwargs
            return [
                QdrantHybridSearchHit(
                    item_id="chunk-1",
                    score=1.0,
                    sources=["bm25"],
                    chunk=chunk,
                )
            ]

    def fake_prepare_qdrant_hybrid_search(**kwargs):
        calls["prepare"] = kwargs
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": kwargs["collection"],
            "selected_vectors": ["text_dense"],
            "query_encoders": {"text_dense": "default_text"},
            "query_encoder_details": {"text_dense": {"backend": "hashing"}},
            "upserted": {"count": 1},
            "chunks": [chunk],
            "assets": [],
            "triples": [],
        }

    monkeypatch.setattr(
        "chunking_docs.cli.prepare_qdrant_hybrid_search",
        fake_prepare_qdrant_hybrid_search,
    )

    result = CliRunner().invoke(
        app,
        [
            "qdrant-rag-context-config",
            str(config_path),
            "redevelopment policy",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["prepare"]["package_dir"] == tmp_path / "package"
    assert calls["prepare"]["collection"] == "config_collection"
    assert calls["prepare"]["vector_names"] == "text_dense"
    assert calls["prepare"]["ngram_max"] == 3
    assert calls["prepare"]["ngram_cjk_only"] is False
    assert calls["prepare"]["deduplicate_tokens"] is True
    assert calls["search"]["query"] == "redevelopment policy"
    assert calls["search"]["top_k"] == 3
    assert calls["search"]["graph_expand"] is True
    assert calls["search"]["collapse_hierarchical"] is True
    assert calls["search"]["fusion_weights"] == {"bm25": 1.2}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["query"] == "redevelopment policy"
    assert payload["chunks"][0]["chunk_id"] == "chunk-1"
    assert payload["metadata"]["backend"] == "qdrant_hybrid_config"
    assert payload["metadata"]["config_selection"]["candidate"] == "balanced"
    assert payload["metadata"]["fusion_weights"] == {"bm25": 1.2}


def fusion_sweep_report():
    balanced = QdrantFusionSweepCandidate(
        name="balanced",
        fusion_weights={"bm25": 1.0, "qdrant:caption_dense": 1.0},
        evaluation=evaluation(
            recall=0.96,
            coverage=0.96,
            ndcg=0.92,
            mrr=0.9,
            precision=0.22,
            failed=0,
            latency=25.0,
            visual_object_group=case_group_metric(
                coverage=0.5,
                ndcg=0.48,
                mrr=0.5,
            ),
            results=[
                retrieval_result(
                    "shared redevelopment query",
                    coverage=1.0,
                    ndcg=0.9,
                    reciprocal_rank=1.0,
                    rank=1,
                )
            ],
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
            results=[
                retrieval_result(
                    "shared redevelopment query",
                    coverage=0.0,
                    ndcg=0.0,
                    reciprocal_rank=0.0,
                    rank=None,
                )
            ],
        ),
    )

    return build_qdrant_fusion_sweep_report(
        [balanced, object_weighted],
        vector_names=["text_dense", "caption_dense", "object_dense"],
        min_recall_at_k=0.8,
        min_target_coverage_at_k=0.8,
        max_failed_queries=1,
        metadata={
            "collection": "seoul_plan",
            "package_dir": "outputs/package",
            "bm25_tokens_path": "outputs/package/bm25_tokens.json",
            "top_k": 7,
            "collapse_hierarchical": True,
            "query_encoders": {
                "text_dense": "BAAI/bge-m3",
                "caption_dense": "BAAI/bge-m3",
                "object_dense": "BAAI/bge-m3",
            },
            "lexical_tokenizer": {
                "strategy": "mixed",
                "min_n": 2,
                "max_n": 4,
                "ngram_cjk_only": True,
                "deduplicate": False,
            },
        },
    )


def evaluation(
    recall: float,
    coverage: float,
    ndcg: float,
    mrr: float,
    precision: float,
    failed: int,
    latency: float,
    visual_object_group: RetrievalCaseGroupMetric,
    results: list[RetrievalCaseResult] | None = None,
) -> RetrievalEvaluation:
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
        top_k=5,
        total_query_latency_ms=latency * 10,
        mean_latency_ms=latency,
        p95_latency_ms=latency,
        failed_queries=[f"failed-{index}" for index in range(failed)],
        case_group_metrics={"case_source": {"visual_object_probe": visual_object_group}},
        results=results or [],
    )


def retrieval_result(
    query: str,
    coverage: float,
    ndcg: float,
    reciprocal_rank: float,
    rank: int | None,
) -> RetrievalCaseResult:
    target_ranks = {"page:1": rank} if rank is not None else {}
    return RetrievalCaseResult(
        query=query,
        passed=rank is not None,
        latency_ms=10.0,
        top_pages=[1] if rank is not None else [],
        top_chunk_ids=["chunk-1"] if rank is not None else [],
        expected_pages=[1],
        expected_chunk_ids=[],
        expected_target_count=1,
        matched_target_count=1 if rank is not None else 0,
        target_coverage_at_k=coverage,
        target_ndcg_at_k=ndcg,
        relevant_hit_count=1 if rank is not None else 0,
        precision_at_k=0.2 if rank is not None else 0.0,
        matched_rank=rank,
        reciprocal_rank=reciprocal_rank,
        target_matches={"page:1": rank is not None},
        target_matched_ranks=target_ranks,
        target_key_matched_ranks=target_ranks,
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
