import json

import pytest
from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.context_quality import (
    evaluate_rag_contexts,
    gate_rag_context_evaluation,
)
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.retrieval.context import (
    RAGContextAsset,
    RAGContextBundle,
    RAGContextChunk,
    RAGContextTriple,
)


def test_evaluate_rag_contexts_scores_final_context_targets():
    case = RetrievalCase(
        query="station access map",
        expected_pages=[4],
        expected_chunk_ids=["source-chunk"],
        expected_asset_ids=["asset-2"],
        expected_triple_ids=["triple-2"],
        excluded_asset_ids=["asset-x"],
        metadata={"case_source": "visual_object_probe"},
    )
    bundle = RAGContextBundle(
        query="station access map",
        chunks=[
            RAGContextChunk(
                chunk_id="chunk-1",
                doc_id="doc",
                page_start=4,
                page_end=4,
                kind="text",
                text="station access evidence",
                metadata={
                    "source_chunk_id": "source-chunk",
                    "retrieved_asset_ids": ["asset-2"],
                    "retrieved_triple_ids": ["triple-2"],
                },
            )
        ],
        assets=[
            RAGContextAsset(
                asset_id="asset-2",
                page_no=4,
                kind="map",
                caption="station access map",
            )
        ],
        triples=[
            RAGContextTriple(
                triple_id="triple-2",
                chunk_id="chunk-1",
                subject="station",
                predicate="connects_to",
                object="corridor",
            )
        ],
    )

    evaluation = evaluate_rag_contexts([case], [bundle], latencies_ms=[12.0])

    assert evaluation.passed_count == 1
    assert evaluation.target_coverage == 1.0
    assert evaluation.excluded_target_hit_rate == 0.0
    assert evaluation.mean_latency_ms == 12.0
    assert evaluation.target_metrics["asset"].coverage == 1.0
    assert evaluation.target_metrics["triple"].coverage == 1.0
    assert (
        evaluation.case_group_metrics["case_source"]["visual_object_probe"].target_coverage
        == 1.0
    )


def test_evaluate_rag_contexts_fails_on_missing_or_excluded_targets():
    case = RetrievalCase(
        query="wrong visual evidence",
        expected_asset_ids=["asset-needed"],
        excluded_pages=[9],
    )
    bundle = RAGContextBundle(
        query="wrong visual evidence",
        chunks=[
            RAGContextChunk(
                chunk_id="chunk-1",
                doc_id="doc",
                page_start=9,
                page_end=9,
                kind="text",
                text="wrong page evidence",
            )
        ],
    )

    evaluation = evaluate_rag_contexts([case], [bundle])

    assert evaluation.passed_count == 0
    assert evaluation.failed_queries == ["wrong visual evidence"]
    assert evaluation.target_coverage == 0.0
    assert evaluation.excluded_target_hit_rate == 1.0
    assert evaluation.results[0].target_key_matches == {"asset:asset-needed": False}
    assert evaluation.results[0].excluded_target_key_matches == {"page:9": True}


def test_evaluate_rag_contexts_requires_aligned_lengths():
    case = RetrievalCase(query="query", expected_pages=[1])

    with pytest.raises(ValueError, match="bundles"):
        evaluate_rag_contexts([case], [])

    with pytest.raises(ValueError, match="latencies_ms"):
        evaluate_rag_contexts(
            [case],
            [
                RAGContextBundle(
                    query="query",
                    chunks=[],
                )
            ],
            latencies_ms=[],
        )


def test_gate_rag_context_evaluation_checks_final_context_metrics():
    case = RetrievalCase(
        query="visual object",
        expected_asset_ids=["asset-1"],
        excluded_asset_ids=["asset-x"],
        metadata={"case_source": "visual_object_probe"},
    )
    bundle = RAGContextBundle(
        query="visual object",
        chunks=[
            RAGContextChunk(
                chunk_id="chunk-1",
                doc_id="doc",
                page_start=1,
                page_end=1,
                kind="text",
                text="visual object context",
                metadata={"retrieved_asset_ids": ["asset-1"]},
            )
        ],
        assets=[
            RAGContextAsset(
                asset_id="asset-1",
                page_no=1,
                kind="map",
            )
        ],
    )
    evaluation = evaluate_rag_contexts([case], [bundle], latencies_ms=[10.0])

    report = gate_rag_context_evaluation(
        evaluation,
        min_case_count=1,
        min_expected_target_count=1,
        min_target_coverage=1.0,
        max_excluded_target_hit_rate=0.0,
        max_mean_context_char_count=100,
        min_target_type_coverage={"asset": 1.0},
        min_case_group_target_coverage={"case_source:visual_object_probe": 1.0},
    )

    assert report.passed is True
    assert report.metrics["target_type.asset.coverage"] == 1.0
    assert (
        report.metrics[
            "case_group.case_source.visual_object_probe.target_coverage"
        ]
        == 1.0
    )

    failed = gate_rag_context_evaluation(
        evaluation,
        min_target_coverage=1.0,
        max_mean_context_char_count=5,
        min_target_type_coverage={"triple": 1.0},
    )

    assert failed.passed is False
    assert "max_mean_context_char_count" in failed.failed_checks
    assert "min_target_type_coverage:triple" in failed.failed_checks


def test_gate_rag_context_cli_exits_nonzero_on_failed_gate(tmp_path):
    case = RetrievalCase(query="missing visual", expected_asset_ids=["asset-needed"])
    bundle = RAGContextBundle(
        query="missing visual",
        chunks=[],
    )
    evaluation = evaluate_rag_contexts([case], [bundle])
    evaluation_path = tmp_path / "context_eval.json"
    output_path = tmp_path / "context_gate.json"
    evaluation_path.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "gate-rag-context",
            str(evaluation_path),
            "--min-target-coverage",
            "1.0",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    assert "min_target_coverage" in result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "min_target_coverage" in payload["failed_checks"]


def test_gate_rag_context_cli_can_report_without_failing(tmp_path):
    case = RetrievalCase(
        query="wrong page",
        expected_pages=[1],
        excluded_pages=[2],
        metadata={"case_source": "visual_image_probe"},
    )
    bundle = RAGContextBundle(
        query="wrong page",
        chunks=[
            RAGContextChunk(
                chunk_id="chunk-1",
                doc_id="doc",
                page_start=2,
                page_end=2,
                kind="text",
                text="wrong page context",
            )
        ],
    )
    evaluation = evaluate_rag_contexts([case], [bundle])
    evaluation_path = tmp_path / "context_eval.json"
    evaluation_path.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "gate-rag-context",
            str(evaluation_path),
            "--max-excluded-target-hit-rate",
            "0",
            "--max-case-group-excluded-target-hit-rate",
            "case_source:visual_image_probe=0",
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "max_excluded_target_hit_rate" in result.output
    assert "max_case_group_excluded_target_hit_rate" in result.output
