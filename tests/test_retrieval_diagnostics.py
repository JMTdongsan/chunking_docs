import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.diagnostics import analyze_retrieval_evaluation
from chunking_docs.evaluation.retrieval import RetrievalCase, evaluate_search_results
from chunking_docs.models import ChunkKind, DocumentChunk


class Hit:
    def __init__(self, chunk, sources=None):
        self.chunk = chunk
        self.sources = sources or ["test"]
        self.evidence_chunks = []
        self.payloads = []


def test_analyze_retrieval_evaluation_reports_failure_reasons():
    chunk_a = DocumentChunk(
        chunk_id="a",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="alpha",
    )
    chunk_b = DocumentChunk(
        chunk_id="b",
        doc_id="doc",
        page_start=9,
        page_end=9,
        kind=ChunkKind.TEXT,
        text="beta",
    )
    cases = [
        RetrievalCase(
            query="partial",
            expected_pages=[1, 2],
            metadata={"case_source": "visual_object_probe", "query_mode": "salient_terms"},
        ),
        RetrievalCase(
            query="empty",
            expected_pages=[3],
            metadata={"case_source": "page_probe"},
        ),
    ]

    evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [Hit(chunk_a), Hit(chunk_b)]
        if case.query == "partial"
        else [],
        top_k=4,
    )
    report = analyze_retrieval_evaluation(evaluation, precision_floor=0.5)

    rows = {row.query: row for row in report.rows}
    assert report.failed_count == 1
    assert report.partial_count == 1
    assert report.no_hit_count == 1
    assert report.low_precision_count == 1
    assert report.low_target_ndcg_count == 1
    assert report.reason_counts["partial_target_coverage"] == 1
    assert report.reason_counts["low_target_ndcg_at_k"] == 1
    assert report.reason_counts["no_hits"] == 1
    assert report.reason_counts["missing_page"] == 2
    assert report.reason_counts_by_case_group["case_source"]["visual_object_probe"] == {
        "low_precision_at_k": 1,
        "low_target_ndcg_at_k": 1,
        "missing_page": 1,
        "partial_target_coverage": 1,
    }
    assert report.source_counts == {"test": 2}
    assert report.source_family_counts == {"test": 2}
    assert report.source_counts_by_case_group["case_source"]["visual_object_probe"] == {
        "test": 2
    }
    assert report.source_family_counts_by_case_group["case_source"]["visual_object_probe"] == {
        "test": 2
    }
    assert report.matched_source_counts == {"test": 1}
    assert report.matched_source_family_counts == {"test": 1}
    assert report.matched_source_counts_by_case_group["case_source"][
        "visual_object_probe"
    ] == {"test": 1}
    assert report.matched_source_family_counts_by_case_group["case_source"][
        "visual_object_probe"
    ] == {"test": 1}
    assert (
        report.missing_target_type_counts_by_case_group["case_source"][
            "visual_object_probe"
        ]["page"]
        == 1
    )
    assert rows["partial"].case_groups == {
        "case_source": "visual_object_probe",
        "query_mode": "salient_terms",
    }
    assert rows["partial"].missing_targets == ["page:2"]
    assert rows["partial"].source_counts == {"test": 2}
    assert rows["partial"].source_family_counts == {"test": 2}
    assert rows["partial"].matched_source_counts == {"test": 1}
    assert rows["partial"].matched_source_family_counts == {"test": 1}
    assert rows["partial"].top_source_families == [["test"], ["test"]]
    assert rows["partial"].target_ndcg_at_k == 0.5
    assert rows["partial"].precision_at_k == 0.25
    assert rows["empty"].reasons == ["no_hits", "no_expected_target_retrieved", "missing_page"]


def test_analyze_retrieval_evaluation_reports_excluded_target_hits():
    chunk = DocumentChunk(
        chunk_id="a",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="alpha",
        asset_ids=["asset-a"],
    )
    evaluation = evaluate_search_results(
        cases=[
            RetrievalCase(
                query="hard negative",
                expected_pages=[1],
                excluded_asset_ids=["asset-a"],
                metadata={"case_source": "hard_negative"},
            )
        ],
        search_fn=lambda case, graph_expand: [
            Hit(chunk, sources=["bm25", "qdrant:image_dense"])
        ],
        top_k=1,
    )

    report = analyze_retrieval_evaluation(evaluation)

    assert report.failed_count == 1
    assert report.reason_counts["excluded_target_retrieved"] == 1
    assert report.reason_counts["excluded_asset_hit"] == 1
    assert report.excluded_source_counts == {"bm25": 1, "qdrant:image_dense": 1}
    assert report.excluded_source_family_counts == {"lexical": 1, "visual": 1}
    assert report.source_counts == {"bm25": 1, "qdrant:image_dense": 1}
    assert report.source_family_counts == {"lexical": 1, "visual": 1}
    assert report.matched_source_counts == {"bm25": 1, "qdrant:image_dense": 1}
    assert report.matched_source_family_counts == {"lexical": 1, "visual": 1}
    assert report.source_counts_by_case_group["case_source"]["hard_negative"] == {
        "bm25": 1,
        "qdrant:image_dense": 1,
    }
    assert report.source_family_counts_by_case_group["case_source"]["hard_negative"] == {
        "lexical": 1,
        "visual": 1,
    }
    assert report.matched_source_counts_by_case_group["case_source"]["hard_negative"] == {
        "bm25": 1,
        "qdrant:image_dense": 1,
    }
    assert report.matched_source_family_counts_by_case_group["case_source"][
        "hard_negative"
    ] == {"lexical": 1, "visual": 1}
    assert report.excluded_source_counts_by_case_group["case_source"]["hard_negative"] == {
        "bm25": 1,
        "qdrant:image_dense": 1,
    }
    assert report.excluded_source_family_counts_by_case_group["case_source"][
        "hard_negative"
    ] == {"lexical": 1, "visual": 1}
    row = report.rows[0]
    assert row.expected_targets == ["page:1"]
    assert row.matched_targets == ["page:1"]
    assert row.excluded_targets == ["asset:asset-a"]
    assert row.matched_excluded_targets == ["asset:asset-a"]
    assert row.excluded_source_counts == {"bm25": 1, "qdrant:image_dense": 1}
    assert row.excluded_source_family_counts == {"lexical": 1, "visual": 1}
    assert row.matched_source_counts == {"bm25": 1, "qdrant:image_dense": 1}
    assert row.matched_source_family_counts == {"lexical": 1, "visual": 1}
    assert row.top_excluded_sources == [["bm25", "qdrant:image_dense"]]
    assert row.top_source_families == [["lexical", "visual"]]
    assert row.reasons == ["excluded_target_retrieved", "excluded_asset_hit"]


def test_diagnose_retrieval_cli_writes_report(tmp_path):
    evaluation_path = tmp_path / "eval.json"
    output_path = tmp_path / "diagnostics.json"
    evaluation = evaluate_search_results(
        cases=[RetrievalCase(query="empty", expected_pages=[3])],
        search_fn=lambda case, graph_expand: [],
        top_k=5,
    )
    evaluation_path.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "diagnose-retrieval",
            str(evaluation_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["no_hit_count"] == 1
    assert payload["low_target_ndcg_count"] == 0
    assert payload["excluded_source_counts"] == {}
    assert payload["excluded_source_family_counts"] == {}
    assert payload["source_counts"] == {}
    assert payload["source_family_counts"] == {}
    assert payload["matched_source_counts"] == {}
    assert payload["matched_source_family_counts"] == {}
    assert payload["source_counts_by_case_group"] == {}
    assert payload["source_family_counts_by_case_group"] == {}
    assert payload["matched_source_counts_by_case_group"] == {}
    assert payload["matched_source_family_counts_by_case_group"] == {}
    assert payload["rows"][0]["query"] == "empty"
    assert payload["reason_counts_by_case_group"] == {}
