import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.diagnostics import analyze_retrieval_evaluation
from chunking_docs.evaluation.retrieval import RetrievalCase, evaluate_search_results
from chunking_docs.models import ChunkKind, DocumentChunk


class Hit:
    def __init__(self, chunk):
        self.chunk = chunk
        self.sources = ["test"]
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
        RetrievalCase(query="partial", expected_pages=[1, 2]),
        RetrievalCase(query="empty", expected_pages=[3]),
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
    assert report.reason_counts["partial_target_coverage"] == 1
    assert report.reason_counts["no_hits"] == 1
    assert report.reason_counts["missing_page"] == 2
    assert rows["partial"].missing_targets == ["page:2"]
    assert rows["partial"].precision_at_k == 0.25
    assert rows["empty"].reasons == ["no_hits", "no_expected_target_retrieved", "missing_page"]


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
    assert payload["rows"][0]["query"] == "empty"
