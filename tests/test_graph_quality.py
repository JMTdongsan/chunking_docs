import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.graph.quality import (
    audit_graph_triples,
    normalize_graph_triple,
    normalize_graph_triples,
)
from chunking_docs.io import read_jsonl, write_jsonl
from chunking_docs.models import ChunkKind, DocumentChunk, GraphTriple


def test_normalize_graph_triple_canonicalizes_text_and_id():
    triple = GraphTriple(
        triple_id="legacy",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="  Transit   Hub  ",
        predicate="Related To",
        object='"Blue Line"',
    )

    normalized = normalize_graph_triple(triple)

    assert normalized.subject == "Transit Hub"
    assert normalized.predicate == "related_to"
    assert normalized.object == "Blue Line"
    assert normalized.triple_id != "legacy"
    assert normalized.qualifiers["original_triple_id"] == "legacy"
    assert normalized.qualifiers["normalized"] is True


def test_normalize_graph_triples_dedupes_semantic_duplicates():
    triples = [
        GraphTriple(
            triple_id="a",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="Transit Hub",
            predicate="related to",
            object="Blue Line",
            confidence=0.2,
        ),
        GraphTriple(
            triple_id="b",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="transit hub",
            predicate="related_to",
            object="blue line",
            confidence=0.9,
        ),
    ]

    normalized = normalize_graph_triples(triples)

    assert len(normalized) == 1
    assert normalized[0].confidence == 0.9
    assert normalized[0].qualifiers["deduped_duplicate_count"] == 1


def test_normalize_graph_triples_drops_empty_normalized_fields():
    triples = [
        GraphTriple(
            triple_id="invalid",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="Index",
            predicate="=",
            object="value expression",
        ),
        GraphTriple(
            triple_id="valid",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="Index",
            predicate="defined_as",
            object="value expression",
        ),
    ]

    normalized = normalize_graph_triples(triples)

    assert len(normalized) == 1
    assert normalized[0].triple_id != "invalid"


def test_normalize_graph_triples_drops_low_information_predicates():
    triples = [
        GraphTriple(
            triple_id="low-info",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="Policy",
            predicate="is",
            object="important",
        ),
        GraphTriple(
            triple_id="valid",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="Policy",
            predicate="supports",
            object="project",
        ),
    ]

    normalized = normalize_graph_triples(triples)

    assert len(normalized) == 1
    assert normalized[0].predicate == "supports"


def test_audit_graph_triples_reports_quality_issues():
    chunk = make_chunk("chunk-1")
    triples = [
        GraphTriple(
            triple_id="a",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="Transit Hub",
            predicate="related to",
            object="Blue Line",
        ),
        GraphTriple(
            triple_id="b",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="transit hub",
            predicate="related_to",
            object="blue line",
        ),
        GraphTriple(
            triple_id="c",
            doc_id="doc",
            chunk_id="missing",
            subject="",
            predicate="---",
            object="Blue Line",
            confidence=1.2,
        ),
        GraphTriple(
            triple_id="d",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="Policy",
            predicate="is",
            object="important",
        ),
    ]

    report = audit_graph_triples(triples, chunks=[chunk])
    codes = {issue.code for issue in report.issues}

    assert not report.passed
    assert report.duplicate_count == 1
    assert report.empty_field_count == 1
    assert report.orphan_count == 1
    assert report.invalid_confidence_count == 1
    assert "duplicate_graph_triple" in codes
    assert "empty_triple_field" in codes
    assert "orphan_triple_chunk" in codes
    assert "invalid_triple_confidence" in codes
    assert "low_information_predicate" in codes


def test_graph_quality_cli_writes_normalized_triples_and_report(tmp_path):
    package_dir = tmp_path / "package"
    chunk = make_chunk("chunk-1")
    write_jsonl(package_dir / "chunks.jsonl", [chunk])
    write_jsonl(
        package_dir / "triples.jsonl",
        [
            GraphTriple(
                triple_id="legacy",
                doc_id="doc",
                chunk_id="chunk-1",
                subject="  Transit Hub ",
                predicate="Related To",
                object="Blue Line",
            ),
            GraphTriple(
                triple_id="duplicate",
                doc_id="doc",
                chunk_id="chunk-1",
                subject="Transit Hub",
                predicate="related_to",
                object="Blue Line",
            ),
            GraphTriple(
                triple_id="invalid",
                doc_id="doc",
                chunk_id="chunk-1",
                subject="Index",
                predicate="=",
                object="value expression",
            )
        ],
    )

    output = package_dir / "triples.normalized.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "normalize-graph-triples",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--export-graph",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["removed_invalid"] == 1
    assert payload["removed_duplicates"] == 1
    normalized = read_jsonl(output, GraphTriple)
    assert len(normalized) == 1
    assert normalized[0].predicate == "related_to"
    assert (package_dir / "graph_nodes.jsonl").exists()
    assert (package_dir / "graph_edges.jsonl").exists()

    report_output = package_dir / "graph_quality.json"
    result = CliRunner().invoke(
        app,
        [
            "audit-graph-triples",
            "--package-dir",
            str(package_dir),
            "--output",
            str(report_output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_output.read_text(encoding="utf-8"))
    assert payload["triple_count"] == 3
    assert payload["empty_field_count"] == 1
    assert payload["duplicate_count"] == 1


def make_chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit Hub Blue Line",
    )
