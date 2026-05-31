import json

from typer.testing import CliRunner

import chunking_docs.cli as cli_module
from chunking_docs.cli import app
from chunking_docs.evaluation.audit import audit_package, degraded_page_ratio
from chunking_docs.evaluation.ablation import evaluate_retrieval_ablation, parse_ablation_modes
from chunking_docs.evaluation.retrieval import RetrievalCase, evaluate_retrieval, evaluate_search_results
from chunking_docs.io import write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    TextQuality,
    VisualAsset,
)
from chunking_docs.retrieval.local_hybrid import HybridSearchHit


def test_audit_package_detects_missing_vlm_annotations():
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=1,
            height=1,
            char_count=0,
            line_count=0,
            text_block_count=0,
            image_block_count=1,
            embedded_image_count=1,
            drawing_count=0,
            text_quality=TextQuality.EMPTY,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.PAGE_SUMMARY,
            text="",
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            metadata={"requires_vlm": True},
        )
    ]

    audit = audit_package(profiles, chunks, assets, [], require_annotations_for_visual_pages=True)

    assert not audit.passed
    assert audit.pages_requiring_vlm == [1]
    assert degraded_page_ratio(profiles) == 1.0


def test_evaluate_retrieval_hit_rate():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=12,
            page_end=12,
            kind=ChunkKind.TEXT,
            text="north district river corridor",
        )
    ]
    triples = [
        GraphTriple(
            triple_id="t",
            doc_id="doc",
            chunk_id="a",
            subject="north district",
            predicate="uses_axis",
            object="river corridor",
        )
    ]
    cases = [RetrievalCase(query="north district", expected_pages=[12], graph_expand=True)]

    result = evaluate_retrieval(chunks, triples, cases, top_k=3, repeat=2)

    assert result.hit_rate == 1.0
    assert result.recall_at_k == 1.0
    assert result.mrr == 1.0
    assert result.repeat == 2
    assert result.mean_latency_ms >= 0.0
    assert result.p95_latency_ms >= 0.0
    assert result.results[0].passed
    assert len(result.results[0].latency_samples_ms) == 2
    assert result.results[0].matched_rank == 1
    assert result.results[0].matched_page == 12


def test_evaluate_retrieval_reports_ranked_failures():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="alpha beta",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=3,
            page_end=4,
            kind=ChunkKind.TEXT,
            text="gamma delta",
        ),
    ]
    cases = [
        RetrievalCase(query="alpha", expected_pages=[2]),
        RetrievalCase(query="missing", expected_pages=[9]),
    ]

    result = evaluate_retrieval(chunks, [], cases, top_k=2)

    assert result.expected_case_count == 2
    assert result.passed_count == 1
    assert result.failed_count == 1
    assert result.recall_at_k == 0.5
    assert result.mrr == 0.5
    assert result.failed_queries == ["missing"]
    assert result.results[0].top_page_ranges == [(1, 2)]


def test_evaluate_retrieval_matches_collapsed_hierarchical_evidence_chunk():
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.PAGE_SUMMARY,
        text="summary",
    )
    child = DocumentChunk(
        chunk_id="child",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access benchmark evidence",
        metadata={"hierarchical_parent_chunk_id": "parent"},
    )
    cases = [RetrievalCase(query="station access", expected_chunk_ids=["child"])]

    result = evaluate_retrieval([parent, child], [], cases, collapse_hierarchical=True)

    assert result.recall_at_k == 1.0
    assert result.results[0].top_chunk_ids == ["parent"]
    assert result.results[0].top_evidence_chunk_ids == [["child"]]
    assert result.results[0].matched_chunk_id == "child"


def test_evaluate_retrieval_matches_visual_asset_id_from_evidence_chunk():
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.PAGE_SUMMARY,
        text="summary",
    )
    child = DocumentChunk(
        chunk_id="child",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access map benchmark evidence",
        asset_ids=["asset-map"],
        metadata={"hierarchical_parent_chunk_id": "parent"},
    )
    cases = [RetrievalCase(query="station access map", expected_asset_ids=["asset-map"])]

    result = evaluate_retrieval([parent, child], [], cases, collapse_hierarchical=True)

    assert result.expected_case_count == 1
    assert result.recall_at_k == 1.0
    assert result.results[0].top_chunk_ids == ["parent"]
    assert result.results[0].top_asset_ids == [["asset-map"]]
    assert result.results[0].matched_asset_id == "asset-map"


def test_evaluate_retrieval_matches_expected_triple_id():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=4,
        page_end=4,
        kind=ChunkKind.TEXT,
        text="unrelated text",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="north district",
        predicate="connects_to",
        object="river corridor",
    )
    cases = [RetrievalCase(query="north district", expected_triple_ids=["triple-1"])]

    result = evaluate_retrieval(
        [chunk],
        [triple],
        cases,
        use_dense=False,
        use_bm25=False,
        use_graph=True,
    )

    assert result.recall_at_k == 1.0
    assert result.results[0].top_triple_ids == [["triple-1"]]
    assert result.results[0].matched_triple_id == "triple-1"
    assert result.results[0].top_sources == [["graph"]]


def test_evaluate_search_results_matches_visual_asset_id_from_payload():
    chunk = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access",
    )

    class PayloadHit:
        def __init__(self):
            self.chunk = chunk
            self.sources = ["qdrant:caption_dense"]
            self.evidence_chunks = []
            self.payloads = [{"asset_id": "asset-map"}]

    result = evaluate_search_results(
        cases=[RetrievalCase(query="station access map", expected_asset_ids=["asset-map"])],
        search_fn=lambda case, graph_expand: [PayloadHit()],
    )

    assert result.recall_at_k == 1.0
    assert result.results[0].top_asset_ids == [["asset-map"]]
    assert result.results[0].matched_asset_id == "asset-map"


def test_evaluate_retrieval_ablation_compares_modes():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="capital budget transit corridor",
        )
    ]
    cases = [RetrievalCase(query="capital budget", expected_pages=[1])]

    report = evaluate_retrieval_ablation(
        chunks,
        [],
        cases,
        modes=parse_ablation_modes("dense,bm25,hybrid"),
        repeat=2,
    )

    assert [row.mode.name for row in report.rows]
    assert report.best_by_recall in {"dense", "bm25", "hybrid"}
    assert report.fastest_by_mean_latency in {"dense", "bm25", "hybrid"}
    assert all(row.evaluation.case_count == 1 for row in report.rows)
    assert all(row.evaluation.repeat == 2 for row in report.rows)


def test_eval_retrieval_cli_writes_latency_report(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    output = tmp_path / "retrieval.json"
    cases_path = tmp_path / "cases.jsonl"
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="capital budget transit corridor",
        )
    ]
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "triples.jsonl", [])
    write_jsonl(cases_path, [RetrievalCase(query="capital budget", expected_pages=[1])])

    result = CliRunner().invoke(
        app,
        [
            "eval-retrieval",
            str(cases_path),
            "--package-dir",
            str(package_dir),
            "--repeat",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["repeat"] == 2
    assert payload["mean_latency_ms"] >= 0.0
    assert len(payload["results"][0]["latency_samples_ms"]) == 2


def test_eval_qdrant_retrieval_cli_writes_report(monkeypatch, tmp_path):
    output = tmp_path / "qdrant_retrieval.json"
    cases_path = tmp_path / "cases.jsonl"
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="capital budget transit corridor",
        asset_ids=["asset-1"],
    )
    write_jsonl(cases_path, [RetrievalCase(query="capital budget", expected_asset_ids=["asset-1"])])

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            assert kwargs["vector_names"] == ["text_dense"]
            assert kwargs["top_k"] == 5
            return [HybridSearchHit(chunk=chunk, score=0.8, sources=["qdrant:text_dense"])]

    def fake_prepare(**kwargs):
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": "documents",
            "selected_vectors": ["text_dense"],
            "query_encoders": {"text_dense": "default_text"},
            "upserted": 1,
            "triples": [],
        }

    monkeypatch.setattr(cli_module, "prepare_qdrant_hybrid_search", fake_prepare)

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-retrieval",
            str(cases_path),
            "--package-dir",
            str(tmp_path),
            "--vector-names",
            "text_dense",
            "--repeat",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["backend"] == "qdrant_hybrid"
    assert payload["metadata"]["collection"] == "documents"
    assert payload["repeat"] == 2
    assert payload["recall_at_k"] == 1.0
    assert payload["results"][0]["matched_asset_id"] == "asset-1"
    assert payload["results"][0]["top_sources"] == [["qdrant:text_dense"]]
    assert len(payload["results"][0]["latency_samples_ms"]) == 2


def test_eval_retrieval_ablation_cli_writes_report(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    output = tmp_path / "ablation.json"
    cases_path = tmp_path / "cases.jsonl"
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="capital budget transit corridor",
        )
    ]
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "triples.jsonl", [])
    write_jsonl(cases_path, [RetrievalCase(query="capital budget", expected_pages=[1])])

    result = CliRunner().invoke(
        app,
        [
            "eval-retrieval-ablation",
            str(cases_path),
            "--package-dir",
            str(package_dir),
            "--modes",
            "bm25,hybrid",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["best_by_recall"] in {"bm25", "hybrid"}
    assert payload["fastest_by_mean_latency"] in {"bm25", "hybrid"}
    assert {row["mode"]["name"] for row in payload["rows"]} == {"bm25", "hybrid"}
