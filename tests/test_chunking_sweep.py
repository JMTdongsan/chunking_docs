import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.evaluation.sweep import dominates, run_chunking_sweep
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


def test_sweep_pareto_dominance_accounts_for_quality_and_cost():
    stronger = {
        "retrieval_recall_at_k": 0.9,
        "target_coverage_at_k": 0.8,
        "target_ndcg_at_k": 0.7,
        "precision_at_k": 0.6,
        "quality_score": 0.85,
        "visual_text_coverage_ratio": 1.0,
        "mean_latency_ms": 10.0,
        "chunk_count": 20.0,
    }
    weaker = {
        "retrieval_recall_at_k": 0.8,
        "target_coverage_at_k": 0.7,
        "target_ndcg_at_k": 0.6,
        "precision_at_k": 0.5,
        "quality_score": 0.8,
        "visual_text_coverage_ratio": 1.0,
        "mean_latency_ms": 12.0,
        "chunk_count": 25.0,
    }
    tradeoff = {**stronger, "target_coverage_at_k": 0.82, "mean_latency_ms": 14.0}

    assert dominates(stronger, weaker)
    assert not dominates(weaker, stronger)
    assert not dominates(stronger, tradeoff)


def test_sweep_chunking_cli_writes_report(tmp_path):
    package_dir = write_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "sweep_report.json"
    candidates_dir = tmp_path / "candidates"
    write_jsonl(cases_path, [RetrievalCase(query="capital investment table", expected_pages=[1])])

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
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparison"]["best_by_retrieval"] is not None
    assert payload["selection"]["recommended"] is not None
    assert payload["selection"]["ranking"][0]["score"] >= payload["selection"]["ranking"][-1]["score"]
    assert len(payload["candidates"]) == 2
    assert any(candidates_dir.glob("chunks.semantic-*.jsonl"))


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
