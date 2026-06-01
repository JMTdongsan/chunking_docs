import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.embeddings.bm25 import BM25LexicalIndex, chunk_lexical_texts
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.ablation import (
    AblationPairwiseComparison,
    QdrantRerankerAblationMode,
    QdrantRerankerAblationRow,
    build_qdrant_reranker_ablation_report,
    QdrantVectorAblationMode,
    QdrantVectorAblationReport,
    QdrantVectorAblationRow,
    RetrievalAblationMode,
    RetrievalAblationReport,
    RetrievalAblationRow,
)
from chunking_docs.evaluation.compare import (
    ChunkingComparison,
    ChunkingComparisonRow,
    ChunkingPairwiseComparison,
)
from chunking_docs.evaluation.context_quality import (
    RAGContextCaseGroupMetric,
    RAGContextEvaluation,
    RAGContextTargetMetric,
)
from chunking_docs.evaluation.readiness import build_ingestion_readiness_report, chunks_with_linked_asset_text
from chunking_docs.evaluation.retrieval import (
    RetrievalCase,
    RetrievalEvaluation,
    evaluate_search_results,
)
from chunking_docs.evaluation.retrieval_config import (
    QdrantRetrievalConfig,
    QdrantRetrievalConfigSelection,
)
from chunking_docs.io import read_jsonl, write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    TextQuality,
    VisualAsset,
)
from chunking_docs.retrieval.local_hybrid import HybridSearchHit
from chunking_docs.runtime import GPUDevice, RuntimeCheck, RuntimeReport
from chunking_docs.storage.records import EmbeddingRecord
from chunking_docs.vision.compare import compare_visual_runs
from chunking_docs.vision.jobs import VisualJobRunResult
from chunking_docs.vision.manual_annotations import AssetAnnotation


def test_ingestion_readiness_passes_ready_package(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(package_dir, manifest)

    assert report.passed is True
    assert report.package_counts == {"pages": 1, "chunks": 1, "assets": 1, "triples": 0}
    assert report.artifact_presence["bm25_tokens.json"] is True
    assert report.postgres_row_counts["embedding_artifacts"] == 1
    assert report.postgres_row_counts["embedding_records"] == 1
    bm25_component = next(component for component in report.components if component.name == "bm25_tokens")
    assert bm25_component.metadata["chunks_with_linked_asset_text"] == 1
    assert bm25_component.metadata["indexed_linked_asset_text_chunk_count"] == 1
    reproducibility_component = next(
        component for component in report.components if component.name == "package_reproducibility"
    )
    assert reproducibility_component.metadata["failed_checks"] == []
    assert reproducibility_component.metadata["bm25_tokenizer"]["matches_package_config"] is True
    assert report.failed_components == []


def test_ingestion_readiness_includes_runtime_report(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        runtime_report=ready_runtime_report(),
        require_runtime_report=True,
    )

    assert report.passed is True
    assert report.runtime_report is not None
    component = next(component for component in report.components if component.name == "runtime_report")
    assert component.passed is True
    assert component.metadata["gpu_count"] == 1
    assert component.metadata["torch_cuda_compute_capabilities"] == ["12.0"]
    assert component.metadata["torch_cuda_compiled_arches"] == ["sm_120"]
    assert component.metadata["torch_bfloat16_supported"] is True
    assert component.metadata["failed_checks"] == []


def test_ingestion_readiness_requires_runtime_report(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_runtime_report=True,
    )

    assert report.passed is False
    assert "runtime_report" in report.failed_components


def test_ingestion_readiness_flags_missing_reproducibility_metadata(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.metadata = {}

    report = build_ingestion_readiness_report(package_dir, manifest)

    component = next(
        component for component in report.components if component.name == "package_reproducibility"
    )
    assert report.passed is False
    assert "package_reproducibility" in report.failed_components
    assert component.metadata["failed_checks"] == [
        "missing_package_config",
        "missing_source_file",
    ]


def test_ingestion_readiness_flags_source_checksum_mismatch(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.metadata["source_file"]["sha256"] = "0" * 64

    report = build_ingestion_readiness_report(package_dir, manifest)

    component = next(
        component for component in report.components if component.name == "package_reproducibility"
    )
    assert report.passed is False
    assert "source_file_sha256_mismatch" in component.metadata["failed_checks"]


def test_ingestion_readiness_flags_package_bm25_tokenizer_mismatch(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.metadata["package_config"]["lexical_tokenizer"]["strategy"] = "word"

    report = build_ingestion_readiness_report(package_dir, manifest)

    component = next(
        component for component in report.components if component.name == "package_reproducibility"
    )
    assert report.passed is False
    assert "bm25_tokenizer_mismatch" in component.metadata["failed_checks"]
    assert component.metadata["bm25_tokenizer"]["matches_package_config"] is False


def test_ingestion_readiness_can_require_visual_derived_triples(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.assets[0] = manifest.assets[0].model_copy(
        update={
            "metadata": {
                "requires_ocr": False,
                "requires_vlm": False,
                "title": "map panel",
                "objects": [{"label": "route line"}],
            }
        }
    )
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_bm25=False,
        require_visual_derived_triples=True,
    )

    component = next(component for component in report.components if component.name == "package_audit")
    assert report.passed is False
    assert "package_audit" in report.failed_components
    assert "missing_visual_derived_triples" in component.metadata["issue_codes"]


def test_ingestion_readiness_cli_can_require_visual_derived_triples(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.assets[0] = manifest.assets[0].model_copy(
        update={
            "metadata": {
                "requires_ocr": False,
                "requires_vlm": False,
                "title": "map panel",
                "objects": [{"label": "route line"}],
            }
        }
    )
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)
    write_bm25_manifest(package_dir, manifest.chunks, manifest.assets)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--require-visual-derived-triples",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "package_audit" in payload["failed_components"]
    component = next(
        component for component in payload["components"] if component["name"] == "package_audit"
    )
    assert "missing_visual_derived_triples" in component["metadata"]["issue_codes"]


def test_chunks_with_linked_asset_text_counts_source_refs():
    manifest = ProcessingManifest(
        doc=SourceDocument(doc_id="doc", title="title", local_path=Path("/tmp/doc.pdf")),
        chunks=[
            DocumentChunk(
                chunk_id="chunk-1",
                doc_id="doc",
                page_start=1,
                page_end=1,
                kind=ChunkKind.TEXT,
                text="visual context",
                source_refs=["asset:asset-1"],
            )
        ],
        assets=[
            VisualAsset(
                asset_id="asset-1",
                doc_id="doc",
                page_no=1,
                kind=AssetKind.MAP,
                caption="visual caption",
            )
        ],
    )

    assert chunks_with_linked_asset_text(manifest) == ["chunk-1"]


def test_ingestion_readiness_can_require_embedding_vectors(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        required_vectors=["text_dense"],
    )

    component = next(component for component in report.components if component.name == "embedding_vectors")
    assert report.passed is True
    assert component.metadata["required_vectors"] == ["text_dense"]
    assert component.metadata["required_vector_details"]["text_dense"]["record_count"] == 1
    assert component.metadata["required_vector_details"]["text_dense"]["dimension"] == 2


def test_ingestion_readiness_flags_missing_required_embedding_vector(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        required_vectors=["text_dense", "caption_dense"],
    )

    component = next(component for component in report.components if component.name == "embedding_vectors")
    assert report.passed is False
    assert "embedding_vectors" in report.failed_components
    assert component.metadata["missing_collection_vectors"] == ["caption_dense"]


def test_ingestion_readiness_warns_when_image_vector_for_visual_asset_is_missing(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_bm25=False,
    )

    component = next(
        component for component in report.components if component.name == "derived_embedding_vectors"
    )
    assert report.passed is True
    assert component.passed is False
    assert component.severity == "warning"
    assert "image_dense" in component.metadata["expected_vectors"]
    assert "image_dense" in component.metadata["missing_collection_vectors"]
    assert component.metadata["expectations"]["image_dense"] == {
        "source": "visual_asset_images",
        "source_count": 1,
        "sample_source_ids": ["asset-1"],
        "record_file": "qdrant_image_records.jsonl",
        "reason": (
            "Rendered visual asset images should have image vectors for visual similarity retrieval."
        ),
    }
    assert "--image-backend clip" in component.metadata["rebuild_commands"][0]
    assert "--image-query-backend clip" in component.metadata["rebuild_commands"][-1]
    assert "--image-query-model openai/clip-vit-large-patch14" in (
        component.metadata["rebuild_commands"][-1]
    )
    assert "image" in component.metadata["recommended_qdrant_vector_modes"]
    assert "text_image" in component.metadata["recommended_qdrant_vector_modes"]
    assert "caption_image" in component.metadata["recommended_qdrant_vector_modes"]


def test_ingestion_readiness_warns_when_derived_triple_vector_is_missing(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="station",
        predicate="connects_to",
        object="corridor",
    )
    manifest.triples = [triple]
    write_jsonl(package_dir / "triples.jsonl", [triple])

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_bm25=False,
    )

    component = next(
        component for component in report.components if component.name == "derived_embedding_vectors"
    )
    assert report.passed is True
    assert component.passed is False
    assert component.severity == "warning"
    assert "triple_dense" in component.metadata["expected_vectors"]
    assert "triple_dense" in component.metadata["missing_collection_vectors"]
    assert component.metadata["missing_expected_vectors"] == [
        "caption_dense",
        "image_dense",
        "triple_dense",
    ]
    assert component.metadata["recommended_qdrant_vector_modes"] == [
        "text",
        "caption",
        "text_caption",
        "image",
        "text_image",
        "caption_image",
        "all",
        "triple",
        "text_triple",
        "all_with_triple",
        "text_caption_graph",
        "text_triple_graph",
        "all_graph",
        "all_with_triple_graph",
    ]
    assert component.metadata["rebuild_commands"] == [
        "chunking-docs normalize-graph-triples --package-dir outputs/package --export-graph",
        (
            "chunking-docs embed-package --package-dir outputs/package "
            "--caption-backend same-as-text --image-backend clip --triple-backend same-as-text"
        ),
        "chunking-docs audit-package --package-dir outputs/package --require-qdrant-records",
        (
            "chunking-docs eval-qdrant-vector-ablation examples/retrieval_cases.jsonl "
            "--package-dir outputs/package "
            "--modes text,caption,text_caption,image,text_image,caption_image,all,triple,"
            "text_triple,all_with_triple,text_caption_graph,text_triple_graph,all_graph,"
            "all_with_triple_graph --image-query-backend clip "
            "--image-query-model openai/clip-vit-large-patch14"
        ),
    ]
    assert component.metadata["expectations"]["triple_dense"]["source_count"] == 1


def test_ingestion_readiness_can_require_derived_object_vector_coverage(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.assets[0] = manifest.assets[0].model_copy(
        update={
            "caption": None,
            "metadata": {
                "requires_ocr": False,
                "requires_vlm": False,
                "objects": [
                    {
                        "label": "legend marker",
                        "attributes": ["red circle"],
                        "bbox_region": "lower right",
                    }
                ],
            },
        }
    )
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_bm25=False,
        require_derived_vector_coverage=True,
    )

    component = next(
        component for component in report.components if component.name == "derived_embedding_vectors"
    )
    assert report.passed is False
    assert "derived_embedding_vectors" in report.failed_components
    assert component.severity == "error"
    assert "object_dense" in component.metadata["expected_vectors"]
    assert "object_dense" in component.metadata["missing_collection_vectors"]
    assert component.metadata["missing_expected_vectors"] == [
        "caption_dense",
        "image_dense",
        "object_dense",
    ]
    assert component.metadata["recommended_qdrant_vector_modes"] == [
        "text",
        "caption",
        "text_caption",
        "object",
        "text_object",
        "caption_object",
        "image",
        "text_image",
        "caption_image",
        "all",
        "all_with_object",
    ]
    assert component.metadata["rebuild_commands"] == [
        (
            "chunking-docs embed-package --package-dir outputs/package "
            "--caption-backend same-as-text --object-backend same-as-caption --image-backend clip"
        ),
        "chunking-docs audit-package --package-dir outputs/package --require-qdrant-records",
        (
            "chunking-docs eval-qdrant-vector-ablation examples/retrieval_cases.jsonl "
            "--package-dir outputs/package --modes text,caption,text_caption,object,text_object,"
            "caption_object,image,text_image,caption_image,all,all_with_object "
            "--image-query-backend clip --image-query-model openai/clip-vit-large-patch14"
        ),
    ]
    assert component.metadata["expectations"]["object_dense"]["source_count"] == 1


def test_ingestion_readiness_detects_stale_bm25_visual_asset_text(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    stale_index = BM25LexicalIndex(manifest.chunks, texts=[chunk.text for chunk in manifest.chunks])
    stale_index.dump_manifest(package_dir / "bm25_tokens.json")

    report = build_ingestion_readiness_report(package_dir, manifest)

    assert report.passed is False
    assert "bm25_tokens" in report.failed_components
    bm25_component = next(component for component in report.components if component.name == "bm25_tokens")
    assert bm25_component.metadata["missing_linked_asset_text_chunk_count"] == 1
    assert bm25_component.metadata["text_char_count_mismatch_count"] == 1


def test_ingestion_readiness_can_gate_package_visual_text_coverage(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        min_visual_text_coverage_ratio=0.8,
    )

    assert report.passed is False
    assert "visual_text_coverage" in report.failed_components
    component = next(component for component in report.components if component.name == "visual_text_coverage")
    assert component.metadata["visual_text_asset_count"] == 1
    assert component.metadata["visual_text_covered_asset_count"] == 0
    assert component.metadata["visual_text_coverage_ratio"] == 0.0
    assert component.metadata["missing_asset_ids"] == ["asset-1"]


def test_ingestion_readiness_can_gate_package_visual_text_part_coverage(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.chunks[0] = manifest.chunks[0].model_copy(
        update={"text": "reference retrieval evidence reference visual evidence"}
    )
    manifest.assets[0] = manifest.assets[0].model_copy(
        update={"ocr_text": "uncovered visual label"}
    )
    write_jsonl(package_dir / "chunks.jsonl", manifest.chunks)
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)
    write_bm25_manifest(package_dir, manifest.chunks, manifest.assets)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        min_visual_text_coverage_ratio=0.8,
        min_visual_text_part_coverage_ratio=0.8,
    )

    assert report.passed is False
    assert "visual_text_coverage" in report.failed_components
    component = next(component for component in report.components if component.name == "visual_text_coverage")
    assert component.metadata["failed_checks"] == ["min_visual_text_part_coverage_ratio"]
    assert component.metadata["visual_text_coverage_ratio"] == 1.0
    assert component.metadata["visual_text_part_count"] == 2
    assert component.metadata["visual_text_covered_part_count"] == 1
    assert component.metadata["visual_text_part_coverage_ratio"] == 0.5
    assert component.metadata["missing_parts"][0]["asset_id"] == "asset-1"


def test_ingestion_readiness_reports_standalone_visual_text_chunks(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    standalone_chunk = DocumentChunk(
        chunk_id="visual-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.MAP,
        text="standalone visual evidence",
        asset_ids=["asset-2"],
        source_refs=["asset:asset-2"],
        metadata={
            "chunking_strategy": "visual_asset_text",
            "visual_asset_unlinked": True,
        },
    )
    standalone_asset = VisualAsset(
        asset_id="asset-2",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="standalone visual evidence",
    )
    manifest.chunks.append(standalone_chunk)
    manifest.assets.append(standalone_asset)
    write_jsonl(package_dir / "chunks.jsonl", manifest.chunks)
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)
    write_bm25_manifest(package_dir, manifest.chunks, manifest.assets)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        min_visual_text_coverage_ratio=0.5,
    )

    component = next(component for component in report.components if component.name == "visual_text_coverage")
    assert component.metadata["standalone_visual_chunk_count"] == 1
    assert component.metadata["standalone_visual_text_asset_count"] == 1
    assert component.metadata["standalone_visual_text_asset_ids"] == ["asset-2"]


def test_ingestion_readiness_includes_retrieval_cases_and_chunking_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    cases = [
        RetrievalCase(query="reference evidence", expected_pages=[1]),
        RetrievalCase(
            query="visual evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe", "object_probe_visual_only": True},
        ),
    ]
    comparison = chunking_comparison()

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_cases=cases,
        retrieval_case_options={
            "min_case_count": 2,
            "min_page_cases": 1,
            "min_asset_cases": 1,
            "min_distinct_asset_targets": 1,
            "max_asset_cases_per_target": 1,
            "min_case_group_distinct_targets": {"case_source:visual_object_probe:asset": 1},
            "require_visual_only_object_probes": True,
            "min_query_terms_per_case": 2,
        },
        chunking_comparison=comparison,
        chunking_gate_options={
            "candidate": "candidate",
            "baseline_candidate": "baseline",
            "min_quality_score": 0.8,
            "min_recall_at_k": 0.8,
            "max_mean_target_rank": 2.0,
            "min_visual_text_coverage_ratio": 0.8,
            "min_target_type_coverage": {"asset": 0.8, "triple": 0.8},
            "min_source_target_coverage": {"bm25": 0.8},
            "min_source_family_target_coverage": {"lexical": 0.8},
            "min_case_group_source_target_coverage": {
                "case_source:visual_lexical_probe:bm25": 0.8
            },
            "min_case_group_source_family_target_coverage": {
                "case_source:visual_lexical_probe:lexical": 0.8
            },
            "max_pairwise_mean_target_rank_delta": 0.0,
            "max_recall_drop": 0.05,
        },
    )

    assert report.passed is True
    assert report.retrieval_case_audit is not None
    assert report.retrieval_case_audit.target_counts["asset"] == 1
    assert report.retrieval_case_audit.distinct_target_counts["asset"] == 1
    assert report.retrieval_case_audit.max_cases_per_target["asset"] == 1
    assert report.retrieval_case_audit.short_query_count == 0
    assert report.retrieval_case_audit.visual_only_object_probe_count == 1
    retrieval_component = next(
        component for component in report.components if component.name == "retrieval_case_audit"
    )
    assert retrieval_component.metadata["check_count"] == len(report.retrieval_case_audit.checks)
    assert retrieval_component.metadata["checks"]
    assert retrieval_component.metadata["visual_object_probe_count"] == 1
    assert retrieval_component.metadata["distinct_target_counts"]["asset"] == 1
    assert retrieval_component.metadata["max_cases_per_target"]["asset"] == 1
    assert retrieval_component.metadata["case_group_distinct_target_counts"]["case_source"][
        "visual_object_probe"
    ]["asset"] == 1
    assert retrieval_component.metadata["min_query_term_count"] == 2
    assert retrieval_component.metadata["non_visual_only_object_probe_count"] == 0
    assert report.chunking_comparison_gate is not None
    assert report.chunking_comparison_gate.candidate == "candidate"
    assert report.chunking_comparison_gate.metrics["retrieval_mean_target_rank"] == 1.0
    assert (
        report.chunking_comparison_gate.pairwise_metrics[
            "pairwise_mean_target_rank_delta"
        ]
        == -1.0
    )
    assert report.chunking_comparison_gate.metrics["visual_text_coverage_ratio"] == 0.9
    assert report.chunking_comparison_gate.metrics["target_type.asset.coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.metrics["target_type.triple.coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.metrics["source.bm25.target_coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.metrics["source_family.lexical.target_coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.target_metrics["asset"]["coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.source_metrics["bm25"]["target_coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.source_family_metrics["lexical"][
        "target_coverage_at_k"
    ] == 0.9
    assert report.chunking_comparison_gate.case_group_source_metrics["case_source"][
        "visual_lexical_probe"
    ]["bm25"]["target_coverage_at_k"] == 0.9
    assert report.chunking_comparison_gate.case_group_source_family_metrics["case_source"][
        "visual_lexical_probe"
    ]["lexical"]["target_coverage_at_k"] == 0.9
    chunking_component = next(
        component for component in report.components if component.name == "chunking_comparison_gate"
    )
    assert chunking_component.metadata["check_count"] == len(report.chunking_comparison_gate.checks)
    assert chunking_component.metadata["checks"]
    assert report.failed_components == []


def test_ingestion_readiness_includes_qdrant_vector_ablation_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        qdrant_vector_ablation=qdrant_vector_ablation_report(),
        qdrant_vector_ablation_mode="text",
        qdrant_vector_ablation_gate_options={
            "baseline_mode": "caption",
            "min_recall_at_k": 1.0,
            "min_target_coverage_at_k": 1.0,
            "max_failed_queries": 0,
            "min_target_type_coverage": {"asset": 1.0},
            "min_source_target_coverage": {"qdrant:text_dense": 1.0},
            "min_source_family_target_coverage": {"dense_text": 1.0},
            "min_source_precision_at_hits": {"qdrant:text_dense": 1.0},
            "min_source_family_precision_at_hits": {"dense_text": 1.0},
            "min_case_group_target_coverage": {"case_source:visual_object_probe": 1.0},
            "min_case_group_source_precision_at_hits": {
                "case_source:visual_object_probe:qdrant:text_dense": 1.0
            },
            "min_case_group_source_family_precision_at_hits": {
                "case_source:visual_object_probe:dense_text": 1.0
            },
            "max_mean_target_rank": 1.0,
            "max_pairwise_mean_target_rank_delta": 0.0,
            "require_best_by_recall": True,
        },
    )

    assert report.passed is True
    assert report.qdrant_vector_ablation_gate is not None
    assert report.qdrant_vector_ablation_gate.mode == "text"
    assert report.qdrant_vector_ablation_gate.baseline_mode == "caption"
    assert report.qdrant_vector_ablation_gate.metrics["failed_query_count"] == 0.0
    assert report.qdrant_vector_ablation_gate.metrics["mean_target_rank"] == 1.0
    assert (
        report.qdrant_vector_ablation_gate.pairwise_metrics[
            "pairwise_mean_target_rank_delta"
        ]
        == -5.0
    )
    assert report.qdrant_vector_ablation_gate.target_metrics["asset"]["coverage_at_k"] == 1.0
    assert (
        report.qdrant_vector_ablation_gate.source_metrics["qdrant:text_dense"][
            "target_coverage_at_k"
        ]
        == 1.0
    )
    assert (
        report.qdrant_vector_ablation_gate.source_metrics["qdrant:text_dense"][
            "precision_at_hits"
        ]
        == 1.0
    )
    assert (
        report.qdrant_vector_ablation_gate.source_family_metrics["dense_text"][
            "target_coverage_at_k"
        ]
        == 1.0
    )
    assert (
        report.qdrant_vector_ablation_gate.source_family_metrics["dense_text"][
            "precision_at_hits"
        ]
        == 1.0
    )
    assert report.qdrant_vector_ablation_gate.case_group_metrics["case_source"][
        "visual_object_probe"
    ]["target_coverage_at_k"] == 1.0
    component = next(
        component
        for component in report.components
        if component.name == "qdrant_vector_ablation_gate"
    )
    assert component.metadata["check_count"] == len(report.qdrant_vector_ablation_gate.checks)
    assert component.metadata["checks"]
    assert report.failed_components == []


def test_ingestion_readiness_includes_qdrant_reranker_ablation_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        qdrant_reranker_ablation=qdrant_reranker_ablation_report(),
        qdrant_reranker_ablation_mode="lexical",
        qdrant_reranker_ablation_gate_options={
            "baseline_mode": "none",
            "min_recall_at_k": 1.0,
            "min_target_coverage_at_k": 1.0,
            "max_failed_queries": 0,
            "min_target_type_coverage": {"asset": 1.0},
            "min_source_target_coverage": {"rerank:lexical": 1.0},
            "min_source_family_target_coverage": {"lexical": 1.0},
            "min_source_precision_at_hits": {"rerank:lexical": 1.0},
            "min_source_family_precision_at_hits": {"lexical": 1.0},
            "min_case_group_target_coverage": {"case_source:visual_object_probe": 1.0},
            "min_case_group_source_precision_at_hits": {
                "case_source:visual_object_probe:rerank:lexical": 1.0
            },
            "min_case_group_source_family_precision_at_hits": {
                "case_source:visual_object_probe:reranker": 1.0
            },
            "max_mean_target_rank": 1.0,
            "max_pairwise_mean_target_rank_delta": 0.0,
            "require_best_by_recall": True,
        },
    )

    assert report.passed is True
    assert report.qdrant_reranker_ablation_gate is not None
    assert report.qdrant_reranker_ablation_gate.mode == "lexical"
    assert report.qdrant_reranker_ablation_gate.baseline_mode == "none"
    assert report.qdrant_reranker_ablation_gate.reranker == "lexical"
    assert report.qdrant_reranker_ablation_gate.rerank_top_k == 20
    assert report.qdrant_reranker_ablation_gate.metrics["failed_query_count"] == 0.0
    assert report.qdrant_reranker_ablation_gate.metrics["mean_target_rank"] == 1.0
    assert (
        report.qdrant_reranker_ablation_gate.pairwise_metrics[
            "pairwise_mean_target_rank_delta"
        ]
        == -5.0
    )
    component = next(
        component
        for component in report.components
        if component.name == "qdrant_reranker_ablation_gate"
    )
    assert component.metadata["reranker"] == "lexical"
    assert component.metadata["rerank_top_k"] == 20
    assert component.metadata["check_count"] == len(report.qdrant_reranker_ablation_gate.checks)
    assert component.metadata["checks"]
    assert component.metadata["source_metrics"]["rerank:lexical"]["target_coverage_at_k"] == 1.0
    assert component.metadata["source_metrics"]["rerank:lexical"]["precision_at_hits"] == 1.0
    assert report.failed_components == []


def test_ingestion_readiness_includes_retrieval_ablation_lift_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_ablation=retrieval_ablation_report(),
        retrieval_ablation_mode="bm25_visual",
        retrieval_ablation_baseline_mode="bm25_text",
        retrieval_ablation_gate_options={
            "min_recall_at_k": 1.0,
            "min_recall_lift": 1.0,
            "min_target_coverage_lift": 1.0,
            "min_target_type_coverage": {"asset": 1.0},
            "min_source_target_coverage": {"bm25": 1.0},
            "min_source_family_target_coverage": {"lexical": 1.0},
            "min_source_precision_at_hits": {"bm25": 1.0},
            "min_source_family_precision_at_hits": {"lexical": 1.0},
            "min_case_group_target_coverage": {"case_source:visual_lexical_probe": 1.0},
            "min_case_group_source_precision_at_hits": {
                "case_source:visual_lexical_probe:bm25": 1.0
            },
            "min_case_group_source_family_precision_at_hits": {
                "case_source:visual_lexical_probe:lexical": 1.0
            },
            "max_mean_target_rank": 1.0,
            "max_pairwise_mean_target_rank_delta": 0.0,
            "require_best_by_recall": True,
        },
    )

    assert report.passed is True
    assert report.retrieval_ablation_gate is not None
    assert report.retrieval_ablation_gate.mode == "bm25_visual"
    assert report.retrieval_ablation_gate.baseline_mode == "bm25_text"
    assert report.retrieval_ablation_gate.metrics["recall_at_k"] == 1.0
    assert report.retrieval_ablation_gate.metrics["mean_target_rank"] == 1.0
    assert report.retrieval_ablation_gate.baseline_metrics["recall_at_k"] == 0.0
    assert report.retrieval_ablation_gate.baseline_metrics["mean_target_rank"] == 6.0
    assert (
        report.retrieval_ablation_gate.pairwise_metrics[
            "pairwise_mean_target_rank_delta"
        ]
        == -5.0
    )
    assert report.retrieval_ablation_gate.source_metrics["bm25"]["target_coverage_at_k"] == 1.0
    assert report.retrieval_ablation_gate.source_metrics["bm25"]["precision_at_hits"] == 1.0
    component = next(
        component for component in report.components if component.name == "retrieval_ablation_gate"
    )
    assert component.metadata["check_count"] == len(report.retrieval_ablation_gate.checks)
    assert component.metadata["checks"]
    assert component.metadata["metrics"]["target_type.asset.coverage_at_k"] == 1.0
    assert component.metadata["source_metrics"]["bm25"]["target_coverage_at_k"] == 1.0
    assert component.metadata["source_metrics"]["bm25"]["precision_at_hits"] == 1.0
    assert (
        component.metadata["metrics"][
            "case_group.case_source.visual_lexical_probe.target_coverage_at_k"
        ]
        == 1.0
    )
    assert component.metadata["baseline_metrics"]["target_coverage_at_k"] == 0.0
    assert report.failed_components == []


def test_ingestion_readiness_requires_retrieval_ablation(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_retrieval_ablation=True,
    )

    assert report.passed is False
    assert "retrieval_ablation_gate" in report.failed_components


def test_ingestion_readiness_requires_qdrant_vector_ablation(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_qdrant_vector_ablation=True,
    )

    assert report.passed is False
    assert "qdrant_vector_ablation_gate" in report.failed_components


def test_ingestion_readiness_requires_qdrant_reranker_ablation(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_qdrant_reranker_ablation=True,
    )

    assert report.passed is False
    assert "qdrant_reranker_ablation_gate" in report.failed_components


def test_ingestion_readiness_includes_rag_context_gate(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        rag_context_evaluation=readiness_rag_context_evaluation(),
        rag_context_gate_options={
            "min_case_count": 2,
            "min_expected_target_count": 2,
            "min_target_coverage": 0.9,
            "max_excluded_target_hit_rate": 0.0,
            "max_mean_context_char_count": 9000,
            "min_target_type_coverage": {"asset": 1.0},
            "min_case_group_target_coverage": {
                "case_source:visual_object_probe": 1.0
            },
        },
    )

    assert report.passed is True
    assert report.rag_context_gate is not None
    assert report.rag_context_gate.metrics["target_coverage"] == 1.0
    assert report.rag_context_gate.target_metrics["asset"]["coverage"] == 1.0
    assert report.rag_context_gate.case_group_metrics["case_source"][
        "visual_object_probe"
    ]["target_coverage"] == 1.0
    component = next(
        component for component in report.components if component.name == "rag_context_gate"
    )
    assert component.metadata["check_count"] == len(report.rag_context_gate.checks)
    assert component.metadata["checks"]
    assert component.metadata["metrics"]["mean_context_char_count"] == 4200.0
    assert report.failed_components == []


def test_ingestion_readiness_requires_rag_context_evaluation(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_rag_context_evaluation=True,
    )

    assert report.passed is False
    assert "rag_context_gate" in report.failed_components


def test_ingestion_readiness_validates_qdrant_retrieval_config(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    config = readiness_qdrant_retrieval_config(package_dir)
    retrieval_eval = readiness_retrieval_evaluation(target_coverage=1.0).model_copy(
        update={"metadata": qdrant_config_metadata(config, "qdrant_hybrid_config")}
    )
    context_eval = readiness_rag_context_evaluation().model_copy(
        update={
            "metadata": qdrant_config_metadata(
                config,
                "qdrant_rag_context_config",
            )
        }
    )

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        qdrant_retrieval_config=config,
        retrieval_evaluation=retrieval_eval,
        rag_context_evaluation=context_eval,
    )

    component = next(
        component
        for component in report.components
        if component.name == "qdrant_retrieval_config"
    )
    assert report.passed is True
    assert component.metadata["failed_checks"] == []
    assert component.metadata["collection_vectors"] == ["text_dense"]
    assert component.metadata["missing_collection_vectors"] == []
    assert component.metadata["bm25_tokenizer_alignment"]["matches"] is True
    assert component.metadata["retrieval_evaluation_alignment"]["matches"] is True
    assert component.metadata["rag_context_evaluation_alignment"]["matches"] is True


def test_ingestion_readiness_flags_qdrant_retrieval_config_mismatch(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    config = readiness_qdrant_retrieval_config(package_dir).model_copy(
        update={
            "vector_names": ["caption_dense"],
            "query_encoders": {},
        }
    )
    retrieval_eval = readiness_retrieval_evaluation(target_coverage=1.0).model_copy(
        update={
            "metadata": qdrant_config_metadata(
                readiness_qdrant_retrieval_config(package_dir),
                "qdrant_hybrid_config",
            )
        }
    )

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        qdrant_retrieval_config=config,
        retrieval_evaluation=retrieval_eval,
    )

    component = next(
        component
        for component in report.components
        if component.name == "qdrant_retrieval_config"
    )
    assert report.passed is False
    assert "qdrant_retrieval_config" in report.failed_components
    assert "missing_collection_vectors" in component.metadata["failed_checks"]
    assert "missing_query_encoders" in component.metadata["failed_checks"]
    assert (
        "retrieval_evaluation_vector_names_mismatch"
        in component.metadata["failed_checks"]
    )


def test_ingestion_readiness_requires_qdrant_retrieval_config(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_qdrant_retrieval_config=True,
    )

    assert report.passed is False
    assert "qdrant_retrieval_config" in report.failed_components


def test_ingestion_readiness_can_gate_retrieval_source(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_evaluation=qdrant_vector_ablation_report().rows[0].evaluation,
        retrieval_gate_options={
            "min_recall_at_k": 1.0,
            "min_source_target_coverage": {"qdrant:text_dense": 1.0},
            "min_source_family_target_coverage": {"dense_text": 1.0},
        },
    )

    assert report.passed is True
    assert report.retrieval_gate is not None
    assert report.retrieval_gate.source_metrics["qdrant:text_dense"]["target_coverage_at_k"] == 1.0
    assert report.retrieval_gate.source_family_metrics["dense_text"]["target_coverage_at_k"] == 1.0
    component = next(component for component in report.components if component.name == "retrieval_gate")
    assert component.metadata["check_count"] == len(report.retrieval_gate.checks)
    assert any(
        check["name"] == "min_source_target_coverage:qdrant:text_dense"
        for check in component.metadata["checks"]
    )
    assert component.metadata["source_metrics"]["qdrant:text_dense"]["target_coverage_at_k"] == 1.0
    assert component.metadata["source_family_metrics"]["dense_text"]["target_coverage_at_k"] == 1.0
    assert report.failed_components == []


def test_ingestion_readiness_can_gate_retrieval_target_type(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_evaluation=qdrant_vector_ablation_report().rows[0].evaluation,
        retrieval_gate_options={
            "min_recall_at_k": 1.0,
            "min_target_type_coverage": {"asset": 1.0},
        },
    )

    assert report.passed is True
    assert report.retrieval_gate is not None
    assert report.retrieval_gate.target_metrics["asset"]["coverage_at_k"] == 1.0
    component = next(component for component in report.components if component.name == "retrieval_gate")
    assert component.metadata["target_metrics"]["asset"]["coverage_at_k"] == 1.0
    assert report.failed_components == []


def test_ingestion_readiness_can_gate_retrieval_target_rank(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        retrieval_evaluation=qdrant_vector_ablation_report().rows[1].evaluation,
        retrieval_gate_options={
            "max_mean_target_rank": 5.0,
        },
    )

    assert report.passed is False
    assert report.retrieval_gate is not None
    assert report.retrieval_gate.metrics["mean_target_rank"] == 6.0
    assert "max_mean_target_rank" in report.retrieval_gate.failed_checks
    component = next(component for component in report.components if component.name == "retrieval_gate")
    assert component.metadata["metrics"]["mean_target_rank"] == 6.0


def test_ingestion_readiness_can_gate_visual_quality_from_assets(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.assets[0] = manifest.assets[0].model_copy(
        update={
            "ocr_text": "recognized visual text",
            "vlm_summary": "structured visual summary",
            "metadata": {
                "requires_ocr": True,
                "requires_vlm": True,
                "vlm_parse_status": "json_object",
            },
        }
    )
    write_bm25_manifest(package_dir, manifest.chunks, manifest.assets)

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        require_visual_annotations=True,
        require_visual_quality=True,
        visual_quality_options={
            "min_completion_rate": 1.0,
            "min_ocr_text_coverage": 1.0,
            "min_vlm_summary_coverage": 1.0,
            "min_vlm_json_parse_rate": 1.0,
        },
    )
    visual_component = next(component for component in report.components if component.name == "visual_quality")

    assert report.passed is True
    assert report.visual_quality is not None
    assert report.visual_quality.vlm_json_parse_rate == 1.0
    assert visual_component.metadata["source"] == "assets"
    assert visual_component.metadata["check_count"] == len(report.visual_quality.checks)
    assert visual_component.metadata["checks"]


def test_ingestion_readiness_can_gate_visual_run_comparison(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    comparison = compare_visual_runs(
        {
            "raw": visual_run_results("raw_text", triple=False),
            "structured": visual_run_results("json_object", triple=True),
        }
    )

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        visual_run_comparison=comparison,
        visual_run_comparison_options={
            "require_same_jobs": True,
            "min_run_count": 2,
            "min_shared_job_count": 1,
            "expected_best_by_quality": "structured",
        },
    )

    component = next(
        component for component in report.components if component.name == "visual_run_comparison"
    )
    assert report.passed is True
    assert report.visual_run_comparison is not None
    assert component.metadata["job_set_mismatch"] is False
    assert component.metadata["best_by_quality"] == "structured"


def test_ingestion_readiness_can_gate_visual_run_retrieval_winner(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    comparison = compare_visual_runs(
        {
            "raw": visual_run_results("raw_text", triple=False),
            "structured": visual_run_results("json_object", triple=True),
        },
        retrieval_evaluations={
            "raw": readiness_retrieval_evaluation(target_coverage=0.8),
            "structured": readiness_retrieval_evaluation(target_coverage=0.95),
        },
    )

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        visual_run_comparison=comparison,
        visual_run_comparison_options={"expected_best_by_retrieval": "structured"},
    )

    component = next(
        component for component in report.components if component.name == "visual_run_comparison"
    )
    assert report.passed is True
    assert component.metadata["best_by_retrieval"] == "structured"
    assert component.metadata["retrieval_evaluation_run_count"] == 2


def test_ingestion_readiness_flags_visual_run_job_mismatch(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    comparison = compare_visual_runs(
        {
            "raw": visual_run_results("raw_text", triple=False, job_id="job-1"),
            "structured": visual_run_results("json_object", triple=True, job_id="job-2"),
        }
    )

    report = build_ingestion_readiness_report(
        package_dir,
        manifest,
        visual_run_comparison=comparison,
        visual_run_comparison_options={"require_same_jobs": True},
    )

    component = next(
        component for component in report.components if component.name == "visual_run_comparison"
    )
    assert report.passed is False
    assert "visual_run_comparison" in report.failed_components
    assert component.metadata["failed_checks"] == ["same_job_set"]
    assert component.metadata["job_set_mismatch"] is True


def test_ingestion_readiness_cli_reports_missing_required_artifact(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    (package_dir / "bm25_tokens.json").unlink()
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "bm25_tokens" in payload["failed_components"]


def test_ingestion_readiness_cli_requires_retrieval_cases(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--require-retrieval-cases",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "retrieval_case_audit" in payload["failed_components"]


def test_ingestion_readiness_cli_can_require_visual_only_object_probes(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output = tmp_path / "readiness.json"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="broad object target",
                expected_asset_ids=["asset-1"],
                metadata={
                    "case_source": "visual_object_probe",
                    "object_probe_visual_only": False,
                },
            )
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-cases",
            str(cases_path),
            "--require-visual-only-object-probes",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "retrieval_case_audit" in payload["failed_components"]
    component = next(
        component for component in payload["components"] if component["name"] == "retrieval_case_audit"
    )
    assert component["metadata"]["failed_checks"] == ["require_visual_only_object_probes"]
    assert component["metadata"]["visual_object_probe_count"] == 1
    assert component["metadata"]["non_visual_only_object_probe_count"] == 1


def test_ingestion_readiness_cli_can_gate_distinct_retrieval_targets(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output = tmp_path / "readiness.json"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(query="visual target one", expected_asset_ids=["asset-1"]),
            RetrievalCase(query="visual target two", expected_asset_ids=["asset-1"]),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-cases",
            str(cases_path),
            "--min-retrieval-distinct-asset-targets",
            "2",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    component = next(
        component for component in payload["components"] if component["name"] == "retrieval_case_audit"
    )
    assert component["metadata"]["failed_checks"] == ["min_distinct_asset_targets"]
    assert component["metadata"]["target_counts"]["asset"] == 2
    assert component["metadata"]["distinct_target_counts"]["asset"] == 1


def test_ingestion_readiness_cli_can_gate_case_group_distinct_targets(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output = tmp_path / "readiness.json"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="object probe one",
                expected_asset_ids=["asset-1"],
                metadata={"case_source": "visual_object_probe"},
            ),
            RetrievalCase(
                query="object probe two",
                expected_asset_ids=["asset-1"],
                metadata={"case_source": "visual_object_probe"},
            ),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-cases",
            str(cases_path),
            "--min-retrieval-case-group-distinct-targets",
            "case_source:visual_object_probe:asset=2",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    component = next(
        component for component in payload["components"] if component["name"] == "retrieval_case_audit"
    )
    assert component["metadata"]["failed_checks"] == [
        "min_case_group_distinct_targets:case_source:visual_object_probe:asset"
    ]
    assert component["metadata"]["case_group_distinct_target_counts"]["case_source"][
        "visual_object_probe"
    ]["asset"] == 1


def test_ingestion_readiness_cli_can_gate_retrieval_target_concentration(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output = tmp_path / "readiness.json"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(query="visual target one", expected_asset_ids=["asset-1"]),
            RetrievalCase(query="visual target two", expected_asset_ids=["asset-1"]),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-cases",
            str(cases_path),
            "--max-retrieval-asset-cases-per-target",
            "1",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    component = next(
        component for component in payload["components"] if component["name"] == "retrieval_case_audit"
    )
    assert component["metadata"]["failed_checks"] == ["max_asset_cases_per_target"]
    assert component["metadata"]["max_cases_per_target"]["asset"] == 2


def test_ingestion_readiness_cli_can_gate_retrieval_query_strength(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output = tmp_path / "readiness.json"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(query="short", expected_pages=[1]),
            RetrievalCase(query="specific visual target", expected_asset_ids=["asset-1"]),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-cases",
            str(cases_path),
            "--min-retrieval-query-terms-per-case",
            "2",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    component = next(
        component for component in payload["components"] if component["name"] == "retrieval_case_audit"
    )
    assert component["metadata"]["failed_checks"] == ["min_query_terms_per_case"]
    assert component["metadata"]["short_query_count"] == 1
    assert component["metadata"]["min_query_term_count"] == 1


def test_ingestion_readiness_cli_can_gate_retrieval_expected_targets_per_case(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output = tmp_path / "readiness.json"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="specific visual target",
                expected_pages=[1],
                expected_asset_ids=["asset-1"],
            ),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-cases",
            str(cases_path),
            "--max-retrieval-expected-targets-per-case",
            "1",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    component = next(
        component for component in payload["components"] if component["name"] == "retrieval_case_audit"
    )
    assert component["metadata"]["failed_checks"] == ["max_expected_targets_per_case"]
    assert component["metadata"]["max_expected_targets_per_case"] == 2
    assert component["metadata"]["oversized_expected_target_case_count"] == 1


def test_ingestion_readiness_cli_can_gate_visual_quality_from_assets(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            path=package_dir / "assets/page.png",
            ocr_text="recognized visual text",
            vlm_summary="structured visual summary",
            metadata={
                "requires_ocr": True,
                "requires_vlm": True,
                "vlm_parse_status": "json_object",
            },
        )
    ]
    write_jsonl(package_dir / "assets.jsonl", assets)
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    write_bm25_manifest(package_dir, chunks, assets)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--require-visual-annotations",
            "--require-visual-quality",
            "--min-visual-completion-rate",
            "1",
            "--min-ocr-text-coverage",
            "1",
            "--min-vlm-summary-coverage",
            "1",
            "--min-vlm-json-parse-rate",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    visual_component = next(
        component for component in payload["components"] if component["name"] == "visual_quality"
    )
    assert visual_component["metadata"]["source"] == "assets"


def test_ingestion_readiness_cli_can_gate_package_visual_text_coverage(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--min-visual-text-coverage-ratio",
            "0.8",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "visual_text_coverage" in payload["failed_components"]
    component = next(
        component for component in payload["components"] if component["name"] == "visual_text_coverage"
    )
    assert component["metadata"]["visual_text_coverage_ratio"] == 0.0
    assert component["metadata"]["missing_asset_ids"] == ["asset-1"]


def test_ingestion_readiness_cli_can_gate_package_visual_text_part_coverage(tmp_path):
    package_dir, manifest = write_ready_package(tmp_path)
    manifest.chunks[0] = manifest.chunks[0].model_copy(
        update={"text": "reference retrieval evidence reference visual evidence"}
    )
    manifest.assets[0] = manifest.assets[0].model_copy(
        update={"ocr_text": "uncovered visual label"}
    )
    write_jsonl(package_dir / "chunks.jsonl", manifest.chunks)
    write_jsonl(package_dir / "assets.jsonl", manifest.assets)
    write_bm25_manifest(package_dir, manifest.chunks, manifest.assets)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--min-visual-text-part-coverage-ratio",
            "0.8",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    component = next(
        component for component in payload["components"] if component["name"] == "visual_text_coverage"
    )
    assert component["metadata"]["failed_checks"] == ["min_visual_text_part_coverage_ratio"]
    assert component["metadata"]["visual_text_coverage_ratio"] == 1.0
    assert component["metadata"]["visual_text_part_coverage_ratio"] == 0.5


def test_ingestion_readiness_cli_can_gate_retrieval_exact_source(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    retrieval_path = tmp_path / "retrieval_eval.json"
    output = tmp_path / "readiness.json"
    retrieval_path.write_text(
        qdrant_vector_ablation_report().rows[0].evaluation.model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-evaluation",
            str(retrieval_path),
            "--min-recall-at-k",
            "1.0",
            "--min-retrieval-source-target-coverage",
            "qdrant:text_dense=1.0",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    component = next(
        component for component in payload["components"] if component["name"] == "retrieval_gate"
    )
    assert component["metadata"]["source_metrics"]["qdrant:text_dense"][
        "target_coverage_at_k"
    ] == 1.0
    assert (
        component["metadata"]["metrics"]["source.qdrant:text_dense.target_coverage_at_k"]
        == 1.0
    )


def test_ingestion_readiness_cli_can_gate_visual_run_comparison(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    comparison_path = tmp_path / "visual_run_comparison.json"
    output = tmp_path / "readiness.json"
    comparison = compare_visual_runs(
        {
            "raw": visual_run_results("raw_text", triple=False),
            "structured": visual_run_results("json_object", triple=True),
        }
    )
    comparison_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--visual-run-comparison",
            str(comparison_path),
            "--require-visual-run-same-jobs",
            "--min-visual-run-count",
            "2",
            "--min-visual-run-shared-jobs",
            "1",
            "--visual-run-best-by-quality",
            "structured",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "visual_run_comparison"
    )
    assert payload["passed"] is True
    assert component["metadata"]["job_set_mismatch"] is False
    assert component["metadata"]["best_by_quality"] == "structured"


def test_ingestion_readiness_cli_can_require_embedding_vectors(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--required-vector",
            "text_dense",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    component = next(
        component for component in payload["components"] if component["name"] == "embedding_vectors"
    )
    assert payload["passed"] is True
    assert component["metadata"]["required_vectors"] == ["text_dense"]
    assert component["metadata"]["required_vector_details"]["text_dense"]["record_count"] == 1


def test_ingestion_readiness_cli_can_require_derived_vector_coverage(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="station",
        predicate="connects_to",
        object="corridor",
    )
    write_jsonl(package_dir / "triples.jsonl", [triple])
    output = tmp_path / "readiness.json"

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--require-derived-vector-coverage",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "derived_embedding_vectors"
    )
    assert payload["passed"] is False
    assert "derived_embedding_vectors" in payload["failed_components"]
    assert "triple_dense" in component["metadata"]["expected_vectors"]
    assert "triple_dense" in component["metadata"]["missing_collection_vectors"]
    assert component["metadata"]["missing_expected_vectors"] == [
        "caption_dense",
        "image_dense",
        "triple_dense",
    ]
    assert "--image-backend clip" in component["metadata"]["rebuild_commands"][1]
    assert "--image-query-backend clip" in component["metadata"]["rebuild_commands"][-1]
    assert "--triple-backend same-as-text" in component["metadata"]["rebuild_commands"][1]
    assert "image_dense" in component["metadata"]["expected_vectors"]
    assert "text_image" in component["metadata"]["recommended_qdrant_vector_modes"]
    assert "text_triple_graph" in component["metadata"]["recommended_qdrant_vector_modes"]


def test_ingestion_readiness_cli_can_require_runtime_report(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    runtime_path = tmp_path / "runtime_doctor.json"
    output = tmp_path / "readiness.json"
    runtime_path.write_text(ready_runtime_report().model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--runtime-report",
            str(runtime_path),
            "--require-runtime-report",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    component = next(
        component for component in payload["components"] if component["name"] == "runtime_report"
    )
    assert component["metadata"]["gpu_count"] == 1
    assert component["metadata"]["torch_cuda_compute_capabilities"] == ["12.0"]


def test_ingestion_readiness_cli_can_gate_qdrant_vector_ablation(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    ablation_path = tmp_path / "qdrant_vector_ablation.json"
    output = tmp_path / "readiness.json"
    ablation_path.write_text(
        qdrant_vector_ablation_report().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--qdrant-vector-ablation",
            str(ablation_path),
            "--qdrant-vector-mode",
            "text",
            "--min-qdrant-vector-recall-at-k",
            "1.0",
            "--min-qdrant-vector-target-coverage-at-k",
            "1.0",
            "--max-qdrant-vector-failed-queries",
            "0",
            "--min-qdrant-vector-target-type-coverage",
            "asset=1.0",
            "--min-qdrant-vector-source-target-coverage",
            "qdrant:text_dense=1.0",
            "--min-qdrant-vector-source-family-target-coverage",
            "dense_text=1.0",
            "--min-qdrant-vector-source-precision-at-hits",
            "qdrant:text_dense=1.0",
            "--min-qdrant-vector-source-family-precision-at-hits",
            "dense_text=1.0",
            "--min-qdrant-vector-case-group-target-coverage",
            "case_source:visual_object_probe=1.0",
            "--min-qdrant-vector-case-group-source-target-coverage",
            "case_source:visual_object_probe:qdrant:text_dense=1.0",
            "--min-qdrant-vector-case-group-source-family-target-coverage",
            "case_source:visual_object_probe:dense_text=1.0",
            "--min-qdrant-vector-case-group-source-precision-at-hits",
            "case_source:visual_object_probe:qdrant:text_dense=1.0",
            "--min-qdrant-vector-case-group-source-family-precision-at-hits",
            "case_source:visual_object_probe:dense_text=1.0",
            "--require-qdrant-vector-best-by-recall",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "qdrant_vector_ablation_gate"
    )
    assert component["metadata"]["mode"] == "text"
    assert component["metadata"]["metrics"]["recall_at_k"] == 1.0
    assert component["metadata"]["target_metrics"]["asset"]["coverage_at_k"] == 1.0
    assert component["metadata"]["source_metrics"]["qdrant:text_dense"][
        "target_coverage_at_k"
    ] == 1.0
    assert component["metadata"]["source_metrics"]["qdrant:text_dense"][
        "precision_at_hits"
    ] == 1.0
    assert component["metadata"]["source_family_metrics"]["dense_text"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["source_family_metrics"]["dense_text"][
        "precision_at_hits"
    ] == 1.0
    assert component["metadata"]["case_group_metrics"]["case_source"]["visual_object_probe"][
        "target_coverage_at_k"
    ] == 1.0
    assert component["metadata"]["case_group_source_metrics"]["case_source"][
        "visual_object_probe"
    ]["qdrant:text_dense"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["case_group_source_metrics"]["case_source"][
        "visual_object_probe"
    ]["qdrant:text_dense"]["precision_at_hits"] == 1.0
    assert component["metadata"]["case_group_source_family_metrics"]["case_source"][
        "visual_object_probe"
    ]["dense_text"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["case_group_source_family_metrics"]["case_source"][
        "visual_object_probe"
    ]["dense_text"]["precision_at_hits"] == 1.0


def test_ingestion_readiness_cli_can_gate_qdrant_reranker_ablation(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    ablation_path = tmp_path / "qdrant_reranker_ablation.json"
    output = tmp_path / "readiness.json"
    ablation_path.write_text(
        qdrant_reranker_ablation_report().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--qdrant-reranker-ablation",
            str(ablation_path),
            "--qdrant-reranker-mode",
            "lexical",
            "--qdrant-reranker-baseline-mode",
            "none",
            "--min-qdrant-reranker-target-coverage-at-k",
            "1.0",
            "--max-qdrant-reranker-failed-queries",
            "0",
            "--min-qdrant-reranker-source-target-coverage",
            "rerank:lexical=1.0",
            "--min-qdrant-reranker-source-precision-at-hits",
            "rerank:lexical=1.0",
            "--min-qdrant-reranker-source-family-precision-at-hits",
            "lexical=1.0",
            "--min-qdrant-reranker-case-group-target-coverage",
            "case_source:visual_object_probe=1.0",
            "--min-qdrant-reranker-case-group-source-target-coverage",
            "case_source:visual_object_probe:rerank:lexical=1.0",
            "--min-qdrant-reranker-case-group-source-family-target-coverage",
            "case_source:visual_object_probe:reranker=1.0",
            "--min-qdrant-reranker-case-group-source-precision-at-hits",
            "case_source:visual_object_probe:rerank:lexical=1.0",
            "--min-qdrant-reranker-case-group-source-family-precision-at-hits",
            "case_source:visual_object_probe:reranker=1.0",
            "--min-qdrant-reranker-pairwise-win-rate",
            "1.0",
            "--max-qdrant-reranker-pairwise-mean-target-rank-delta",
            "0.0",
            "--require-qdrant-reranker-best-by-recall",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "qdrant_reranker_ablation_gate"
    )
    assert component["metadata"]["mode"] == "lexical"
    assert component["metadata"]["baseline_mode"] == "none"
    assert component["metadata"]["reranker"] == "lexical"
    assert component["metadata"]["rerank_top_k"] == 20
    assert component["metadata"]["metrics"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["source_metrics"]["rerank:lexical"][
        "target_coverage_at_k"
    ] == 1.0
    assert component["metadata"]["source_metrics"]["rerank:lexical"][
        "precision_at_hits"
    ] == 1.0
    assert component["metadata"]["case_group_source_metrics"]["case_source"][
        "visual_object_probe"
    ]["rerank:lexical"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["case_group_source_metrics"]["case_source"][
        "visual_object_probe"
    ]["rerank:lexical"]["precision_at_hits"] == 1.0
    assert component["metadata"]["case_group_source_family_metrics"]["case_source"][
        "visual_object_probe"
    ]["reranker"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["case_group_source_family_metrics"]["case_source"][
        "visual_object_probe"
    ]["reranker"]["precision_at_hits"] == 1.0
    assert component["metadata"]["pairwise_metrics"]["pairwise_candidate_win_rate"] == 1.0


def test_ingestion_readiness_cli_can_gate_rag_context(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    context_eval_path = tmp_path / "rag_context_eval.json"
    output = tmp_path / "readiness.json"
    context_eval_path.write_text(
        readiness_rag_context_evaluation().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--rag-context-evaluation",
            str(context_eval_path),
            "--min-rag-context-case-count",
            "2",
            "--min-rag-context-target-coverage",
            "1.0",
            "--min-rag-context-target-type-coverage",
            "asset=1.0",
            "--min-rag-context-case-group-target-coverage",
            "case_source:visual_object_probe=1.0",
            "--max-rag-context-excluded-target-hit-rate",
            "0",
            "--max-rag-context-mean-context-char-count",
            "9000",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    component = next(
        component for component in payload["components"] if component["name"] == "rag_context_gate"
    )
    assert component["metadata"]["metrics"]["target_coverage"] == 1.0
    assert component["metadata"]["target_metrics"]["asset"]["coverage"] == 1.0
    assert component["metadata"]["case_group_metrics"]["case_source"][
        "visual_object_probe"
    ]["target_coverage"] == 1.0


def test_ingestion_readiness_cli_can_validate_qdrant_retrieval_config(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    config = readiness_qdrant_retrieval_config(package_dir)
    config_path = tmp_path / "qdrant_retrieval_config.json"
    retrieval_eval_path = tmp_path / "qdrant_retrieval_config_eval.json"
    context_eval_path = tmp_path / "rag_context_eval.json"
    output = tmp_path / "readiness.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    retrieval_eval_path.write_text(
        readiness_retrieval_evaluation(target_coverage=1.0)
        .model_copy(
            update={
                "metadata": qdrant_config_metadata(
                    config,
                    "qdrant_hybrid_config",
                )
            }
        )
        .model_dump_json(indent=2),
        encoding="utf-8",
    )
    context_eval_path.write_text(
        readiness_rag_context_evaluation()
        .model_copy(
            update={
                "metadata": qdrant_config_metadata(
                    config,
                    "qdrant_rag_context_config",
                )
            }
        )
        .model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--qdrant-retrieval-config",
            str(config_path),
            "--retrieval-evaluation",
            str(retrieval_eval_path),
            "--require-retrieval-evaluation",
            "--min-target-coverage-at-k",
            "1.0",
            "--max-retrieval-failed-queries",
            "0",
            "--rag-context-evaluation",
            str(context_eval_path),
            "--min-rag-context-target-coverage",
            "1.0",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "qdrant_retrieval_config"
    )
    assert payload["passed"] is True
    assert component["metadata"]["failed_checks"] == []
    assert component["metadata"]["retrieval_evaluation_alignment"]["matches"] is True
    assert component["metadata"]["rag_context_evaluation_alignment"]["matches"] is True
    retrieval_component = next(
        component
        for component in payload["components"]
        if component["name"] == "retrieval_gate"
    )
    assert retrieval_component["metadata"]["metrics"]["target_coverage_at_k"] == 1.0


def test_ingestion_readiness_cli_can_gate_retrieval_ablation_lift(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    ablation_path = tmp_path / "retrieval_ablation.json"
    output = tmp_path / "readiness.json"
    ablation_path.write_text(
        retrieval_ablation_report().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--retrieval-ablation",
            str(ablation_path),
            "--retrieval-ablation-mode",
            "bm25_visual",
            "--retrieval-ablation-baseline-mode",
            "bm25_text",
            "--min-retrieval-ablation-recall-at-k",
            "1.0",
            "--min-retrieval-ablation-recall-lift",
            "1.0",
            "--min-retrieval-ablation-target-coverage-lift",
            "1.0",
            "--min-retrieval-ablation-target-type-coverage",
            "asset=1.0",
            "--min-retrieval-ablation-source-target-coverage",
            "bm25=1.0",
            "--min-retrieval-ablation-source-family-target-coverage",
            "lexical=1.0",
            "--min-retrieval-ablation-source-precision-at-hits",
            "bm25=1.0",
            "--min-retrieval-ablation-source-family-precision-at-hits",
            "lexical=1.0",
            "--min-retrieval-ablation-case-group-target-coverage",
            "case_source:visual_lexical_probe=1.0",
            "--min-retrieval-ablation-case-group-source-target-coverage",
            "case_source:visual_lexical_probe:bm25=1.0",
            "--min-retrieval-ablation-case-group-source-family-target-coverage",
            "case_source:visual_lexical_probe:lexical=1.0",
            "--min-retrieval-ablation-case-group-source-precision-at-hits",
            "case_source:visual_lexical_probe:bm25=1.0",
            "--min-retrieval-ablation-case-group-source-family-precision-at-hits",
            "case_source:visual_lexical_probe:lexical=1.0",
            "--require-retrieval-ablation-best-by-recall",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "retrieval_ablation_gate"
    )
    assert component["metadata"]["mode"] == "bm25_visual"
    assert component["metadata"]["baseline_mode"] == "bm25_text"
    assert component["metadata"]["metrics"]["recall_at_k"] == 1.0
    assert component["metadata"]["baseline_metrics"]["recall_at_k"] == 0.0
    assert component["metadata"]["source_metrics"]["bm25"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["source_metrics"]["bm25"]["precision_at_hits"] == 1.0
    assert component["metadata"]["case_group_metrics"]["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ] == 1.0
    assert component["metadata"]["case_group_source_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["bm25"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["case_group_source_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["bm25"]["precision_at_hits"] == 1.0
    assert component["metadata"]["case_group_source_family_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["lexical"]["target_coverage_at_k"] == 1.0
    assert component["metadata"]["case_group_source_family_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["lexical"]["precision_at_hits"] == 1.0


def test_ingestion_readiness_cli_can_gate_chunking_target_coverage(tmp_path):
    package_dir, _ = write_ready_package(tmp_path)
    comparison_path = tmp_path / "chunking_comparison.json"
    output = tmp_path / "readiness.json"
    comparison_path.write_text(
        chunking_comparison().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingestion-readiness",
            "--package-dir",
            str(package_dir),
            "--chunking-comparison",
            str(comparison_path),
            "--chunking-candidate",
            "candidate",
            "--baseline-chunking-candidate",
            "baseline",
            "--min-chunking-recall-at-k",
            "0.8",
            "--max-chunking-mean-target-rank",
            "2.0",
            "--max-chunking-total-chunk-chars",
            "2000",
            "--min-chunking-visual-text-coverage-ratio",
            "0.8",
            "--min-chunking-retrieval-score-per-mean-latency-ms",
            "0.1",
            "--min-chunking-target-coverage-per-p95-latency-ms",
            "0.1",
            "--min-chunking-target-type-coverage",
            "asset=0.8",
            "--min-chunking-target-type-coverage",
            "triple=0.8",
            "--min-chunking-source-target-coverage",
            "bm25=0.8",
            "--min-chunking-source-family-target-coverage",
            "lexical=0.8",
            "--min-chunking-case-group-source-target-coverage",
            "case_source:visual_lexical_probe:bm25=0.8",
            "--min-chunking-case-group-source-family-target-coverage",
            "case_source:visual_lexical_probe:lexical=0.8",
            "--max-chunking-pairwise-mean-target-rank-delta",
            "0.0",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    component = next(
        component
        for component in payload["components"]
        if component["name"] == "chunking_comparison_gate"
    )
    assert component["metadata"]["candidate"] == "candidate"
    assert component["metadata"]["metrics"]["retrieval_mean_target_rank"] == 1.0
    assert component["metadata"]["metrics"]["retrieval_score_per_mean_latency_ms"] == 0.18
    assert component["metadata"]["metrics"]["target_coverage_per_p95_latency_ms"] == (
        0.9 / 7.0
    )
    assert component["metadata"]["pairwise_metrics"]["pairwise_mean_target_rank_delta"] == -1.0
    assert component["metadata"]["metrics"]["visual_text_coverage_ratio"] == 0.9
    assert component["metadata"]["metrics"]["target_type.asset.coverage_at_k"] == 0.9
    assert component["metadata"]["metrics"]["target_type.triple.coverage_at_k"] == 0.9
    assert component["metadata"]["metrics"]["source.bm25.target_coverage_at_k"] == 0.9
    assert component["metadata"]["metrics"]["source_family.lexical.target_coverage_at_k"] == 0.9
    assert component["metadata"]["target_metrics"]["asset"]["coverage_at_k"] == 0.9
    assert component["metadata"]["source_metrics"]["bm25"]["target_coverage_at_k"] == 0.9
    assert component["metadata"]["source_family_metrics"]["lexical"]["target_coverage_at_k"] == 0.9
    assert component["metadata"]["case_group_source_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["bm25"]["target_coverage_at_k"] == 0.9
    assert component["metadata"]["case_group_source_family_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["lexical"]["target_coverage_at_k"] == 0.9


def write_ready_package(tmp_path: Path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    source_path = tmp_path / "reference.pdf"
    source_content = b"%PDF-1.4\nreference document\n"
    source_path.write_bytes(source_content)
    doc = SourceDocument(
        doc_id="doc",
        title="Reference Document",
        local_path=source_path,
    )
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=100,
            height=100,
            char_count=120,
            line_count=4,
            text_block_count=1,
            image_block_count=1,
            embedded_image_count=0,
            drawing_count=0,
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
            text="reference retrieval evidence",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            path=package_dir / "assets/page.png",
            caption="reference visual evidence",
            metadata={"requires_ocr": False, "requires_vlm": False},
        )
    ]
    manifest = ProcessingManifest(
        doc=doc,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        metadata={
            "source_file": {
                "name": source_path.name,
                "bytes": len(source_content),
                "sha256": hashlib.sha256(source_content).hexdigest(),
            },
            "package_config": {
                "base_chunking_strategy": "page",
                "render_zoom": 2.0,
                "dry_run_embeddings": True,
                "section_map_count": 0,
                "extract_tables": True,
                "lexical_tokenizer": LexicalTokenizerConfig().model_dump(),
            },
        },
    )
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", profiles)
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", [])
    write_bm25_manifest(package_dir, chunks, assets)
    (package_dir / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "document_chunks",
                "named_vectors": {"text_dense": {"size": 2, "distance": "Cosine"}},
                "payload_indexes": [
                    {"field": "doc_id", "schema": "keyword"},
                    {"field": "chunk_id", "schema": "keyword"},
                    {"field": "asset_id", "schema": "keyword"},
                    {"field": "kind", "schema": "keyword"},
                    {"field": "page_no", "schema": "integer"},
                    {"field": "page_start", "schema": "integer"},
                    {"field": "page_end", "schema": "integer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        package_dir / "qdrant_text_records.jsonl",
        [
            EmbeddingRecord(
                point_id="00000000-0000-0000-0000-000000000001",
                chunk_id="chunk-1",
                doc_id="doc",
                vector_name="text_dense",
                vector=[0.1, 0.2],
                payload={
                    "chunk_id": "chunk-1",
                    "doc_id": "doc",
                    "page_start": 1,
                    "page_end": 1,
                    "kind": "text",
                    "text": "reference retrieval evidence",
                },
            )
        ],
    )
    record_content = (package_dir / "qdrant_text_records.jsonl").read_bytes()
    (package_dir / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": "document_chunks",
                "vectors": {
                    "text_dense": {
                        "file": "qdrant_text_records.jsonl",
                        "record_count": 1,
                        "dimension": 2,
                        "distance": "Cosine",
                        "exists": True,
                        "bytes": len(record_content),
                        "sha256": hashlib.sha256(record_content).hexdigest(),
                    }
                },
                "payload_indexes": [{"field": "doc_id", "schema": "keyword"}],
            }
        ),
        encoding="utf-8",
    )
    return package_dir, manifest


def write_bm25_manifest(
    package_dir: Path,
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
) -> None:
    bm25 = BM25LexicalIndex(
        chunks,
        texts=chunk_lexical_texts(chunks, assets),
    )
    bm25.dump_manifest(package_dir / "bm25_tokens.json")


def chunking_comparison():
    return ChunkingComparison(
        rows=[
            chunking_row("candidate", quality_score=0.9, recall=0.9, mean_target_rank=1.0),
            chunking_row("baseline", quality_score=0.85, recall=0.88, mean_target_rank=2.0),
        ],
        best_by_quality="candidate",
        best_by_retrieval="candidate",
        fastest_by_mean_latency="candidate",
        pairwise=[
            ChunkingPairwiseComparison(
                candidate="candidate",
                baseline="baseline",
                shared_query_count=2,
                candidate_win_count=2,
                baseline_win_count=0,
                candidate_win_rate=1.0,
                baseline_win_rate=0.0,
                mean_target_rank_delta=-1.0,
                target_rank_delta_ci_high=-1.0,
            )
        ],
    )


def chunking_row(name: str, quality_score: float, recall: float, mean_target_rank: float):
    retrieval_score = recall
    total_chunk_chars = 1000.0
    embedding_text_kchars = total_chunk_chars / 1000.0
    mean_latency_ms = 5.0
    p95_latency_ms = 7.0
    return ChunkingComparisonRow(
        name=name,
        chunk_count=1,
        total_chunk_chars=total_chunk_chars,
        mean_chunk_chars=total_chunk_chars,
        p95_chunk_chars=total_chunk_chars,
        embedding_text_kchars=embedding_text_kchars,
        quality_score=quality_score,
        retrieval_score=retrieval_score,
        retrieval_score_per_embedding_kchar=(
            retrieval_score / embedding_text_kchars
        ),
        target_coverage_per_embedding_kchar=recall / embedding_text_kchars,
        target_ndcg_per_embedding_kchar=recall / embedding_text_kchars,
        retrieval_score_per_mean_latency_ms=retrieval_score / mean_latency_ms,
        target_coverage_per_mean_latency_ms=recall / mean_latency_ms,
        target_ndcg_per_mean_latency_ms=recall / mean_latency_ms,
        retrieval_score_per_p95_latency_ms=retrieval_score / p95_latency_ms,
        target_coverage_per_p95_latency_ms=recall / p95_latency_ms,
        target_ndcg_per_p95_latency_ms=recall / p95_latency_ms,
        retrieval_hit_rate=recall,
        retrieval_recall_at_k=recall,
        retrieval_mrr=recall,
        retrieval_target_coverage_at_k=recall,
        retrieval_mean_target_ndcg_at_k=recall,
        retrieval_mean_precision_at_k=recall,
        retrieval_mean_latency_ms=mean_latency_ms,
        retrieval_p95_latency_ms=p95_latency_ms,
        retrieval_mean_first_relevant_rank=mean_target_rank,
        retrieval_p95_first_relevant_rank=mean_target_rank,
        retrieval_mean_target_rank=mean_target_rank,
        retrieval_p95_target_rank=mean_target_rank,
        retrieval_ranked_expected_case_count=2.0,
        retrieval_ranked_target_count=2.0,
        target_metrics={
            "asset": {"coverage_at_k": recall},
            "triple": {"coverage_at_k": recall},
        },
        source_metrics={"bm25": {"target_coverage_at_k": recall}},
        source_family_metrics={"lexical": {"target_coverage_at_k": recall}},
        case_group_metrics={
            "case_source": {"visual_lexical_probe": {"target_coverage_at_k": recall}}
        },
        case_group_source_metrics={
            "case_source": {
                "visual_lexical_probe": {
                    "bm25": {"target_coverage_at_k": recall}
                }
            }
        },
        case_group_source_family_metrics={
            "case_source": {
                "visual_lexical_probe": {
                    "lexical": {"target_coverage_at_k": recall}
                }
            }
        },
        failed_queries=[],
        page_coverage_ratio=1.0,
        visual_annotation_ratio=1.0,
        visual_text_asset_count=10,
        visual_text_covered_asset_count=round(10 * recall),
        visual_text_coverage_ratio=recall,
        visual_text_part_count=20,
        visual_text_covered_part_count=round(20 * recall),
        visual_text_part_coverage_ratio=recall,
        chunks_under_min_chars=0,
        chunks_over_max_chars=0,
        issue_codes=[],
    )


def retrieval_ablation_report():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="reference retrieval evidence",
        asset_ids=["asset-1"],
    )
    cases = [
        RetrievalCase(
            query="reference evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_lexical_probe"},
        )
    ]
    passing_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=chunk,
                score=0.9,
                sources=["bm25"],
            )
        ],
        top_k=5,
    )
    failing_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [],
        top_k=5,
    )
    return RetrievalAblationReport(
        rows=[
            RetrievalAblationRow(
                mode=RetrievalAblationMode(
                    name="bm25_visual",
                    use_dense=False,
                    use_bm25=True,
                    include_asset_text=True,
                ),
                evaluation=passing_evaluation,
            ),
            RetrievalAblationRow(
                mode=RetrievalAblationMode(
                    name="bm25_text",
                    use_dense=False,
                    use_bm25=True,
                    include_asset_text=False,
                ),
                evaluation=failing_evaluation,
            ),
        ],
        best_by_recall="bm25_visual",
        best_by_target_coverage="bm25_visual",
        best_by_target_ndcg="bm25_visual",
        best_by_mrr="bm25_visual",
        fastest_by_mean_latency="bm25_text",
        pairwise=[
            AblationPairwiseComparison(
                candidate="bm25_visual",
                baseline="bm25_text",
                shared_query_count=1,
                candidate_win_count=1,
                candidate_win_rate=1.0,
                mean_target_rank_delta=-5.0,
            )
        ],
    )


def qdrant_vector_ablation_report():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="reference retrieval evidence",
        asset_ids=["asset-1"],
    )
    cases = [
        RetrievalCase(
            query="reference evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe"},
        )
    ]
    passing_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=chunk,
                score=0.9,
                sources=["qdrant:text_dense"],
            )
        ],
        top_k=5,
    )
    failing_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [],
        top_k=5,
    )
    return QdrantVectorAblationReport(
        rows=[
            QdrantVectorAblationRow(
                mode=QdrantVectorAblationMode(name="text", vector_names=["text_dense"]),
                evaluation=passing_evaluation,
            ),
            QdrantVectorAblationRow(
                mode=QdrantVectorAblationMode(
                    name="caption",
                    vector_names=["caption_dense"],
                ),
                evaluation=failing_evaluation,
            ),
        ],
        best_by_recall="text",
        best_by_target_coverage="text",
        best_by_target_ndcg="text",
        best_by_mrr="text",
        fastest_by_mean_latency="caption",
        pairwise=[
            AblationPairwiseComparison(
                candidate="text",
                baseline="caption",
                shared_query_count=1,
                candidate_win_count=1,
                candidate_win_rate=1.0,
                mean_target_rank_delta=-5.0,
            )
        ],
    )


def qdrant_reranker_ablation_report():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="reference retrieval evidence",
        asset_ids=["asset-1"],
    )
    cases = [
        RetrievalCase(
            query="reference evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe"},
        )
    ]
    lexical_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=chunk,
                score=0.9,
                sources=["bm25", "rerank:lexical"],
            )
        ],
        top_k=5,
    )
    none_evaluation = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [],
        top_k=5,
    )
    return build_qdrant_reranker_ablation_report(
        [
            QdrantRerankerAblationRow(
                mode=QdrantRerankerAblationMode(
                    name="lexical",
                    reranker="lexical",
                    rerank_top_k=20,
                ),
                evaluation=lexical_evaluation,
            ),
            QdrantRerankerAblationRow(
                mode=QdrantRerankerAblationMode(name="none"),
                evaluation=none_evaluation,
            ),
        ]
    )


def ready_runtime_report():
    return RuntimeReport(
        passed=True,
        gpus=[
            GPUDevice(
                name="NVIDIA GeForce RTX 5090",
                memory_total_mib=32607,
                driver_version="580.159.03",
            )
        ],
        torch_cuda_available=True,
        torch_cuda_device_count=1,
        torch_cuda_device_names=["NVIDIA GeForce RTX 5090"],
        torch_cuda_compute_capabilities=["12.0"],
        torch_cuda_version="13.0",
        torch_cuda_compiled_arches=["sm_120"],
        torch_bfloat16_supported=True,
        checks=[
            RuntimeCheck(
                name="gpu_available",
                passed=True,
                message="At least one NVIDIA GPU is visible through nvidia-smi.",
            ),
            RuntimeCheck(
                name="torch_cuda_arch:12.0",
                passed=True,
                message="Torch CUDA build includes an architecture target for the visible GPU.",
            ),
        ],
    )


def visual_run_results(
    parse_status: str,
    triple: bool,
    job_id: str = "job-1",
) -> list[VisualJobRunResult]:
    triples = (
        [{"subject": "subject", "predicate": "relates_to", "object": "object"}]
        if triple
        else []
    )
    return [
        VisualJobRunResult(
            job_id=job_id,
            asset_id="asset-1",
            page_no=1,
            status="completed",
            annotation=AssetAnnotation(
                asset_id="asset-1",
                page_no=1,
                ocr_text="recognized visual text",
                vlm_summary="structured visual evidence" if triple else "plain visual evidence",
                triples=triples,
                metadata={"vlm_parse_status": parse_status},
            ),
            metadata={
                "operations": ["ocr", "vlm"],
                "ocr_text_chars": 22,
                "vlm_parse_status": parse_status,
                "total_duration_ms": 40.0 if triple else 20.0,
            },
        )
    ]


def readiness_retrieval_evaluation(target_coverage: float) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        case_count=2,
        expected_case_count=2,
        passed_count=2,
        failed_count=0,
        hit_rate=1.0,
        recall_at_k=0.8,
        mrr=0.7,
        target_coverage_at_k=target_coverage,
        mean_target_ndcg_at_k=0.75,
        mean_precision_at_k=0.6,
        top_k=5,
        failed_queries=[],
        results=[],
    )


def readiness_qdrant_retrieval_config(package_dir: Path) -> QdrantRetrievalConfig:
    return QdrantRetrievalConfig(
        collection_name="document_chunks",
        package_dir=str(package_dir),
        bm25_tokens_path="bm25_tokens.json",
        vector_names=["text_dense"],
        graph_expand=False,
        fusion_weights={"bm25": 1.0, "qdrant:text_dense": 1.0},
        top_k=5,
        collapse_hierarchical=False,
        query_encoders={"text_dense": "default_text"},
        lexical_tokenizer=LexicalTokenizerConfig().model_dump(),
        selection=QdrantRetrievalConfigSelection(
            candidate="bm25_1__qdrant_text_dense_1",
            source="global_recommended",
            candidate_rank=1,
            candidate_eligible=True,
        ),
    )


def qdrant_config_metadata(
    config: QdrantRetrievalConfig,
    backend: str,
) -> dict[str, object]:
    return {
        "backend": backend,
        "config_selection": config.selection.model_dump(),
        "collection": config.collection_name,
        "vector_names": config.vector_names,
        "graph_expand": config.graph_expand,
        "query_encoders": config.query_encoders,
        "fusion_weights": config.fusion_weights,
        "collapse_hierarchical": config.collapse_hierarchical,
        "lexical_tokenizer": config.lexical_tokenizer,
    }


def readiness_rag_context_evaluation(target_coverage: float = 1.0) -> RAGContextEvaluation:
    return RAGContextEvaluation(
        case_count=2,
        expected_case_count=2,
        passed_count=2 if target_coverage >= 1.0 else 1,
        failed_count=0 if target_coverage >= 1.0 else 1,
        hit_rate=1.0 if target_coverage >= 1.0 else 0.5,
        target_coverage=target_coverage,
        excluded_query_count=1,
        excluded_hit_query_count=0,
        excluded_query_hit_rate=0.0,
        excluded_target_count=1,
        excluded_matched_target_count=0,
        excluded_target_hit_rate=0.0,
        mean_latency_ms=25.0,
        mean_context_char_count=4200.0,
        max_context_char_count=6500,
        mean_chunk_count=4.0,
        mean_asset_count=1.0,
        mean_triple_count=2.0,
        target_metrics={
            "asset": RAGContextTargetMetric(
                expected_count=2,
                passed_count=2 if target_coverage >= 1.0 else 1,
                target_count=2,
                matched_target_count=2 if target_coverage >= 1.0 else 1,
                coverage=target_coverage,
            )
        },
        case_group_metrics={
            "case_source": {
                "visual_object_probe": RAGContextCaseGroupMetric(
                    case_count=2,
                    expected_case_count=2,
                    passed_count=2 if target_coverage >= 1.0 else 1,
                    failed_count=0 if target_coverage >= 1.0 else 1,
                    target_count=2,
                    matched_target_count=2 if target_coverage >= 1.0 else 1,
                    target_coverage=target_coverage,
                    excluded_target_count=1,
                    excluded_matched_target_count=0,
                    excluded_target_hit_rate=0.0,
                    mean_latency_ms=25.0,
                    mean_context_char_count=4200.0,
                )
            }
        },
        failed_queries=[] if target_coverage >= 1.0 else ["missing visual object"],
        results=[],
    )
