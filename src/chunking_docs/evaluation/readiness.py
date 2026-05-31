from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.evaluation.audit import PackageAudit, audit_package
from chunking_docs.evaluation.case_audit import RetrievalCaseAuditReport, audit_retrieval_cases
from chunking_docs.evaluation.chunking_gate import (
    ChunkingComparisonGateReport,
    gate_chunking_comparison,
)
from chunking_docs.evaluation.compare import ChunkingComparison
from chunking_docs.evaluation.gate import RetrievalGateReport, gate_retrieval_evaluation
from chunking_docs.evaluation.retrieval import RetrievalCase, RetrievalEvaluation
from chunking_docs.models import ProcessingManifest
from chunking_docs.storage.postgres_store import manifest_rows
from chunking_docs.vision.jobs import VisualJobRunResult
from chunking_docs.vision.quality import (
    VisualQualityReport,
    evaluate_visual_results,
    visual_results_from_assets,
)


class ReadinessComponent(BaseModel):
    name: str
    passed: bool
    severity: str = "error"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionReadinessReport(BaseModel):
    package_dir: str
    passed: bool
    package_counts: dict[str, int]
    artifact_presence: dict[str, bool]
    postgres_row_counts: dict[str, int] = Field(default_factory=dict)
    audit: PackageAudit
    visual_quality: VisualQualityReport | None = None
    retrieval_case_audit: RetrievalCaseAuditReport | None = None
    retrieval_gate: RetrievalGateReport | None = None
    chunking_comparison_gate: ChunkingComparisonGateReport | None = None
    components: list[ReadinessComponent] = Field(default_factory=list)
    failed_components: list[str] = Field(default_factory=list)


def build_ingestion_readiness_report(
    package_dir: Path,
    manifest: ProcessingManifest,
    require_qdrant_records: bool = True,
    require_bm25: bool = True,
    require_embedding_manifest: bool = True,
    require_postgres_rows: bool = True,
    require_visual_annotations: bool = False,
    visual_results: list[VisualJobRunResult] | None = None,
    require_visual_quality: bool = False,
    visual_quality_options: dict[str, Any] | None = None,
    retrieval_cases: list[RetrievalCase] | None = None,
    require_retrieval_cases: bool = False,
    retrieval_case_options: dict[str, Any] | None = None,
    retrieval_evaluation: RetrievalEvaluation | None = None,
    require_retrieval_evaluation: bool = False,
    retrieval_gate_options: dict[str, Any] | None = None,
    chunking_comparison: ChunkingComparison | None = None,
    require_chunking_comparison: bool = False,
    chunking_gate_options: dict[str, Any] | None = None,
) -> IngestionReadinessReport:
    artifact_presence = package_artifact_presence(package_dir)
    audit = audit_package(
        manifest.profiles,
        manifest.chunks,
        manifest.assets,
        manifest.triples,
        require_annotations_for_visual_pages=require_visual_annotations,
        package_dir=package_dir,
        require_qdrant_records=require_qdrant_records,
    )
    components = [
        ReadinessComponent(
            name="package_audit",
            passed=audit.passed,
            message="Package structure, visual coverage, triples, and Qdrant records are valid.",
            metadata={"issue_codes": [issue.code for issue in audit.issues]},
        )
    ]

    if require_bm25:
        components.append(required_artifact_component("bm25_tokens", artifact_presence, "bm25_tokens.json"))
    if require_embedding_manifest:
        components.append(
            required_artifact_component(
                "embedding_manifest",
                artifact_presence,
                "embedding_manifest.json",
            )
        )

    postgres_row_counts: dict[str, int] = {}
    if require_postgres_rows:
        postgres_component, postgres_row_counts = postgres_rows_component(manifest, package_dir)
        components.append(postgres_component)

    visual_quality = None
    if visual_results is not None or require_visual_quality:
        visual_quality_source = "visual_results"
        evaluated_visual_results = visual_results
        if evaluated_visual_results is None:
            visual_quality_source = "assets"
            evaluated_visual_results = visual_results_from_assets(manifest.assets)
        visual_quality = evaluate_visual_results(
            evaluated_visual_results,
            **(visual_quality_options or {}),
        )
        components.append(
            ReadinessComponent(
                name="visual_quality",
                passed=visual_quality.passed,
                message="Visual OCR/VLM annotations meet configured quality thresholds.",
                metadata={
                    "source": visual_quality_source,
                    "failed_checks": visual_quality.failed_checks,
                    "completion_rate": visual_quality.completion_rate,
                    "ocr_text_coverage": visual_quality.ocr_text_coverage,
                    "vlm_summary_coverage": visual_quality.vlm_summary_coverage,
                    "vlm_json_parse_rate": visual_quality.vlm_json_parse_rate,
                },
            )
        )

    retrieval_case_audit = None
    if retrieval_cases is not None:
        retrieval_case_audit = audit_retrieval_cases(
            retrieval_cases,
            profiles=manifest.profiles,
            chunks=manifest.chunks,
            assets=manifest.assets,
            triples=manifest.triples,
            **(retrieval_case_options or {}),
        )
        components.append(
            ReadinessComponent(
                name="retrieval_case_audit",
                passed=retrieval_case_audit.passed,
                message="Retrieval benchmark cases are valid for the package and configured target coverage.",
                metadata={
                    "failed_checks": retrieval_case_audit.failed_checks,
                    "target_counts": retrieval_case_audit.target_counts,
                    "missing_target_counts": retrieval_case_audit.missing_target_counts,
                },
            )
        )
    elif require_retrieval_cases:
        components.append(
            ReadinessComponent(
                name="retrieval_case_audit",
                passed=False,
                message="Retrieval benchmark cases are required but were not supplied.",
            )
        )

    retrieval_gate = None
    if retrieval_evaluation is not None:
        retrieval_gate = gate_retrieval_evaluation(
            retrieval_evaluation,
            **(retrieval_gate_options or {}),
        )
        components.append(
            ReadinessComponent(
                name="retrieval_gate",
                passed=retrieval_gate.passed,
                message="Retrieval evaluation meets configured quality thresholds.",
                metadata={
                    "failed_checks": retrieval_gate.failed_checks,
                    "metrics": retrieval_gate.metrics,
                },
            )
        )
    elif require_retrieval_evaluation:
        components.append(
            ReadinessComponent(
                name="retrieval_gate",
                passed=False,
                message="Retrieval evaluation is required but was not supplied.",
            )
        )

    chunking_comparison_gate = None
    if chunking_comparison is not None:
        chunking_comparison_gate = gate_chunking_comparison(
            chunking_comparison,
            **(chunking_gate_options or {}),
        )
        components.append(
            ReadinessComponent(
                name="chunking_comparison_gate",
                passed=chunking_comparison_gate.passed,
                message="Selected chunking candidate meets configured quality, retrieval, and latency thresholds.",
                metadata={
                    "candidate": chunking_comparison_gate.candidate,
                    "baseline_candidate": chunking_comparison_gate.baseline_candidate,
                    "failed_checks": chunking_comparison_gate.failed_checks,
                    "metrics": chunking_comparison_gate.metrics,
                },
            )
        )
    elif require_chunking_comparison:
        components.append(
            ReadinessComponent(
                name="chunking_comparison_gate",
                passed=False,
                message="Chunking comparison gate is required but no comparison was supplied.",
            )
        )

    failed_components = [
        component.name
        for component in components
        if not component.passed and component.severity == "error"
    ]
    return IngestionReadinessReport(
        package_dir=str(package_dir),
        passed=not failed_components,
        package_counts={
            "pages": len(manifest.profiles),
            "chunks": len(manifest.chunks),
            "assets": len(manifest.assets),
            "triples": len(manifest.triples),
        },
        artifact_presence=artifact_presence,
        postgres_row_counts=postgres_row_counts,
        audit=audit,
        visual_quality=visual_quality,
        retrieval_case_audit=retrieval_case_audit,
        retrieval_gate=retrieval_gate,
        chunking_comparison_gate=chunking_comparison_gate,
        components=components,
        failed_components=failed_components,
    )


def package_artifact_presence(package_dir: Path) -> dict[str, bool]:
    names = [
        "manifest.json",
        "pages.jsonl",
        "chunks.jsonl",
        "assets.jsonl",
        "triples.jsonl",
        "bm25_tokens.json",
        "embedding_manifest.json",
        "qdrant_collection.json",
        "qdrant_text_records.jsonl",
        "qdrant_caption_records.jsonl",
        "qdrant_image_records.jsonl",
    ]
    return {name: (package_dir / name).exists() for name in names}


def required_artifact_component(
    name: str,
    artifact_presence: dict[str, bool],
    filename: str,
) -> ReadinessComponent:
    return ReadinessComponent(
        name=name,
        passed=artifact_presence.get(filename, False),
        message=f"Required package artifact exists: {filename}.",
        metadata={"file": filename},
    )


def postgres_rows_component(
    manifest: ProcessingManifest,
    package_dir: Path,
) -> tuple[ReadinessComponent, dict[str, int]]:
    try:
        rows = manifest_rows(manifest, base_dir=package_dir)
    except Exception as exc:
        return (
            ReadinessComponent(
                name="postgres_rows",
                passed=False,
                message="PostgreSQL row conversion failed.",
                metadata={"error": str(exc)},
            ),
            {},
        )
    counts = {
        "documents": 1,
        "pages": len(rows["pages"]),
        "chunks": len(rows["chunks"]),
        "assets": len(rows["assets"]),
        "triples": len(rows["triples"]),
        "embedding_artifacts": len(rows["embedding_artifacts"]),
    }
    return (
        ReadinessComponent(
            name="postgres_rows",
            passed=counts["chunks"] > 0 and counts["pages"] > 0,
            message="Package can be converted into PostgreSQL metadata rows.",
            metadata={"row_counts": counts},
        ),
        counts,
    )
