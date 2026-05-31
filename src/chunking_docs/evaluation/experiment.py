from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.compare import ChunkingComparison, compare_chunking_reports
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.io import read_jsonl
from chunking_docs.models import DocumentChunk, ProcessingManifest


class ArtifactSummary(BaseModel):
    path: str
    exists: bool
    bytes: int = 0
    sha256: str | None = None
    record_count: int | None = None


class ValidationSummary(BaseModel):
    path: str
    kind: str
    passed: bool | None = None
    failed_checks: list[str] = Field(default_factory=list)
    failed_components: list[str] = Field(default_factory=list)
    candidate: str | None = None
    mode: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentReport(BaseModel):
    generated_at: str
    package_dir: str
    doc_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    candidate_files: dict[str, str] = Field(default_factory=dict)
    package_counts: dict[str, int] = Field(default_factory=dict)
    profile_summary: dict[str, Any] = Field(default_factory=dict)
    qdrant_collection: dict[str, Any] = Field(default_factory=dict)
    bm25_tokenizer: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    validation_summaries: list[ValidationSummary] = Field(default_factory=list)
    comparison: ChunkingComparison | None = None


DEFAULT_ARTIFACTS = [
    "manifest.json",
    "pages.jsonl",
    "chunks.jsonl",
    "assets.jsonl",
    "triples.jsonl",
    "bm25_tokens.json",
    "embedding_manifest.json",
    "postgres_schema_contract.json",
    "qdrant_collection.json",
    "qdrant_collection_contract.json",
    "qdrant_text_records.jsonl",
    "qdrant_caption_records.jsonl",
    "qdrant_image_records.jsonl",
    "graph_nodes.jsonl",
    "graph_edges.jsonl",
    "visual_job_summary.json",
    "visual_run_comparison.json",
    "visual_quality.json",
    "document_characteristics.json",
    "ingestion_readiness.json",
    "chunking_comparison.json",
    "chunking_comparison_gate.json",
    "chunking_sweep.json",
    "qdrant_retrieval_eval.json",
    "qdrant_vector_ablation.json",
    "retrieval_case_audit.json",
    "retrieval_gate.json",
    "retrieval_ablation.json",
    "rag_context.json",
    "rag_context.qdrant.json",
]

DEFAULT_ARTIFACT_GLOBS = [
    "ingestion_readiness*.json",
    "chunking_comparison*.json",
    "chunking_gate*.json",
    "retrieval_gate*.json",
    "qdrant_retrieval_eval*.json",
    "qdrant_vector_ablation*.json",
    "qdrant_vector_ablation_gate*.json",
]

SUMMARY_METRIC_KEYS = {
    "recall_at_k",
    "retrieval_recall_at_k",
    "target_coverage_at_k",
    "retrieval_target_coverage_at_k",
    "mean_target_ndcg_at_k",
    "retrieval_mean_target_ndcg_at_k",
    "mean_precision_at_k",
    "retrieval_mean_precision_at_k",
    "failed_query_count",
    "target_type.asset.coverage_at_k",
    "target_type.triple.coverage_at_k",
    "source_family.lexical.target_coverage_at_k",
    "source_family.visual.target_coverage_at_k",
    "source_family.graph.target_coverage_at_k",
}


def build_experiment_report(
    package_dir: Path,
    manifest: ProcessingManifest,
    candidates: dict[str, Path] | None = None,
    retrieval_cases: list[RetrievalCase] | None = None,
    top_k: int = 5,
    min_chars: int = 120,
    max_chars: int = 1800,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = False,
    retrieval_repeat: int = 1,
    fusion_weights: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> ExperimentReport:
    comparison = None
    artifact_paths = package_artifact_paths(package_dir, candidates or {})
    if candidates:
        reports = {}
        for name, path in candidates.items():
            chunks = read_jsonl(path, DocumentChunk)
            reports[name] = evaluate_chunking_quality(
                chunks=chunks,
                profiles=manifest.profiles,
                assets=manifest.assets,
                triples=manifest.triples,
                retrieval_cases=retrieval_cases,
                top_k=top_k,
                min_chars=min_chars,
                max_chars=max_chars,
                tokenizer_config=tokenizer_config,
                collapse_hierarchical=collapse_hierarchical,
                retrieval_repeat=retrieval_repeat,
                fusion_weights=fusion_weights,
            )
        comparison = compare_chunking_reports(reports)

    return ExperimentReport(
        generated_at=datetime.now(UTC).isoformat(),
        package_dir=str(package_dir),
        doc_id=manifest.doc.doc_id,
        config=config or {},
        candidate_files={name: str(path) for name, path in (candidates or {}).items()},
        package_counts={
            "pages": len(manifest.profiles),
            "chunks": len(manifest.chunks),
            "assets": len(manifest.assets),
            "triples": len(manifest.triples),
        },
        profile_summary=manifest.metadata.get("profile_summary", {}),
        qdrant_collection=read_json(package_dir / "qdrant_collection.json"),
        bm25_tokenizer=read_json(package_dir / "bm25_tokens.json").get("tokenizer", {}),
        artifacts=[artifact_summary(path, root=package_dir) for path in artifact_paths],
        validation_summaries=validation_artifact_summaries(package_dir, artifact_paths),
        comparison=comparison,
    )


def package_artifact_paths(package_dir: Path, candidates: dict[str, Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for name in DEFAULT_ARTIFACTS:
        append_unique_path(paths, seen, package_dir / name)
    for pattern in DEFAULT_ARTIFACT_GLOBS:
        for path in sorted(package_dir.glob(pattern)):
            append_unique_path(paths, seen, path)
    for path in candidates.values():
        append_unique_path(paths, seen, path)
    return paths


def package_artifact_summaries(package_dir: Path, candidates: dict[str, Path]) -> list[ArtifactSummary]:
    paths = package_artifact_paths(package_dir, candidates)
    return [artifact_summary(path, root=package_dir) for path in paths]


def append_unique_path(paths: list[Path], seen: set[Path], path: Path) -> None:
    normalized = path.resolve() if path.exists() else path
    if normalized in seen:
        return
    seen.add(normalized)
    paths.append(path)


def artifact_summary(path: Path, root: Path | None = None) -> ArtifactSummary:
    display_path = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
    if not path.exists():
        return ArtifactSummary(path=display_path, exists=False)
    content = path.read_bytes()
    return ArtifactSummary(
        path=display_path,
        exists=True,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        record_count=count_records(path),
    )


def count_records(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return None


def validation_artifact_summaries(package_dir: Path, paths: list[Path]) -> list[ValidationSummary]:
    summaries = []
    for path in paths:
        summary = validation_artifact_summary(path, root=package_dir)
        if summary is not None:
            summaries.append(summary)
    return summaries


def validation_artifact_summary(path: Path, root: Path | None = None) -> ValidationSummary | None:
    if not path.exists() or path.suffix != ".json":
        return None
    payload = read_json(path)
    if not is_validation_payload(payload):
        return None
    return ValidationSummary(
        path=str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path),
        kind=validation_kind(path),
        passed=payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
        failed_checks=string_list(payload.get("failed_checks")),
        failed_components=string_list(payload.get("failed_components")),
        candidate=payload.get("candidate") if isinstance(payload.get("candidate"), str) else None,
        mode=payload.get("mode") if isinstance(payload.get("mode"), str) else None,
        metrics=summary_metrics(payload.get("metrics")),
    )


def is_validation_payload(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("passed", "failed_checks", "failed_components", "metrics"))


def validation_kind(path: Path) -> str:
    name = path.name
    for prefix in (
        "ingestion_readiness",
        "chunking_gate",
        "retrieval_gate",
        "qdrant_vector_ablation_gate",
        "visual_gate",
        "visual_quality",
    ):
        if name.startswith(prefix):
            return prefix
    return path.stem.split(".", 1)[0]


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def summary_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        key: float(metric)
        for key, metric in value.items()
        if key in SUMMARY_METRIC_KEYS and isinstance(metric, int | float)
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}
