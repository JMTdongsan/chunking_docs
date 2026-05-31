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
    comparison: ChunkingComparison | None = None


DEFAULT_ARTIFACTS = [
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
    "graph_nodes.jsonl",
    "graph_edges.jsonl",
    "visual_job_summary.json",
    "chunking_sweep.json",
    "qdrant_retrieval_eval.json",
    "qdrant_vector_ablation.json",
    "retrieval_ablation.json",
    "rag_context.json",
    "rag_context.qdrant.json",
]


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
        artifacts=package_artifact_summaries(package_dir, candidates or {}),
        comparison=comparison,
    )


def package_artifact_summaries(package_dir: Path, candidates: dict[str, Path]) -> list[ArtifactSummary]:
    paths = [package_dir / name for name in DEFAULT_ARTIFACTS]
    for path in candidates.values():
        if path not in paths:
            paths.append(path)
    return [artifact_summary(path, root=package_dir) for path in paths]


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}
