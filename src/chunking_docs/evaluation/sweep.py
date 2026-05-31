from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.chunking.multimodal import ChunkStrategy, build_strategy_chunks
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.chunking_quality import ChunkingQualityReport, evaluate_chunking_quality
from chunking_docs.evaluation.compare import ChunkingComparison, compare_chunking_reports
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.io import write_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, VisualAsset


class ChunkingSweepCandidate(BaseModel):
    name: str
    strategy: str
    config: dict[str, Any] = Field(default_factory=dict)
    chunks_file: str | None = None
    chunk_count: int
    report: ChunkingQualityReport


class ChunkingSweepReport(BaseModel):
    generated_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    candidates: list[ChunkingSweepCandidate]
    comparison: ChunkingComparison


def run_chunking_sweep(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    profiles: list[PageProfile],
    triples: list[GraphTriple],
    strategies: list[ChunkStrategy],
    max_chars_values: list[int],
    overlap_chars_values: list[int],
    min_chars: int = 180,
    parent_max_chars_values: list[int] | None = None,
    visual_context_chars_values: list[int] | None = None,
    retrieval_cases: list[RetrievalCase] | None = None,
    top_k: int = 5,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = True,
    retrieval_repeat: int = 1,
    output_dir: Path | None = None,
    write_candidates: bool = True,
) -> ChunkingSweepReport:
    reports: dict[str, ChunkingQualityReport] = {}
    candidates: list[ChunkingSweepCandidate] = []

    for config in sweep_configs(
        strategies=strategies,
        max_chars_values=max_chars_values,
        overlap_chars_values=overlap_chars_values,
        min_chars=min_chars,
        parent_max_chars_values=parent_max_chars_values or [900],
        visual_context_chars_values=visual_context_chars_values or [700],
    ):
        candidate_chunks = build_strategy_chunks(chunks, assets, **config)
        name = candidate_name(config)
        chunks_file = None
        if output_dir is not None and write_candidates:
            output_path = output_dir / f"chunks.{name}.jsonl"
            write_jsonl(output_path, candidate_chunks)
            chunks_file = str(output_path)

        report = evaluate_chunking_quality(
            chunks=candidate_chunks,
            profiles=profiles,
            assets=assets,
            triples=triples,
            retrieval_cases=retrieval_cases,
            top_k=top_k,
            min_chars=min_chars,
            max_chars=int(config["max_chars"]),
            tokenizer_config=tokenizer_config,
            collapse_hierarchical=collapse_hierarchical,
            retrieval_repeat=retrieval_repeat,
        )
        reports[name] = report
        candidates.append(
            ChunkingSweepCandidate(
                name=name,
                strategy=str(config["strategy"]),
                config=serializable_config(config),
                chunks_file=chunks_file,
                chunk_count=len(candidate_chunks),
                report=report,
            )
        )

    comparison = compare_chunking_reports(reports)
    candidates.sort(
        key=lambda candidate: (
            candidate.report.retrieval.recall_at_k if candidate.report.retrieval else -1.0,
            candidate.report.retrieval.mrr if candidate.report.retrieval else -1.0,
            candidate.report.quality_score,
        ),
        reverse=True,
    )
    return ChunkingSweepReport(
        generated_at=datetime.now(UTC).isoformat(),
        config={
            "top_k": top_k,
            "min_chars": min_chars,
            "collapse_hierarchical": collapse_hierarchical,
            "retrieval_repeat": retrieval_repeat,
            "tokenizer": tokenizer_config.model_dump() if tokenizer_config else None,
        },
        candidates=candidates,
        comparison=comparison,
    )


def sweep_configs(
    strategies: list[ChunkStrategy],
    max_chars_values: list[int],
    overlap_chars_values: list[int],
    min_chars: int,
    parent_max_chars_values: list[int],
    visual_context_chars_values: list[int],
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    seen = set()
    for strategy in strategies:
        if strategy == "page":
            config = {
                "strategy": strategy,
                "max_chars": max_chars_values[0],
                "overlap_chars": overlap_chars_values[0],
                "min_chars": min_chars,
            }
            add_unique_config(configs, seen, config)
            continue
        for max_chars in max_chars_values:
            for overlap_chars in overlap_chars_values:
                base = {
                    "strategy": strategy,
                    "max_chars": max_chars,
                    "overlap_chars": overlap_chars,
                    "min_chars": min_chars,
                }
                if strategy != "hierarchical":
                    add_unique_config(configs, seen, base)
                    continue
                for parent_max_chars in parent_max_chars_values:
                    for visual_context_chars in visual_context_chars_values:
                        add_unique_config(
                            configs,
                            seen,
                            {
                                **base,
                                "parent_max_chars": parent_max_chars,
                                "visual_context_chars": visual_context_chars,
                            },
                        )
    return configs


def add_unique_config(configs: list[dict[str, Any]], seen: set[tuple[tuple[str, Any], ...]], config):
    key = tuple(sorted(config.items()))
    if key in seen:
        return
    seen.add(key)
    configs.append(config)


def candidate_name(config: dict[str, Any]) -> str:
    strategy = str(config["strategy"])
    parts = [
        strategy,
        f"max{config['max_chars']}",
        f"ov{config['overlap_chars']}",
        f"min{config['min_chars']}",
    ]
    if strategy == "hierarchical":
        parts.extend(
            [
                f"parent{config.get('parent_max_chars')}",
                f"visual{config.get('visual_context_chars')}",
            ]
        )
    return safe_name("-".join(parts))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if key == "strategy" else value for key, value in config.items()}
