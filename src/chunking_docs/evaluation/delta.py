from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.io import read_jsonl
from chunking_docs.models import ProcessingManifest
from chunking_docs.storage.records import EmbeddingRecord


class PackageDeltaObservation(BaseModel):
    code: str
    severity: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PackageDeltaReport(BaseModel):
    before_dir: str
    after_dir: str
    before_counts: dict[str, int]
    after_counts: dict[str, int]
    count_delta: dict[str, int]
    added_ids: dict[str, list[str]] = Field(default_factory=dict)
    removed_ids: dict[str, list[str]] = Field(default_factory=dict)
    changed_ids: dict[str, list[str]] = Field(default_factory=dict)
    qdrant_record_counts_before: dict[str, int] = Field(default_factory=dict)
    qdrant_record_counts_after: dict[str, int] = Field(default_factory=dict)
    qdrant_record_count_delta: dict[str, int] = Field(default_factory=dict)
    observations: list[PackageDeltaObservation] = Field(default_factory=list)


def compare_processing_packages(
    before: ProcessingManifest,
    after: ProcessingManifest,
    before_dir: Path,
    after_dir: Path,
    max_ids: int = 50,
) -> PackageDeltaReport:
    before_counts = package_counts(before)
    after_counts = package_counts(after)
    before_records = qdrant_record_counts(before_dir)
    after_records = qdrant_record_counts(after_dir)
    report = PackageDeltaReport(
        before_dir=str(before_dir),
        after_dir=str(after_dir),
        before_counts=before_counts,
        after_counts=after_counts,
        count_delta=delta_counts(before_counts, after_counts),
        added_ids={
            "chunks": sorted_ids(chunk_ids(after) - chunk_ids(before), max_ids),
            "assets": sorted_ids(asset_ids(after) - asset_ids(before), max_ids),
            "triples": sorted_ids(triple_ids(after) - triple_ids(before), max_ids),
        },
        removed_ids={
            "chunks": sorted_ids(chunk_ids(before) - chunk_ids(after), max_ids),
            "assets": sorted_ids(asset_ids(before) - asset_ids(after), max_ids),
            "triples": sorted_ids(triple_ids(before) - triple_ids(after), max_ids),
        },
        changed_ids={
            "chunks": changed_model_ids(
                {chunk.chunk_id: chunk.model_dump(mode="json") for chunk in before.chunks},
                {chunk.chunk_id: chunk.model_dump(mode="json") for chunk in after.chunks},
                max_ids,
            ),
            "assets": changed_model_ids(
                {
                    asset.asset_id: asset.model_dump(mode="json", exclude={"path"})
                    for asset in before.assets
                },
                {
                    asset.asset_id: asset.model_dump(mode="json", exclude={"path"})
                    for asset in after.assets
                },
                max_ids,
            ),
            "triples": changed_model_ids(
                {triple.triple_id: triple.model_dump(mode="json") for triple in before.triples},
                {triple.triple_id: triple.model_dump(mode="json") for triple in after.triples},
                max_ids,
            ),
        },
        qdrant_record_counts_before=before_records,
        qdrant_record_counts_after=after_records,
        qdrant_record_count_delta=delta_counts(before_records, after_records),
    )
    report.observations = delta_observations(report)
    return report


def package_counts(manifest: ProcessingManifest) -> dict[str, int]:
    return {
        "pages": len(manifest.profiles),
        "chunks": len(manifest.chunks),
        "assets": len(manifest.assets),
        "triples": len(manifest.triples),
        "annotated_assets": sum(1 for asset in manifest.assets if asset.ocr_text or asset.vlm_summary),
        "chunks_with_visual_annotations": sum(
            1 for chunk in manifest.chunks if chunk.metadata.get("has_visual_annotations")
        ),
        "visual_triples": sum(
            1 for triple in manifest.triples if triple.qualifiers.get("source") == "visual_annotation"
        ),
    }


def delta_counts(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def qdrant_record_counts(package_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(package_dir.glob("qdrant_*_records.jsonl")):
        records = read_jsonl(path, EmbeddingRecord)
        for record in records:
            counts[record.vector_name] = counts.get(record.vector_name, 0) + 1
    return counts


def chunk_ids(manifest: ProcessingManifest) -> set[str]:
    return {chunk.chunk_id for chunk in manifest.chunks}


def asset_ids(manifest: ProcessingManifest) -> set[str]:
    return {asset.asset_id for asset in manifest.assets}


def triple_ids(manifest: ProcessingManifest) -> set[str]:
    return {triple.triple_id for triple in manifest.triples}


def sorted_ids(values: set[str], max_ids: int) -> list[str]:
    return sorted(values)[:max_ids]


def changed_model_ids(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    max_ids: int,
) -> list[str]:
    shared_ids = set(before) & set(after)
    return sorted(identifier for identifier in shared_ids if before[identifier] != after[identifier])[:max_ids]


def delta_observations(report: PackageDeltaReport) -> list[PackageDeltaObservation]:
    observations = []
    if report.count_delta.get("annotated_assets", 0) > 0:
        observations.append(
            PackageDeltaObservation(
                code="annotations_added",
                severity="info",
                message="The after package contains more OCR/VLM-annotated assets.",
                metadata={"delta": report.count_delta["annotated_assets"]},
            )
        )
    if report.count_delta.get("chunks_with_visual_annotations", 0) > 0:
        observations.append(
            PackageDeltaObservation(
                code="chunks_enriched",
                severity="info",
                message="The after package has more chunks enriched with visual annotation text.",
                metadata={"delta": report.count_delta["chunks_with_visual_annotations"]},
            )
        )
    if report.count_delta.get("visual_triples", 0) > 0:
        observations.append(
            PackageDeltaObservation(
                code="visual_triples_added",
                severity="info",
                message="The after package contains more graph triples from visual annotations.",
                metadata={"delta": report.count_delta["visual_triples"]},
            )
        )
    if any(value > 0 for value in report.qdrant_record_count_delta.values()):
        observations.append(
            PackageDeltaObservation(
                code="qdrant_records_added",
                severity="info",
                message="The after package contains more Qdrant embedding records for at least one vector.",
                metadata={"delta": report.qdrant_record_count_delta},
            )
        )
    if any(value < 0 for value in report.qdrant_record_count_delta.values()):
        observations.append(
            PackageDeltaObservation(
                code="qdrant_records_removed",
                severity="warning",
                message="Some Qdrant record counts decreased between packages.",
                metadata={"delta": report.qdrant_record_count_delta},
            )
        )
    return observations
