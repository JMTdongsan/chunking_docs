from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.io import read_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, TextQuality, VisualAsset
from chunking_docs.storage.records import EmbeddingRecord


class AuditIssue(BaseModel):
    severity: str
    code: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PackageAudit(BaseModel):
    page_count: int
    chunk_count: int
    asset_count: int
    triple_count: int
    text_quality_counts: dict[str, int]
    asset_kind_counts: dict[str, int]
    annotated_asset_count: int
    chunks_with_assets: int
    chunks_with_visual_annotations: int
    pages_requiring_ocr: list[int]
    pages_requiring_vlm: list[int]
    qdrant_collection: dict[str, Any] = Field(default_factory=dict)
    qdrant_record_counts: dict[str, int] = Field(default_factory=dict)
    qdrant_vector_sizes: dict[str, int] = Field(default_factory=dict)
    issues: list[AuditIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def audit_package(
    profiles: list[PageProfile],
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    require_annotations_for_visual_pages: bool = False,
    package_dir: Path | None = None,
    require_qdrant_records: bool = False,
) -> PackageAudit:
    issues: list[AuditIssue] = []
    profile_pages = {profile.page_no for profile in profiles}
    chunk_pages = {page for chunk in chunks for page in range(chunk.page_start, chunk.page_end + 1)}
    asset_pages = {asset.page_no for asset in assets}

    missing_chunk_pages = sorted(profile_pages - chunk_pages)
    if missing_chunk_pages:
        issues.append(
            AuditIssue(
                severity="error",
                code="missing_chunk_pages",
                message="Some pages do not have any chunk.",
                metadata={"pages": missing_chunk_pages[:50], "count": len(missing_chunk_pages)},
            )
        )

    missing_asset_pages = sorted(profile_pages - asset_pages)
    if missing_asset_pages:
        issues.append(
            AuditIssue(
                severity="warning",
                code="missing_asset_pages",
                message="Some pages do not have a rendered asset.",
                metadata={"pages": missing_asset_pages[:50], "count": len(missing_asset_pages)},
            )
        )

    chunk_ids = {chunk.chunk_id for chunk in chunks}
    orphan_triples = [triple.triple_id for triple in triples if triple.chunk_id not in chunk_ids]
    if orphan_triples:
        issues.append(
            AuditIssue(
                severity="error",
                code="orphan_triples",
                message="Some triples point to missing chunks.",
                metadata={"triple_ids": orphan_triples[:50], "count": len(orphan_triples)},
            )
        )

    pages_requiring_ocr = sorted(
        asset.page_no for asset in assets if asset.metadata.get("requires_ocr") and not asset.ocr_text
    )
    pages_requiring_vlm = sorted(
        asset.page_no for asset in assets if asset.metadata.get("requires_vlm") and not asset.vlm_summary
    )
    if require_annotations_for_visual_pages and pages_requiring_vlm:
        issues.append(
            AuditIssue(
                severity="error",
                code="missing_vlm_annotations",
                message="Some visual pages still require VLM annotations.",
                metadata={"pages": pages_requiring_vlm[:50], "count": len(pages_requiring_vlm)},
            )
        )

    text_quality_counts = Counter(profile.text_quality for profile in profiles)
    asset_kind_counts = Counter(asset.kind for asset in assets)
    annotated_asset_count = sum(1 for asset in assets if asset.ocr_text or asset.vlm_summary)
    chunks_with_assets = sum(1 for chunk in chunks if chunk.asset_ids)
    chunks_with_visual_annotations = sum(
        1 for chunk in chunks if chunk.metadata.get("has_visual_annotations")
    )
    qdrant_collection, qdrant_record_counts, qdrant_vector_sizes = audit_qdrant_artifacts(
        package_dir=package_dir,
        issues=issues,
        require_qdrant_records=require_qdrant_records,
    )

    return PackageAudit(
        page_count=len(profiles),
        chunk_count=len(chunks),
        asset_count=len(assets),
        triple_count=len(triples),
        text_quality_counts={str(key): value for key, value in text_quality_counts.items()},
        asset_kind_counts={str(key): value for key, value in asset_kind_counts.items()},
        annotated_asset_count=annotated_asset_count,
        chunks_with_assets=chunks_with_assets,
        chunks_with_visual_annotations=chunks_with_visual_annotations,
        pages_requiring_ocr=pages_requiring_ocr,
        pages_requiring_vlm=pages_requiring_vlm,
        qdrant_collection=qdrant_collection,
        qdrant_record_counts=qdrant_record_counts,
        qdrant_vector_sizes=qdrant_vector_sizes,
        issues=issues,
    )


def audit_qdrant_artifacts(
    package_dir: Path | None,
    issues: list[AuditIssue],
    require_qdrant_records: bool = False,
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    if package_dir is None:
        return {}, {}, {}
    collection_path = package_dir / "qdrant_collection.json"
    embedding_manifest_path = package_dir / "embedding_manifest.json"
    record_files = sorted(package_dir.glob("qdrant_*_records.jsonl"))
    if not collection_path.exists():
        if record_files or require_qdrant_records:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="missing_qdrant_collection",
                    message="Qdrant record files require qdrant_collection.json.",
                    metadata={"record_files": [path.name for path in record_files]},
                )
            )
        return {}, {}, {}

    collection_config = json.loads(collection_path.read_text(encoding="utf-8"))
    if record_files and not embedding_manifest_path.exists():
        issues.append(
            AuditIssue(
                severity="warning",
                code="missing_embedding_manifest",
                message="Qdrant record files should include embedding_manifest.json for reproducibility.",
                metadata={"record_files": [path.name for path in record_files]},
            )
        )
    named_vectors = {
        name: int(config["size"])
        for name, config in collection_config.get("named_vectors", {}).items()
    }
    payload_indexes = normalize_payload_indexes(collection_config.get("payload_indexes", []))
    missing_payload_indexes = sorted(required_payload_indexes() - set(payload_indexes))
    if missing_payload_indexes:
        issues.append(
            AuditIssue(
                severity="warning",
                code="missing_qdrant_payload_indexes",
                message="Some recommended Qdrant payload indexes are missing.",
                metadata={"fields": missing_payload_indexes},
            )
        )

    record_counts: dict[str, int] = {}
    observed_sizes: dict[str, int] = {}
    if require_qdrant_records:
        missing_vectors = [
            vector_name
            for vector_name in named_vectors
            if not (package_dir / qdrant_record_filename(vector_name)).exists()
        ]
        if missing_vectors:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="missing_qdrant_records",
                    message="Some configured Qdrant vectors do not have record files.",
                    metadata={"vector_names": missing_vectors},
                )
            )

    for record_file in record_files:
        records = read_jsonl(record_file, EmbeddingRecord)
        for record in records:
            record_counts[record.vector_name] = record_counts.get(record.vector_name, 0) + 1
            observed_sizes.setdefault(record.vector_name, len(record.vector))
            validate_qdrant_record(record, record_file.name, named_vectors, issues)

    return collection_config, record_counts, observed_sizes


def validate_qdrant_record(
    record: EmbeddingRecord,
    filename: str,
    named_vectors: dict[str, int],
    issues: list[AuditIssue],
) -> None:
    expected_size = named_vectors.get(record.vector_name)
    if expected_size is None:
        issues.append(
            AuditIssue(
                severity="error",
                code="qdrant_unknown_vector",
                message="A Qdrant record uses a vector name that is not in qdrant_collection.json.",
                metadata={"file": filename, "point_id": record.point_id, "vector_name": record.vector_name},
            )
        )
    elif len(record.vector) != expected_size:
        issues.append(
            AuditIssue(
                severity="error",
                code="qdrant_vector_size_mismatch",
                message="A Qdrant record vector length does not match qdrant_collection.json.",
                metadata={
                    "file": filename,
                    "point_id": record.point_id,
                    "vector_name": record.vector_name,
                    "expected_size": expected_size,
                    "actual_size": len(record.vector),
                },
            )
        )
    missing_payload = [
        field
        for field in required_payload_fields(record.vector_name)
        if record.payload.get(field) in {None, ""}
    ]
    if missing_payload:
        issues.append(
            AuditIssue(
                severity="error",
                code="qdrant_missing_payload",
                message="A Qdrant record is missing required payload fields.",
                metadata={
                    "file": filename,
                    "point_id": record.point_id,
                    "vector_name": record.vector_name,
                    "fields": missing_payload,
                },
            )
        )


def normalize_payload_indexes(payload_indexes: list[str | dict[str, Any]]) -> set[str]:
    fields: set[str] = set()
    for index in payload_indexes:
        if isinstance(index, str):
            fields.add(index)
        else:
            field = index.get("field") or index.get("field_name")
            if field:
                fields.add(str(field))
    return fields


def required_payload_indexes() -> set[str]:
    return {"doc_id", "chunk_id", "asset_id", "kind", "page_no", "page_start", "page_end"}


def qdrant_record_filename(vector_name: str) -> str:
    return f"qdrant_{vector_name.removesuffix('_dense')}_records.jsonl"


def required_payload_fields(vector_name: str) -> set[str]:
    if vector_name == "text_dense":
        return {"chunk_id", "doc_id", "page_start", "page_end", "kind", "text"}
    if vector_name == "caption_dense":
        return {"asset_id", "doc_id", "page_no", "kind", "text"}
    if vector_name == "image_dense":
        return {"asset_id", "doc_id", "page_no", "kind"}
    return {"doc_id"}


def degraded_page_ratio(profiles: list[PageProfile]) -> float:
    if not profiles:
        return 0.0
    degraded = sum(
        1
        for profile in profiles
        if profile.text_quality in {TextQuality.DEGRADED, TextQuality.EMPTY}
    )
    return degraded / len(profiles)
