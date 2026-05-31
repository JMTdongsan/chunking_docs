from __future__ import annotations

import hashlib
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
    embedding_manifest: dict[str, Any] = Field(default_factory=dict)
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
        asset.page_no for asset in assets if asset.metadata.get("requires_ocr") and asset_requires_ocr(asset)
    )
    pages_requiring_vlm = sorted(
        asset.page_no for asset in assets if asset.metadata.get("requires_vlm") and asset_requires_vlm(asset)
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
    qdrant_collection, embedding_manifest, qdrant_record_counts, qdrant_vector_sizes = audit_qdrant_artifacts(
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
        embedding_manifest=embedding_manifest,
        qdrant_record_counts=qdrant_record_counts,
        qdrant_vector_sizes=qdrant_vector_sizes,
        issues=issues,
    )


def asset_requires_vlm(asset: VisualAsset) -> bool:
    if not asset.vlm_summary:
        return True
    parse_status = str(asset.metadata.get("vlm_parse_status", "")).strip()
    return parse_status not in {"json_object", "json_list", "json_repaired"}


def asset_requires_ocr(asset: VisualAsset) -> bool:
    if asset.ocr_text:
        return False
    return "ocr_text_chars" not in asset.metadata and not asset.metadata.get("ocr_backend")


def audit_qdrant_artifacts(
    package_dir: Path | None,
    issues: list[AuditIssue],
    require_qdrant_records: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], dict[str, int]]:
    if package_dir is None:
        return {}, {}, {}, {}
    collection_path = package_dir / "qdrant_collection.json"
    embedding_manifest_path = package_dir / "embedding_manifest.json"
    record_files = sorted(package_dir.glob("qdrant_*_records.jsonl"))
    if not collection_path.exists():
        if record_files or require_qdrant_records or embedding_manifest_path.exists():
            issues.append(
                AuditIssue(
                    severity="error",
                    code="missing_qdrant_collection",
                    message="Qdrant record files require qdrant_collection.json.",
                    metadata={"record_files": [path.name for path in record_files]},
                )
            )
        return {}, {}, {}, {}

    collection_config = json.loads(collection_path.read_text(encoding="utf-8"))
    embedding_manifest = {}
    if record_files and not embedding_manifest_path.exists():
        issues.append(
            AuditIssue(
                severity="warning",
                code="missing_embedding_manifest",
                message="Qdrant record files should include embedding_manifest.json for reproducibility.",
                metadata={"record_files": [path.name for path in record_files]},
            )
        )
    elif embedding_manifest_path.exists():
        try:
            embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="invalid_embedding_manifest",
                    message="embedding_manifest.json is not valid JSON.",
                    metadata={"error": str(exc)},
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

    if embedding_manifest:
        validate_embedding_manifest(
            embedding_manifest=embedding_manifest,
            package_dir=package_dir,
            collection_config=collection_config,
            record_counts=record_counts,
            observed_sizes=observed_sizes,
            issues=issues,
        )

    return collection_config, embedding_manifest, record_counts, observed_sizes


def validate_embedding_manifest(
    embedding_manifest: dict[str, Any],
    package_dir: Path,
    collection_config: dict[str, Any],
    record_counts: dict[str, int],
    observed_sizes: dict[str, int],
    issues: list[AuditIssue],
) -> None:
    manifest_collection = embedding_manifest.get("collection")
    expected_collection = collection_config.get("collection")
    if manifest_collection and expected_collection and manifest_collection != expected_collection:
        issues.append(
            AuditIssue(
                severity="error",
                code="embedding_manifest_collection_mismatch",
                message="embedding_manifest.json collection does not match qdrant_collection.json.",
                metadata={"expected": expected_collection, "actual": manifest_collection},
            )
        )

    vectors = embedding_manifest.get("vectors")
    if not isinstance(vectors, dict):
        issues.append(
            AuditIssue(
                severity="error",
                code="invalid_embedding_manifest_vectors",
                message="embedding_manifest.json must contain a vectors object.",
            )
        )
        return

    named_vectors = collection_config.get("named_vectors", {})
    missing_manifest_vectors = sorted(set(record_counts) - set(vectors))
    if missing_manifest_vectors:
        issues.append(
            AuditIssue(
                severity="error",
                code="embedding_manifest_missing_vectors",
                message="Some Qdrant record vectors are missing from embedding_manifest.json.",
                metadata={"vector_names": missing_manifest_vectors},
            )
        )

    for vector_name, vector_payload in sorted(vectors.items()):
        if not isinstance(vector_payload, dict):
            issues.append(
                AuditIssue(
                    severity="error",
                    code="invalid_embedding_manifest_vector",
                    message="Each embedding manifest vector entry must be an object.",
                    metadata={"vector_name": vector_name},
                )
            )
            continue
        validate_embedding_manifest_vector(
            vector_name=str(vector_name),
            vector_payload=vector_payload,
            package_dir=package_dir,
            named_vectors=named_vectors,
            record_counts=record_counts,
            observed_sizes=observed_sizes,
            issues=issues,
        )


def validate_embedding_manifest_vector(
    vector_name: str,
    vector_payload: dict[str, Any],
    package_dir: Path,
    named_vectors: dict[str, Any],
    record_counts: dict[str, int],
    observed_sizes: dict[str, int],
    issues: list[AuditIssue],
) -> None:
    if vector_name not in named_vectors:
        issues.append(
            AuditIssue(
                severity="error",
                code="embedding_manifest_unknown_vector",
                message="embedding_manifest.json includes a vector not configured in qdrant_collection.json.",
                metadata={"vector_name": vector_name},
            )
        )

    record_file_name = vector_payload.get("file")
    if not record_file_name:
        issues.append(
            AuditIssue(
                severity="error",
                code="embedding_manifest_missing_file",
                message="An embedding manifest vector entry is missing the record file name.",
                metadata={"vector_name": vector_name},
            )
        )
        return
    record_file = package_dir / str(record_file_name)
    if not record_file.exists():
        issues.append(
            AuditIssue(
                severity="error",
                code="embedding_manifest_file_missing",
                message="An embedding manifest vector file does not exist.",
                metadata={"vector_name": vector_name, "file": str(record_file_name)},
            )
        )
        return

    actual_summary = file_summary(record_file)
    compare_manifest_value(
        issues,
        code="embedding_manifest_record_count_mismatch",
        message="embedding_manifest.json record_count does not match the record file.",
        vector_name=vector_name,
        field="record_count",
        expected=record_counts.get(vector_name, 0),
        actual=vector_payload.get("record_count"),
    )
    expected_dimension = vector_config_size(named_vectors.get(vector_name))
    if expected_dimension is not None:
        compare_manifest_value(
            issues,
            code="embedding_manifest_dimension_mismatch",
            message="embedding_manifest.json dimension does not match qdrant_collection.json.",
            vector_name=vector_name,
            field="dimension",
            expected=expected_dimension,
            actual=vector_payload.get("dimension"),
        )
    observed_dimension = observed_sizes.get(vector_name)
    if observed_dimension is not None and vector_payload.get("dimension") != observed_dimension:
        issues.append(
            AuditIssue(
                severity="error",
                code="embedding_manifest_observed_dimension_mismatch",
                message="embedding_manifest.json dimension does not match the observed record vectors.",
                metadata={
                    "vector_name": vector_name,
                    "expected": observed_dimension,
                    "actual": vector_payload.get("dimension"),
                },
            )
        )
    compare_manifest_value(
        issues,
        code="embedding_manifest_bytes_mismatch",
        message="embedding_manifest.json byte count does not match the record file.",
        vector_name=vector_name,
        field="bytes",
        expected=actual_summary["bytes"],
        actual=vector_payload.get("bytes"),
    )
    compare_manifest_value(
        issues,
        code="embedding_manifest_sha256_mismatch",
        message="embedding_manifest.json sha256 does not match the record file.",
        vector_name=vector_name,
        field="sha256",
        expected=actual_summary["sha256"],
        actual=vector_payload.get("sha256"),
    )


def compare_manifest_value(
    issues: list[AuditIssue],
    code: str,
    message: str,
    vector_name: str,
    field: str,
    expected: Any,
    actual: Any,
) -> None:
    if actual != expected:
        issues.append(
            AuditIssue(
                severity="error",
                code=code,
                message=message,
                metadata={"vector_name": vector_name, "field": field, "expected": expected, "actual": actual},
            )
        )


def vector_config_size(config: Any) -> int | None:
    if isinstance(config, dict) and "size" in config:
        return int(config["size"])
    return None


def file_summary(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


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
