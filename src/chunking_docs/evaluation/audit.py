from __future__ import annotations

import hashlib
import json
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.embeddings.records import asset_text, triple_text
from chunking_docs.graph.provenance import (
    chunk_asset_ids,
    chunk_id_alias_map,
    chunk_ids_by_asset_id,
    triple_asset_ids,
    triple_resolved_chunk_ids,
)
from chunking_docs.io import read_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, TextQuality, VisualAsset
from chunking_docs.storage.qdrant_config import qdrant_payload_index_fields
from chunking_docs.storage.records import EmbeddingRecord
from chunking_docs.vision.vlm_output import visual_triples_from_payload


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


class PublicArtifactAudit(BaseModel):
    root: str
    scanned_file_count: int
    skipped_file_count: int
    forbidden_match_count: int
    blocked_extension_count: int
    large_file_count: int
    issues: list[AuditIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


DEFAULT_PUBLIC_AUDIT_EXCLUDES = [
    ".git/**",
    ".venv/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "__pycache__/**",
    "dist/**",
    "*.egg-info/**",
    "data/raw/**",
    "data/analysis/**",
    "outputs/**",
]

DEFAULT_BLOCKED_PUBLIC_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".bin",
    ".parquet",
    ".sqlite",
    ".db",
}

DEFAULT_REQUIRED_GITIGNORE_PATTERNS = [
    "data/raw/*.pdf",
    "outputs/",
]


def audit_public_artifacts(
    root: Path,
    forbidden_patterns: list[str] | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    blocked_extensions: list[str] | None = None,
    max_file_bytes: int = 2_000_000,
    max_text_scan_bytes: int = 512_000,
    required_gitignore_patterns: list[str] | None = None,
) -> PublicArtifactAudit:
    root = root.resolve()
    issues: list[AuditIssue] = []
    include_globs = include_globs or []
    exclude_globs = [*DEFAULT_PUBLIC_AUDIT_EXCLUDES, *(exclude_globs or [])]
    blocked_extensions = sorted(
        normalize_extension(value)
        for value in (blocked_extensions or sorted(DEFAULT_BLOCKED_PUBLIC_EXTENSIONS))
        if value
    )
    forbidden_terms = [pattern.casefold() for pattern in forbidden_patterns or [] if pattern]
    scanned_file_count = 0
    skipped_file_count = 0
    forbidden_match_count = 0
    blocked_extension_count = 0
    large_file_count = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = public_relative_path(path, root)
        if should_skip_public_audit_path(relative_path, include_globs, exclude_globs):
            skipped_file_count += 1
            continue
        scanned_file_count += 1
        size = path.stat().st_size
        suffix = path.suffix.lower()
        if suffix in blocked_extensions:
            blocked_extension_count += 1
            issues.append(
                AuditIssue(
                    severity="error",
                    code="blocked_public_extension",
                    message="A public repository file uses a blocked artifact extension.",
                    metadata={"path": relative_path, "extension": suffix, "bytes": size},
                )
            )
        if size > max_file_bytes:
            large_file_count += 1
            issues.append(
                AuditIssue(
                    severity="error",
                    code="large_public_file",
                    message="A public repository file is larger than the configured limit.",
                    metadata={"path": relative_path, "bytes": size, "max_file_bytes": max_file_bytes},
                )
            )
        if forbidden_terms and size <= max_text_scan_bytes:
            matches = scan_forbidden_terms(path, relative_path, forbidden_terms)
            forbidden_match_count += len(matches)
            issues.extend(matches)

    required_patterns = required_gitignore_patterns or DEFAULT_REQUIRED_GITIGNORE_PATTERNS
    issues.extend(required_gitignore_checks(root, required_patterns))

    return PublicArtifactAudit(
        root=str(root),
        scanned_file_count=scanned_file_count,
        skipped_file_count=skipped_file_count,
        forbidden_match_count=forbidden_match_count,
        blocked_extension_count=blocked_extension_count,
        large_file_count=large_file_count,
        issues=issues,
    )


def public_relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_skip_public_audit_path(
    relative_path: str,
    include_globs: list[str],
    exclude_globs: list[str],
) -> bool:
    if include_globs and not matches_any_public_glob(relative_path, include_globs):
        return True
    return matches_any_public_glob(relative_path, exclude_globs)


def matches_any_public_glob(relative_path: str, patterns: list[str]) -> bool:
    return any(matches_public_glob(relative_path, pattern) for pattern in patterns)


def matches_public_glob(relative_path: str, pattern: str) -> bool:
    normalized = pattern.strip().replace("\\", "/")
    if not normalized:
        return False
    if normalized.endswith("/"):
        normalized = f"{normalized}**"
    if normalized.endswith("/**"):
        prefix = normalized[:-3].rstrip("/")
        return relative_path == prefix or relative_path.startswith(f"{prefix}/")
    return fnmatch(relative_path, normalized) or fnmatch(Path(relative_path).name, normalized)


def normalize_extension(value: str) -> str:
    stripped = value.strip().lower()
    if not stripped:
        return ""
    return stripped if stripped.startswith(".") else f".{stripped}"


def scan_forbidden_terms(
    path: Path,
    relative_path: str,
    forbidden_terms: list[str],
) -> list[AuditIssue]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
    except OSError:
        return []
    issues = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.casefold()
        matches = sorted({term for term in forbidden_terms if term in lowered})
        if not matches:
            continue
        issues.append(
            AuditIssue(
                severity="error",
                code="forbidden_public_text",
                message="A public repository file contains a forbidden text pattern.",
                metadata={
                    "path": relative_path,
                    "line": line_number,
                    "patterns": matches,
                    "sample": line.strip()[:160],
                },
            )
        )
    return issues


def required_gitignore_checks(root: Path, patterns: list[str]) -> list[AuditIssue]:
    if not patterns:
        return []
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return [
            AuditIssue(
                severity="error",
                code="missing_gitignore",
                message="A public repository should include .gitignore for generated artifacts.",
                metadata={"required_patterns": patterns},
            )
        ]
    lines = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return [
        AuditIssue(
            severity="error",
            code="missing_gitignore_pattern",
            message="A generated-artifact ignore pattern is missing from .gitignore.",
            metadata={"pattern": pattern},
        )
        for pattern in patterns
        if pattern not in lines
    ]


def audit_package(
    profiles: list[PageProfile],
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    require_annotations_for_visual_pages: bool = False,
    package_dir: Path | None = None,
    require_qdrant_records: bool = False,
    require_visual_derived_triples: bool = False,
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
    chunk_id_by_alias = chunk_id_alias_map(chunks)
    chunk_ids_by_asset = chunk_ids_by_asset_id(chunks)
    orphan_triples = [
        triple.triple_id
        for triple in triples
        if not triple_resolved_chunk_ids(triple, chunk_ids, chunk_id_by_alias, chunk_ids_by_asset)
    ]
    if orphan_triples:
        issues.append(
            AuditIssue(
                severity="error",
                code="orphan_triples",
                message="Some triples point to missing chunks.",
                metadata={"triple_ids": orphan_triples[:50], "count": len(orphan_triples)},
            )
        )

    validate_visual_derived_triple_coverage(
        assets=assets,
        triples=triples,
        issues=issues,
        require_visual_derived_triples=require_visual_derived_triples,
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
    chunks_with_assets = sum(1 for chunk in chunks if chunk_asset_ids(chunk))
    chunks_with_visual_annotations = sum(
        1 for chunk in chunks if chunk.metadata.get("has_visual_annotations")
    )
    qdrant_collection, embedding_manifest, qdrant_record_counts, qdrant_vector_sizes = audit_qdrant_artifacts(
        package_dir=package_dir,
        issues=issues,
        require_qdrant_records=require_qdrant_records,
    )
    if package_dir is not None and require_qdrant_records and qdrant_collection:
        validate_qdrant_target_coverage(
            chunks=chunks,
            assets=assets,
            triples=triples,
            package_dir=package_dir,
            collection_config=qdrant_collection,
            issues=issues,
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


def validate_visual_derived_triple_coverage(
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    issues: list[AuditIssue],
    require_visual_derived_triples: bool = False,
) -> None:
    existing_keys_by_asset = triple_keys_by_asset_id(triples)
    missing_assets = []
    for asset in assets:
        expected_keys = {
            normalized_graph_triple_key(triple.get("subject"), triple.get("predicate"), triple.get("object"))
            for triple in visual_triples_from_payload(visual_triple_payload(asset))
            if triple.get("derived_from_vlm_field")
        }
        expected_keys.discard(("", "", ""))
        if not expected_keys:
            continue
        missing_keys = sorted(expected_keys - existing_keys_by_asset.get(asset.asset_id, set()))
        if not missing_keys:
            continue
        missing_assets.append(
            {
                "asset_id": asset.asset_id,
                "page_no": asset.page_no,
                "missing_triple_count": len(missing_keys),
                "missing_triples": [format_graph_triple_key(key) for key in missing_keys[:10]],
            }
        )

    if missing_assets:
        issues.append(
            AuditIssue(
                severity="error" if require_visual_derived_triples else "warning",
                code="missing_visual_derived_triples",
                message=(
                    "Some structured VLM metadata is not represented by graph triples with visual asset provenance."
                ),
                metadata={
                    "asset_count": len(missing_assets),
                    "assets": missing_assets[:50],
                    "requires_visual_derived_triples": require_visual_derived_triples,
                },
            )
        )


def triple_keys_by_asset_id(triples: list[GraphTriple]) -> dict[str, set[tuple[str, str, str]]]:
    indexed: dict[str, set[tuple[str, str, str]]] = {}
    for triple in triples:
        key = normalized_graph_triple_key(triple.subject, triple.predicate, triple.object)
        if key == ("", "", ""):
            continue
        for asset_id in sorted(triple_asset_ids(triple)):
            indexed.setdefault(asset_id, set()).add(key)
    return indexed


def visual_triple_payload(asset: VisualAsset) -> dict[str, Any]:
    payload = dict(asset.metadata)
    if asset.caption and not any(payload.get(key) for key in ("title", "caption", "name")):
        payload["caption"] = asset.caption
    return payload


def normalized_graph_triple_key(subject: Any, predicate: Any, object_: Any) -> tuple[str, str, str]:
    return (
        str(subject or "").strip().casefold(),
        str(predicate or "").strip().casefold(),
        str(object_ or "").strip().casefold(),
    )


def format_graph_triple_key(key: tuple[str, str, str]) -> dict[str, str]:
    subject, predicate, object_ = key
    return {"subject": subject, "predicate": predicate, "object": object_}


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


def validate_qdrant_target_coverage(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    package_dir: Path,
    collection_config: dict[str, Any],
    issues: list[AuditIssue],
) -> None:
    named_vectors = set(collection_config.get("named_vectors", {}))
    if "text_dense" in named_vectors:
        validate_target_record_ids(
            vector_name="text_dense",
            target_label="chunk",
            payload_key="chunk_id",
            expected_ids={chunk.chunk_id for chunk in chunks},
            package_dir=package_dir,
            issues=issues,
        )
        validate_target_payload_text(
            vector_name="text_dense",
            target_label="chunk",
            payload_key="chunk_id",
            expected_text_by_id={chunk.chunk_id: chunk.text for chunk in chunks},
            package_dir=package_dir,
            issues=issues,
        )
    if "image_dense" in named_vectors:
        image_payload_fields_by_asset_id = {
            asset.asset_id: image_asset_payload_fields(asset)
            for asset in assets
            if asset.path is not None
        }
        validate_target_record_ids(
            vector_name="image_dense",
            target_label="image_asset",
            payload_key="asset_id",
            expected_ids=set(image_payload_fields_by_asset_id),
            package_dir=package_dir,
            issues=issues,
        )
        validate_target_payload_fields(
            vector_name="image_dense",
            target_label="image_asset",
            payload_key="asset_id",
            expected_fields_by_id=image_payload_fields_by_asset_id,
            package_dir=package_dir,
            issues=issues,
        )
    if "caption_dense" in named_vectors:
        caption_text_by_asset_id = {
            asset.asset_id: text
            for asset in assets
            for text in [asset_text(asset)]
            if text
        }
        caption_payload_fields_by_asset_id = {
            asset.asset_id: caption_asset_payload_fields(asset)
            for asset in assets
            if asset.asset_id in caption_text_by_asset_id
        }
        validate_target_record_ids(
            vector_name="caption_dense",
            target_label="caption_asset",
            payload_key="asset_id",
            expected_ids=set(caption_text_by_asset_id),
            package_dir=package_dir,
            issues=issues,
        )
        validate_target_payload_fields(
            vector_name="caption_dense",
            target_label="caption_asset",
            payload_key="asset_id",
            expected_fields_by_id=caption_payload_fields_by_asset_id,
            package_dir=package_dir,
            issues=issues,
        )
        validate_target_payload_text(
            vector_name="caption_dense",
            target_label="caption_asset",
            payload_key="asset_id",
            expected_text_by_id=caption_text_by_asset_id,
            package_dir=package_dir,
            issues=issues,
        )
    if "triple_dense" in named_vectors:
        triple_text_by_id = {triple.triple_id: triple_text(triple) for triple in triples}
        triple_payload_fields_by_id = {
            triple.triple_id: triple_payload_fields(triple)
            for triple in triples
        }
        validate_target_record_ids(
            vector_name="triple_dense",
            target_label="triple",
            payload_key="triple_id",
            expected_ids=set(triple_text_by_id),
            package_dir=package_dir,
            issues=issues,
        )
        validate_target_payload_fields(
            vector_name="triple_dense",
            target_label="triple",
            payload_key="triple_id",
            expected_fields_by_id=triple_payload_fields_by_id,
            package_dir=package_dir,
            issues=issues,
        )
        validate_target_payload_text(
            vector_name="triple_dense",
            target_label="triple",
            payload_key="triple_id",
            expected_text_by_id=triple_text_by_id,
            package_dir=package_dir,
            issues=issues,
        )


def validate_target_record_ids(
    vector_name: str,
    target_label: str,
    payload_key: str,
    expected_ids: set[str],
    package_dir: Path,
    issues: list[AuditIssue],
) -> None:
    record_file = package_dir / qdrant_record_filename(vector_name)
    if not record_file.exists():
        return
    observed_ids = {
        str(value)
        for record in read_jsonl(record_file, EmbeddingRecord)
        if record.vector_name == vector_name
        for value in [record.payload.get(payload_key)]
        if value not in {None, ""}
    }
    missing_ids = sorted(expected_ids - observed_ids)
    if missing_ids:
        issues.append(
            AuditIssue(
                severity="error",
                code=f"qdrant_missing_{target_label}_records",
                message=f"Some {target_label} IDs do not have {vector_name} records.",
                metadata={
                    "vector_name": vector_name,
                    "ids": missing_ids[:50],
                    "count": len(missing_ids),
                },
            )
        )
    stale_ids = sorted(observed_ids - expected_ids)
    if stale_ids:
        issues.append(
            AuditIssue(
                severity="error",
                code=f"qdrant_stale_{target_label}_records",
                message=f"Some {vector_name} records point to unknown {target_label} IDs.",
                metadata={
                    "vector_name": vector_name,
                    "ids": stale_ids[:50],
                    "count": len(stale_ids),
                },
            )
        )


def validate_target_payload_text(
    vector_name: str,
    target_label: str,
    payload_key: str,
    expected_text_by_id: dict[str, str],
    package_dir: Path,
    issues: list[AuditIssue],
) -> None:
    record_file = package_dir / qdrant_record_filename(vector_name)
    if not record_file.exists():
        return
    mismatches = []
    for record in read_jsonl(record_file, EmbeddingRecord):
        if record.vector_name != vector_name:
            continue
        target_id = record.payload.get(payload_key)
        if target_id in {None, ""}:
            continue
        expected_text = expected_text_by_id.get(str(target_id))
        if expected_text is None:
            continue
        actual_text = str(record.payload.get("text") or "")
        if normalized_payload_text(actual_text) == normalized_payload_text(expected_text):
            continue
        mismatches.append(
            {
                "id": str(target_id),
                "point_id": record.point_id,
                "expected_text_chars": len(expected_text),
                "actual_text_chars": len(actual_text),
            }
        )
    if mismatches:
        issues.append(
            AuditIssue(
                severity="error",
                code=f"qdrant_stale_{target_label}_payload_text",
                message=f"Some {vector_name} records have stale payload text for current {target_label} data.",
                metadata={
                    "vector_name": vector_name,
                    "mismatches": mismatches[:50],
                    "count": len(mismatches),
                },
            )
        )


def validate_target_payload_fields(
    vector_name: str,
    target_label: str,
    payload_key: str,
    expected_fields_by_id: dict[str, dict[str, Any]],
    package_dir: Path,
    issues: list[AuditIssue],
) -> None:
    record_file = package_dir / qdrant_record_filename(vector_name)
    if not record_file.exists():
        return
    mismatches = []
    for record in read_jsonl(record_file, EmbeddingRecord):
        if record.vector_name != vector_name:
            continue
        target_id = record.payload.get(payload_key)
        if target_id in {None, ""}:
            continue
        expected_fields = expected_fields_by_id.get(str(target_id))
        if expected_fields is None:
            continue
        stale_fields = [
            field
            for field, expected_value in sorted(expected_fields.items())
            if not payload_values_equal(record.payload.get(field), expected_value)
        ]
        if stale_fields:
            mismatches.append(
                {
                    "id": str(target_id),
                    "point_id": record.point_id,
                    "fields": stale_fields,
                }
            )
    if mismatches:
        issues.append(
            AuditIssue(
                severity="error",
                code=f"qdrant_stale_{target_label}_payload_fields",
                message=f"Some {vector_name} records have stale payload fields for current {target_label} data.",
                metadata={
                    "vector_name": vector_name,
                    "mismatches": mismatches[:50],
                    "count": len(mismatches),
                },
            )
        )


def image_asset_payload_fields(asset: VisualAsset) -> dict[str, Any]:
    return non_empty_payload_fields(
        {
            "caption": asset.caption,
            "ocr_text": asset.ocr_text,
            "vlm_summary": asset.vlm_summary,
            **asset.metadata,
        }
    )


def caption_asset_payload_fields(asset: VisualAsset) -> dict[str, Any]:
    return non_empty_payload_fields(
        {
            "caption": asset.caption,
            **asset.metadata,
        }
    )


def triple_payload_fields(triple: GraphTriple) -> dict[str, Any]:
    return non_empty_payload_fields(
        {
            "triple_id": triple.triple_id,
            "chunk_id": triple.chunk_id,
            "doc_id": triple.doc_id,
            "kind": "graph_triple",
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "confidence": triple.confidence,
            "qualifiers": triple.qualifiers,
        }
    )


def non_empty_payload_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fields.items()
        if not payload_value_is_empty(value)
    }


def payload_values_equal(actual: Any, expected: Any) -> bool:
    return normalized_payload_value(actual) == normalized_payload_value(expected)


def normalized_payload_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalized_payload_text(value)
    if isinstance(value, list):
        return [normalized_payload_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): normalized_payload_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def payload_value_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def normalized_payload_text(text: str) -> str:
    return " ".join(text.split())


def normalize_payload_indexes(payload_indexes: list[str | dict[str, Any]]) -> set[str]:
    return qdrant_payload_index_fields(payload_indexes)


def required_payload_indexes() -> set[str]:
    return qdrant_payload_index_fields()


def qdrant_record_filename(vector_name: str) -> str:
    return f"qdrant_{vector_name.removesuffix('_dense')}_records.jsonl"


def required_payload_fields(vector_name: str) -> set[str]:
    if vector_name == "text_dense":
        return {"chunk_id", "doc_id", "page_start", "page_end", "kind", "text"}
    if vector_name == "caption_dense":
        return {"asset_id", "doc_id", "page_no", "kind", "text"}
    if vector_name == "image_dense":
        return {"asset_id", "doc_id", "page_no", "kind"}
    if vector_name == "triple_dense":
        return {"triple_id", "chunk_id", "doc_id", "kind", "subject", "predicate", "object", "text"}
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
