from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, TextQuality, VisualAsset


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
        issues=issues,
    )


def degraded_page_ratio(profiles: list[PageProfile]) -> float:
    if not profiles:
        return 0.0
    degraded = sum(
        1
        for profile in profiles
        if profile.text_quality in {TextQuality.DEGRADED, TextQuality.EMPTY}
    )
    return degraded / len(profiles)
