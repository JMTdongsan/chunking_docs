from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.graph.provenance import chunk_asset_ids
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, TextQuality, VisualAsset


class CharacteristicObservation(BaseModel):
    code: str
    severity: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessingRecommendation(BaseModel):
    code: str
    area: str
    priority: str
    message: str
    commands: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextLayerCharacteristics(BaseModel):
    page_count: int
    quality_counts: dict[str, int]
    degraded_or_empty_ratio: float
    char_total: int
    char_mean: float
    char_median: float
    low_text_pages: list[int] = Field(default_factory=list)


class VisualCharacteristics(BaseModel):
    asset_kind_counts: dict[str, int]
    visual_heavy_pages: list[int] = Field(default_factory=list)
    top_visual_pages: list[dict[str, Any]] = Field(default_factory=list)
    pages_requiring_ocr_count: int = 0
    pages_requiring_vlm_count: int = 0
    annotated_asset_count: int = 0
    annotation_coverage: float = 0.0


class ChunkCharacteristics(BaseModel):
    chunk_count: int
    chunk_kind_counts: dict[str, int]
    chunks_with_assets: int
    chunks_with_visual_annotations: int
    table_chunk_count: int = 0


class GraphCharacteristics(BaseModel):
    triple_count: int
    visual_triple_count: int = 0
    predicate_counts: dict[str, int] = Field(default_factory=dict)


class ArtifactCharacteristics(BaseModel):
    bm25_tokens: bool = False
    embedding_manifest: bool = False
    qdrant_collection: bool = False
    qdrant_record_files: list[str] = Field(default_factory=list)


class PackageCharacteristics(BaseModel):
    text_layer: TextLayerCharacteristics
    visual: VisualCharacteristics
    chunks: ChunkCharacteristics
    graph: GraphCharacteristics
    artifacts: ArtifactCharacteristics = Field(default_factory=ArtifactCharacteristics)
    observations: list[CharacteristicObservation] = Field(default_factory=list)
    recommendations: list[ProcessingRecommendation] = Field(default_factory=list)


def characterize_package(
    profiles: list[PageProfile],
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    package_dir: Path | None = None,
    max_pages: int = 25,
) -> PackageCharacteristics:
    text_layer = text_layer_characteristics(profiles, max_pages=max_pages)
    visual = visual_characteristics(profiles, assets, max_pages=max_pages)
    chunk_summary = chunk_characteristics(chunks)
    graph = graph_characteristics(triples)
    artifacts = artifact_characteristics(package_dir) if package_dir else ArtifactCharacteristics()
    return PackageCharacteristics(
        text_layer=text_layer,
        visual=visual,
        chunks=chunk_summary,
        graph=graph,
        artifacts=artifacts,
        observations=observations(text_layer, visual, chunk_summary, graph, artifacts),
        recommendations=recommendations(text_layer, visual, chunk_summary, graph, artifacts),
    )


def text_layer_characteristics(
    profiles: list[PageProfile],
    max_pages: int,
) -> TextLayerCharacteristics:
    counts = Counter(str(profile.text_quality) for profile in profiles)
    chars = [profile.char_count for profile in profiles]
    low_text_pages = [
        profile.page_no
        for profile in profiles
        if profile.text_quality in {TextQuality.DEGRADED, TextQuality.EMPTY}
    ][:max_pages]
    degraded_or_empty = counts.get(str(TextQuality.DEGRADED), 0) + counts.get(str(TextQuality.EMPTY), 0)
    return TextLayerCharacteristics(
        page_count=len(profiles),
        quality_counts=dict(sorted(counts.items())),
        degraded_or_empty_ratio=degraded_or_empty / len(profiles) if profiles else 0.0,
        char_total=sum(chars),
        char_mean=round(statistics.fmean(chars), 3) if chars else 0.0,
        char_median=float(statistics.median(chars)) if chars else 0.0,
        low_text_pages=low_text_pages,
    )


def visual_characteristics(
    profiles: list[PageProfile],
    assets: list[VisualAsset],
    max_pages: int,
) -> VisualCharacteristics:
    kind_counts = Counter(str(asset.kind) for asset in assets)
    pages_requiring_ocr = {asset.page_no for asset in assets if asset.metadata.get("requires_ocr") and not asset.ocr_text}
    pages_requiring_vlm = {asset.page_no for asset in assets if asset.metadata.get("requires_vlm") and not asset.vlm_summary}
    annotated_asset_count = sum(1 for asset in assets if asset.ocr_text or asset.vlm_summary)
    visual_scores = [
        {
            "page_no": profile.page_no,
            "score": page_visual_score(profile),
            "image_block_count": profile.image_block_count,
            "embedded_image_count": profile.embedded_image_count,
            "drawing_count": profile.drawing_count,
            "text_quality": str(profile.text_quality),
        }
        for profile in profiles
    ]
    visual_scores.sort(key=lambda item: (-int(item["score"]), int(item["page_no"])))
    visual_heavy_pages = [
        int(item["page_no"])
        for item in visual_scores
        if int(item["score"]) >= 20
    ][:max_pages]
    return VisualCharacteristics(
        asset_kind_counts=dict(sorted(kind_counts.items())),
        visual_heavy_pages=visual_heavy_pages,
        top_visual_pages=visual_scores[:max_pages],
        pages_requiring_ocr_count=len(pages_requiring_ocr),
        pages_requiring_vlm_count=len(pages_requiring_vlm),
        annotated_asset_count=annotated_asset_count,
        annotation_coverage=annotated_asset_count / len(assets) if assets else 0.0,
    )


def chunk_characteristics(chunks: list[DocumentChunk]) -> ChunkCharacteristics:
    kind_counts = Counter(str(chunk.kind) for chunk in chunks)
    return ChunkCharacteristics(
        chunk_count=len(chunks),
        chunk_kind_counts=dict(sorted(kind_counts.items())),
        chunks_with_assets=sum(1 for chunk in chunks if chunk_asset_ids(chunk)),
        chunks_with_visual_annotations=sum(1 for chunk in chunks if chunk.metadata.get("has_visual_annotations")),
        table_chunk_count=kind_counts.get("table", 0),
    )


def graph_characteristics(triples: list[GraphTriple]) -> GraphCharacteristics:
    predicate_counts = Counter(triple.predicate for triple in triples)
    visual_triple_count = sum(
        1 for triple in triples if triple.qualifiers.get("source") == "visual_annotation"
    )
    return GraphCharacteristics(
        triple_count=len(triples),
        visual_triple_count=visual_triple_count,
        predicate_counts=dict(sorted(predicate_counts.items())),
    )


def artifact_characteristics(package_dir: Path) -> ArtifactCharacteristics:
    return ArtifactCharacteristics(
        bm25_tokens=(package_dir / "bm25_tokens.json").exists(),
        embedding_manifest=(package_dir / "embedding_manifest.json").exists(),
        qdrant_collection=(package_dir / "qdrant_collection.json").exists(),
        qdrant_record_files=sorted(path.name for path in package_dir.glob("qdrant_*_records.jsonl")),
    )


def observations(
    text_layer: TextLayerCharacteristics,
    visual: VisualCharacteristics,
    chunks: ChunkCharacteristics,
    graph: GraphCharacteristics,
    artifacts: ArtifactCharacteristics,
) -> list[CharacteristicObservation]:
    result = []
    if text_layer.degraded_or_empty_ratio >= 0.25:
        result.append(
            CharacteristicObservation(
                code="text_layer_degraded",
                severity="warning",
                message="A large share of pages has degraded or empty text; OCR/VLM annotations should be prioritized.",
                metadata={"ratio": text_layer.degraded_or_empty_ratio},
            )
        )
    if visual.visual_heavy_pages or visual.asset_kind_counts.get("map", 0) or visual.asset_kind_counts.get("table", 0):
        result.append(
            CharacteristicObservation(
                code="visual_retrieval_required",
                severity="info",
                message="The package contains visual-heavy pages or structured visual assets; caption and image vectors should be evaluated.",
                metadata={"asset_kind_counts": visual.asset_kind_counts},
            )
        )
    if visual.pages_requiring_ocr_count or visual.pages_requiring_vlm_count:
        result.append(
            CharacteristicObservation(
                code="visual_annotation_pending",
                severity="warning",
                message="Some visual assets still need OCR or VLM summaries before final retrieval evaluation.",
                metadata={
                    "pages_requiring_ocr_count": visual.pages_requiring_ocr_count,
                    "pages_requiring_vlm_count": visual.pages_requiring_vlm_count,
                },
            )
        )
    if graph.triple_count == 0:
        result.append(
            CharacteristicObservation(
                code="graph_triples_missing",
                severity="info",
                message="No graph triples are present; VLM JSON or external annotations can add graph retrieval signals.",
            )
        )
    if not artifacts.bm25_tokens:
        result.append(
            CharacteristicObservation(
                code="bm25_missing",
                severity="warning",
                message="BM25 token artifacts are missing; lexical retrieval should be generated before evaluation.",
            )
        )
    if not artifacts.embedding_manifest or not artifacts.qdrant_record_files:
        result.append(
            CharacteristicObservation(
                code="embedding_artifacts_missing",
                severity="warning",
                message="Embedding artifact provenance or Qdrant records are missing.",
            )
        )
    if chunks.chunk_count and not chunks.chunks_with_assets:
        result.append(
            CharacteristicObservation(
                code="asset_linkage_missing",
                severity="warning",
                message="Chunks are not linked to visual assets; multimodal context assembly will be limited.",
            )
        )
    return result


def recommendations(
    text_layer: TextLayerCharacteristics,
    visual: VisualCharacteristics,
    chunks: ChunkCharacteristics,
    graph: GraphCharacteristics,
    artifacts: ArtifactCharacteristics,
) -> list[ProcessingRecommendation]:
    result = []
    if (
        text_layer.degraded_or_empty_ratio > 0
        or visual.pages_requiring_ocr_count
        or visual.pages_requiring_vlm_count
    ):
        result.append(
            ProcessingRecommendation(
                code="prioritize_visual_annotations",
                area="vision",
                priority="required",
                message=(
                    "Run prioritized OCR/VLM jobs for degraded text pages and visual-heavy assets before final "
                    "retrieval evaluation."
                ),
                commands=[
                    "chunking-docs plan-visual-jobs --package-dir outputs/package",
                    "chunking-docs run-visual-jobs --package-dir outputs/package --jobs outputs/package/visual_jobs.jsonl --apply",
                    "chunking-docs gate-visual-results --results outputs/package/visual_job_results.jsonl",
                ],
                metadata={
                    "low_text_pages": text_layer.low_text_pages,
                    "visual_heavy_pages": visual.visual_heavy_pages,
                    "pages_requiring_ocr_count": visual.pages_requiring_ocr_count,
                    "pages_requiring_vlm_count": visual.pages_requiring_vlm_count,
                },
            )
        )
    if visual.asset_kind_counts.get("map", 0) or visual.asset_kind_counts.get("chart", 0):
        result.append(
            ProcessingRecommendation(
                code="evaluate_visual_vectors",
                area="embeddings",
                priority="required",
                message=(
                    "Compare text, caption, image, and graph-expanded Qdrant modes so visual evidence can be "
                    "measured instead of assumed."
                ),
                commands=[
                    "chunking-docs embed-package --package-dir outputs/package --image-backend clip",
                    "chunking-docs eval-qdrant-vector-ablation examples/retrieval_cases.jsonl --package-dir outputs/package --modes text,caption,text_caption,all_graph",
                ],
                metadata={"asset_kind_counts": visual.asset_kind_counts},
            )
        )
    if visual.asset_kind_counts.get("table", 0) or chunks.table_chunk_count:
        result.append(
            ProcessingRecommendation(
                code="preserve_table_structure",
                area="chunking",
                priority="recommended",
                message=(
                    "Keep table chunks and table assets separate from prose chunks, then include table-specific "
                    "targets in retrieval cases."
                ),
                commands=[
                    "chunking-docs audit-package --package-dir outputs/package --require-qdrant-records",
                    "chunking-docs generate-retrieval-cases --package-dir outputs/package",
                ],
                metadata={"table_assets": visual.asset_kind_counts.get("table", 0), "table_chunks": chunks.table_chunk_count},
            )
        )
    if chunks.chunks_with_assets or visual.asset_kind_counts:
        result.append(
            ProcessingRecommendation(
                code="compare_multimodal_hierarchical_chunking",
                area="chunking",
                priority="required",
                message=(
                    "Compare semantic, multimodal, and hierarchical chunk candidates with the same benchmark cases "
                    "before changing default chunking settings."
                ),
                commands=[
                    "chunking-docs compare-chunking --package-dir outputs/package --candidate semantic=outputs/package/chunks.semantic.jsonl --candidate multimodal=outputs/package/chunks.multimodal.jsonl",
                    "chunking-docs gate-chunking-comparison outputs/package/chunking_comparison.json",
                ],
                metadata={
                    "chunks_with_assets": chunks.chunks_with_assets,
                    "asset_kind_counts": visual.asset_kind_counts,
                },
            )
        )
    if graph.triple_count == 0 or graph.visual_triple_count == 0:
        result.append(
            ProcessingRecommendation(
                code="add_graph_signals",
                area="graph",
                priority="recommended",
                message=(
                    "Add section-derived or visual-annotation triples and evaluate graph-expanded retrieval against "
                    "non-graph baselines."
                ),
                commands=[
                    "chunking-docs normalize-graph-triples --package-dir outputs/package --export-graph",
                    "chunking-docs eval-retrieval-ablation examples/retrieval_cases.jsonl --package-dir outputs/package --modes dense,bm25,hybrid,hybrid_graph",
                ],
                metadata={
                    "triple_count": graph.triple_count,
                    "visual_triple_count": graph.visual_triple_count,
                },
            )
        )
    if not artifacts.embedding_manifest or not artifacts.qdrant_record_files:
        result.append(
            ProcessingRecommendation(
                code="build_embedding_artifacts",
                area="storage",
                priority="required",
                message=(
                    "Build embedding records with manifest provenance before loading Qdrant or comparing retrieval "
                    "runs."
                ),
                commands=["chunking-docs embed-package --package-dir outputs/package"],
                metadata={
                    "embedding_manifest": artifacts.embedding_manifest,
                    "qdrant_record_files": artifacts.qdrant_record_files,
                },
            )
        )
    result.append(
        ProcessingRecommendation(
            code="maintain_retrieval_benchmark",
            area="evaluation",
            priority="required",
            message=(
                "Maintain benchmark cases with page, chunk, asset, and graph targets; use gates to decide whether "
                "a chunking strategy is an improvement."
            ),
            commands=[
                "chunking-docs audit-retrieval-cases examples/retrieval_cases.jsonl --package-dir outputs/package",
                "chunking-docs eval-retrieval examples/retrieval_cases.jsonl --package-dir outputs/package --repeat 3",
                "chunking-docs gate-retrieval outputs/package/retrieval_eval.json",
            ],
            metadata={
                "page_count": text_layer.page_count,
                "chunk_count": chunks.chunk_count,
                "asset_count": sum(visual.asset_kind_counts.values()),
                "triple_count": graph.triple_count,
            },
        )
    )
    return result


def page_visual_score(profile: PageProfile) -> int:
    return (
        profile.image_block_count * 5
        + profile.embedded_image_count * 3
        + profile.drawing_count
    )
