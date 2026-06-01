from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.embeddings.records import (
    VISUAL_FEATURE_METADATA_KEYS,
    VISUAL_OBJECT_METADATA_KEYS,
    dedupe_visual_object_records,
    metadata_object_records,
)
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
    quality_reason_counts: dict[str, int] = Field(default_factory=dict)
    degraded_or_empty_ratio: float
    char_total: int
    char_mean: float
    char_median: float
    control_char_ratio_mean: float = 0.0
    letter_or_number_ratio_mean: float = 0.0
    cjk_char_ratio_mean: float = 0.0
    low_text_pages: list[int] = Field(default_factory=list)


class VisualCharacteristics(BaseModel):
    asset_kind_counts: dict[str, int]
    rendered_asset_count: int = 0
    visual_heavy_pages: list[int] = Field(default_factory=list)
    tile_candidate_pages: list[int] = Field(default_factory=list)
    tile_candidate_count: int = 0
    top_visual_pages: list[dict[str, Any]] = Field(default_factory=list)
    pages_requiring_ocr_count: int = 0
    pages_requiring_vlm_count: int = 0
    annotated_asset_count: int = 0
    annotation_coverage: float = 0.0
    vlm_object_asset_count: int = 0
    vlm_object_count: int = 0
    vlm_object_bbox_count: int = 0
    vlm_visual_feature_asset_count: int = 0
    vlm_visual_feature_count: int = 0


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
    reason_counts = Counter(reason for profile in profiles for reason in profile.text_quality_reasons)
    chars = [profile.char_count for profile in profiles]
    control_ratios = [profile.control_char_ratio for profile in profiles]
    letter_or_number_ratios = [profile.letter_or_number_ratio for profile in profiles]
    cjk_ratios = [profile.cjk_char_ratio for profile in profiles]
    low_text_pages = [
        profile.page_no
        for profile in profiles
        if profile.text_quality in {TextQuality.DEGRADED, TextQuality.EMPTY}
    ][:max_pages]
    degraded_or_empty = counts.get(str(TextQuality.DEGRADED), 0) + counts.get(str(TextQuality.EMPTY), 0)
    return TextLayerCharacteristics(
        page_count=len(profiles),
        quality_counts=dict(sorted(counts.items())),
        quality_reason_counts=dict(sorted(reason_counts.items())),
        degraded_or_empty_ratio=degraded_or_empty / len(profiles) if profiles else 0.0,
        char_total=sum(chars),
        char_mean=round(statistics.fmean(chars), 3) if chars else 0.0,
        char_median=float(statistics.median(chars)) if chars else 0.0,
        control_char_ratio_mean=round(statistics.fmean(control_ratios), 6) if control_ratios else 0.0,
        letter_or_number_ratio_mean=round(statistics.fmean(letter_or_number_ratios), 6)
        if letter_or_number_ratios
        else 0.0,
        cjk_char_ratio_mean=round(statistics.fmean(cjk_ratios), 6) if cjk_ratios else 0.0,
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
    asset_objects = [asset_visual_objects(asset) for asset in assets]
    object_count = sum(len(objects) for objects in asset_objects)
    visual_feature_count = sum(
        1
        for objects in asset_objects
        for item in objects
        if item.get("visual_feature_type")
    )
    object_bbox_count = sum(
        1
        for objects in asset_objects
        for item in objects
        if item.get("bbox")
    )
    visual_scores = [
        {
            "page_no": profile.page_no,
            "score": page_visual_score(profile),
            "image_block_count": profile.image_block_count,
            "embedded_image_count": profile.embedded_image_count,
            "drawing_count": profile.drawing_count,
            "text_quality": str(profile.text_quality),
            "tile_candidate": page_needs_tiling(profile),
            "tile_reasons": tile_candidate_reasons(profile),
        }
        for profile in profiles
    ]
    visual_scores.sort(key=lambda item: (-int(item["score"]), int(item["page_no"])))
    tile_candidates = [
        int(item["page_no"])
        for item in visual_scores
        if item["tile_candidate"]
    ]
    visual_heavy_pages = [
        int(item["page_no"])
        for item in visual_scores
        if int(item["score"]) >= 20
    ][:max_pages]
    return VisualCharacteristics(
        asset_kind_counts=dict(sorted(kind_counts.items())),
        rendered_asset_count=sum(1 for asset in assets if asset.path is not None),
        visual_heavy_pages=visual_heavy_pages,
        tile_candidate_pages=tile_candidates[:max_pages],
        tile_candidate_count=len(tile_candidates),
        top_visual_pages=visual_scores[:max_pages],
        pages_requiring_ocr_count=len(pages_requiring_ocr),
        pages_requiring_vlm_count=len(pages_requiring_vlm),
        annotated_asset_count=annotated_asset_count,
        annotation_coverage=annotated_asset_count / len(assets) if assets else 0.0,
        vlm_object_asset_count=sum(1 for objects in asset_objects if objects),
        vlm_object_count=object_count,
        vlm_object_bbox_count=object_bbox_count,
        vlm_visual_feature_asset_count=sum(
            1 for objects in asset_objects if any(item.get("visual_feature_type") for item in objects)
        ),
        vlm_visual_feature_count=visual_feature_count,
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
                metadata={
                    "ratio": text_layer.degraded_or_empty_ratio,
                    "quality_reason_counts": text_layer.quality_reason_counts,
                    "control_char_ratio_mean": text_layer.control_char_ratio_mean,
                    "letter_or_number_ratio_mean": text_layer.letter_or_number_ratio_mean,
                },
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
    if visual.tile_candidate_pages:
        result.append(
            CharacteristicObservation(
                code="dense_visual_pages_need_tiling",
                severity="info",
                message="Some visual-heavy pages should be processed as overlapping tiles before OCR/VLM evaluation.",
                metadata={
                    "tile_candidate_pages": visual.tile_candidate_pages,
                    "tile_candidate_page_ranges": page_range_spec(visual.tile_candidate_pages),
                    "tile_candidate_count": visual.tile_candidate_count,
                },
            )
        )
    if visual.vlm_object_count:
        result.append(
            CharacteristicObservation(
                code="vlm_objects_available",
                severity="info",
                message=(
                    "Structured VLM object or visual-element metadata is available and should be evaluated "
                    "as its own retrieval case group."
                ),
                metadata={
                    "vlm_object_asset_count": visual.vlm_object_asset_count,
                    "vlm_object_count": visual.vlm_object_count,
                    "vlm_object_bbox_count": visual.vlm_object_bbox_count,
                    "vlm_visual_feature_asset_count": visual.vlm_visual_feature_asset_count,
                    "vlm_visual_feature_count": visual.vlm_visual_feature_count,
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
    if graph.triple_count and not qdrant_record_present(artifacts, "triple_dense"):
        result.append(
            CharacteristicObservation(
                code="triple_vector_records_missing",
                severity="warning",
                message="Graph triples are present but triple_dense Qdrant records are missing.",
                metadata={
                    "triple_count": graph.triple_count,
                    "qdrant_record_files": artifacts.qdrant_record_files,
                },
            )
        )
    if visual.vlm_object_count and not qdrant_record_present(artifacts, "object_dense"):
        result.append(
            CharacteristicObservation(
                code="object_vector_records_missing",
                severity="warning",
                message="Structured VLM objects or visual elements are present but object_dense Qdrant records are missing.",
                metadata={
                    "vlm_object_count": visual.vlm_object_count,
                    "vlm_visual_feature_count": visual.vlm_visual_feature_count,
                    "qdrant_record_files": artifacts.qdrant_record_files,
                },
            )
        )
    if visual.rendered_asset_count and not qdrant_record_present(artifacts, "image_dense"):
        result.append(
            CharacteristicObservation(
                code="image_vector_records_missing",
                severity="warning",
                message="Rendered visual assets are present but image_dense Qdrant records are missing.",
                metadata={
                    "rendered_asset_count": visual.rendered_asset_count,
                    "qdrant_record_files": artifacts.qdrant_record_files,
                },
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
    if visual.tile_candidate_pages:
        tile_pages = page_range_spec(visual.tile_candidate_pages)
        result.append(
            ProcessingRecommendation(
                code="build_page_tiles",
                area="vision",
                priority="required",
                message=(
                    "Create overlapping tiles for dense visual pages before final OCR/VLM runs so small map, "
                    "table, and diagram details are not hidden inside a full-page image."
                ),
                commands=[
                    (
                        "chunking-docs build-tile-assets --package-dir outputs/package "
                        f"--pages {tile_pages} --rows 2 --cols 2 --overlap-ratio 0.08"
                    ),
                    (
                        "chunking-docs plan-visual-jobs --package-dir outputs/package "
                        f"--pages {tile_pages} --output outputs/package/visual_jobs.tiled.jsonl"
                    ),
                ],
                metadata={
                    "tile_candidate_pages": visual.tile_candidate_pages,
                    "tile_candidate_page_ranges": tile_pages,
                    "tile_candidate_count": visual.tile_candidate_count,
                },
            )
        )
    if visual.asset_kind_counts.get("map", 0) or visual.asset_kind_counts.get("chart", 0):
        embed_command = "chunking-docs embed-package --package-dir outputs/package --image-backend clip"
        if visual.vlm_object_count or graph.triple_count:
            embed_command = (
                "chunking-docs embed-package --package-dir outputs/package --image-backend clip "
                "--object-backend same-as-caption --triple-backend same-as-text"
            )
        result.append(
            ProcessingRecommendation(
                code="evaluate_visual_vectors",
                area="embeddings",
                priority="required",
                message=(
                    "Compare text, caption, object, image, and graph-expanded Qdrant modes so visual evidence "
                    "can be measured instead of assumed."
                ),
                commands=[
                    embed_command,
                    (
                        "chunking-docs eval-qdrant-vector-ablation examples/retrieval_cases.jsonl "
                        "--package-dir outputs/package "
                        "--modes text,caption,image,text_image,caption_image,all "
                        "--image-query-backend clip --image-query-model openai/clip-vit-large-patch14"
                    ),
                ],
                metadata={
                    "asset_kind_counts": visual.asset_kind_counts,
                    "rendered_asset_count": visual.rendered_asset_count,
                },
            )
        )
    if visual.rendered_asset_count:
        image_probe_threshold = bounded_threshold(visual.rendered_asset_count)
        result.append(
            ProcessingRecommendation(
                code="generate_visual_image_probe_cases",
                area="evaluation",
                priority="required",
                message=(
                    "Generate and gate rendered-image retrieval probes so image_dense contribution is measured "
                    "separately from caption, object, and text vectors."
                ),
                commands=[
                    (
                        "chunking-docs generate-retrieval-cases --package-dir outputs/package "
                        "--query-mode salient_terms --selection-strategy salience "
                        "--image-probe-limit 20 --output examples/retrieval_cases.jsonl"
                    ),
                    (
                        "chunking-docs audit-retrieval-cases examples/retrieval_cases.jsonl "
                        "--package-dir outputs/package "
                        f"--min-case-group-count case_source:visual_image_probe={image_probe_threshold} "
                        f"--min-distinct-asset-targets {image_probe_threshold} "
                        f"--min-case-group-distinct-targets case_source:visual_image_probe:asset={image_probe_threshold} "
                        "--min-query-terms-per-case 3"
                    ),
                    (
                        "chunking-docs gate-qdrant-vector-ablation outputs/package/qdrant_vector_ablation.json "
                        "--mode caption_image --baseline-mode caption "
                        "--min-source-target-coverage qdrant:image_dense=0.5 "
                        "--min-case-group-target-coverage case_source:visual_image_probe=0.7"
                    ),
                ],
                metadata={
                    "rendered_asset_count": visual.rendered_asset_count,
                    "recommended_image_probe_case_threshold": image_probe_threshold,
                    "recommended_distinct_asset_threshold": image_probe_threshold,
                },
            )
        )
    if visual.vlm_object_count:
        object_probe_case_threshold = bounded_threshold(visual.vlm_object_count)
        object_probe_asset_threshold = bounded_threshold(visual.vlm_object_asset_count)
        result.append(
            ProcessingRecommendation(
                code="generate_visual_object_probe_cases",
                area="evaluation",
                priority="required",
                message=(
                    "Generate and gate visual object probe retrieval cases so VLM object detections and "
                    "visual elements are measured separately from text and caption averages."
                ),
                commands=[
                    "chunking-docs generate-retrieval-cases --package-dir outputs/package --query-mode salient_terms --selection-strategy salience --object-probe-limit 20 --output examples/retrieval_cases.jsonl",
                    (
                        "chunking-docs audit-retrieval-cases examples/retrieval_cases.jsonl "
                        "--package-dir outputs/package "
                        f"--min-case-group-count case_source:visual_object_probe={object_probe_case_threshold} "
                        f"--min-distinct-asset-targets {object_probe_asset_threshold} "
                        f"--min-case-group-distinct-targets case_source:visual_object_probe:asset={object_probe_asset_threshold} "
                        "--max-asset-cases-per-target 3 "
                        "--min-query-terms-per-case 3 "
                        "--require-visual-only-object-probes"
                    ),
                    "chunking-docs gate-qdrant-vector-ablation outputs/package/qdrant_vector_ablation.json --mode text_caption --min-case-group-target-coverage case_source:visual_object_probe=0.7",
                ],
                metadata={
                    "vlm_object_asset_count": visual.vlm_object_asset_count,
                    "vlm_object_count": visual.vlm_object_count,
                    "vlm_object_bbox_count": visual.vlm_object_bbox_count,
                    "vlm_visual_feature_asset_count": visual.vlm_visual_feature_asset_count,
                    "vlm_visual_feature_count": visual.vlm_visual_feature_count,
                    "recommended_object_probe_case_threshold": object_probe_case_threshold,
                    "recommended_distinct_asset_threshold": object_probe_asset_threshold,
                },
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
                    "Compare semantic, multimodal, object-aware, and hierarchical chunk candidates with the same "
                    "benchmark cases before changing default chunking settings."
                ),
                commands=[
                    (
                        "chunking-docs sweep-chunking --package-dir outputs/package "
                        "--cases examples/retrieval_cases.jsonl "
                        "--output outputs/package/chunking_sweep.json"
                    ),
                    "chunking-docs compare-chunking --package-dir outputs/package --candidate semantic=outputs/package/chunks.semantic.jsonl --candidate multimodal=outputs/package/chunks.multimodal.jsonl --candidate object_aware=outputs/package/chunks.object_aware.jsonl",
                    "chunking-docs gate-chunking-comparison outputs/package/chunking_comparison.json",
                    (
                        "chunking-docs apply-chunking-sweep --package-dir outputs/package "
                        "--report outputs/package/chunking_sweep.json"
                    ),
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
    if graph.triple_count and not qdrant_record_present(artifacts, "triple_dense"):
        result.append(
            ProcessingRecommendation(
                code="build_triple_vector_artifacts",
                area="embeddings",
                priority="required",
                message=(
                    "Build triple_dense records so graph relationships can be evaluated as a vector source and not "
                    "only through symbolic graph expansion."
                ),
                commands=[
                    "chunking-docs normalize-graph-triples --package-dir outputs/package --export-graph",
                    "chunking-docs embed-package --package-dir outputs/package --triple-backend same-as-text",
                    "chunking-docs eval-qdrant-vector-ablation examples/retrieval_cases.jsonl --package-dir outputs/package --modes text,triple,text_triple,all_with_triple_graph",
                ],
                metadata={
                    "triple_count": graph.triple_count,
                    "qdrant_record_files": artifacts.qdrant_record_files,
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
                "chunking-docs audit-retrieval-cases examples/retrieval_cases.jsonl --package-dir outputs/package --min-query-terms-per-case 3 --max-duplicate-queries 0",
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


def bounded_threshold(value: int, cap: int = 5) -> int:
    return max(1, min(cap, value))


def qdrant_record_present(artifacts: ArtifactCharacteristics, vector_name: str) -> bool:
    record_file_by_vector = {
        "text_dense": "qdrant_text_records.jsonl",
        "caption_dense": "qdrant_caption_records.jsonl",
        "object_dense": "qdrant_object_records.jsonl",
        "image_dense": "qdrant_image_records.jsonl",
        "triple_dense": "qdrant_triple_records.jsonl",
    }
    record_file = record_file_by_vector.get(vector_name)
    return bool(record_file and record_file in artifacts.qdrant_record_files)


def page_visual_score(profile: PageProfile) -> int:
    return (
        profile.image_block_count * 5
        + profile.embedded_image_count * 3
        + profile.drawing_count
    )


def page_needs_tiling(profile: PageProfile) -> bool:
    return bool(tile_candidate_reasons(profile))


def tile_candidate_reasons(profile: PageProfile) -> list[str]:
    image_count = profile.image_block_count + profile.embedded_image_count
    reasons = []
    if image_count >= 8:
        reasons.append("many_images")
    if profile.drawing_count >= 30:
        reasons.append("many_drawings")
    if profile.text_quality == TextQuality.EMPTY and page_visual_score(profile) >= 20:
        reasons.append("empty_visual_page")
    return reasons


def page_range_spec(pages: list[int]) -> str:
    ranges = []
    ordered_pages = sorted(set(pages))
    index = 0
    while index < len(ordered_pages):
        start = ordered_pages[index]
        end = start
        while index + 1 < len(ordered_pages) and ordered_pages[index + 1] == end + 1:
            index += 1
            end = ordered_pages[index]
        ranges.append(str(start) if start == end else f"{start}-{end}")
        index += 1
    return ",".join(ranges)


def asset_visual_objects(asset: VisualAsset) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for source_key in VISUAL_OBJECT_METADATA_KEYS:
        objects.extend(
            metadata_object_records(
                asset.metadata.get(source_key),
                source_key=source_key,
                limit=64,
            )
        )
    for source_key in VISUAL_FEATURE_METADATA_KEYS:
        for item in metadata_object_records(
            asset.metadata.get(source_key),
            source_key=source_key,
            limit=64,
        ):
            objects.append(
                {
                    **item,
                    "visual_feature_type": source_key.removesuffix("s"),
                }
            )
    return dedupe_visual_object_records(objects, limit=64)
