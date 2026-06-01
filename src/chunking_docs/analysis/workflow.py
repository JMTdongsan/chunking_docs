from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.analysis.chunking_defaults import CHUNKING_READINESS_GATE_ARGS
from chunking_docs.analysis.characterize import PackageCharacteristics, ProcessingRecommendation
from chunking_docs.analysis.qdrant_defaults import QDRANT_RAG_READINESS_GATE_ARGS


class WorkflowStep(BaseModel):
    step_id: str
    title: str
    area: str
    priority: str
    reason: str
    commands: list[str] = Field(default_factory=list)
    recommendation_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionWorkflowPlan(BaseModel):
    package_dir: str
    retrieval_cases: str
    vlm_profiles: list[str] = Field(default_factory=list)
    observation_codes: list[str] = Field(default_factory=list)
    recommendation_codes: list[str] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


RECOMMENDATION_ORDER = [
    "build_page_tiles",
    "prioritize_visual_annotations",
    "preserve_table_structure",
    "add_graph_signals",
    "generate_visual_image_probe_cases",
    "generate_visual_object_probe_cases",
    "compare_multimodal_hierarchical_chunking",
    "maintain_retrieval_benchmark",
    "build_embedding_artifacts",
    "evaluate_visual_vectors",
    "build_triple_vector_artifacts",
    "validate_qdrant_rag_context",
]

POST_INDEX_RECOMMENDATIONS = {
    "build_embedding_artifacts",
    "evaluate_visual_vectors",
    "build_triple_vector_artifacts",
    "validate_qdrant_rag_context",
}

EMBEDDING_REBUILD_RECOMMENDATIONS = {
    "build_embedding_artifacts",
    "evaluate_visual_vectors",
    "build_triple_vector_artifacts",
}


def build_ingestion_workflow_plan(
    characteristics: PackageCharacteristics,
    package_dir: Path = Path("outputs/package"),
    retrieval_cases: Path = Path("examples/retrieval_cases.jsonl"),
    vlm_profiles: list[str] | None = None,
) -> IngestionWorkflowPlan:
    profiles = vlm_profiles or ["qwen2_5_vl_7b", "qwen2_vl_7b", "llava_next_7b"]
    recommendations_by_code = {item.code: item for item in characteristics.recommendations}
    steps = [
        runtime_check_step(
            package_dir=package_dir,
            vlm_profiles=profiles,
            require_ocr="prioritize_visual_annotations" in recommendations_by_code,
        ),
        WorkflowStep(
            step_id="characterize_package",
            title="Characterize package",
            area="analysis",
            priority="required",
            reason="Record document, visual, graph, and artifact characteristics before changing the package.",
            commands=[
                f"chunking-docs characterize-package --package-dir {path_arg(package_dir)} "
                f"--output {path_arg(package_dir / 'package_characteristics.json')}"
            ],
        ),
    ]

    handled_codes: set[str] = set()
    if "build_page_tiles" in recommendations_by_code:
        steps.append(recommendation_step(recommendations_by_code["build_page_tiles"], package_dir, retrieval_cases))
        handled_codes.add("build_page_tiles")
    if "prioritize_visual_annotations" in recommendations_by_code:
        steps.append(visual_annotation_step(package_dir, retrieval_cases, profiles, recommendations_by_code["prioritize_visual_annotations"]))
        handled_codes.add("prioritize_visual_annotations")

    for code in RECOMMENDATION_ORDER:
        recommendation = recommendations_by_code.get(code)
        if recommendation is None or code in handled_codes or code in POST_INDEX_RECOMMENDATIONS:
            continue
        steps.append(recommendation_step(recommendation, package_dir, retrieval_cases))
        handled_codes.add(code)

    for recommendation in characteristics.recommendations:
        if recommendation.code in handled_codes or recommendation.code in POST_INDEX_RECOMMENDATIONS:
            continue
        steps.append(recommendation_step(recommendation, package_dir, retrieval_cases))
        handled_codes.add(recommendation.code)

    steps.append(index_refresh_step(package_dir))
    needs_qdrant_rag_context = "validate_qdrant_rag_context" in recommendations_by_code
    if needs_qdrant_rag_context and not (
        EMBEDDING_REBUILD_RECOMMENDATIONS & recommendations_by_code.keys()
    ):
        steps.append(embedding_rebuild_step(package_dir))
    for code in RECOMMENDATION_ORDER:
        recommendation = recommendations_by_code.get(code)
        if recommendation is None or code in handled_codes or code not in POST_INDEX_RECOMMENDATIONS:
            continue
        steps.append(recommendation_step(recommendation, package_dir, retrieval_cases))
        handled_codes.add(code)
    for recommendation in characteristics.recommendations:
        if recommendation.code in handled_codes or recommendation.code not in POST_INDEX_RECOMMENDATIONS:
            continue
        steps.append(recommendation_step(recommendation, package_dir, retrieval_cases))
        handled_codes.add(recommendation.code)

    steps.append(metadata_refresh_step(package_dir))
    steps.append(
        readiness_step(
            package_dir,
            retrieval_cases,
            include_chunking_comparison="compare_multimodal_hierarchical_chunking"
            in recommendations_by_code,
            include_qdrant_rag_context=needs_qdrant_rag_context,
        )
    )
    return IngestionWorkflowPlan(
        package_dir=path_text(package_dir),
        retrieval_cases=path_text(retrieval_cases),
        vlm_profiles=profiles,
        observation_codes=[item.code for item in characteristics.observations],
        recommendation_codes=[item.code for item in characteristics.recommendations],
        steps=steps,
        metadata={
            "step_count": len(steps),
            "required_step_count": sum(1 for step in steps if step.priority == "required"),
        },
    )


def runtime_check_step(package_dir: Path, vlm_profiles: list[str], require_ocr: bool) -> WorkflowStep:
    command_parts = [
        "chunking-docs doctor",
        "--require-gpu",
        "--require-qdrant",
        "--require-postgres",
        "--require-embeddings",
        "--require-vision",
    ]
    if require_ocr:
        command_parts.append("--require-ocr")
    if vlm_profiles:
        command_parts.extend(["--vlm-profile", vlm_profiles[0]])
    command_parts.extend(
        [
            "--vlm-memory-margin-ratio",
            "0.1",
            "--output",
            path_arg(package_dir / "runtime_doctor.json"),
        ]
    )
    return WorkflowStep(
        step_id="runtime_check",
        title="Check local runtime",
        area="runtime",
        priority="required",
        reason="Verify GPU, embedding, VLM, Qdrant, and PostgreSQL dependencies before long runs.",
        commands=[" ".join(command_parts)],
        metadata={"require_ocr": require_ocr},
    )


def visual_annotation_step(
    package_dir: Path,
    retrieval_cases: Path,
    vlm_profiles: list[str],
    recommendation: ProcessingRecommendation,
) -> WorkflowStep:
    jobs_path = package_dir / "visual_jobs.jsonl"
    ocr_backend = "paddleocr" if int(recommendation.metadata.get("pages_requiring_ocr_count") or 0) else "none"
    commands = [
        f"chunking-docs plan-visual-jobs --package-dir {path_arg(package_dir)} --output {path_arg(jobs_path)}",
        (
            f"chunking-docs plan-vlm-experiments --package-dir {path_arg(package_dir)} "
            f"--jobs {path_arg(jobs_path)} --profiles {','.join(vlm_profiles)} --ocr {ocr_backend} "
            f"--batch-size 25 --output {path_arg(package_dir / 'vlm_experiment_plan.json')}"
        ),
        (
            f"chunking-docs run-visual-jobs --package-dir {path_arg(package_dir)} "
            f"--jobs {path_arg(jobs_path)} --ocr {ocr_backend} --vlm hf "
            f"--vlm-profile {vlm_profiles[0] if vlm_profiles else 'qwen2_5_vl_7b'} --apply"
        ),
        f"chunking-docs gate-visual-results --results {path_arg(package_dir / 'visual_job_results.jsonl')}",
    ]
    return WorkflowStep(
        step_id="visual_annotations",
        title="Run OCR/VLM visual annotations",
        area="vision",
        priority=recommendation.priority,
        reason=recommendation.message,
        commands=[rewrite_command_paths(command, package_dir, retrieval_cases) for command in commands],
        recommendation_codes=[recommendation.code],
        metadata=recommendation.metadata,
    )


def recommendation_step(
    recommendation: ProcessingRecommendation,
    package_dir: Path,
    retrieval_cases: Path,
) -> WorkflowStep:
    return WorkflowStep(
        step_id=recommendation.code,
        title=title_from_code(recommendation.code),
        area=recommendation.area,
        priority=recommendation.priority,
        reason=recommendation.message,
        commands=[
            rewrite_command_paths(command, package_dir, retrieval_cases)
            for command in recommendation.commands
        ],
        recommendation_codes=[recommendation.code],
        metadata=recommendation.metadata,
    )


def readiness_step(
    package_dir: Path,
    retrieval_cases: Path,
    include_chunking_comparison: bool = False,
    include_qdrant_rag_context: bool = False,
) -> WorkflowStep:
    command_parts = [
        "chunking-docs ingestion-readiness",
        "--package-dir",
        path_arg(package_dir),
        "--require-derived-vector-coverage",
        "--retrieval-cases",
        path_arg(retrieval_cases),
        "--require-retrieval-cases",
        "--min-retrieval-query-terms-per-case",
        "3",
    ]
    if include_chunking_comparison:
        command_parts.extend(
            option.format(
                chunking_comparison=path_arg(package_dir / "chunking_sweep.json")
            )
            for option in CHUNKING_READINESS_GATE_ARGS
        )
    if include_qdrant_rag_context:
        command_parts.extend(
            option.format(
                qdrant_retrieval_config=path_arg(package_dir / "qdrant_retrieval_config.json"),
                rag_context_evaluation=path_arg(
                    package_dir / "qdrant_rag_context_config_eval.json"
                ),
            )
            for option in QDRANT_RAG_READINESS_GATE_ARGS
        )
    command_parts.extend(["--output", path_arg(package_dir / "ingestion_readiness.json")])
    return WorkflowStep(
        step_id="ingestion_readiness",
        title="Gate package before ingestion",
        area="readiness",
        priority="required",
        reason="Run the combined readiness gate before loading Qdrant, PostgreSQL, or a RAG service.",
        commands=[" ".join(command_parts)],
        metadata={
            "include_chunking_comparison": include_chunking_comparison,
            "include_qdrant_rag_context": include_qdrant_rag_context,
        },
    )


def metadata_refresh_step(package_dir: Path) -> WorkflowStep:
    return WorkflowStep(
        step_id="refresh_package_metadata",
        title="Refresh package metadata",
        area="readiness",
        priority="required",
        reason="Record current source checksum, tokenizer config, and chunking selection before final ingestion gates.",
        commands=[
            f"chunking-docs refresh-package-metadata --package-dir {path_arg(package_dir)}"
        ],
    )


def embedding_rebuild_step(package_dir: Path) -> WorkflowStep:
    return WorkflowStep(
        step_id="rebuild_embedding_artifacts",
        title="Rebuild embedding artifacts",
        area="embeddings",
        priority="required",
        reason=(
            "Rebuild Qdrant records after index refresh invalidates stale vector artifacts and before "
            "Qdrant/RAG validation commands run."
        ),
        commands=[
            (
                f"chunking-docs embed-package --package-dir {path_arg(package_dir)} "
                "--caption-backend same-as-text --object-backend same-as-caption "
                "--triple-backend same-as-text"
            )
        ],
    )


def index_refresh_step(package_dir: Path) -> WorkflowStep:
    return WorkflowStep(
        step_id="refresh_package_indexes",
        title="Refresh package indexes",
        area="readiness",
        priority="required",
        reason="Regenerate BM25 tokens from current chunk and visual asset text, then invalidate stale vector artifacts before final gates.",
        commands=[
            f"chunking-docs refresh-package-indexes --package-dir {path_arg(package_dir)}"
        ],
    )


def rewrite_command_paths(command: str, package_dir: Path, retrieval_cases: Path) -> str:
    return command.replace("outputs/package", path_arg(package_dir)).replace(
        "examples/retrieval_cases.jsonl",
        path_arg(retrieval_cases),
    )


def title_from_code(code: str) -> str:
    return code.replace("_", " ").capitalize()


def path_text(path: Path) -> str:
    return path.as_posix()


def path_arg(path: Path) -> str:
    return shlex.quote(path_text(path))
