from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.analysis.chunking_defaults import CHUNKING_READINESS_GATE_ARGS
from chunking_docs.analysis.characterize import PackageCharacteristics, ProcessingRecommendation
from chunking_docs.analysis.qdrant_defaults import (
    QDRANT_ADAPTIVE_ROUTE_READINESS_GATE_ARGS,
    QDRANT_RAG_READINESS_GATE_ARGS,
    qdrant_source_precision_readiness_gate_args,
)


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

BASE_VISUAL_READINESS_GATE_ARGS = [
    "--require-visual-annotations",
    "--require-visual-quality",
    "--max-visual-failed-count 0",
]

VLM_VISUAL_READINESS_GATE_ARGS = [
    "--min-vlm-summary-coverage 0.9",
    "--min-vlm-json-parse-rate 0.9",
    "--min-vlm-object-coverage 0.5",
    "--min-object-bbox-coverage 0.5",
]

VISUAL_RUN_COMPARISON_READINESS_GATE_ARGS = [
    "--visual-run-comparison {visual_run_comparison}",
    "--require-visual-run-comparison",
    "--require-visual-run-same-jobs",
    "--min-visual-run-count 2",
]

def build_ingestion_workflow_plan(
    characteristics: PackageCharacteristics,
    package_dir: Path = Path("outputs/package"),
    retrieval_cases: Path | None = None,
    vlm_profiles: list[str] | None = None,
) -> IngestionWorkflowPlan:
    case_path = retrieval_cases or package_dir / "retrieval_cases.jsonl"
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
        steps.append(recommendation_step(recommendations_by_code["build_page_tiles"], package_dir, case_path))
        handled_codes.add("build_page_tiles")
    if "prioritize_visual_annotations" in recommendations_by_code:
        steps.append(
            visual_annotation_step(
                package_dir,
                case_path,
                profiles,
                recommendations_by_code["prioritize_visual_annotations"],
            )
        )
        handled_codes.add("prioritize_visual_annotations")

    for code in RECOMMENDATION_ORDER:
        recommendation = recommendations_by_code.get(code)
        if recommendation is None or code in handled_codes or code in POST_INDEX_RECOMMENDATIONS:
            continue
        steps.append(recommendation_step(recommendation, package_dir, case_path))
        handled_codes.add(code)

    for recommendation in characteristics.recommendations:
        if recommendation.code in handled_codes or recommendation.code in POST_INDEX_RECOMMENDATIONS:
            continue
        steps.append(recommendation_step(recommendation, package_dir, case_path))
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
        steps.append(recommendation_step(recommendation, package_dir, case_path))
        handled_codes.add(code)
    for recommendation in characteristics.recommendations:
        if recommendation.code in handled_codes or recommendation.code not in POST_INDEX_RECOMMENDATIONS:
            continue
        steps.append(recommendation_step(recommendation, package_dir, case_path))
        handled_codes.add(recommendation.code)

    steps.append(metadata_refresh_step(package_dir))
    qdrant_route_preset = qdrant_retrieval_route_preset(
        recommendations_by_code.get("validate_qdrant_rag_context")
    )
    qdrant_vector_names = qdrant_retrieval_vector_names(
        recommendations_by_code.get("validate_qdrant_rag_context")
    )
    visual_probe_gate_args = visual_probe_readiness_gate_args(recommendations_by_code)
    steps.append(
        readiness_step(
            package_dir,
            case_path,
            include_visual_quality=requires_visual_quality_gate(
                characteristics,
                recommendations_by_code.get("prioritize_visual_annotations"),
            ),
            include_vlm_quality=requires_vlm_quality_gate(
                characteristics,
                recommendations_by_code.get("prioritize_visual_annotations"),
            ),
            include_visual_run_comparison=requires_visual_run_comparison(
                recommendations_by_code.get("prioritize_visual_annotations")
            ),
            include_chunking_comparison="compare_multimodal_hierarchical_chunking"
            in recommendations_by_code,
            include_qdrant_rag_context=needs_qdrant_rag_context,
            qdrant_route_preset=qdrant_route_preset,
            qdrant_vector_names=qdrant_vector_names,
            visual_probe_gate_args=visual_probe_gate_args,
        )
    )
    return IngestionWorkflowPlan(
        package_dir=path_text(package_dir),
        retrieval_cases=path_text(case_path),
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
    pending_ocr = pending_visual_ocr_count(recommendation) > 0
    pending_vlm = pending_visual_vlm_count(recommendation) > 0
    ocr_backend = "paddleocr" if pending_ocr else "none"
    commands = [plan_visual_jobs_command(package_dir, jobs_path, pending_ocr, pending_vlm)]
    profiles = vlm_profiles or ["qwen2_5_vl_7b"]
    if pending_vlm:
        commands.append(
            f"chunking-docs plan-vlm-experiments --package-dir {path_arg(package_dir)} "
            f"--jobs {path_arg(jobs_path)} --profiles {','.join(profiles)} --ocr {ocr_backend} "
            f"--batch-size 25 --output {path_arg(package_dir / 'vlm_experiment_plan.json')}"
        )
        commands.extend(
            visual_profile_doctor_command(package_dir, ocr_backend, profile)
            for profile in profiles
        )
        commands.append(
            vlm_experiment_plan_gate_command(
                package_dir,
                min_profile_count=len(profiles),
                require_doctor_outputs=True,
                output_name="vlm_experiment_plan_gate.runtime.json",
            )
        )
        commands.extend(
            visual_profile_run_command(package_dir, jobs_path, ocr_backend, profile)
            for profile in profiles
        )
        commands.extend(
            [
                vlm_experiment_plan_gate_command(
                    package_dir,
                    min_profile_count=len(profiles),
                    require_doctor_outputs=True,
                    require_results=True,
                    require_annotations=True,
                    min_completed_result_profiles=len(profiles),
                    require_same_result_jobs=True,
                    output_name="vlm_experiment_plan_gate.results.json",
                ),
                f"chunking-docs gate-visual-results --results {path_arg(visual_profile_results_path(package_dir, profiles[0]))}",
                visual_run_comparison_command(package_dir, profiles),
                (
                    f"chunking-docs apply-annotations "
                    f"{path_arg(visual_profile_annotations_path(package_dir, profiles[0]))} "
                    f"--package-dir {path_arg(package_dir)}"
                ),
            ]
        )
    else:
        commands.extend(
            [
                visual_ocr_run_command(package_dir, jobs_path, ocr_backend),
                f"chunking-docs gate-visual-results --results {path_arg(visual_ocr_results_path(package_dir))}",
                (
                    f"chunking-docs apply-annotations "
                    f"{path_arg(visual_ocr_annotations_path(package_dir))} "
                    f"--package-dir {path_arg(package_dir)}"
                ),
            ]
        )
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


def pending_visual_ocr_count(recommendation: ProcessingRecommendation | None) -> int:
    if recommendation is None:
        return 0
    return int(recommendation.metadata.get("pages_requiring_ocr_count") or 0)


def pending_visual_vlm_count(recommendation: ProcessingRecommendation | None) -> int:
    if recommendation is None:
        return 0
    return int(recommendation.metadata.get("pages_requiring_vlm_count") or 0)


def requires_visual_run_comparison(recommendation: ProcessingRecommendation | None) -> bool:
    return pending_visual_vlm_count(recommendation) > 0


def requires_visual_quality_gate(
    characteristics: PackageCharacteristics,
    recommendation: ProcessingRecommendation | None,
) -> bool:
    return recommendation is not None or bool(characteristics.visual.asset_kind_counts)


def requires_vlm_quality_gate(
    characteristics: PackageCharacteristics,
    recommendation: ProcessingRecommendation | None,
) -> bool:
    return pending_visual_vlm_count(recommendation) > 0 or characteristics.visual.vlm_object_count > 0


def qdrant_retrieval_route_preset(recommendation: ProcessingRecommendation | None) -> str:
    if recommendation is None:
        return ""
    value = recommendation.metadata.get("retrieval_route_preset")
    return str(value).strip() if value is not None else ""


def qdrant_retrieval_vector_names(
    recommendation: ProcessingRecommendation | None,
) -> list[str]:
    if recommendation is None:
        return []
    value = recommendation.metadata.get("recommended_vector_names")
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def visual_probe_readiness_gate_args(
    recommendations_by_code: dict[str, ProcessingRecommendation],
) -> list[str]:
    args: list[str] = []
    distinct_asset_thresholds: list[int] = []
    image_probe = recommendations_by_code.get("generate_visual_image_probe_cases")
    if image_probe is not None:
        case_threshold = positive_metadata_int(
            image_probe,
            "recommended_image_probe_case_threshold",
        )
        asset_threshold = positive_metadata_int(
            image_probe,
            "recommended_distinct_asset_threshold",
        )
        if case_threshold:
            args.append(
                f"--min-retrieval-case-group-count case_source:visual_image_probe={case_threshold}"
            )
        if asset_threshold:
            distinct_asset_thresholds.append(asset_threshold)
            args.append(
                "--min-retrieval-case-group-distinct-targets "
                f"case_source:visual_image_probe:asset={asset_threshold}"
            )

    object_probe = recommendations_by_code.get("generate_visual_object_probe_cases")
    if object_probe is not None:
        case_threshold = positive_metadata_int(
            object_probe,
            "recommended_object_probe_case_threshold",
        )
        asset_threshold = positive_metadata_int(
            object_probe,
            "recommended_distinct_asset_threshold",
        )
        if case_threshold:
            args.append(
                f"--min-retrieval-case-group-count case_source:visual_object_probe={case_threshold}"
            )
        if asset_threshold:
            distinct_asset_thresholds.append(asset_threshold)
            args.append(
                "--min-retrieval-case-group-distinct-targets "
                f"case_source:visual_object_probe:asset={asset_threshold}"
            )
        args.extend(
            [
                "--max-retrieval-case-group-cases-per-target "
                "case_source:visual_object_probe:asset=3",
                "--require-visual-only-object-probes",
            ]
        )

    if distinct_asset_thresholds:
        args.append(
            f"--min-retrieval-distinct-asset-targets {max(distinct_asset_thresholds)}"
        )
    return args


def positive_metadata_int(recommendation: ProcessingRecommendation, key: str) -> int:
    value = recommendation.metadata.get(key)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def plan_visual_jobs_command(
    package_dir: Path,
    jobs_path: Path,
    include_ocr: bool,
    include_vlm: bool,
) -> str:
    command_parts = [
        "chunking-docs",
        "plan-visual-jobs",
        "--package-dir",
        path_arg(package_dir),
        "--output",
        path_arg(jobs_path),
    ]
    if not include_ocr:
        command_parts.append("--no-include-ocr")
    if not include_vlm:
        command_parts.append("--no-include-vlm")
    return " ".join(command_parts)


def visual_profile_run_command(
    package_dir: Path,
    jobs_path: Path,
    ocr_backend: str,
    profile: str,
) -> str:
    return (
        f"chunking-docs run-visual-jobs --package-dir {path_arg(package_dir)} "
        f"--jobs {path_arg(jobs_path)} "
        f"--results-output {path_arg(visual_profile_results_path(package_dir, profile))} "
        f"--annotations-output {path_arg(visual_profile_annotations_path(package_dir, profile))} "
        f"--ocr {shlex.quote(ocr_backend)} --vlm hf --vlm-profile {shlex.quote(profile)}"
    )


def visual_profile_doctor_command(package_dir: Path, ocr_backend: str, profile: str) -> str:
    command_parts = [
        "chunking-docs",
        "doctor",
        "--output",
        path_arg(package_dir / f"runtime_doctor.{profile}.json"),
        "--require-gpu",
        "--require-vision",
    ]
    if ocr_backend != "none":
        command_parts.append("--require-ocr")
    command_parts.extend(
        [
            "--vlm-profile",
            shlex.quote(profile),
            "--vlm-memory-margin-ratio",
            "0.1",
        ]
    )
    return " ".join(command_parts)


def vlm_experiment_plan_gate_command(
    package_dir: Path,
    min_profile_count: int,
    output_name: str,
    require_doctor_outputs: bool = False,
    require_results: bool = False,
    require_annotations: bool = False,
    min_completed_result_profiles: int = 0,
    require_same_result_jobs: bool = False,
) -> str:
    command_parts = [
        "chunking-docs",
        "gate-vlm-experiment-plan",
        "--plan",
        path_arg(package_dir / "vlm_experiment_plan.json"),
        "--min-profile-count",
        str(min_profile_count),
    ]
    if require_doctor_outputs:
        command_parts.append("--require-doctor-outputs")
    if require_results:
        command_parts.append("--require-results")
    if require_annotations:
        command_parts.append("--require-annotations")
    if min_completed_result_profiles:
        command_parts.extend(
            [
                "--min-completed-result-profiles",
                str(min_completed_result_profiles),
            ]
        )
    if require_same_result_jobs:
        command_parts.append("--require-same-result-jobs")
    command_parts.extend(["--output", path_arg(package_dir / output_name)])
    return " ".join(command_parts)


def visual_ocr_run_command(
    package_dir: Path,
    jobs_path: Path,
    ocr_backend: str,
) -> str:
    return (
        f"chunking-docs run-visual-jobs --package-dir {path_arg(package_dir)} "
        f"--jobs {path_arg(jobs_path)} "
        f"--results-output {path_arg(visual_ocr_results_path(package_dir))} "
        f"--annotations-output {path_arg(visual_ocr_annotations_path(package_dir))} "
        f"--ocr {shlex.quote(ocr_backend)} --vlm none"
    )


def visual_profile_results_path(package_dir: Path, profile: str) -> Path:
    return package_dir / f"visual_job_results.{profile}.jsonl"


def visual_profile_annotations_path(package_dir: Path, profile: str) -> Path:
    return package_dir / f"visual_annotations.{profile}.jsonl"


def visual_ocr_results_path(package_dir: Path) -> Path:
    return package_dir / "visual_job_results.ocr.jsonl"


def visual_ocr_annotations_path(package_dir: Path) -> Path:
    return package_dir / "visual_annotations.ocr.jsonl"


def visual_run_comparison_command(package_dir: Path, vlm_profiles: list[str]) -> str:
    profiles = vlm_profiles or ["qwen2_5_vl_7b"]
    command_parts = ["chunking-docs compare-visual-runs"]
    for profile in profiles:
        run_path = visual_profile_results_path(package_dir, profile)
        command_parts.extend(["--run", shlex.quote(f"{profile}={path_text(run_path)}")])
    command_parts.extend(
        [
            "--output",
            path_arg(package_dir / "visual_run_comparison.json"),
            "--require-same-jobs",
        ]
    )
    return " ".join(command_parts)


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
    include_visual_quality: bool = False,
    include_vlm_quality: bool = False,
    include_visual_run_comparison: bool = False,
    include_chunking_comparison: bool = False,
    include_qdrant_rag_context: bool = False,
    qdrant_route_preset: str = "",
    qdrant_vector_names: list[str] | None = None,
    visual_probe_gate_args: list[str] | None = None,
) -> WorkflowStep:
    command_parts = [
        "chunking-docs ingestion-readiness",
        "--package-dir",
        path_arg(package_dir),
        "--runtime-report",
        path_arg(package_dir / "runtime_doctor.json"),
        "--require-runtime-report",
        "--require-derived-vector-coverage",
        "--retrieval-cases",
        path_arg(retrieval_cases),
        "--require-retrieval-cases",
        "--min-retrieval-query-terms-per-case",
        "3",
        "--max-retrieval-expected-targets-per-case",
        "5",
    ]
    if visual_probe_gate_args:
        command_parts.extend(visual_probe_gate_args)
    if include_visual_quality:
        command_parts.extend(BASE_VISUAL_READINESS_GATE_ARGS)
    if include_vlm_quality:
        command_parts.extend(VLM_VISUAL_READINESS_GATE_ARGS)
    if include_visual_run_comparison:
        command_parts.extend(
            option.format(
                visual_run_comparison=path_arg(package_dir / "visual_run_comparison.json")
            )
            for option in VISUAL_RUN_COMPARISON_READINESS_GATE_ARGS
        )
    if include_chunking_comparison:
        command_parts.extend(
            option.format(
                chunking_comparison=path_arg(package_dir / "chunking_sweep.json")
            )
            for option in CHUNKING_READINESS_GATE_ARGS
        )
    if include_qdrant_rag_context:
        qdrant_gate_args = list(QDRANT_RAG_READINESS_GATE_ARGS)
        qdrant_gate_args.extend(qdrant_source_precision_readiness_gate_args(qdrant_vector_names or []))
        if qdrant_route_preset in {"adaptive", "visual-object-graph"}:
            qdrant_gate_args.extend(QDRANT_ADAPTIVE_ROUTE_READINESS_GATE_ARGS)
        command_parts.extend(
            option.format(
                qdrant_retrieval_config=path_arg(package_dir / "qdrant_retrieval_config.json"),
                qdrant_retrieval_config_evaluation=path_arg(
                    package_dir / "qdrant_retrieval_config_eval.json"
                ),
                rag_context_evaluation=path_arg(
                    package_dir / "qdrant_rag_context_config_eval.json"
                ),
            )
            for option in qdrant_gate_args
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
            "include_visual_quality": include_visual_quality,
            "include_chunking_comparison": include_chunking_comparison,
            "include_qdrant_rag_context": include_qdrant_rag_context,
            "qdrant_route_preset": qdrant_route_preset or None,
            "visual_probe_gate_args": visual_probe_gate_args or [],
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
