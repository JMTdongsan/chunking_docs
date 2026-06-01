from __future__ import annotations

import shlex
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from chunking_docs.io import read_jsonl
from chunking_docs.vision.hf_vlm import get_vlm_model_profile
from chunking_docs.vision.jobs import VisualAnnotationJob


class VLMExperimentRecipe(BaseModel):
    name: str
    profile: str
    model_name: str
    model_class: str
    device_map: str
    torch_dtype: str
    max_new_tokens: int
    attn_implementation: str = ""
    jobs_file: str
    doctor_output: str
    results_output: str
    annotations_output: str
    doctor_command: str
    command: str
    batch_commands: list[str] = Field(default_factory=list)
    merge_command: str = ""
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class VLMExperimentJobSummary(BaseModel):
    jobs_file: str
    exists: bool
    total_job_count: int = 0
    selected_job_count: int = 0
    selected_pending_job_count: int = 0
    skipped_by_limit_count: int = 0
    operation_counts: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    asset_kind_counts: dict[str, int] = Field(default_factory=dict)
    asset_scope_counts: dict[str, int] = Field(default_factory=dict)
    page_count: int = 0
    page_min: int | None = None
    page_max: int | None = None
    priority_min: int | None = None
    priority_max: int | None = None


class VLMExperimentBatch(BaseModel):
    batch_id: str
    offset: int
    limit: int
    job_count: int
    pending_job_count: int = 0
    operation_counts: dict[str, int] = Field(default_factory=dict)
    asset_kind_counts: dict[str, int] = Field(default_factory=dict)
    asset_scope_counts: dict[str, int] = Field(default_factory=dict)
    page_count: int = 0
    page_min: int | None = None
    page_max: int | None = None
    priority_min: int | None = None
    priority_max: int | None = None


class VLMExperimentPlan(BaseModel):
    package_dir: str
    jobs_file: str
    profiles: list[str]
    limit: int | None = None
    batch_size: int | None = None
    job_summary: VLMExperimentJobSummary
    batches: list[VLMExperimentBatch] = Field(default_factory=list)
    recipes: list[VLMExperimentRecipe] = Field(default_factory=list)
    compare_command: str
    batch_compare_commands: list[str] = Field(default_factory=list)


def build_vlm_experiment_plan(
    package_dir: Path,
    jobs_file: Path,
    profiles: list[str],
    output_dir: Path | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    ocr: str = "paddleocr",
    ocr_model_lang: str = "korean",
    ocr_device: str = "cpu",
    ocr_min_confidence: float = 0.3,
    ocr_use_gpu: bool = False,
    ocr_enable_mkldnn: bool = False,
    vlm_device_map: str = "auto",
    vlm_torch_dtype: str = "auto",
    vlm_max_new_tokens: int | None = None,
    vlm_attn_implementation: str = "",
    vlm_memory_margin_ratio: float = 0.1,
) -> VLMExperimentPlan:
    output_dir = output_dir or package_dir
    normalized_profiles = [normalize_profile_name(profile) for profile in profiles]
    job_summary = summarize_vlm_experiment_jobs(jobs_file, limit=limit)
    batches = build_vlm_experiment_batches(jobs_file, limit=limit, batch_size=batch_size)
    recipes = [
        vlm_experiment_recipe(
            package_dir=package_dir,
            jobs_file=jobs_file,
            output_dir=output_dir,
            profile_name=profile_name,
            limit=limit,
            batches=batches,
            job_summary=job_summary,
            ocr=ocr,
            ocr_model_lang=ocr_model_lang,
            ocr_device=ocr_device,
            ocr_min_confidence=ocr_min_confidence,
            ocr_use_gpu=ocr_use_gpu,
            ocr_enable_mkldnn=ocr_enable_mkldnn,
            vlm_device_map=vlm_device_map,
            vlm_torch_dtype=vlm_torch_dtype,
            vlm_max_new_tokens=vlm_max_new_tokens,
            vlm_attn_implementation=vlm_attn_implementation,
            vlm_memory_margin_ratio=vlm_memory_margin_ratio,
        )
        for profile_name in normalized_profiles
    ]
    compare_command = build_compare_visual_runs_command(output_dir, recipes)
    batch_compare_commands = build_batch_compare_visual_runs_commands(output_dir, recipes, batches)
    return VLMExperimentPlan(
        package_dir=str(package_dir),
        jobs_file=str(jobs_file),
        profiles=normalized_profiles,
        limit=limit,
        batch_size=normalized_batch_size(batch_size),
        job_summary=job_summary,
        batches=batches,
        recipes=recipes,
        compare_command=compare_command,
        batch_compare_commands=batch_compare_commands,
    )


def vlm_experiment_recipe(
    package_dir: Path,
    jobs_file: Path,
    output_dir: Path,
    profile_name: str,
    limit: int | None,
    batches: list[VLMExperimentBatch],
    job_summary: VLMExperimentJobSummary,
    ocr: str,
    ocr_model_lang: str,
    ocr_device: str,
    ocr_min_confidence: float,
    ocr_use_gpu: bool,
    ocr_enable_mkldnn: bool,
    vlm_device_map: str,
    vlm_torch_dtype: str,
    vlm_max_new_tokens: int | None,
    vlm_attn_implementation: str,
    vlm_memory_margin_ratio: float,
) -> VLMExperimentRecipe:
    profile = get_vlm_model_profile(profile_name)
    device_map = vlm_device_map if vlm_device_map != "auto" else profile.device_map
    torch_dtype = vlm_torch_dtype if vlm_torch_dtype != "auto" else profile.torch_dtype
    max_new_tokens = vlm_max_new_tokens or profile.max_new_tokens
    doctor_output = output_dir / f"runtime_doctor.{profile.name}.json"
    results_output = output_dir / f"visual_job_results.{profile.name}.jsonl"
    annotations_output = output_dir / f"visual_annotations.{profile.name}.jsonl"
    selected_vlm_job_count = job_summary.operation_counts.get("vlm", 0)
    selected_ocr_job_count = job_summary.operation_counts.get("ocr", 0)
    effective_ocr = ocr if selected_ocr_job_count > 0 else "none"
    normalized_effective_ocr = normalize_ocr_name(effective_ocr)
    doctor_args = [
        "chunking-docs",
        "doctor",
        "--output",
        str(doctor_output),
        "--require-gpu",
        "--require-vision",
    ]
    if normalized_effective_ocr in {"paddle", "paddleocr"}:
        doctor_args.append("--require-ocr")
    if normalized_effective_ocr in {"paddle", "paddleocr"} and requires_ocr_gpu(
        ocr_device,
        ocr_use_gpu,
    ):
        doctor_args.append("--require-ocr-gpu")
    doctor_args.extend(
        [
            "--vlm-profile",
            profile.name,
            "--vlm-memory-margin-ratio",
            str(vlm_memory_margin_ratio),
        ]
    )
    doctor_command = quote_command(doctor_args)
    def visual_job_command(
        result_path: Path,
        annotation_path: Path,
        command_limit: int | None = None,
        offset: int = 0,
    ) -> str:
        command_args = [
            "chunking-docs",
            "run-visual-jobs",
            "--package-dir",
            str(package_dir),
            "--jobs",
            str(jobs_file),
            "--results-output",
            str(result_path),
            "--annotations-output",
            str(annotation_path),
            "--ocr",
            effective_ocr,
            "--ocr-model-lang",
            ocr_model_lang,
            "--ocr-device",
            ocr_device,
            "--ocr-min-confidence",
            str(ocr_min_confidence),
            "--vlm",
            "hf",
            "--vlm-profile",
            profile.name,
            "--vlm-device-map",
            device_map,
            "--vlm-torch-dtype",
            torch_dtype,
            "--vlm-max-new-tokens",
            str(max_new_tokens),
        ]
        if offset:
            command_args.extend(["--offset", str(offset)])
        if ocr_use_gpu:
            command_args.append("--ocr-use-gpu")
        if ocr_enable_mkldnn:
            command_args.append("--ocr-enable-mkldnn")
        if vlm_attn_implementation:
            command_args.extend(["--vlm-attn-implementation", vlm_attn_implementation])
        if command_limit is not None:
            command_args.extend(["--limit", str(command_limit)])
        return quote_command(command_args)

    batch_commands = [
        visual_job_command(
            result_path=batch_results_path(output_dir, profile.name, batch.batch_id),
            annotation_path=batch_annotations_path(output_dir, profile.name, batch.batch_id),
            command_limit=batch.limit,
            offset=batch.offset,
        )
        for batch in batches
    ]
    merge_command = build_merge_visual_results_command(
        output_dir,
        profile.name,
        results_output,
        annotations_output,
        batches,
    )
    return VLMExperimentRecipe(
        name=profile.name,
        profile=profile.name,
        model_name=profile.model_name,
        model_class=profile.model_class,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        attn_implementation=vlm_attn_implementation or profile.attn_implementation,
        jobs_file=str(jobs_file),
        doctor_output=str(doctor_output),
        results_output=str(results_output),
        annotations_output=str(annotations_output),
        doctor_command=doctor_command,
        command=visual_job_command(results_output, annotations_output, command_limit=limit),
        batch_commands=batch_commands,
        merge_command=merge_command,
        metadata={
            "profile_notes": profile.notes,
            "min_gpu_memory_mib": profile.min_gpu_memory_mib,
            "selected_job_count": job_summary.selected_job_count,
            "selected_vlm_job_count": selected_vlm_job_count,
            "selected_ocr_job_count": selected_ocr_job_count,
            "requested_ocr_backend": ocr,
            "effective_ocr_backend": effective_ocr,
            "max_generation_tokens_upper_bound": selected_vlm_job_count * max_new_tokens,
            "batch_count": len(batches),
            "batch_size": batches[0].limit if batches else None,
            "vlm_memory_margin_ratio": vlm_memory_margin_ratio,
        },
    )


def build_compare_visual_runs_command(
    output_dir: Path,
    recipes: list[VLMExperimentRecipe],
) -> str:
    args = ["chunking-docs", "compare-visual-runs"]
    for recipe in recipes:
        args.extend(["--run", f"{recipe.name}={recipe.results_output}"])
    args.extend(["--output", str(output_dir / "visual_run_comparison.json"), "--require-same-jobs"])
    return quote_command(args)


def build_batch_compare_visual_runs_commands(
    output_dir: Path,
    recipes: list[VLMExperimentRecipe],
    batches: list[VLMExperimentBatch],
) -> list[str]:
    commands = []
    for batch in batches:
        args = ["chunking-docs", "compare-visual-runs"]
        for recipe in recipes:
            args.extend(
                [
                    "--run",
                    f"{recipe.name}={batch_results_path(output_dir, recipe.name, batch.batch_id)}",
                ]
            )
        args.extend(
            [
                "--output",
                str(output_dir / f"visual_run_comparison.{batch.batch_id}.json"),
                "--require-same-jobs",
            ]
        )
        commands.append(quote_command(args))
    return commands


def build_merge_visual_results_command(
    output_dir: Path,
    profile: str,
    results_output: Path,
    annotations_output: Path,
    batches: list[VLMExperimentBatch],
) -> str:
    if not batches:
        return ""
    args = ["chunking-docs", "merge-visual-results"]
    for batch in batches:
        args.extend(["--results", str(batch_results_path(output_dir, profile, batch.batch_id))])
    args.extend(
        [
            "--output",
            str(results_output),
            "--annotations-output",
            str(annotations_output),
        ]
    )
    return quote_command(args)


def summarize_vlm_experiment_jobs(
    jobs_file: Path,
    limit: int | None = None,
) -> VLMExperimentJobSummary:
    if not jobs_file.exists():
        return VLMExperimentJobSummary(jobs_file=str(jobs_file), exists=False)

    jobs = read_jsonl(jobs_file, VisualAnnotationJob)
    selected_limit = max(0, limit) if limit is not None else None
    selected_jobs = jobs[:selected_limit] if selected_limit is not None else jobs
    operation_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    asset_kind_counts: Counter[str] = Counter()
    asset_scope_counts: Counter[str] = Counter()
    pages = []
    priorities = []
    pending_count = 0
    for job in selected_jobs:
        operation_counts.update(str(operation) for operation in job.operations)
        status_counts[str(job.status)] += 1
        asset_kind_counts[str(job.kind)] += 1
        asset_scope = str(job.metadata.get("asset_scope") or "asset")
        asset_scope_counts[asset_scope] += 1
        pages.append(job.page_no)
        priorities.append(job.priority)
        if job.status == "pending":
            pending_count += 1

    return VLMExperimentJobSummary(
        jobs_file=str(jobs_file),
        exists=True,
        total_job_count=len(jobs),
        selected_job_count=len(selected_jobs),
        selected_pending_job_count=pending_count,
        skipped_by_limit_count=max(0, len(jobs) - len(selected_jobs)),
        operation_counts=dict(sorted(operation_counts.items())),
        status_counts=dict(sorted(status_counts.items())),
        asset_kind_counts=dict(sorted(asset_kind_counts.items())),
        asset_scope_counts=dict(sorted(asset_scope_counts.items())),
        page_count=len(set(pages)),
        page_min=min(pages) if pages else None,
        page_max=max(pages) if pages else None,
        priority_min=min(priorities) if priorities else None,
        priority_max=max(priorities) if priorities else None,
    )


def build_vlm_experiment_batches(
    jobs_file: Path,
    limit: int | None = None,
    batch_size: int | None = None,
) -> list[VLMExperimentBatch]:
    size = normalized_batch_size(batch_size)
    if size is None or not jobs_file.exists():
        return []
    jobs = read_jsonl(jobs_file, VisualAnnotationJob)
    selected_limit = max(0, limit) if limit is not None else None
    selected_jobs = jobs[:selected_limit] if selected_limit is not None else jobs
    batches = []
    for batch_index, offset in enumerate(range(0, len(selected_jobs), size), start=1):
        batch_jobs = selected_jobs[offset : offset + size]
        batches.append(
            summarize_vlm_experiment_batch(
                batch_id=f"batch_{batch_index:03d}",
                offset=offset,
                jobs=batch_jobs,
            )
        )
    return batches


def summarize_vlm_experiment_batch(
    batch_id: str,
    offset: int,
    jobs: list[VisualAnnotationJob],
) -> VLMExperimentBatch:
    operation_counts: Counter[str] = Counter()
    asset_kind_counts: Counter[str] = Counter()
    asset_scope_counts: Counter[str] = Counter()
    pages = []
    priorities = []
    pending_count = 0
    for job in jobs:
        operation_counts.update(str(operation) for operation in job.operations)
        asset_kind_counts[str(job.kind)] += 1
        asset_scope = str(job.metadata.get("asset_scope") or "asset")
        asset_scope_counts[asset_scope] += 1
        pages.append(job.page_no)
        priorities.append(job.priority)
        if job.status == "pending":
            pending_count += 1
    return VLMExperimentBatch(
        batch_id=batch_id,
        offset=offset,
        limit=len(jobs),
        job_count=len(jobs),
        pending_job_count=pending_count,
        operation_counts=dict(sorted(operation_counts.items())),
        asset_kind_counts=dict(sorted(asset_kind_counts.items())),
        asset_scope_counts=dict(sorted(asset_scope_counts.items())),
        page_count=len(set(pages)),
        page_min=min(pages) if pages else None,
        page_max=max(pages) if pages else None,
        priority_min=min(priorities) if priorities else None,
        priority_max=max(priorities) if priorities else None,
    )


def normalized_batch_size(batch_size: int | None) -> int | None:
    if batch_size is None:
        return None
    return batch_size if batch_size > 0 else None


def batch_results_path(output_dir: Path, profile: str, batch_id: str) -> Path:
    return output_dir / f"visual_job_results.{profile}.{batch_id}.jsonl"


def batch_annotations_path(output_dir: Path, profile: str, batch_id: str) -> Path:
    return output_dir / f"visual_annotations.{profile}.{batch_id}.jsonl"


def parse_profile_list(value: str) -> list[str]:
    return [normalize_profile_name(profile) for profile in value.split(",") if profile.strip()]


def normalize_profile_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def normalize_ocr_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def requires_ocr_gpu(ocr_device: str, ocr_use_gpu: bool) -> bool:
    normalized_device = ocr_device.strip().lower()
    return ocr_use_gpu or normalized_device.startswith(("gpu", "cuda"))


def quote_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)
