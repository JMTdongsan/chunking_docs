from __future__ import annotations

import shlex
from pathlib import Path

from pydantic import BaseModel, Field

from chunking_docs.vision.hf_vlm import get_vlm_model_profile


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
    results_output: str
    annotations_output: str
    command: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class VLMExperimentPlan(BaseModel):
    package_dir: str
    jobs_file: str
    profiles: list[str]
    limit: int | None = None
    recipes: list[VLMExperimentRecipe] = Field(default_factory=list)
    compare_command: str


def build_vlm_experiment_plan(
    package_dir: Path,
    jobs_file: Path,
    profiles: list[str],
    output_dir: Path | None = None,
    limit: int | None = None,
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
) -> VLMExperimentPlan:
    output_dir = output_dir or package_dir
    normalized_profiles = [normalize_profile_name(profile) for profile in profiles]
    recipes = [
        vlm_experiment_recipe(
            package_dir=package_dir,
            jobs_file=jobs_file,
            output_dir=output_dir,
            profile_name=profile_name,
            limit=limit,
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
        )
        for profile_name in normalized_profiles
    ]
    compare_command = build_compare_visual_runs_command(output_dir, recipes)
    return VLMExperimentPlan(
        package_dir=str(package_dir),
        jobs_file=str(jobs_file),
        profiles=normalized_profiles,
        limit=limit,
        recipes=recipes,
        compare_command=compare_command,
    )


def vlm_experiment_recipe(
    package_dir: Path,
    jobs_file: Path,
    output_dir: Path,
    profile_name: str,
    limit: int | None,
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
) -> VLMExperimentRecipe:
    profile = get_vlm_model_profile(profile_name)
    device_map = vlm_device_map if vlm_device_map != "auto" else profile.device_map
    torch_dtype = vlm_torch_dtype if vlm_torch_dtype != "auto" else profile.torch_dtype
    max_new_tokens = vlm_max_new_tokens or profile.max_new_tokens
    results_output = output_dir / f"visual_job_results.{profile.name}.jsonl"
    annotations_output = output_dir / f"visual_annotations.{profile.name}.jsonl"
    command_args = [
        "chunking-docs",
        "run-visual-jobs",
        "--package-dir",
        str(package_dir),
        "--jobs",
        str(jobs_file),
        "--results-output",
        str(results_output),
        "--annotations-output",
        str(annotations_output),
        "--ocr",
        ocr,
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
    if ocr_use_gpu:
        command_args.append("--ocr-use-gpu")
    if ocr_enable_mkldnn:
        command_args.append("--ocr-enable-mkldnn")
    if vlm_attn_implementation:
        command_args.extend(["--vlm-attn-implementation", vlm_attn_implementation])
    if limit is not None:
        command_args.extend(["--limit", str(limit)])
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
        results_output=str(results_output),
        annotations_output=str(annotations_output),
        command=quote_command(command_args),
        metadata={"profile_notes": profile.notes},
    )


def build_compare_visual_runs_command(
    output_dir: Path,
    recipes: list[VLMExperimentRecipe],
) -> str:
    args = ["chunking-docs", "compare-visual-runs"]
    for recipe in recipes:
        args.extend(["--run", f"{recipe.name}={recipe.results_output}"])
    args.extend(["--output", str(output_dir / "visual_run_comparison.json")])
    return quote_command(args)


def parse_profile_list(value: str) -> list[str]:
    return [normalize_profile_name(profile) for profile in value.split(",") if profile.strip()]


def normalize_profile_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def quote_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)
