from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.io import read_jsonl
from chunking_docs.runtime import RuntimeReport
from chunking_docs.vision.jobs import VisualJobRunResult


class VLMExperimentPlanGateCheck(BaseModel):
    name: str
    passed: bool
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VLMExperimentPlanGateReport(BaseModel):
    plan_path: str
    passed: bool
    failed_checks: list[str] = Field(default_factory=list)
    profile_count: int
    recipe_count: int
    doctor_output_count: int = 0
    existing_doctor_output_count: int = 0
    passed_doctor_output_count: int = 0
    results_output_count: int = 0
    existing_results_output_count: int = 0
    completed_result_profile_count: int = 0
    annotations_output_count: int = 0
    existing_annotations_output_count: int = 0
    union_job_count: int = 0
    shared_job_count: int = 0
    job_set_mismatch: bool = False
    checks: list[VLMExperimentPlanGateCheck] = Field(default_factory=list)


def gate_vlm_experiment_plan(
    plan_path: Path,
    min_profile_count: int = 1,
    require_doctor_outputs: bool = False,
    require_results: bool = False,
    require_annotations: bool = False,
    min_completed_result_profiles: int = 0,
    require_same_result_jobs: bool = False,
) -> VLMExperimentPlanGateReport:
    payload = read_json_dict(plan_path)
    profiles = [profile for profile in payload.get("profiles", []) if isinstance(profile, str)]
    recipes = [recipe for recipe in payload.get("recipes", []) if isinstance(recipe, dict)]
    recipe_paths = [recipe_output_paths(recipe, plan_path) for recipe in recipes]
    doctor_reports = load_runtime_reports(recipe_paths)
    result_runs = load_result_runs(recipe_paths)
    job_set_report = compare_result_job_sets(result_runs)
    checks = [
        gate_check(
            "min_profile_count",
            len(profiles) >= min_profile_count,
            f"Plan should include at least {min_profile_count} VLM profile(s).",
            {"profile_count": len(profiles), "min_profile_count": min_profile_count},
        ),
        gate_check(
            "recipe_profile_count",
            len(recipes) == len(profiles),
            "Plan should contain one recipe per VLM profile.",
            {"profile_count": len(profiles), "recipe_count": len(recipes)},
        ),
    ]
    checks.extend(
        doctor_output_checks(
            recipe_paths,
            doctor_reports,
            require_doctor_outputs=require_doctor_outputs,
        )
    )
    checks.extend(
        result_output_checks(
            recipe_paths,
            result_runs,
            require_results=require_results,
            min_completed_result_profiles=min_completed_result_profiles,
        )
    )
    checks.extend(
        annotation_output_checks(
            recipe_paths,
            require_annotations=require_annotations,
        )
    )
    checks.append(
        gate_check(
            "same_result_jobs",
            not require_same_result_jobs or not job_set_report["job_set_mismatch"],
            "Existing result files should contain the same visual job IDs.",
            job_set_report,
        )
    )
    failed_checks = [check.name for check in checks if not check.passed]
    return VLMExperimentPlanGateReport(
        plan_path=str(plan_path),
        passed=not failed_checks,
        failed_checks=failed_checks,
        profile_count=len(profiles),
        recipe_count=len(recipes),
        doctor_output_count=count_declared(recipe_paths, "doctor_output"),
        existing_doctor_output_count=sum(
            1
            for paths in recipe_paths
            if paths.doctor_output is not None and paths.doctor_output.exists()
        ),
        passed_doctor_output_count=sum(1 for report in doctor_reports.values() if report.passed),
        results_output_count=count_declared(recipe_paths, "results_output"),
        existing_results_output_count=len(result_runs),
        completed_result_profile_count=sum(
            1 for results in result_runs.values() if any(result.status == "completed" for result in results)
        ),
        annotations_output_count=count_declared(recipe_paths, "annotations_output"),
        existing_annotations_output_count=sum(
            1
            for paths in recipe_paths
            if paths.annotations_output is not None and paths.annotations_output.exists()
        ),
        union_job_count=job_set_report["union_job_count"],
        shared_job_count=job_set_report["shared_job_count"],
        job_set_mismatch=job_set_report["job_set_mismatch"],
        checks=checks,
    )


class VLMRecipeOutputPaths(BaseModel):
    name: str
    doctor_output: Path | None = None
    results_output: Path | None = None
    annotations_output: Path | None = None

    model_config = {"arbitrary_types_allowed": True}


def recipe_output_paths(recipe: dict[str, Any], plan_path: Path) -> VLMRecipeOutputPaths:
    name = recipe.get("name")
    return VLMRecipeOutputPaths(
        name=name if isinstance(name, str) else "",
        doctor_output=resolve_declared_path(recipe.get("doctor_output"), plan_path),
        results_output=resolve_declared_path(recipe.get("results_output"), plan_path),
        annotations_output=resolve_declared_path(recipe.get("annotations_output"), plan_path),
    )


def resolve_declared_path(value: Any, plan_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return plan_path.parent / path


def doctor_output_checks(
    recipe_paths: list[VLMRecipeOutputPaths],
    doctor_reports: dict[str, RuntimeReport],
    require_doctor_outputs: bool,
) -> list[VLMExperimentPlanGateCheck]:
    declared = [paths for paths in recipe_paths if paths.doctor_output is not None]
    existing = [
        paths
        for paths in declared
        if paths.doctor_output is not None and paths.doctor_output.exists()
    ]
    passed_reports = [report for report in doctor_reports.values() if report.passed]
    checks = [
        gate_check(
            "doctor_outputs_declared",
            not require_doctor_outputs or len(declared) == len(recipe_paths),
            "Every recipe should declare a runtime doctor output path.",
            {"declared_count": len(declared), "recipe_count": len(recipe_paths)},
        ),
        gate_check(
            "doctor_outputs_exist",
            not require_doctor_outputs or len(existing) == len(recipe_paths),
            "Every recipe runtime doctor output should exist.",
            {"existing_count": len(existing), "recipe_count": len(recipe_paths)},
        ),
        gate_check(
            "doctor_outputs_passed",
            not require_doctor_outputs or len(passed_reports) == len(recipe_paths),
            "Every recipe runtime doctor output should pass.",
            {"passed_count": len(passed_reports), "recipe_count": len(recipe_paths)},
        ),
    ]
    return checks


def result_output_checks(
    recipe_paths: list[VLMRecipeOutputPaths],
    result_runs: dict[str, list[VisualJobRunResult]],
    require_results: bool,
    min_completed_result_profiles: int,
) -> list[VLMExperimentPlanGateCheck]:
    declared = [paths for paths in recipe_paths if paths.results_output is not None]
    existing_count = len(result_runs)
    completed_profile_count = sum(
        1 for results in result_runs.values() if any(result.status == "completed" for result in results)
    )
    return [
        gate_check(
            "results_outputs_declared",
            not require_results or len(declared) == len(recipe_paths),
            "Every recipe should declare a visual job results output path.",
            {"declared_count": len(declared), "recipe_count": len(recipe_paths)},
        ),
        gate_check(
            "results_outputs_exist",
            not require_results or existing_count == len(recipe_paths),
            "Every recipe visual job results output should exist.",
            {"existing_count": existing_count, "recipe_count": len(recipe_paths)},
        ),
        gate_check(
            "min_completed_result_profiles",
            completed_profile_count >= min_completed_result_profiles,
            "Enough profile result files should contain at least one completed visual job.",
            {
                "completed_profile_count": completed_profile_count,
                "min_completed_result_profiles": min_completed_result_profiles,
            },
        ),
    ]


def annotation_output_checks(
    recipe_paths: list[VLMRecipeOutputPaths],
    require_annotations: bool,
) -> list[VLMExperimentPlanGateCheck]:
    declared = [paths for paths in recipe_paths if paths.annotations_output is not None]
    existing = [
        paths
        for paths in declared
        if paths.annotations_output is not None and paths.annotations_output.exists()
    ]
    return [
        gate_check(
            "annotations_outputs_declared",
            not require_annotations or len(declared) == len(recipe_paths),
            "Every recipe should declare a visual annotation output path.",
            {"declared_count": len(declared), "recipe_count": len(recipe_paths)},
        ),
        gate_check(
            "annotations_outputs_exist",
            not require_annotations or len(existing) == len(recipe_paths),
            "Every recipe visual annotation output should exist.",
            {"existing_count": len(existing), "recipe_count": len(recipe_paths)},
        ),
    ]


def load_runtime_reports(recipe_paths: list[VLMRecipeOutputPaths]) -> dict[str, RuntimeReport]:
    reports = {}
    for paths in recipe_paths:
        if paths.doctor_output is None or not paths.doctor_output.exists():
            continue
        try:
            reports[paths.name] = RuntimeReport.model_validate_json(
                paths.doctor_output.read_text(encoding="utf-8")
            )
        except ValueError:
            continue
    return reports


def load_result_runs(
    recipe_paths: list[VLMRecipeOutputPaths],
) -> dict[str, list[VisualJobRunResult]]:
    result_runs = {}
    for paths in recipe_paths:
        if paths.results_output is None or not paths.results_output.exists():
            continue
        try:
            result_runs[paths.name] = read_jsonl(paths.results_output, VisualJobRunResult)
        except ValueError:
            continue
    return result_runs


def compare_result_job_sets(
    result_runs: dict[str, list[VisualJobRunResult]],
) -> dict[str, int | bool]:
    job_sets = [
        {str(result.job_id).strip() for result in results if str(result.job_id).strip()}
        for results in result_runs.values()
    ]
    if not job_sets:
        return {"union_job_count": 0, "shared_job_count": 0, "job_set_mismatch": False}
    union_jobs = set().union(*job_sets)
    shared_jobs = set.intersection(*job_sets)
    return {
        "union_job_count": len(union_jobs),
        "shared_job_count": len(shared_jobs),
        "job_set_mismatch": any(job_set != shared_jobs for job_set in job_sets),
    }


def count_declared(recipe_paths: list[VLMRecipeOutputPaths], field_name: str) -> int:
    return sum(1 for paths in recipe_paths if getattr(paths, field_name) is not None)


def gate_check(
    name: str,
    passed: bool,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> VLMExperimentPlanGateCheck:
    return VLMExperimentPlanGateCheck(
        name=name,
        passed=passed,
        message=message,
        metadata=metadata or {},
    )


def read_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VLM experiment plan must be a JSON object.")
    return payload
