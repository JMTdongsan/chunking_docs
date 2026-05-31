from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
from typing import Any

from pydantic import BaseModel, Field


class DependencyStatus(BaseModel):
    name: str
    module: str
    package: str
    installed: bool
    version: str | None = None


class GPUDevice(BaseModel):
    name: str
    memory_total_mib: int | None = None
    driver_version: str | None = None


class RuntimeCheck(BaseModel):
    name: str
    passed: bool
    severity: str = "error"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeReport(BaseModel):
    passed: bool
    gpus: list[GPUDevice] = Field(default_factory=list)
    torch_cuda_available: bool | None = None
    torch_cuda_device_count: int | None = None
    paddle_cuda_available: bool | None = None
    paddle_cuda_device_count: int | None = None
    dependencies: dict[str, DependencyStatus] = Field(default_factory=dict)
    checks: list[RuntimeCheck] = Field(default_factory=list)


DEPENDENCIES = {
    "qdrant": ("qdrant_client", "qdrant-client"),
    "postgres": ("psycopg", "psycopg"),
    "pgvector": ("pgvector", "pgvector"),
    "sentence_transformers": ("sentence_transformers", "sentence-transformers"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
    "accelerate": ("accelerate", "accelerate"),
    "paddleocr": ("paddleocr", "paddleocr"),
    "paddlepaddle": ("paddle", "paddlepaddle"),
}


def inspect_runtime(
    require_gpu: bool = False,
    require_qdrant: bool = False,
    require_postgres: bool = False,
    require_embeddings: bool = False,
    require_ocr: bool = False,
    require_vision: bool = False,
) -> RuntimeReport:
    dependencies = {name: dependency_status(name, module, package) for name, (module, package) in DEPENDENCIES.items()}
    return build_runtime_report(
        dependencies=dependencies,
        gpus=detect_gpus(),
        torch_cuda=detect_torch_cuda(dependencies.get("torch")),
        paddle_cuda=detect_paddle_cuda(dependencies.get("paddlepaddle"))
        if require_gpu and require_ocr
        else (None, None),
        require_gpu=require_gpu,
        require_qdrant=require_qdrant,
        require_postgres=require_postgres,
        require_embeddings=require_embeddings,
        require_ocr=require_ocr,
        require_vision=require_vision,
    )


def build_runtime_report(
    dependencies: dict[str, DependencyStatus],
    gpus: list[GPUDevice],
    torch_cuda: tuple[bool | None, int | None] = (None, None),
    paddle_cuda: tuple[bool | None, int | None] = (None, None),
    require_gpu: bool = False,
    require_qdrant: bool = False,
    require_postgres: bool = False,
    require_embeddings: bool = False,
    require_ocr: bool = False,
    require_vision: bool = False,
) -> RuntimeReport:
    checks = []
    if require_gpu:
        checks.append(
            RuntimeCheck(
                name="gpu_available",
                passed=bool(gpus),
                message="At least one NVIDIA GPU is visible through nvidia-smi.",
                metadata={"gpu_count": len(gpus)},
            )
        )
    if require_qdrant:
        checks.extend(dependency_checks(dependencies, ["qdrant"]))
    if require_postgres:
        checks.extend(dependency_checks(dependencies, ["postgres", "pgvector"]))
    if require_embeddings:
        checks.extend(dependency_checks(dependencies, ["sentence_transformers"]))
    if require_ocr:
        checks.extend(dependency_checks(dependencies, ["paddleocr", "paddlepaddle"]))
    if require_vision:
        checks.extend(dependency_checks(dependencies, ["torch", "transformers", "accelerate"]))

    torch_cuda_available, torch_cuda_device_count = torch_cuda
    paddle_cuda_available, paddle_cuda_device_count = paddle_cuda
    if require_gpu and dependencies.get("torch") and dependencies["torch"].installed:
        checks.append(
            RuntimeCheck(
                name="torch_cuda_available",
                passed=bool(torch_cuda_available),
                message="Torch can access CUDA devices.",
                metadata={"torch_cuda_device_count": torch_cuda_device_count or 0},
            )
        )
    if require_gpu and require_ocr and dependencies.get("paddlepaddle") and dependencies["paddlepaddle"].installed:
        checks.append(
            RuntimeCheck(
                name="paddle_cuda_available",
                passed=bool(paddle_cuda_available),
                message="PaddlePaddle can access CUDA devices for GPU OCR.",
                metadata={"paddle_cuda_device_count": paddle_cuda_device_count or 0},
            )
        )

    return RuntimeReport(
        passed=not any(not check.passed and check.severity == "error" for check in checks),
        gpus=gpus,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_device_count=torch_cuda_device_count,
        paddle_cuda_available=paddle_cuda_available,
        paddle_cuda_device_count=paddle_cuda_device_count,
        dependencies=dependencies,
        checks=checks,
    )


def dependency_checks(
    dependencies: dict[str, DependencyStatus],
    names: list[str],
) -> list[RuntimeCheck]:
    checks = []
    for name in names:
        dependency = dependencies[name]
        checks.append(
            RuntimeCheck(
                name=f"dependency:{name}",
                passed=dependency.installed,
                message=f"Python dependency is installed: {dependency.package}",
                metadata={"module": dependency.module, "version": dependency.version},
            )
        )
    return checks


def dependency_status(name: str, module: str, package: str) -> DependencyStatus:
    installed = importlib.util.find_spec(module) is not None
    version = None
    if installed:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return DependencyStatus(
        name=name,
        module=module,
        package=package,
        installed=installed,
        version=version,
    )


def detect_gpus() -> list[GPUDevice]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        gpus.append(
            GPUDevice(
                name=parts[0],
                memory_total_mib=parse_int(parts[1]) if len(parts) > 1 else None,
                driver_version=parts[2] if len(parts) > 2 else None,
            )
        )
    return gpus


def detect_torch_cuda(dependency: DependencyStatus | None = None) -> tuple[bool | None, int | None]:
    if dependency is not None and not dependency.installed:
        return None, None
    try:
        import torch
    except ImportError:
        return None, None
    return bool(torch.cuda.is_available()), int(torch.cuda.device_count())


def detect_paddle_cuda(dependency: DependencyStatus | None = None) -> tuple[bool | None, int | None]:
    if dependency is not None and not dependency.installed:
        return None, None
    try:
        import paddle
    except ImportError:
        return None, None
    try:
        compiled = bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        return False, 0
    if not compiled:
        return False, 0
    try:
        count = int(paddle.device.cuda.device_count())
    except Exception:
        count = 0
    return count > 0, count


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
