from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.vision.hf_vlm import get_vlm_model_profile


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


class TorchCudaStatus(BaseModel):
    available: bool | None = None
    device_count: int | None = None
    device_names: list[str] = Field(default_factory=list)
    compute_capabilities: list[str] = Field(default_factory=list)
    cuda_version: str | None = None
    compiled_arches: list[str] = Field(default_factory=list)
    bfloat16_supported: bool | None = None


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
    torch_cuda_device_names: list[str] = Field(default_factory=list)
    torch_cuda_compute_capabilities: list[str] = Field(default_factory=list)
    torch_cuda_version: str | None = None
    torch_cuda_compiled_arches: list[str] = Field(default_factory=list)
    torch_bfloat16_supported: bool | None = None
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
    "torchvision": ("torchvision", "torchvision"),
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
    require_ocr_gpu: bool = False,
    require_vision: bool = False,
    vlm_profiles: list[str] | None = None,
    vlm_memory_margin_ratio: float = 0.0,
) -> RuntimeReport:
    dependencies = {name: dependency_status(name, module, package) for name, (module, package) in DEPENDENCIES.items()}
    torch_cuda_status = detect_torch_cuda_status(dependencies.get("torch"))
    return build_runtime_report(
        dependencies=dependencies,
        gpus=detect_gpus(),
        torch_cuda_status=torch_cuda_status,
        paddle_cuda=detect_paddle_cuda(dependencies.get("paddlepaddle"))
        if require_ocr_gpu
        else (None, None),
        require_gpu=require_gpu,
        require_qdrant=require_qdrant,
        require_postgres=require_postgres,
        require_embeddings=require_embeddings,
        require_ocr=require_ocr,
        require_ocr_gpu=require_ocr_gpu,
        require_vision=require_vision,
        vlm_profiles=vlm_profiles,
        vlm_memory_margin_ratio=vlm_memory_margin_ratio,
    )


def build_runtime_report(
    dependencies: dict[str, DependencyStatus],
    gpus: list[GPUDevice],
    torch_cuda: tuple[bool | None, int | None] = (None, None),
    torch_cuda_status: TorchCudaStatus | None = None,
    paddle_cuda: tuple[bool | None, int | None] = (None, None),
    require_gpu: bool = False,
    require_qdrant: bool = False,
    require_postgres: bool = False,
    require_embeddings: bool = False,
    require_ocr: bool = False,
    require_ocr_gpu: bool = False,
    require_vision: bool = False,
    vlm_profiles: list[str] | None = None,
    vlm_memory_margin_ratio: float = 0.0,
) -> RuntimeReport:
    checks = []
    if require_gpu or require_ocr_gpu:
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
    if require_ocr or require_ocr_gpu:
        checks.extend(dependency_checks(dependencies, ["paddleocr", "paddlepaddle"]))
    if require_vision:
        checks.extend(dependency_checks(dependencies, ["torch", "torchvision", "transformers", "accelerate"]))
    if vlm_profiles:
        checks.extend(
            vlm_profile_checks(
                gpus,
                vlm_profiles,
                torch_cuda_status=torch_cuda_status,
                memory_margin_ratio=vlm_memory_margin_ratio,
            )
        )

    if torch_cuda_status is None:
        torch_cuda_status = TorchCudaStatus(
            available=torch_cuda[0],
            device_count=torch_cuda[1],
        )
    torch_cuda_available = torch_cuda_status.available
    torch_cuda_device_count = torch_cuda_status.device_count
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
        if torch_cuda_available:
            checks.extend(
                torch_cuda_arch_checks(
                    torch_cuda_status,
                    strict=require_embeddings or require_vision,
                )
            )
    if (
        require_ocr_gpu
        and dependencies.get("paddlepaddle")
        and dependencies["paddlepaddle"].installed
    ):
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
        torch_cuda_device_names=torch_cuda_status.device_names,
        torch_cuda_compute_capabilities=torch_cuda_status.compute_capabilities,
        torch_cuda_version=torch_cuda_status.cuda_version,
        torch_cuda_compiled_arches=torch_cuda_status.compiled_arches,
        torch_bfloat16_supported=torch_cuda_status.bfloat16_supported,
        paddle_cuda_available=paddle_cuda_available,
        paddle_cuda_device_count=paddle_cuda_device_count,
        dependencies=dependencies,
        checks=checks,
    )


def vlm_profile_checks(
    gpus: list[GPUDevice],
    profile_names: list[str],
    torch_cuda_status: TorchCudaStatus | None = None,
    memory_margin_ratio: float = 0.0,
) -> list[RuntimeCheck]:
    checks = []
    max_gpu_memory_mib = max((gpu.memory_total_mib or 0 for gpu in gpus), default=0)
    margin_ratio = max(float(memory_margin_ratio), 0.0)
    for profile_name in profile_names:
        try:
            profile = get_vlm_model_profile(profile_name)
        except ValueError as exc:
            checks.append(
                RuntimeCheck(
                    name=f"vlm_profile_memory:{profile_name}",
                    passed=False,
                    message="Requested VLM profile is not supported.",
                    metadata={"error": str(exc)},
                )
            )
            continue
        required_mib = profile.min_gpu_memory_mib or 0
        required_with_margin_mib = int(required_mib * (1.0 + margin_ratio))
        matching_gpus = [
            gpu.name
            for gpu in gpus
            if gpu.memory_total_mib is not None and gpu.memory_total_mib >= required_mib
        ]
        margin_matching_gpus = [
            gpu.name
            for gpu in gpus
            if gpu.memory_total_mib is not None and gpu.memory_total_mib >= required_with_margin_mib
        ]
        checks.append(
            RuntimeCheck(
                name=f"vlm_profile_memory:{profile.name}",
                passed=required_mib <= 0 or bool(matching_gpus),
                message="GPU memory is sufficient for the selected VLM profile.",
                metadata={
                    "profile": profile.name,
                    "model_name": profile.model_name,
                    "required_memory_mib": required_mib,
                    "required_memory_with_margin_mib": required_with_margin_mib,
                    "memory_margin_ratio": margin_ratio,
                    "max_gpu_memory_mib": max_gpu_memory_mib,
                    "matching_gpus": matching_gpus,
                },
            )
        )
        if required_mib > 0 and margin_ratio > 0 and matching_gpus:
            checks.append(
                RuntimeCheck(
                    name=f"vlm_profile_memory_margin:{profile.name}",
                    passed=bool(margin_matching_gpus),
                    severity="warning",
                    message="GPU memory meets the profile minimum but not the configured safety margin.",
                    metadata={
                        "profile": profile.name,
                        "model_name": profile.model_name,
                        "required_memory_mib": required_mib,
                        "required_memory_with_margin_mib": required_with_margin_mib,
                        "memory_margin_ratio": margin_ratio,
                        "max_gpu_memory_mib": max_gpu_memory_mib,
                        "matching_gpus": margin_matching_gpus,
                    },
                )
            )
        checks.extend(vlm_profile_dtype_checks(profile, torch_cuda_status))
    return checks


def vlm_profile_dtype_checks(
    profile,
    torch_cuda_status: TorchCudaStatus | None,
) -> list[RuntimeCheck]:
    dtype = profile.torch_dtype.strip().lower()
    if dtype not in {"bfloat16", "bf16"}:
        return []
    if torch_cuda_status is None:
        return []
    if torch_cuda_status.bfloat16_supported is None:
        return [
            RuntimeCheck(
                name=f"vlm_profile_dtype:{profile.name}",
                passed=False,
                severity="warning",
                message="Torch CUDA bfloat16 support could not be confirmed for this VLM profile.",
                metadata={
                    "profile": profile.name,
                    "torch_dtype": profile.torch_dtype,
                },
            )
        ]
    return [
        RuntimeCheck(
            name=f"vlm_profile_dtype:{profile.name}",
            passed=torch_cuda_status.bfloat16_supported,
            message="Torch CUDA bfloat16 support is compatible with the selected VLM profile.",
            metadata={
                "profile": profile.name,
                "torch_dtype": profile.torch_dtype,
                "torch_cuda_device_names": torch_cuda_status.device_names,
                "torch_cuda_compute_capabilities": torch_cuda_status.compute_capabilities,
                "torch_bfloat16_supported": torch_cuda_status.bfloat16_supported,
            },
        )
    ]


def torch_cuda_arch_checks(
    torch_cuda_status: TorchCudaStatus,
    strict: bool = False,
) -> list[RuntimeCheck]:
    capabilities = [value for value in torch_cuda_status.compute_capabilities if value]
    if not capabilities:
        return []
    arches = sorted(set(torch_cuda_status.compiled_arches))
    severity = "error" if strict else "warning"
    if not arches:
        return [
            RuntimeCheck(
                name="torch_cuda_arches_known",
                passed=False,
                severity="warning",
                message="Torch CUDA compiled architecture list could not be inspected.",
                metadata={
                    "torch_cuda_version": torch_cuda_status.cuda_version,
                    "torch_cuda_compute_capabilities": capabilities,
                },
            )
        ]

    checks = []
    for capability in capabilities:
        expected = capability_arch_tokens(capability)
        matching_arches = sorted(set(expected) & set(arches))
        checks.append(
            RuntimeCheck(
                name=f"torch_cuda_arch:{capability}",
                passed=bool(matching_arches),
                severity=severity,
                message="Torch CUDA build includes an architecture target for the visible GPU.",
                metadata={
                    "compute_capability": capability,
                    "expected_arches": expected,
                    "matching_arches": matching_arches,
                    "compiled_arches": arches,
                    "torch_cuda_version": torch_cuda_status.cuda_version,
                },
            )
        )
    return checks


def capability_arch_tokens(capability: str) -> list[str]:
    parts = capability.split(".", 1)
    if len(parts) != 2:
        return []
    major, minor = (part.strip() for part in parts)
    if not major.isdigit() or not minor.isdigit():
        return []
    suffix = f"{int(major)}{int(minor)}"
    return [f"sm_{suffix}", f"compute_{suffix}"]


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
    status = detect_torch_cuda_status(dependency)
    return status.available, status.device_count


def detect_torch_cuda_status(dependency: DependencyStatus | None = None) -> TorchCudaStatus:
    if dependency is not None and not dependency.installed:
        return TorchCudaStatus()
    try:
        import torch
    except ImportError:
        return TorchCudaStatus()
    available = bool(torch.cuda.is_available())
    try:
        device_count = int(torch.cuda.device_count()) if available else 0
    except Exception:
        device_count = 0
    device_names = []
    compute_capabilities = []
    for index in range(device_count):
        try:
            device_names.append(str(torch.cuda.get_device_name(index)))
        except Exception:
            device_names.append(f"cuda:{index}")
        try:
            major, minor = torch.cuda.get_device_capability(index)
            compute_capabilities.append(f"{major}.{minor}")
        except Exception:
            compute_capabilities.append("")
    try:
        bfloat16_supported = bool(torch.cuda.is_bf16_supported()) if available else False
    except Exception:
        bfloat16_supported = None
    try:
        cuda_version = str(torch.version.cuda) if torch.version.cuda else None
    except Exception:
        cuda_version = None
    try:
        arch_getter = getattr(torch.cuda, "get_arch_list", None)
        compiled_arches = list(arch_getter()) if available and callable(arch_getter) else []
    except Exception:
        compiled_arches = []
    return TorchCudaStatus(
        available=available,
        device_count=device_count,
        device_names=device_names,
        compute_capabilities=compute_capabilities,
        cuda_version=cuda_version,
        compiled_arches=[str(arch) for arch in compiled_arches],
        bfloat16_supported=bfloat16_supported,
    )


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
