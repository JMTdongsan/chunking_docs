import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.runtime import (
    DependencyStatus,
    GPUDevice,
    TorchCudaStatus,
    build_runtime_report,
)


def dep(name: str, installed: bool) -> DependencyStatus:
    return DependencyStatus(
        name=name,
        module=name,
        package=name,
        installed=installed,
        version="1.0" if installed else None,
    )


def test_runtime_report_checks_requested_capabilities():
    dependencies = {
        "qdrant": dep("qdrant", True),
        "postgres": dep("postgres", False),
        "pgvector": dep("pgvector", False),
        "sentence_transformers": dep("sentence_transformers", True),
        "torch": dep("torch", True),
        "torchvision": dep("torchvision", False),
        "transformers": dep("transformers", False),
        "accelerate": dep("accelerate", True),
        "paddleocr": dep("paddleocr", False),
        "paddlepaddle": DependencyStatus(
            name="paddlepaddle",
            module="paddle",
            package="paddlepaddle",
            installed=False,
            version=None,
        ),
    }

    report = build_runtime_report(
        dependencies=dependencies,
        gpus=[GPUDevice(name="GPU", memory_total_mib=24000, driver_version="1")],
        torch_cuda=(True, 1),
        require_gpu=True,
        require_qdrant=True,
        require_postgres=True,
        require_embeddings=True,
        require_ocr=True,
        require_vision=True,
    )

    failed = [check.name for check in report.checks if not check.passed]
    assert report.passed is False
    assert "dependency:postgres" in failed
    assert "dependency:pgvector" in failed
    assert "dependency:torchvision" in failed
    assert "dependency:transformers" in failed
    assert "dependency:paddleocr" in failed
    assert "dependency:paddlepaddle" in failed
    assert "dependency:qdrant" not in failed
    assert "torch_cuda_available" not in failed


def test_doctor_cli_writes_report_without_requirements(tmp_path):
    output = tmp_path / "doctor.json"

    result = CliRunner().invoke(app, ["doctor", "--output", str(output)])

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "dependencies" in payload
    assert "gpus" in payload


def test_runtime_report_checks_paddle_cuda_when_gpu_ocr_is_required():
    dependencies = {
        "paddlepaddle": DependencyStatus(
            name="paddlepaddle",
            module="paddle",
            package="paddlepaddle",
            installed=True,
            version="1.0",
        ),
        "paddleocr": dep("paddleocr", True),
    }

    report = build_runtime_report(
        dependencies=dependencies,
        gpus=[GPUDevice(name="GPU")],
        paddle_cuda=(False, 0),
        require_gpu=True,
        require_ocr=True,
        require_ocr_gpu=True,
    )

    failed = [check.name for check in report.checks if not check.passed]
    assert report.passed is False
    assert "paddle_cuda_available" in failed


def test_runtime_report_allows_cpu_ocr_with_gpu_vlm():
    dependencies = {
        "paddlepaddle": DependencyStatus(
            name="paddlepaddle",
            module="paddle",
            package="paddlepaddle",
            installed=True,
            version="1.0",
        ),
        "paddleocr": dep("paddleocr", True),
        **vision_dependencies(),
    }

    report = build_runtime_report(
        dependencies=dependencies,
        gpus=[GPUDevice(name="GPU")],
        paddle_cuda=(False, 0),
        torch_cuda_status=TorchCudaStatus(
            available=True,
            device_count=1,
            device_names=["GPU"],
            compute_capabilities=["8.0"],
            cuda_version="12.8",
            compiled_arches=["sm_80"],
            bfloat16_supported=True,
        ),
        require_gpu=True,
        require_ocr=True,
        require_vision=True,
    )

    failed = [check.name for check in report.checks if not check.passed]
    assert report.passed is True
    assert "paddle_cuda_available" not in failed


def test_runtime_report_checks_vlm_profile_gpu_memory():
    dependencies = {
        "torch": dep("torch", True),
        "torchvision": dep("torchvision", True),
        "transformers": dep("transformers", True),
        "accelerate": dep("accelerate", True),
    }

    report = build_runtime_report(
        dependencies=dependencies,
        gpus=[GPUDevice(name="RTX", memory_total_mib=24576, driver_version="1")],
        require_vision=True,
        vlm_profiles=["qwen2_5_vl_7b", "phi3_5_vision"],
    )

    checks = {check.name: check for check in report.checks}
    assert report.passed is True
    assert checks["vlm_profile_memory:qwen2_5_vl_7b"].passed is True
    assert checks["vlm_profile_memory:qwen2_5_vl_7b"].metadata["required_memory_mib"] == 24576
    assert checks["vlm_profile_memory:phi3_5_vision"].metadata["matching_gpus"] == ["RTX"]


def test_runtime_report_warns_when_vlm_profile_lacks_memory_margin():
    report = build_runtime_report(
        dependencies={},
        gpus=[GPUDevice(name="RTX", memory_total_mib=24576)],
        vlm_profiles=["qwen2_5_vl_7b"],
        vlm_memory_margin_ratio=0.1,
    )

    checks = {check.name: check for check in report.checks}
    assert report.passed is True
    assert checks["vlm_profile_memory:qwen2_5_vl_7b"].passed is True
    margin_check = checks["vlm_profile_memory_margin:qwen2_5_vl_7b"]
    assert margin_check.passed is False
    assert margin_check.severity == "warning"
    assert margin_check.metadata["required_memory_with_margin_mib"] == 27033


def test_runtime_report_checks_vlm_profile_bfloat16_support():
    report = build_runtime_report(
        dependencies={},
        gpus=[GPUDevice(name="old-gpu", memory_total_mib=24576)],
        torch_cuda_status=TorchCudaStatus(
            available=True,
            device_count=1,
            device_names=["old-gpu"],
            compute_capabilities=["7.5"],
            cuda_version="12.4",
            compiled_arches=["sm_75"],
            bfloat16_supported=False,
        ),
        vlm_profiles=["qwen2_5_vl_7b"],
    )

    checks = {check.name: check for check in report.checks}
    assert report.passed is False
    dtype_check = checks["vlm_profile_dtype:qwen2_5_vl_7b"]
    assert dtype_check.passed is False
    assert dtype_check.metadata["torch_bfloat16_supported"] is False
    assert report.torch_cuda_device_names == ["old-gpu"]
    assert report.torch_cuda_compute_capabilities == ["7.5"]
    assert report.torch_cuda_version == "12.4"
    assert report.torch_cuda_compiled_arches == ["sm_75"]
    assert report.torch_bfloat16_supported is False


def test_runtime_report_checks_torch_cuda_arch_support_for_gpu_vision():
    report = build_runtime_report(
        dependencies=vision_dependencies(),
        gpus=[GPUDevice(name="RTX 5090", memory_total_mib=32768)],
        torch_cuda_status=TorchCudaStatus(
            available=True,
            device_count=1,
            device_names=["RTX 5090"],
            compute_capabilities=["12.0"],
            cuda_version="12.8",
            compiled_arches=["sm_120", "compute_120"],
            bfloat16_supported=True,
        ),
        require_gpu=True,
        require_vision=True,
    )

    checks = {check.name: check for check in report.checks}
    assert report.passed is True
    arch_check = checks["torch_cuda_arch:12.0"]
    assert arch_check.passed is True
    assert arch_check.metadata["matching_arches"] == ["compute_120", "sm_120"]
    assert report.torch_cuda_version == "12.8"
    assert report.torch_cuda_compiled_arches == ["sm_120", "compute_120"]


def test_runtime_report_fails_gpu_vision_when_torch_arch_is_missing():
    report = build_runtime_report(
        dependencies=vision_dependencies(),
        gpus=[GPUDevice(name="RTX 5090", memory_total_mib=32768)],
        torch_cuda_status=TorchCudaStatus(
            available=True,
            device_count=1,
            device_names=["RTX 5090"],
            compute_capabilities=["12.0"],
            cuda_version="12.4",
            compiled_arches=["sm_90"],
            bfloat16_supported=True,
        ),
        require_gpu=True,
        require_vision=True,
    )

    checks = {check.name: check for check in report.checks}
    assert report.passed is False
    arch_check = checks["torch_cuda_arch:12.0"]
    assert arch_check.passed is False
    assert arch_check.severity == "error"
    assert arch_check.metadata["expected_arches"] == ["sm_120", "compute_120"]
    assert arch_check.metadata["compiled_arches"] == ["sm_90"]


def test_runtime_report_warns_when_torch_arch_list_is_unknown():
    report = build_runtime_report(
        dependencies=vision_dependencies(),
        gpus=[GPUDevice(name="RTX 5090", memory_total_mib=32768)],
        torch_cuda_status=TorchCudaStatus(
            available=True,
            device_count=1,
            device_names=["RTX 5090"],
            compute_capabilities=["12.0"],
            cuda_version="12.8",
            compiled_arches=[],
            bfloat16_supported=True,
        ),
        require_gpu=True,
        require_vision=True,
    )

    checks = {check.name: check for check in report.checks}
    assert report.passed is True
    arch_check = checks["torch_cuda_arches_known"]
    assert arch_check.passed is False
    assert arch_check.severity == "warning"


def test_runtime_report_fails_when_vlm_profile_needs_more_memory():
    report = build_runtime_report(
        dependencies={},
        gpus=[GPUDevice(name="small-gpu", memory_total_mib=12288)],
        vlm_profiles=["qwen2_5_vl_7b"],
    )

    failed = [check.name for check in report.checks if not check.passed]
    assert report.passed is False
    assert failed == ["vlm_profile_memory:qwen2_5_vl_7b"]
    assert report.checks[0].metadata["max_gpu_memory_mib"] == 12288


def test_runtime_report_fails_unknown_vlm_profile():
    report = build_runtime_report(
        dependencies={},
        gpus=[GPUDevice(name="RTX", memory_total_mib=24576)],
        vlm_profiles=["unknown_profile"],
    )

    assert report.passed is False
    assert report.checks[0].name == "vlm_profile_memory:unknown_profile"


def vision_dependencies() -> dict[str, DependencyStatus]:
    return {
        "torch": dep("torch", True),
        "torchvision": dep("torchvision", True),
        "transformers": dep("transformers", True),
        "accelerate": dep("accelerate", True),
    }
