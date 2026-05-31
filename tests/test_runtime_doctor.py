import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.runtime import DependencyStatus, GPUDevice, build_runtime_report


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
    )

    failed = [check.name for check in report.checks if not check.passed]
    assert report.passed is False
    assert "paddle_cuda_available" in failed
