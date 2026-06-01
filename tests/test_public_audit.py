import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.audit import audit_public_artifacts


def write_default_gitignore(root):
    (root / ".gitignore").write_text("data/raw/*.pdf\noutputs/\n", encoding="utf-8")


def test_public_audit_passes_clean_repository_shape(tmp_path):
    write_default_gitignore(tmp_path)
    (tmp_path / "README.md").write_text("Reusable document chunking library.\n", encoding="utf-8")
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "source.pdf").write_bytes(b"%PDF ignored generated input")

    report = audit_public_artifacts(tmp_path, forbidden_patterns=["private target"])

    assert report.passed is True
    assert report.scanned_file_count == 2
    assert report.skipped_file_count == 1
    assert report.issues == []


def test_public_audit_detects_forbidden_public_text(tmp_path):
    write_default_gitignore(tmp_path)
    (tmp_path / "README.md").write_text(
        "This line mentions a private target document.\n",
        encoding="utf-8",
    )

    report = audit_public_artifacts(tmp_path, forbidden_patterns=["PRIVATE TARGET"])

    assert report.passed is False
    assert report.forbidden_match_count == 1
    assert report.issues[0].code == "forbidden_public_text"
    assert report.issues[0].metadata["path"] == "README.md"
    assert report.issues[0].metadata["line"] == 1


def test_public_audit_detects_blocked_public_artifact(tmp_path):
    write_default_gitignore(tmp_path)
    (tmp_path / "reference.pdf").write_bytes(b"%PDF public artifact")

    report = audit_public_artifacts(tmp_path)

    assert report.passed is False
    assert report.blocked_extension_count == 1
    assert report.issues[0].code == "blocked_public_extension"
    assert report.issues[0].metadata["path"] == "reference.pdf"


def test_public_audit_requires_generated_artifact_gitignore_patterns(tmp_path):
    (tmp_path / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Library docs.\n", encoding="utf-8")

    report = audit_public_artifacts(tmp_path)

    assert report.passed is False
    assert [issue.code for issue in report.issues] == ["missing_gitignore_pattern"]
    assert report.issues[0].metadata["pattern"] == "data/raw/*.pdf"


def test_audit_publication_cli_writes_report(tmp_path):
    write_default_gitignore(tmp_path)
    (tmp_path / "README.md").write_text("private target\n", encoding="utf-8")
    output = tmp_path / "audit.json"

    result = CliRunner().invoke(
        app,
        [
            "audit-publication",
            str(tmp_path),
            "--forbidden-pattern",
            "private target",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["issues"][0]["code"] == "forbidden_public_text"
