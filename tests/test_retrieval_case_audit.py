import json

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.case_audit import audit_retrieval_cases
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.io import write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    TextQuality,
    VisualAsset,
)


def test_audit_retrieval_cases_passes_valid_target_distribution():
    profiles, chunks, assets, triples = package_records()
    cases = [
        RetrievalCase(
            query="overview target",
            expected_pages=[1],
            expected_chunk_ids=["chunk-1"],
            metadata={"case_source": "page"},
        ),
        RetrievalCase(
            query="visual target",
            expected_asset_ids=["asset-1"],
            metadata={
                "case_source": "visual_object_probe",
                "modality": "vision_object",
                "object_probe_visual_only": True,
            },
        ),
        RetrievalCase(
            query="graph target",
            expected_triple_ids=["triple-1"],
            graph_expand=True,
            metadata={"case_source": "triple"},
        ),
    ]

    report = audit_retrieval_cases(
        cases,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        min_case_count=3,
        min_page_cases=1,
        min_asset_cases=1,
        min_triple_cases=1,
        min_distinct_page_targets=1,
        min_distinct_asset_targets=1,
        min_distinct_triple_targets=1,
        max_page_cases_per_target=1,
        max_chunk_cases_per_target=1,
        max_asset_cases_per_target=1,
        max_triple_cases_per_target=1,
        min_case_group_counts={"case_source:visual_object_probe": 1, "modality:vision_object": 1},
        min_case_group_distinct_targets={"case_source:visual_object_probe:asset": 1},
        require_visual_only_object_probes=True,
        min_query_terms_per_case=2,
    )

    assert report.passed is True
    assert report.target_counts == {"page": 1, "chunk": 1, "asset": 1, "triple": 1}
    assert report.distinct_target_counts == {"page": 1, "chunk": 1, "asset": 1, "triple": 1}
    assert report.max_cases_per_target == {"page": 1, "chunk": 1, "asset": 1, "triple": 1}
    assert report.case_group_counts["case_source"]["visual_object_probe"] == 1
    assert report.case_group_distinct_target_counts["case_source"]["visual_object_probe"]["asset"] == 1
    assert report.case_group_counts["modality"]["vision_object"] == 1
    assert report.case_group_counts["graph_expand"]["true"] == 1
    assert report.visual_object_probe_count == 1
    assert report.visual_only_object_probe_count == 1
    assert report.non_visual_only_object_probe_count == 0
    assert report.short_query_count == 0
    assert report.min_query_term_count == 2
    assert report.max_query_term_count == 2
    assert report.missing_target_counts == {"page": 0, "chunk": 0, "asset": 0, "triple": 0}


def test_audit_retrieval_cases_flags_bad_cases():
    profiles, chunks, assets, triples = package_records()
    cases = [
        RetrievalCase(query="", expected_pages=[1]),
        RetrievalCase(query="TODO: write query", expected_pages=[99]),
        RetrievalCase(query="duplicate", expected_chunk_ids=["missing"]),
        RetrievalCase(query="duplicate", expected_triple_ids=["triple-1"]),
        RetrievalCase(query="no target"),
    ]

    report = audit_retrieval_cases(
        cases,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        max_duplicate_queries=0,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert report.passed is False
    assert "empty_query" in issue_codes
    assert "todo_query" in issue_codes
    assert "unknown_page_target" in issue_codes
    assert "unknown_chunk_target" in issue_codes
    assert "missing_expected_target" in issue_codes
    assert "triple_case_without_graph_expand" in issue_codes
    assert "duplicate_query" in issue_codes
    assert "max_duplicate_queries" in report.failed_checks


def test_audit_retrieval_cases_checks_case_group_counts():
    profiles, chunks, assets, triples = package_records()
    cases = [
        RetrievalCase(
            query="visual target",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_lexical_probe"},
        )
    ]

    report = audit_retrieval_cases(
        cases,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        min_case_group_counts={"case_source:visual_object_probe": 1},
    )

    assert report.passed is False
    assert "min_case_group_count:case_source:visual_object_probe" in report.failed_checks
    check = next(
        check
        for check in report.checks
        if check.name == "min_case_group_count:case_source:visual_object_probe"
    )
    assert check.actual == 0


def test_audit_retrieval_cases_checks_case_group_distinct_targets():
    profiles, chunks, assets, triples = package_records()
    assets.append(
        VisualAsset(
            asset_id="asset-2",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            caption="other visual target",
        )
    )
    cases = [
        RetrievalCase(
            query="object probe one",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe"},
        ),
        RetrievalCase(
            query="object probe two",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe"},
        ),
        RetrievalCase(
            query="regular asset target",
            expected_asset_ids=["asset-2"],
            metadata={"case_source": "asset"},
        ),
    ]

    report = audit_retrieval_cases(
        cases,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        min_distinct_asset_targets=2,
        min_case_group_distinct_targets={"case_source:visual_object_probe:asset": 2},
    )

    assert report.passed is False
    assert report.distinct_target_counts["asset"] == 2
    assert report.case_group_distinct_target_counts["case_source"]["visual_object_probe"]["asset"] == 1
    assert "min_case_group_distinct_targets:case_source:visual_object_probe:asset" in report.failed_checks
    check = next(
        check
        for check in report.checks
        if check.name == "min_case_group_distinct_targets:case_source:visual_object_probe:asset"
    )
    assert check.actual == 1
    assert check.threshold == 2


def test_audit_retrieval_cases_checks_distinct_target_counts():
    profiles, chunks, assets, triples = package_records()
    cases = [
        RetrievalCase(query="visual target one", expected_asset_ids=["asset-1"]),
        RetrievalCase(query="visual target two", expected_asset_ids=["asset-1"]),
    ]

    report = audit_retrieval_cases(
        cases,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        min_distinct_asset_targets=2,
    )

    assert report.passed is False
    assert report.target_counts["asset"] == 2
    assert report.distinct_target_counts["asset"] == 1
    assert "min_distinct_asset_targets" in report.failed_checks
    check = next(check for check in report.checks if check.name == "min_distinct_asset_targets")
    assert check.actual == 1
    assert check.threshold == 2


def test_audit_retrieval_cases_checks_target_concentration():
    profiles, chunks, assets, triples = package_records()
    cases = [
        RetrievalCase(query="visual target one", expected_asset_ids=["asset-1"]),
        RetrievalCase(query="visual target two", expected_asset_ids=["asset-1"]),
    ]

    report = audit_retrieval_cases(
        cases,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        max_asset_cases_per_target=1,
    )

    assert report.passed is False
    assert report.max_cases_per_target["asset"] == 2
    assert "max_asset_cases_per_target" in report.failed_checks
    check = next(check for check in report.checks if check.name == "max_asset_cases_per_target")
    assert check.actual == 2
    assert check.threshold == 1


def test_audit_retrieval_cases_checks_query_term_strength():
    profiles, chunks, assets, triples = package_records()
    cases = [
        RetrievalCase(query="short", expected_pages=[1]),
        RetrievalCase(query="specific visual target", expected_asset_ids=["asset-1"]),
    ]

    report = audit_retrieval_cases(
        cases,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        min_query_terms_per_case=2,
    )

    assert report.passed is False
    assert report.short_query_count == 1
    assert report.min_query_term_count == 1
    assert report.max_query_term_count == 3
    assert "min_query_terms_per_case" in report.failed_checks
    assert "short_query" in {issue.code for issue in report.issues}
    check = next(check for check in report.checks if check.name == "min_query_terms_per_case")
    assert check.actual == 1
    assert check.threshold == 2


def test_audit_retrieval_cases_can_require_visual_only_object_probes():
    profiles, chunks, assets, triples = package_records()
    cases = [
        RetrievalCase(
            query="visual only object target",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe", "object_probe_visual_only": True},
        ),
        RetrievalCase(
            query="broad object target",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe", "object_probe_visual_only": False},
        ),
    ]

    report = audit_retrieval_cases(
        cases,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        require_visual_only_object_probes=True,
    )

    assert report.passed is False
    assert report.visual_object_probe_count == 2
    assert report.visual_only_object_probe_count == 1
    assert report.non_visual_only_object_probe_count == 1
    assert "require_visual_only_object_probes" in report.failed_checks
    assert {issue.code for issue in report.issues} == {"non_visual_only_object_probe"}
    check = next(
        check for check in report.checks if check.name == "require_visual_only_object_probes"
    )
    assert check.actual == 1
    assert check.threshold == 0


def test_audit_retrieval_cases_cli_writes_report(tmp_path):
    package_dir = write_package(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "case_audit.json"
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="overview target",
                expected_pages=[1],
                metadata={"case_source": "page"},
            ),
            RetrievalCase(
                query="visual target",
                expected_asset_ids=["asset-1"],
                metadata={"case_source": "visual_object_probe", "object_probe_visual_only": True},
            ),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "audit-retrieval-cases",
            str(cases_path),
            "--package-dir",
            str(package_dir),
            "--min-case-count",
            "2",
            "--min-page-cases",
            "1",
            "--min-asset-cases",
            "1",
            "--min-distinct-asset-targets",
            "1",
            "--max-asset-cases-per-target",
            "1",
            "--min-case-group-count",
            "case_source:visual_object_probe=1",
            "--min-case-group-distinct-targets",
            "case_source:visual_object_probe:asset=1",
            "--require-visual-only-object-probes",
            "--min-query-terms-per-case",
            "2",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["target_counts"]["asset"] == 1
    assert payload["distinct_target_counts"]["asset"] == 1
    assert payload["max_cases_per_target"]["asset"] == 1
    assert payload["case_group_counts"]["case_source"]["visual_object_probe"] == 1
    assert payload["case_group_distinct_target_counts"]["case_source"]["visual_object_probe"]["asset"] == 1
    assert payload["visual_object_probe_count"] == 1
    assert payload["visual_only_object_probe_count"] == 1
    assert payload["non_visual_only_object_probe_count"] == 0
    assert payload["short_query_count"] == 0
    assert payload["min_query_term_count"] == 2


def write_package(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    profiles, chunks, assets, triples = package_records()
    doc = SourceDocument(
        doc_id="doc",
        title="Reference Document",
        local_path=tmp_path / "reference.pdf",
    )
    manifest = ProcessingManifest(doc=doc, profiles=profiles, chunks=chunks, assets=assets, triples=triples)
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", profiles)
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", triples)
    return package_dir


def package_records():
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=100,
            height=100,
            char_count=120,
            line_count=4,
            text_block_count=1,
            image_block_count=1,
            embedded_image_count=0,
            drawing_count=1,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="overview target visual target graph target",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            caption="visual target",
        )
    ]
    triples = [
        GraphTriple(
            triple_id="triple-1",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="graph",
            predicate="relates_to",
            object="target",
        )
    ]
    return profiles, chunks, assets, triples
