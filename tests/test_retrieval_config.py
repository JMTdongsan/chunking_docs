import json

import pytest
from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.fusion_sweep import (
    QdrantFusionSweepCandidate,
    build_qdrant_fusion_sweep_report,
)
from chunking_docs.evaluation.retrieval import (
    RetrievalCaseGroupMetric,
    RetrievalCaseResult,
    RetrievalEvaluation,
    RetrievalSourceMetric,
)
from chunking_docs.evaluation.retrieval_config import (
    QdrantRetrievalRoute,
    apply_qdrant_retrieval_route_preset,
    build_qdrant_retrieval_config_from_fusion_sweep,
    qdrant_retrieval_config_vector_names,
    select_qdrant_retrieval_route,
)
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.retrieval.qdrant_hybrid import QdrantHybridSearchHit


def test_qdrant_retrieval_config_exports_global_recommended_candidate():
    report = fusion_sweep_report()

    config = build_qdrant_retrieval_config_from_fusion_sweep(report)

    assert config.backend == "qdrant_hybrid"
    assert config.collection_name == "seoul_plan"
    assert config.package_dir == "outputs/package"
    assert config.bm25_tokens_path == "outputs/package/bm25_tokens.json"
    assert config.vector_names == ["text_dense", "caption_dense", "object_dense"]
    assert config.top_k == 7
    assert config.collapse_hierarchical is True
    assert config.reranker == "none"
    assert config.rerank_top_k == 0
    assert config.fusion_weights == {"bm25": 1.0, "qdrant:caption_dense": 1.0}
    assert config.query_encoders["object_dense"] == "BAAI/bge-m3"
    assert config.lexical_tokenizer["strategy"] == "mixed"
    assert config.selection.source == "global_recommended"
    assert config.selection.candidate == "balanced"
    assert config.selection.metrics["target_coverage_at_k"] == pytest.approx(0.96)
    assert config.selection.min_source_precision_at_hits == pytest.approx(0.82)
    assert config.selection.min_source_precision_at_hits_name == "qdrant:caption_dense"
    assert config.selection.min_source_family_precision_at_hits == pytest.approx(0.88)
    assert config.selection.min_source_family_precision_at_hits_name == "visual"
    assert config.selection.source_precision_at_hits["qdrant:caption_dense"] == pytest.approx(
        0.82
    )
    assert config.selection.source_family_precision_at_hits["visual"] == pytest.approx(0.88)
    assert config.selection.metrics["min_source_precision_at_hits"] == pytest.approx(0.82)
    assert config.selection.metrics["min_source_family_precision_at_hits"] == pytest.approx(0.88)
    assert config.selection.pairwise_comparisons[0]["baseline"] == "object_weighted"
    assert config.selection.pairwise_comparisons[0]["candidate_win_rate"] == 1.0
    assert config.selection.pairwise_comparisons[0]["mean_target_coverage_delta"] == 1.0
    assert config.metadata["sweep_eligible_count"] == 2


def test_qdrant_retrieval_config_exports_case_group_recommendation():
    report = fusion_sweep_report()

    config = build_qdrant_retrieval_config_from_fusion_sweep(
        report,
        case_group="visual_object_probe",
        source_report="qdrant_fusion_sweep.json",
    )

    assert config.selection.source == "case_group_recommended"
    assert config.selection.source_report == "qdrant_fusion_sweep.json"
    assert config.selection.global_recommended == "balanced"
    assert config.selection.case_group == "case_source:visual_object_probe"
    assert config.selection.candidate == "object_weighted"
    assert config.selection.case_group_recommended_from_globally_eligible is True
    assert config.fusion_weights == {"bm25": 1.0, "qdrant:object_dense": 1.4}
    assert config.selection.case_group_metrics["target_coverage_at_k"] == pytest.approx(0.9)
    assert config.selection.min_source_precision_at_hits == pytest.approx(0.74)
    assert config.selection.min_source_precision_at_hits_name == "qdrant:object_dense"


def test_qdrant_retrieval_config_route_preset_selects_visual_and_graph_routes():
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())

    routed = apply_qdrant_retrieval_route_preset(config, "adaptive")

    assert [route.name for route in routed.routes] == ["visual_object", "graph_triple"]
    assert qdrant_retrieval_config_vector_names(routed) == [
        "text_dense",
        "caption_dense",
        "object_dense",
        "triple_dense",
    ]
    assert routed.query_encoders["triple_dense"] == "default_text"

    visual = select_qdrant_retrieval_route(routed, "색상 기호 의미")
    assert visual.name == "visual_object"
    assert visual.vector_names == ["object_dense"]
    assert visual.fusion_weights == {"bm25": 1.2, "qdrant:object_dense": 0.8}

    graph = select_qdrant_retrieval_route(routed, "목표와 전략의 관계")
    assert graph.name == "graph_triple"
    assert graph.vector_names == ["text_dense", "triple_dense"]
    assert graph.graph_expand is True

    default = select_qdrant_retrieval_route(routed, "서울 재개발 정책")
    assert default.name is None
    assert default.vector_names == config.vector_names
    assert default.fusion_weights == config.fusion_weights


def test_qdrant_retrieval_config_exports_reranker_settings():
    report = fusion_sweep_report()
    report = report.model_copy(
        update={
            "metadata": {
                **report.metadata,
                "reranker": "lexical",
                "reranker_model": "",
                "reranker_max_length": 0,
                "rerank_top_k": 20,
            }
        }
    )

    config = build_qdrant_retrieval_config_from_fusion_sweep(report)

    assert config.reranker == "lexical"
    assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert config.reranker_max_length == 0
    assert config.rerank_top_k == 20


def test_export_qdrant_retrieval_config_cli_writes_json(tmp_path):
    report_path = tmp_path / "qdrant_fusion_sweep.json"
    output_path = tmp_path / "qdrant_retrieval_config.json"
    report_path.write_text(
        fusion_sweep_report().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "export-qdrant-retrieval-config",
            str(report_path),
            "--output",
            str(output_path),
            "--case-group",
            "case_source:visual_object_probe",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "qdrant_hybrid"
    assert payload["selection"]["candidate"] == "object_weighted"
    assert payload["selection"]["min_source_precision_at_hits"] == pytest.approx(0.74)
    assert payload["selection"]["min_source_precision_at_hits_name"] == "qdrant:object_dense"
    assert payload["selection"]["source_precision_at_hits"]["qdrant:object_dense"] == pytest.approx(
        0.74
    )
    assert payload["selection"]["pairwise_comparisons"][0]["baseline"] == "balanced"
    assert payload["fusion_weights"] == {"bm25": 1.0, "qdrant:object_dense": 1.4}


def test_eval_qdrant_retrieval_config_cli_uses_exported_settings(tmp_path, monkeypatch):
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())
    config = config.model_copy(
        update={
            "package_dir": str(tmp_path / "package"),
            "collection_name": "config_collection",
            "vector_names": ["text_dense"],
            "graph_expand": True,
            "fusion_weights": {"bm25": 1.2},
            "top_k": 3,
            "collapse_hierarchical": True,
            "reranker": "lexical",
            "rerank_top_k": 11,
            "lexical_tokenizer": {
                "strategy": "mixed",
                "min_n": 2,
                "max_n": 3,
                "ngram_cjk_only": False,
                "deduplicate": True,
            },
        }
    )
    config_path = tmp_path / "qdrant_retrieval_config.json"
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "eval.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    cases_path.write_text(
        json.dumps({"query": "redevelopment policy", "expected_pages": [1]}) + "\n",
        encoding="utf-8",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="redevelopment policy",
        metadata={"chunking_strategy": "semantic", "retrieval_role": "child"},
    )
    calls = {}

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            calls["search"] = kwargs
            return [
                QdrantHybridSearchHit(
                    item_id="chunk-1",
                    score=1.0,
                    sources=["bm25"],
                    chunk=chunk,
                )
            ]

    def fake_prepare_qdrant_hybrid_search(**kwargs):
        calls["prepare"] = kwargs
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": kwargs["collection"],
            "selected_vectors": ["text_dense"],
            "query_encoders": {"text_dense": "default_text"},
            "query_encoder_details": {"text_dense": {"backend": "hashing"}},
            "upserted": {"count": 1},
            "chunks": [chunk],
            "assets": [],
            "triples": [],
        }

    monkeypatch.setattr(
        "chunking_docs.cli.prepare_qdrant_hybrid_search",
        fake_prepare_qdrant_hybrid_search,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-retrieval-config",
            str(config_path),
            str(cases_path),
            "--output",
            str(output_path),
            "--repeat",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["prepare"]["package_dir"] == tmp_path / "package"
    assert calls["prepare"]["collection"] == "config_collection"
    assert calls["prepare"]["vector_names"] == "text_dense"
    assert calls["prepare"]["ngram_max"] == 3
    assert calls["prepare"]["ngram_cjk_only"] is False
    assert calls["prepare"]["deduplicate_tokens"] is True
    assert calls["search"]["top_k"] == 3
    assert calls["search"]["graph_expand"] is True
    assert calls["search"]["collapse_hierarchical"] is True
    assert calls["search"]["fusion_weights"] == {"bm25": 1.2}
    assert calls["search"]["reranker"].source == "rerank:lexical"
    assert calls["search"]["rerank_top_k"] == 11
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["recall_at_k"] == 1.0
    assert payload["metadata"]["backend"] == "qdrant_hybrid_config"
    assert payload["metadata"]["reranker"] == "rerank:lexical"
    assert payload["metadata"]["rerank_top_k"] == 11
    assert payload["metadata"]["config_selection"]["candidate"] == "balanced"


def test_eval_qdrant_retrieval_config_cli_routes_visual_queries(tmp_path, monkeypatch):
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())
    config = config.model_copy(
        update={
            "package_dir": str(tmp_path / "package"),
            "collection_name": "config_collection",
            "vector_names": ["text_dense"],
            "query_encoders": {
                "text_dense": "default_text",
                "object_dense": "default_text",
            },
            "fusion_weights": {"bm25": 1.0, "qdrant:text_dense": 1.0},
            "routes": [
                QdrantRetrievalRoute(
                    name="visual_object",
                    match_query_terms=["색상"],
                    vector_names=["object_dense"],
                    graph_expand=False,
                    fusion_weights={"bm25": 1.2, "qdrant:object_dense": 0.8},
                )
            ],
        }
    )
    config_path = tmp_path / "qdrant_retrieval_config.json"
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "eval.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    cases_path.write_text(
        "\n".join(
            [
                json.dumps({"query": "색상 기호 의미", "expected_pages": [2]}),
                json.dumps({"query": "redevelopment policy", "expected_pages": [1]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    text_chunk = DocumentChunk(
        chunk_id="chunk-text",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="redevelopment policy",
    )
    visual_chunk = DocumentChunk(
        chunk_id="chunk-visual",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.FIGURE,
        text="색상 기호 의미",
    )
    calls = {"searches": []}

    class FakeStore:
        def count(self):
            return 2

    class FakeSearcher:
        def search(self, **kwargs):
            calls["searches"].append(kwargs)
            chunk = visual_chunk if kwargs["vector_names"] == ["object_dense"] else text_chunk
            return [
                QdrantHybridSearchHit(
                    item_id=chunk.chunk_id,
                    score=1.0,
                    sources=["bm25"],
                    chunk=chunk,
                )
            ]

    def fake_prepare_qdrant_hybrid_search(**kwargs):
        calls["prepare"] = kwargs
        selected_vectors = [
            item.strip() for item in kwargs["vector_names"].split(",") if item.strip()
        ]
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": kwargs["collection"],
            "selected_vectors": selected_vectors,
            "query_encoders": {vector: "default_text" for vector in selected_vectors},
            "query_encoder_details": {
                vector: {"backend": "hashing"} for vector in selected_vectors
            },
            "upserted": {"count": 2},
            "chunks": [text_chunk, visual_chunk],
            "assets": [],
            "triples": [],
        }

    monkeypatch.setattr(
        "chunking_docs.cli.prepare_qdrant_hybrid_search",
        fake_prepare_qdrant_hybrid_search,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-retrieval-config",
            str(config_path),
            str(cases_path),
            "--output",
            str(output_path),
            "--repeat",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["prepare"]["vector_names"] == "text_dense,object_dense"
    assert calls["searches"][0]["vector_names"] == ["object_dense"]
    assert calls["searches"][0]["fusion_weights"] == {
        "bm25": 1.2,
        "qdrant:object_dense": 0.8,
    }
    assert calls["searches"][1]["vector_names"] == ["text_dense"]
    assert calls["searches"][1]["fusion_weights"] == {
        "bm25": 1.0,
        "qdrant:text_dense": 1.0,
    }
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["recall_at_k"] == 1.0
    assert payload["metadata"]["route_usage"]["counts"] == {
        "visual_object": 1,
        "default": 1,
    }
    assert payload["metadata"]["route_decisions"][0]["name"] == "visual_object"
    assert payload["results"][0]["case_metadata"]["retrieval_route"] == "visual_object"
    assert payload["results"][1]["case_metadata"]["retrieval_route"] == "default"
    assert payload["case_group_metrics"]["retrieval_route"]["visual_object"][
        "target_coverage_at_k"
    ] == 1.0
    assert payload["case_group_metrics"]["retrieval_route"]["default"][
        "target_coverage_at_k"
    ] == 1.0


def test_eval_qdrant_retrieval_config_cli_auto_detects_text_query_encoder(
    tmp_path,
    monkeypatch,
):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "vectors": {
                    "text_dense": {
                        "dimension": 1024,
                        "embedding": {
                            "backend": "sentence-transformers",
                            "model": "BAAI/bge-m3",
                        },
                    },
                    "caption_dense": {
                        "dimension": 1024,
                        "embedding": {
                            "backend": "sentence-transformers",
                            "model": "BAAI/bge-m3",
                            "same_as": "text_dense",
                        },
                    },
                    "object_dense": {
                        "dimension": 1024,
                        "embedding": {
                            "same_as": "caption_dense",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())
    config = config.model_copy(
        update={
            "package_dir": str(package_dir),
            "collection_name": "config_collection",
            "vector_names": ["text_dense", "caption_dense", "object_dense"],
            "top_k": 3,
        }
    )
    config_path = tmp_path / "qdrant_retrieval_config.json"
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "eval.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    cases_path.write_text(
        json.dumps({"query": "redevelopment policy", "expected_pages": [1]}) + "\n",
        encoding="utf-8",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="redevelopment policy",
    )
    calls = {}

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            return [
                QdrantHybridSearchHit(
                    item_id="chunk-1",
                    score=1.0,
                    sources=["qdrant:text_dense"],
                    chunk=chunk,
                )
            ]

    def fake_prepare_qdrant_hybrid_search(**kwargs):
        calls["prepare"] = kwargs
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": kwargs["collection"],
            "selected_vectors": ["text_dense", "caption_dense", "object_dense"],
            "query_encoders": {
                "text_dense": "default_text",
                "caption_dense": "default_text",
                "object_dense": "default_text",
            },
            "query_encoder_details": {
                "text_dense": {
                    "backend": kwargs["text_backend"],
                    "model": kwargs["text_model"],
                }
            },
            "upserted": {"count": 1},
            "chunks": [chunk],
            "assets": [],
            "triples": [],
        }

    monkeypatch.setattr(
        "chunking_docs.cli.prepare_qdrant_hybrid_search",
        fake_prepare_qdrant_hybrid_search,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-retrieval-config",
            str(config_path),
            str(cases_path),
            "--output",
            str(output_path),
            "--repeat",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["prepare"]["text_backend"] == "sentence-transformers"
    assert calls["prepare"]["text_model"] == "BAAI/bge-m3"
    assert calls["prepare"]["image_query_backend"] == "none"


def test_qdrant_rag_context_config_cli_uses_exported_settings(tmp_path, monkeypatch):
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())
    config = config.model_copy(
        update={
            "package_dir": str(tmp_path / "package"),
            "collection_name": "config_collection",
            "vector_names": ["text_dense"],
            "graph_expand": True,
            "fusion_weights": {"bm25": 1.2},
            "top_k": 3,
            "collapse_hierarchical": True,
            "lexical_tokenizer": {
                "strategy": "mixed",
                "min_n": 2,
                "max_n": 3,
                "ngram_cjk_only": False,
                "deduplicate": True,
            },
        }
    )
    config_path = tmp_path / "qdrant_retrieval_config.json"
    output_path = tmp_path / "context.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="redevelopment policy",
        metadata={"chunking_strategy": "semantic", "retrieval_role": "child"},
    )
    calls = {}

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            calls["search"] = kwargs
            return [
                QdrantHybridSearchHit(
                    item_id="chunk-1",
                    score=1.0,
                    sources=["bm25"],
                    chunk=chunk,
                )
            ]

    def fake_prepare_qdrant_hybrid_search(**kwargs):
        calls["prepare"] = kwargs
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": kwargs["collection"],
            "selected_vectors": ["text_dense"],
            "query_encoders": {"text_dense": "default_text"},
            "query_encoder_details": {"text_dense": {"backend": "hashing"}},
            "upserted": {"count": 1},
            "chunks": [chunk],
            "assets": [],
            "triples": [],
        }

    monkeypatch.setattr(
        "chunking_docs.cli.prepare_qdrant_hybrid_search",
        fake_prepare_qdrant_hybrid_search,
    )

    result = CliRunner().invoke(
        app,
        [
            "qdrant-rag-context-config",
            str(config_path),
            "redevelopment policy",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["prepare"]["package_dir"] == tmp_path / "package"
    assert calls["prepare"]["collection"] == "config_collection"
    assert calls["prepare"]["vector_names"] == "text_dense"
    assert calls["prepare"]["ngram_max"] == 3
    assert calls["prepare"]["ngram_cjk_only"] is False
    assert calls["prepare"]["deduplicate_tokens"] is True
    assert calls["search"]["query"] == "redevelopment policy"
    assert calls["search"]["top_k"] == 3
    assert calls["search"]["graph_expand"] is True
    assert calls["search"]["collapse_hierarchical"] is True
    assert calls["search"]["fusion_weights"] == {"bm25": 1.2}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["query"] == "redevelopment policy"
    assert payload["chunks"][0]["chunk_id"] == "chunk-1"
    assert payload["metadata"]["backend"] == "qdrant_hybrid_config"
    assert payload["metadata"]["config_selection"]["candidate"] == "balanced"
    assert payload["metadata"]["fusion_weights"] == {"bm25": 1.2}


def test_qdrant_rag_context_config_cli_auto_detects_image_query_encoder(
    tmp_path,
    monkeypatch,
):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "vectors": {
                    "text_dense": {
                        "dimension": 1024,
                        "embedding": {
                            "backend": "sentence-transformers",
                            "model": "BAAI/bge-m3",
                        },
                    },
                    "image_dense": {
                        "dimension": 768,
                        "embedding": {
                            "backend": "clip",
                            "model": "openai/clip-vit-large-patch14",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())
    config = config.model_copy(
        update={
            "package_dir": str(package_dir),
            "collection_name": "config_collection",
            "vector_names": ["text_dense", "image_dense"],
            "top_k": 3,
        }
    )
    config_path = tmp_path / "qdrant_retrieval_config.json"
    output_path = tmp_path / "context.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="redevelopment policy",
    )
    calls = {}

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            return [
                QdrantHybridSearchHit(
                    item_id="chunk-1",
                    score=1.0,
                    sources=["qdrant:image_dense"],
                    chunk=chunk,
                )
            ]

    def fake_prepare_qdrant_hybrid_search(**kwargs):
        calls["prepare"] = kwargs
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": kwargs["collection"],
            "selected_vectors": ["text_dense", "image_dense"],
            "query_encoders": {
                "text_dense": "default_text",
                "image_dense": kwargs["image_query_backend"],
            },
            "query_encoder_details": {
                "image_dense": {
                    "backend": kwargs["image_query_backend"],
                    "model": kwargs["image_query_model"],
                }
            },
            "upserted": {"count": 1},
            "chunks": [chunk],
            "assets": [],
            "triples": [],
        }

    monkeypatch.setattr(
        "chunking_docs.cli.prepare_qdrant_hybrid_search",
        fake_prepare_qdrant_hybrid_search,
    )

    result = CliRunner().invoke(
        app,
        [
            "qdrant-rag-context-config",
            str(config_path),
            "redevelopment policy",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["prepare"]["text_backend"] == "sentence-transformers"
    assert calls["prepare"]["text_model"] == "BAAI/bge-m3"
    assert calls["prepare"]["image_query_backend"] == "clip"
    assert calls["prepare"]["image_query_model"] == "openai/clip-vit-large-patch14"


def test_eval_qdrant_rag_context_config_cli_scores_generated_contexts(
    tmp_path,
    monkeypatch,
):
    config = build_qdrant_retrieval_config_from_fusion_sweep(fusion_sweep_report())
    config = config.model_copy(
        update={
            "package_dir": str(tmp_path / "package"),
            "collection_name": "config_collection",
            "vector_names": ["text_dense"],
            "graph_expand": True,
            "fusion_weights": {"bm25": 1.2},
            "top_k": 3,
            "collapse_hierarchical": True,
            "lexical_tokenizer": {
                "strategy": "mixed",
                "min_n": 2,
                "max_n": 3,
                "ngram_cjk_only": False,
                "deduplicate": True,
            },
            "routes": [
                QdrantRetrievalRoute(
                    name="graph_triple",
                    match_query_terms=["visual"],
                    vector_names=["text_dense"],
                    graph_expand=True,
                    fusion_weights={"bm25": 1.2},
                )
            ],
        }
    )
    config_path = tmp_path / "qdrant_retrieval_config.json"
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "context_eval.json"
    contexts_path = tmp_path / "contexts.jsonl"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    cases_path.write_text(
        json.dumps(
            {
                "query": "redevelopment visual policy",
                "expected_pages": [1],
                "expected_asset_ids": ["asset-1"],
                "expected_triple_ids": ["triple-1"],
                "metadata": {"case_source": "visual_object_probe"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="redevelopment visual policy",
        metadata={"chunking_strategy": "semantic", "retrieval_role": "child"},
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="redevelopment map",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="redevelopment",
        predicate="uses",
        object="station corridor",
        qualifiers={"asset_id": "asset-1"},
    )
    calls = {}

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            calls["search"] = kwargs
            return [
                QdrantHybridSearchHit(
                    item_id="chunk-1",
                    score=1.0,
                    sources=["qdrant:text_dense"],
                    chunk=chunk,
                    payloads=[
                        {
                            "asset_id": ["asset-1"],
                            "triple_ids": ["triple-1"],
                            "doc_id": "doc",
                            "page_no": 1,
                            "kind": "map",
                        }
                    ],
                )
            ]

    def fake_prepare_qdrant_hybrid_search(**kwargs):
        calls["prepare"] = kwargs
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": kwargs["collection"],
            "selected_vectors": ["text_dense"],
            "query_encoders": {"text_dense": "default_text"},
            "query_encoder_details": {"text_dense": {"backend": "hashing"}},
            "upserted": {"count": 1},
            "chunks": [chunk],
            "assets": [asset],
            "triples": [triple],
        }

    monkeypatch.setattr(
        "chunking_docs.cli.prepare_qdrant_hybrid_search",
        fake_prepare_qdrant_hybrid_search,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-rag-context-config",
            str(config_path),
            str(cases_path),
            "--output",
            str(output_path),
            "--contexts-output",
            str(contexts_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["prepare"]["package_dir"] == tmp_path / "package"
    assert calls["prepare"]["collection"] == "config_collection"
    assert calls["prepare"]["vector_names"] == "text_dense"
    assert calls["prepare"]["ngram_max"] == 3
    assert calls["search"]["query"] == "redevelopment visual policy"
    assert calls["search"]["top_k"] == 3
    assert calls["search"]["graph_expand"] is True
    assert calls["search"]["collapse_hierarchical"] is True
    assert calls["search"]["fusion_weights"] == {"bm25": 1.2}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed_count"] == 1
    assert payload["target_coverage"] == 1.0
    assert payload["target_metrics"]["asset"]["coverage"] == 1.0
    assert payload["target_metrics"]["triple"]["coverage"] == 1.0
    assert payload["metadata"]["backend"] == "qdrant_rag_context_config"
    assert payload["metadata"]["config_selection"]["candidate"] == "balanced"
    assert payload["case_group_metrics"]["retrieval_route"]["graph_triple"][
        "target_coverage"
    ] == 1.0
    context_payload = json.loads(contexts_path.read_text(encoding="utf-8").splitlines()[0])
    assert context_payload["metadata"]["case_metadata"]["case_source"] == "visual_object_probe"
    assert context_payload["metadata"]["case_metadata"]["retrieval_route"] == "graph_triple"
    assert context_payload["metadata"]["retrieved_asset_ids"] == ["asset-1"]


def fusion_sweep_report():
    balanced = QdrantFusionSweepCandidate(
        name="balanced",
        fusion_weights={"bm25": 1.0, "qdrant:caption_dense": 1.0},
        evaluation=evaluation(
            recall=0.96,
            coverage=0.96,
            ndcg=0.92,
            mrr=0.9,
            precision=0.22,
            failed=0,
            latency=25.0,
            visual_object_group=case_group_metric(
                coverage=0.5,
                ndcg=0.48,
                mrr=0.5,
            ),
            results=[
                retrieval_result(
                    "shared redevelopment query",
                    coverage=1.0,
                    ndcg=0.9,
                    reciprocal_rank=1.0,
                    rank=1,
                )
            ],
            source_metrics={
                "qdrant:caption_dense": source_metric(precision=0.82),
            },
            source_family_metrics={
                "visual": source_metric(precision=0.88),
            },
        ),
    )
    object_weighted = QdrantFusionSweepCandidate(
        name="object_weighted",
        fusion_weights={"bm25": 1.0, "qdrant:object_dense": 1.4},
        evaluation=evaluation(
            recall=0.9,
            coverage=0.9,
            ndcg=0.86,
            mrr=0.82,
            precision=0.2,
            failed=0,
            latency=35.0,
            visual_object_group=case_group_metric(
                coverage=0.9,
                ndcg=0.88,
                mrr=0.84,
            ),
            results=[
                retrieval_result(
                    "shared redevelopment query",
                    coverage=0.0,
                    ndcg=0.0,
                    reciprocal_rank=0.0,
                    rank=None,
                )
            ],
            source_metrics={
                "qdrant:object_dense": source_metric(precision=0.74),
            },
            source_family_metrics={
                "visual": source_metric(precision=0.79),
            },
        ),
    )

    return build_qdrant_fusion_sweep_report(
        [balanced, object_weighted],
        vector_names=["text_dense", "caption_dense", "object_dense"],
        min_recall_at_k=0.8,
        min_target_coverage_at_k=0.8,
        max_failed_queries=1,
        metadata={
            "collection": "seoul_plan",
            "package_dir": "outputs/package",
            "bm25_tokens_path": "outputs/package/bm25_tokens.json",
            "top_k": 7,
            "collapse_hierarchical": True,
            "query_encoders": {
                "text_dense": "BAAI/bge-m3",
                "caption_dense": "BAAI/bge-m3",
                "object_dense": "BAAI/bge-m3",
            },
            "lexical_tokenizer": {
                "strategy": "mixed",
                "min_n": 2,
                "max_n": 4,
                "ngram_cjk_only": True,
                "deduplicate": False,
            },
        },
    )


def evaluation(
    recall: float,
    coverage: float,
    ndcg: float,
    mrr: float,
    precision: float,
    failed: int,
    latency: float,
    visual_object_group: RetrievalCaseGroupMetric,
    results: list[RetrievalCaseResult] | None = None,
    source_metrics: dict[str, RetrievalSourceMetric] | None = None,
    source_family_metrics: dict[str, RetrievalSourceMetric] | None = None,
) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        case_count=10,
        expected_case_count=10,
        passed_count=10 - failed,
        failed_count=failed,
        hit_rate=recall,
        recall_at_k=recall,
        mrr=mrr,
        target_coverage_at_k=coverage,
        mean_target_ndcg_at_k=ndcg,
        mean_precision_at_k=precision,
        top_k=5,
        total_query_latency_ms=latency * 10,
        mean_latency_ms=latency,
        p95_latency_ms=latency,
        failed_queries=[f"failed-{index}" for index in range(failed)],
        source_metrics=source_metrics or {},
        source_family_metrics=source_family_metrics or {},
        case_group_metrics={"case_source": {"visual_object_probe": visual_object_group}},
        results=results or [],
    )


def source_metric(precision: float) -> RetrievalSourceMetric:
    return RetrievalSourceMetric(
        query_count=1,
        relevant_query_count=1 if precision else 0,
        hit_count=1,
        relevant_hit_count=1 if precision else 0,
        precision_at_hits=precision,
        target_coverage_at_k=precision,
    )


def retrieval_result(
    query: str,
    coverage: float,
    ndcg: float,
    reciprocal_rank: float,
    rank: int | None,
) -> RetrievalCaseResult:
    target_ranks = {"page:1": rank} if rank is not None else {}
    return RetrievalCaseResult(
        query=query,
        passed=rank is not None,
        latency_ms=10.0,
        top_pages=[1] if rank is not None else [],
        top_chunk_ids=["chunk-1"] if rank is not None else [],
        expected_pages=[1],
        expected_chunk_ids=[],
        expected_target_count=1,
        matched_target_count=1 if rank is not None else 0,
        target_coverage_at_k=coverage,
        target_ndcg_at_k=ndcg,
        relevant_hit_count=1 if rank is not None else 0,
        precision_at_k=0.2 if rank is not None else 0.0,
        matched_rank=rank,
        reciprocal_rank=reciprocal_rank,
        target_matches={"page:1": rank is not None},
        target_matched_ranks=target_ranks,
        target_key_matched_ranks=target_ranks,
    )


def case_group_metric(
    coverage: float,
    ndcg: float,
    mrr: float,
    recall: float = 1.0,
    precision: float = 0.2,
    failed: int = 0,
    latency: float = 20.0,
) -> RetrievalCaseGroupMetric:
    return RetrievalCaseGroupMetric(
        case_count=5,
        expected_case_count=5,
        passed_count=5 - failed,
        failed_count=failed,
        recall_at_k=recall,
        mrr=mrr,
        target_count=10,
        matched_target_count=int(round(coverage * 10)),
        target_coverage_at_k=coverage,
        ndcg_at_k=ndcg,
        precision_at_k=precision,
        mean_latency_ms=latency,
        failed_queries=[f"group-failed-{index}" for index in range(failed)],
    )
