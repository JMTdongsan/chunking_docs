from __future__ import annotations


def qdrant_rag_validation_commands(
    vector_names: list[str],
    route_preset: str | None = None,
) -> list[str]:
    vector_csv = ",".join(vector_names)
    has_visual = any(name in vector_names for name in ["caption_dense", "object_dense", "image_dense"])
    has_image = "image_dense" in vector_names
    has_object = "object_dense" in vector_names
    has_triple = "triple_dense" in vector_names
    selected_route_preset = (
        qdrant_route_preset_for_vectors(vector_names)
        if route_preset is None
        else route_preset.strip()
    )
    image_query_args = [
        "--image-query-backend clip",
        "--image-query-model openai/clip-vit-large-patch14",
    ] if has_image else []
    route_preset_args = [
        f"--route-preset {selected_route_preset}"
    ] if selected_route_preset else []
    fusion_weight_args = [
        "--weight-grid bm25=0.8,1.0,1.2",
        "--fixed-fusion-weight qdrant:text_dense=1.0",
    ]
    if "caption_dense" in vector_names:
        fusion_weight_args.append("--weight-grid qdrant:caption_dense=0.8,1.0,1.2")
    if has_object:
        fusion_weight_args.append("--weight-grid qdrant:object_dense=0.5,1.0,1.5")
    if has_image:
        fusion_weight_args.append("--weight-grid qdrant:image_dense=0.0,0.25,0.5")
    if has_triple:
        fusion_weight_args.append("--fixed-fusion-weight qdrant:triple_dense=0.5")

    fusion_gate_args = [
        "--min-target-coverage-at-k 0.8",
        "--min-target-ndcg-at-k 0.7",
        "--max-failed-queries 3",
        "--max-p95-latency-ms 250",
        "--reranker lexical",
        "--rerank-top-k 20",
        "--pairwise-top-k 10",
    ]
    if has_visual:
        fusion_gate_args.extend(
            [
                "--max-source-family-excluded-target-hit-rate visual=0.0",
                "--source-family-excluded-target-hit-penalty 1.0",
            ]
        )
    if has_image:
        fusion_gate_args.append("--max-source-excluded-target-hit-rate qdrant:image_dense=0.0")

    rag_context_gate_args = [
        "--min-target-coverage 0.8",
        "--max-excluded-target-hit-rate 0",
        "--max-mean-context-char-count 12000",
    ]
    if has_visual:
        rag_context_gate_args.append("--min-target-type-coverage asset=0.75")
    if has_triple:
        rag_context_gate_args.append("--min-target-type-coverage triple=0.7")
    if has_object:
        rag_context_gate_args.append(
            "--min-case-group-target-coverage case_source:visual_object_probe=0.7"
        )

    return [
        " ".join(
            [
                "chunking-docs eval-qdrant-retrieval examples/retrieval_cases.jsonl",
                "--package-dir outputs/package",
                "--location ':memory:'",
                f"--vector-names {vector_csv}",
                "--top-k 5",
                "--repeat 3",
                "--output outputs/package/qdrant_retrieval_eval.json",
                *image_query_args,
            ]
        ),
        " ".join(
            [
                "chunking-docs sweep-qdrant-fusion examples/retrieval_cases.jsonl",
                "--package-dir outputs/package",
                "--location ':memory:'",
                f"--vector-names {vector_csv}",
                *image_query_args,
                *fusion_weight_args,
                *fusion_gate_args,
                "--output outputs/package/qdrant_fusion_sweep.json",
            ]
        ),
        (
            " ".join(
                [
                    "chunking-docs export-qdrant-retrieval-config",
                    "outputs/package/qdrant_fusion_sweep.json",
                    *route_preset_args,
                    "--output outputs/package/qdrant_retrieval_config.json",
                ]
            )
        ),
        (
            " ".join(
                [
                    "chunking-docs eval-qdrant-retrieval-config",
                    "outputs/package/qdrant_retrieval_config.json examples/retrieval_cases.jsonl",
                    "--package-dir outputs/package --location ':memory:' --repeat 3",
                    "--output outputs/package/qdrant_retrieval_config_eval.json",
                    *image_query_args,
                ]
            )
        ),
        (
            " ".join(
                [
                    "chunking-docs eval-qdrant-rag-context-config",
                    "outputs/package/qdrant_retrieval_config.json examples/retrieval_cases.jsonl",
                    "--package-dir outputs/package --location ':memory:'",
                    "--contexts-output outputs/package/rag_context.config.cases.jsonl",
                    "--output outputs/package/qdrant_rag_context_config_eval.json",
                    *image_query_args,
                ]
            )
        ),
        " ".join(
            [
                "chunking-docs gate-rag-context",
                "outputs/package/qdrant_rag_context_config_eval.json",
                *rag_context_gate_args,
                "--output outputs/package/qdrant_rag_context_gate.json",
            ]
        ),
    ]


def qdrant_route_preset_for_vectors(vector_names: list[str]) -> str:
    names = set(vector_names)
    if {"object_dense", "triple_dense"}.issubset(names):
        return "adaptive"
    return ""


QDRANT_RAG_READINESS_GATE_ARGS = [
    "--qdrant-retrieval-config {qdrant_retrieval_config}",
    "--require-qdrant-retrieval-config",
    "--retrieval-evaluation {qdrant_retrieval_config_evaluation}",
    "--require-retrieval-evaluation",
    "--min-target-coverage-at-k 0.8",
    "--min-target-ndcg-at-k 0.7",
    "--max-retrieval-failed-queries 3",
    "--max-p95-latency-ms 250",
    "--rag-context-evaluation {rag_context_evaluation}",
    "--require-rag-context-evaluation",
    "--min-rag-context-target-coverage 0.8",
    "--max-rag-context-excluded-target-hit-rate 0",
    "--max-rag-context-mean-context-char-count 12000",
]


QDRANT_ADAPTIVE_ROUTE_READINESS_GATE_ARGS = [
    "--min-retrieval-case-group-target-coverage retrieval_route:graph_triple=0.7",
    "--min-retrieval-case-group-target-coverage retrieval_route:visual_object=0.7",
    "--min-retrieval-case-group-source-target-coverage retrieval_route:graph_triple:qdrant:triple_dense=0.7",
    "--min-retrieval-case-group-source-target-coverage retrieval_route:visual_object:qdrant:object_dense=0.3",
    "--min-retrieval-case-group-source-family-target-coverage retrieval_route:graph_triple:graph=0.7",
    "--min-retrieval-case-group-source-family-target-coverage retrieval_route:visual_object:visual=0.3",
    "--min-rag-context-case-group-target-coverage retrieval_route:graph_triple=0.7",
    "--min-rag-context-case-group-target-coverage retrieval_route:visual_object=0.7",
]
