from __future__ import annotations

from pydantic import BaseModel, Field

from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.retrieval import RetrievalCase, RetrievalEvaluation, evaluate_retrieval
from chunking_docs.models import DocumentChunk, GraphTriple


class RetrievalAblationMode(BaseModel):
    name: str
    use_dense: bool = True
    use_bm25: bool = True
    use_graph: bool = False
    graph_expand: bool = False


class RetrievalAblationRow(BaseModel):
    mode: RetrievalAblationMode
    evaluation: RetrievalEvaluation


class RetrievalAblationReport(BaseModel):
    rows: list[RetrievalAblationRow]
    best_by_recall: str | None
    best_by_mrr: str | None
    fastest_by_mean_latency: str | None


class QdrantVectorAblationMode(BaseModel):
    name: str
    vector_names: list[str] = Field(default_factory=list)
    graph_expand: bool = False


class QdrantVectorAblationRow(BaseModel):
    mode: QdrantVectorAblationMode
    evaluation: RetrievalEvaluation


class QdrantVectorAblationReport(BaseModel):
    rows: list[QdrantVectorAblationRow]
    best_by_recall: str | None
    best_by_mrr: str | None
    fastest_by_mean_latency: str | None


DEFAULT_ABLATION_MODES = {
    "dense": RetrievalAblationMode(name="dense", use_dense=True, use_bm25=False),
    "bm25": RetrievalAblationMode(name="bm25", use_dense=False, use_bm25=True),
    "hybrid": RetrievalAblationMode(name="hybrid", use_dense=True, use_bm25=True),
    "graph": RetrievalAblationMode(
        name="graph",
        use_dense=False,
        use_bm25=False,
        use_graph=True,
    ),
    "hybrid_graph": RetrievalAblationMode(
        name="hybrid_graph",
        use_dense=True,
        use_bm25=True,
        use_graph=True,
        graph_expand=True,
    ),
}


DEFAULT_QDRANT_VECTOR_ABLATION_MODES = {
    "text": QdrantVectorAblationMode(name="text", vector_names=["text_dense"]),
    "caption": QdrantVectorAblationMode(name="caption", vector_names=["caption_dense"]),
    "image": QdrantVectorAblationMode(name="image", vector_names=["image_dense"]),
    "text_caption": QdrantVectorAblationMode(
        name="text_caption",
        vector_names=["text_dense", "caption_dense"],
    ),
    "text_image": QdrantVectorAblationMode(
        name="text_image",
        vector_names=["text_dense", "image_dense"],
    ),
    "caption_image": QdrantVectorAblationMode(
        name="caption_image",
        vector_names=["caption_dense", "image_dense"],
    ),
    "all": QdrantVectorAblationMode(
        name="all",
        vector_names=["text_dense", "caption_dense", "image_dense"],
    ),
    "text_caption_graph": QdrantVectorAblationMode(
        name="text_caption_graph",
        vector_names=["text_dense", "caption_dense"],
        graph_expand=True,
    ),
    "all_graph": QdrantVectorAblationMode(
        name="all_graph",
        vector_names=["text_dense", "caption_dense", "image_dense"],
        graph_expand=True,
    ),
}


def evaluate_retrieval_ablation(
    chunks: list[DocumentChunk],
    triples: list[GraphTriple],
    cases: list[RetrievalCase],
    modes: list[RetrievalAblationMode] | None = None,
    top_k: int = 5,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = False,
    repeat: int = 1,
) -> RetrievalAblationReport:
    rows = [
        RetrievalAblationRow(
            mode=mode,
            evaluation=evaluate_retrieval(
                chunks=chunks,
                triples=triples,
                cases=cases,
                top_k=top_k,
                tokenizer_config=tokenizer_config,
                collapse_hierarchical=collapse_hierarchical,
                graph_expand_override=mode.graph_expand,
                use_dense=mode.use_dense,
                use_bm25=mode.use_bm25,
                use_graph=mode.use_graph,
                repeat=repeat,
            ),
        )
        for mode in (modes or list(DEFAULT_ABLATION_MODES.values()))
    ]
    rows.sort(
        key=lambda row: (
            row.evaluation.recall_at_k,
            row.evaluation.mrr,
            row.evaluation.hit_rate,
        ),
        reverse=True,
    )
    return RetrievalAblationReport(
        rows=rows,
        best_by_recall=rows[0].mode.name if rows else None,
        best_by_mrr=max(rows, key=lambda row: row.evaluation.mrr).mode.name if rows else None,
        fastest_by_mean_latency=min(rows, key=lambda row: row.evaluation.mean_latency_ms).mode.name
        if rows
        else None,
    )


def parse_ablation_modes(value: str) -> list[RetrievalAblationMode]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        return list(DEFAULT_ABLATION_MODES.values())
    unknown = sorted(set(names) - set(DEFAULT_ABLATION_MODES))
    if unknown:
        raise ValueError(f"Unsupported ablation modes: {', '.join(unknown)}")
    return [DEFAULT_ABLATION_MODES[name] for name in names]


def parse_qdrant_vector_ablation_modes(value: str) -> list[QdrantVectorAblationMode]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        names = ["text", "caption", "text_caption", "text_caption_graph"]
    unknown = sorted(set(names) - set(DEFAULT_QDRANT_VECTOR_ABLATION_MODES))
    if unknown:
        raise ValueError(f"Unsupported Qdrant vector ablation modes: {', '.join(unknown)}")
    return [DEFAULT_QDRANT_VECTOR_ABLATION_MODES[name].model_copy(deep=True) for name in names]


def qdrant_vector_names_for_modes(modes: list[QdrantVectorAblationMode]) -> list[str]:
    seen: set[str] = set()
    vector_names: list[str] = []
    for mode in modes:
        for vector_name in mode.vector_names:
            if vector_name not in seen:
                seen.add(vector_name)
                vector_names.append(vector_name)
    return vector_names


def build_qdrant_vector_ablation_report(
    rows: list[QdrantVectorAblationRow],
) -> QdrantVectorAblationReport:
    rows = sorted(
        rows,
        key=lambda row: (
            row.evaluation.recall_at_k,
            row.evaluation.mrr,
            row.evaluation.hit_rate,
        ),
        reverse=True,
    )
    return QdrantVectorAblationReport(
        rows=rows,
        best_by_recall=rows[0].mode.name if rows else None,
        best_by_mrr=max(rows, key=lambda row: row.evaluation.mrr).mode.name if rows else None,
        fastest_by_mean_latency=min(rows, key=lambda row: row.evaluation.mean_latency_ms).mode.name
        if rows
        else None,
    )
