from __future__ import annotations

import re

from chunking_docs.embeddings.records import asset_text
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset

_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_RE = re.compile(r"[|#*_`>\[\]()]")
_PLACEHOLDER_PATTERNS = [
    "OCR/VLM processing required",
    "Full page render for page",
]


def generate_retrieval_case_skeleton(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    page_limit: int = 20,
    asset_limit: int = 20,
    triple_limit: int = 20,
    include_pages: bool = True,
    include_assets: bool = True,
    include_triples: bool = True,
    include_todo: bool = False,
    query_max_chars: int = 120,
) -> list[RetrievalCase]:
    cases: list[RetrievalCase] = []
    if include_pages:
        cases.extend(page_cases(chunks, limit=page_limit, include_todo=include_todo, query_max_chars=query_max_chars))
    if include_assets:
        cases.extend(asset_cases(assets, limit=asset_limit, include_todo=include_todo, query_max_chars=query_max_chars))
    if include_triples:
        cases.extend(triple_cases(triples, limit=triple_limit, query_max_chars=query_max_chars))
    return cases


def page_cases(
    chunks: list[DocumentChunk],
    limit: int,
    include_todo: bool,
    query_max_chars: int,
) -> list[RetrievalCase]:
    cases = []
    seen_pages: set[int] = set()
    for chunk in sorted(chunks, key=lambda item: (item.page_start, item.page_end, item.chunk_id)):
        if len(cases) >= limit:
            break
        if chunk.page_start in seen_pages:
            continue
        query = query_from_text(chunk.text, max_chars=query_max_chars)
        if not query and include_todo:
            query = f"TODO: write query for page {chunk.page_start}"
        if not query:
            continue
        seen_pages.add(chunk.page_start)
        cases.append(
            RetrievalCase(
                query=query,
                expected_pages=[chunk.page_start],
                expected_chunk_ids=[chunk.chunk_id],
            )
        )
    return cases


def asset_cases(
    assets: list[VisualAsset],
    limit: int,
    include_todo: bool,
    query_max_chars: int,
) -> list[RetrievalCase]:
    cases = []
    for asset in sorted(assets, key=lambda item: (asset_priority(item), item.page_no, item.asset_id)):
        if len(cases) >= limit:
            break
        query = query_from_text(asset_text(asset), max_chars=query_max_chars)
        if not query and include_todo:
            query = f"TODO: write query for {asset.kind} asset on page {asset.page_no}"
        if not query:
            continue
        cases.append(
            RetrievalCase(
                query=query,
                expected_pages=[asset.page_no],
                expected_asset_ids=[asset.asset_id],
            )
        )
    return cases


def triple_cases(
    triples: list[GraphTriple],
    limit: int,
    query_max_chars: int,
) -> list[RetrievalCase]:
    cases = []
    for triple in sorted(triples, key=lambda item: (item.chunk_id, item.triple_id)):
        if len(cases) >= limit:
            break
        query = query_from_text(
            f"{triple.subject} {triple.predicate} {triple.object}",
            max_chars=query_max_chars,
        )
        if not query:
            continue
        cases.append(
            RetrievalCase(
                query=query,
                expected_chunk_ids=[triple.chunk_id],
                expected_triple_ids=[triple.triple_id],
                graph_expand=True,
            )
        )
    return cases


def query_from_text(text: str, max_chars: int = 120) -> str:
    normalized = normalize_query_text(text)
    if not normalized or is_placeholder_text(normalized):
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip() or normalized[:max_chars].strip()


def normalize_query_text(text: str) -> str:
    text = _MARKDOWN_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_placeholder_text(text: str) -> bool:
    return any(pattern in text for pattern in _PLACEHOLDER_PATTERNS)


def asset_priority(asset: VisualAsset) -> int:
    return {
        "map": 0,
        "table": 1,
        "chart": 2,
        "figure": 3,
        "page_image": 4,
    }.get(str(asset.kind), 5)
