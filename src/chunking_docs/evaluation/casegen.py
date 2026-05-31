from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from chunking_docs.embeddings.records import asset_text
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset

_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_RE = re.compile(r"[|#*_`>\[\]()]")
_TERM_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_./%+-]{2,}|"
    r"[0-9]+(?:[.,][0-9]+)*(?:[%A-Za-z가-힣㎡²³]*)?|"
    r"[가-힣]{2,}|"
    r"[\u4e00-\u9fff]{2,}"
)
_ID_LIKE_RE = re.compile(r"(?=.*[0-9])(?=.*[a-f])[a-f0-9]{2,}", re.IGNORECASE)
_PLACEHOLDER_PATTERNS = [
    "OCR/VLM processing required",
    "Full page render for page",
]
_STOP_TERMS = {
    "about",
    "above",
    "appendix",
    "belongs",
    "belong",
    "chapter",
    "contents",
    "defines",
    "document",
    "example",
    "figure",
    "includes",
    "introduction",
    "overview",
    "page",
    "range",
    "reference",
    "report",
    "section",
    "summary",
    "table",
    "본문",
    "보고서",
    "부록",
    "요약",
    "개요",
    "그림",
    "목차",
    "자료",
    "참고",
    "페이지",
}

QueryMode = Literal["snippet", "salient_terms"]
SelectionStrategy = Literal["document_order", "salience"]


@dataclass(frozen=True)
class QueryDraft:
    query: str
    score: float = 0.0
    terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseCandidate:
    case: RetrievalCase
    score: float
    order: tuple


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
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    min_query_terms: int = 3,
    max_query_terms: int = 8,
    dedupe_queries: bool = True,
) -> list[RetrievalCase]:
    validate_casegen_options(query_mode, selection_strategy, min_query_terms, max_query_terms)
    corpus_texts = case_source_texts(chunks, assets, triples)
    term_df = term_document_frequencies(corpus_texts)
    cases: list[RetrievalCase] = []
    if include_pages:
        cases.extend(
            page_cases(
                chunks,
                limit=page_limit,
                include_todo=include_todo,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
            )
        )
    if include_assets:
        cases.extend(
            asset_cases(
                assets,
                limit=asset_limit,
                include_todo=include_todo,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
            )
        )
    if include_triples:
        cases.extend(
            triple_cases(
                triples,
                limit=triple_limit,
                query_max_chars=query_max_chars,
                query_mode=query_mode,
                selection_strategy=selection_strategy,
                term_df=term_df,
                document_count=len(corpus_texts),
                min_query_terms=min_query_terms,
                max_query_terms=max_query_terms,
            )
        )
    if dedupe_queries:
        cases = dedupe_cases_by_query(cases)
    return cases


def page_cases(
    chunks: list[DocumentChunk],
    limit: int,
    include_todo: bool,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> list[RetrievalCase]:
    candidates: list[CaseCandidate] = []
    for chunk in sorted(chunks, key=lambda item: (item.page_start, item.page_end, item.chunk_id)):
        draft = query_from_text(
            chunk.text,
            max_chars=query_max_chars,
            mode=query_mode,
            term_df=term_df,
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        query = draft.query
        if not query and include_todo:
            query = f"TODO: write query for page {chunk.page_start}"
        if not query:
            continue
        case = with_case_metadata(
            RetrievalCase(
                query=query,
                expected_pages=[chunk.page_start],
                expected_chunk_ids=[chunk.chunk_id],
            ),
            case_source="page",
            query_mode=query_mode,
            draft=draft,
        )
        candidates.append(
            CaseCandidate(
                case=case,
                score=draft.score,
                order=(chunk.page_start, chunk.page_end, chunk.chunk_id),
            )
        )
    return select_candidates(dedupe_page_candidates(candidates, selection_strategy), limit, selection_strategy)


def asset_cases(
    assets: list[VisualAsset],
    limit: int,
    include_todo: bool,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> list[RetrievalCase]:
    candidates: list[CaseCandidate] = []
    for asset in sorted(assets, key=lambda item: (asset_priority(item), item.page_no, item.asset_id)):
        draft = query_from_text(
            asset_text(asset),
            max_chars=query_max_chars,
            mode=query_mode,
            term_df=term_df,
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        query = draft.query
        if not query and include_todo:
            query = f"TODO: write query for {asset.kind} asset on page {asset.page_no}"
        if not query:
            continue
        case = with_case_metadata(
            RetrievalCase(
                query=query,
                expected_pages=[asset.page_no],
                expected_asset_ids=[asset.asset_id],
            ),
            case_source="asset",
            query_mode=query_mode,
            draft=draft,
        )
        candidates.append(
            CaseCandidate(
                case=case,
                score=draft.score,
                order=(asset_priority(asset), asset.page_no, asset.asset_id),
            )
        )
    return select_candidates(merge_case_candidates_by_query(candidates), limit, selection_strategy)


def triple_cases(
    triples: list[GraphTriple],
    limit: int,
    query_max_chars: int,
    query_mode: QueryMode = "snippet",
    selection_strategy: SelectionStrategy = "document_order",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> list[RetrievalCase]:
    candidates: list[CaseCandidate] = []
    for triple in sorted(triples, key=lambda item: (item.chunk_id, item.triple_id)):
        draft = query_from_text(
            triple_query_text(triple),
            max_chars=query_max_chars,
            mode=query_mode,
            term_df=term_df,
            document_count=document_count,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
        query = draft.query
        if not query:
            continue
        case = with_case_metadata(
            RetrievalCase(
                query=query,
                expected_chunk_ids=[triple.chunk_id],
                expected_triple_ids=[triple.triple_id],
                graph_expand=True,
            ),
            case_source="triple",
            query_mode=query_mode,
            draft=draft,
        )
        candidates.append(
            CaseCandidate(
                case=case,
                score=draft.score,
                order=(triple.chunk_id, triple.triple_id),
            )
        )
    return select_candidates(merge_case_candidates_by_query(candidates), limit, selection_strategy)


def query_from_text(
    text: str,
    max_chars: int = 120,
    mode: QueryMode = "snippet",
    term_df: dict[str, int] | None = None,
    document_count: int = 0,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> QueryDraft:
    normalized = meaningful_query_text(text)
    if not normalized:
        return QueryDraft(query="")
    if mode == "salient_terms":
        return salient_terms_query(
            normalized,
            term_df=term_df or {},
            document_count=document_count,
            max_chars=max_chars,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
        )
    return QueryDraft(query=trim_query(normalized, max_chars=max_chars))


def normalize_query_text(text: str) -> str:
    text = _MARKDOWN_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_placeholder_text(text: str) -> bool:
    return not meaningful_query_text(text) and any(pattern in text for pattern in _PLACEHOLDER_PATTERNS)


def meaningful_query_text(text: str) -> str:
    lines = []
    for line in text.splitlines() or [text]:
        if any(pattern in line for pattern in _PLACEHOLDER_PATTERNS):
            continue
        lines.append(line)
    return normalize_query_text(" ".join(lines))


def salient_terms_query(
    text: str,
    term_df: dict[str, int],
    document_count: int,
    max_chars: int,
    min_query_terms: int = 3,
    max_query_terms: int = 8,
) -> QueryDraft:
    terms = extracted_terms(text)
    if not terms:
        return QueryDraft(query="")
    scored_terms = [
        (term_score(key, term_df=term_df, document_count=document_count), position, term, key)
        for position, term, key in terms
    ]
    selected = select_salient_terms(scored_terms, min_query_terms=min_query_terms, max_query_terms=max_query_terms)
    if len(selected) < min_query_terms:
        return QueryDraft(query="")
    selected_by_position = sorted(selected, key=lambda item: item[1])
    query_terms = [term for _, _, term, _ in selected_by_position]
    query = trim_query(" ".join(query_terms), max_chars=max_chars)
    return QueryDraft(
        query=query,
        score=sum(score for score, _, _, _ in selected),
        terms=tuple(query_terms),
    )


def extracted_terms(text: str) -> list[tuple[int, str, str]]:
    terms = []
    seen_keys = set()
    for match in _TERM_RE.finditer(text):
        term = match.group(0).strip(".,;:!?/\\-+")
        key = normalize_term(term)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        terms.append((match.start(), term, key))
    return terms


def normalize_term(term: str) -> str:
    key = term.lower().strip()
    if len(key) < 2 or key in _STOP_TERMS:
        return ""
    if is_identifier_like(key):
        return ""
    if key.isdigit() and len(key) != 4:
        return ""
    return key


def is_identifier_like(value: str) -> bool:
    return bool(_ID_LIKE_RE.fullmatch(value)) or (
        len(value) >= 12 and any(character.isdigit() for character in value)
    )


def term_document_frequencies(texts: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        normalized = meaningful_query_text(text)
        if not normalized:
            continue
        counts.update({key for _, _, key in extracted_terms(normalized)})
    return dict(counts)


def term_score(term: str, term_df: dict[str, int], document_count: int) -> float:
    df = max(1, term_df.get(term, 1))
    idf = math.log((document_count + 1) / df) + 1 if document_count else 1.0
    length_score = min(len(term), 16) / 8
    digit_penalty = 0.75 if term.replace(".", "").replace(",", "").isdigit() else 1.0
    return idf * length_score * digit_penalty


def select_salient_terms(
    scored_terms: list[tuple[float, int, str, str]],
    min_query_terms: int,
    max_query_terms: int,
) -> list[tuple[float, int, str, str]]:
    selected: list[tuple[float, int, str, str]] = []
    selected_keys: set[str] = set()
    for score, position, term, key in sorted(scored_terms, key=lambda item: (-item[0], item[1])):
        if len(selected) >= max_query_terms:
            break
        if any(key in selected_key or selected_key in key for selected_key in selected_keys):
            continue
        selected.append((score, position, term, key))
        selected_keys.add(key)
    if len(selected) >= min_query_terms:
        return selected
    for score, position, term, key in sorted(scored_terms, key=lambda item: item[1]):
        if len(selected) >= min_query_terms:
            break
        if key in selected_keys:
            continue
        selected.append((score, position, term, key))
        selected_keys.add(key)
    return selected


def trim_query(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() or text[:max_chars].strip()


def case_source_texts(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
) -> list[str]:
    texts = [chunk.text for chunk in chunks]
    texts.extend(asset_text(asset) for asset in assets)
    texts.extend(triple_query_text(triple) for triple in triples)
    return [text for text in texts if meaningful_query_text(text)]


def triple_query_text(triple: GraphTriple) -> str:
    components = [triple.subject, triple.predicate.replace("_", " "), triple.object]
    return " ".join(component for component in components if meaningful_component(component))


def meaningful_component(value: str) -> bool:
    normalized = normalize_query_text(value)
    if not normalized:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", normalized)
    if compact and is_identifier_like(compact):
        return False
    terms = [key for _, _, key in extracted_terms(normalized)]
    return any(term for term in terms if term not in _STOP_TERMS)


def select_candidates(
    candidates: list[CaseCandidate],
    limit: int,
    strategy: SelectionStrategy,
) -> list[RetrievalCase]:
    ordered = (
        sorted(candidates, key=lambda item: (-item.score, item.order))
        if strategy == "salience"
        else sorted(candidates, key=lambda item: item.order)
    )
    return [candidate.case for candidate in ordered[:limit]]


def merge_case_candidates_by_query(candidates: list[CaseCandidate]) -> list[CaseCandidate]:
    merged: dict[str, CaseCandidate] = {}
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = normalize_query_text(candidate.case.query).lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            counts[key] = 1
            continue
        counts[key] += 1
        merged_case = merge_retrieval_cases(existing.case, candidate.case, counts[key])
        merged[key] = CaseCandidate(
            case=merged_case,
            score=max(existing.score, candidate.score),
            order=min(existing.order, candidate.order),
        )
    return list(merged.values())


def merge_retrieval_cases(
    left: RetrievalCase,
    right: RetrievalCase,
    merged_count: int,
) -> RetrievalCase:
    metadata = {
        **left.metadata,
        "merged_case_count": merged_count,
    }
    return left.model_copy(
        update={
            "expected_pages": merge_stable_values(left.expected_pages, right.expected_pages),
            "expected_chunk_ids": merge_stable_values(left.expected_chunk_ids, right.expected_chunk_ids),
            "expected_asset_ids": merge_stable_values(left.expected_asset_ids, right.expected_asset_ids),
            "expected_triple_ids": merge_stable_values(left.expected_triple_ids, right.expected_triple_ids),
            "graph_expand": left.graph_expand or right.graph_expand,
            "metadata": metadata,
        }
    )


def merge_stable_values(left: list, right: list) -> list:
    seen = set()
    merged = []
    for value in [*left, *right]:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def dedupe_page_candidates(
    candidates: list[CaseCandidate],
    strategy: SelectionStrategy,
) -> list[CaseCandidate]:
    ordered = (
        sorted(candidates, key=lambda item: (-item.score, item.order))
        if strategy == "salience"
        else sorted(candidates, key=lambda item: item.order)
    )
    seen_pages = set()
    selected = []
    for candidate in ordered:
        page = candidate.case.expected_pages[0] if candidate.case.expected_pages else None
        if page in seen_pages:
            continue
        seen_pages.add(page)
        selected.append(candidate)
    return selected


def with_case_metadata(
    case: RetrievalCase,
    case_source: str,
    query_mode: QueryMode,
    draft: QueryDraft,
) -> RetrievalCase:
    metadata = {
        **case.metadata,
        "case_source": case_source,
        "query_mode": query_mode,
    }
    if draft.terms:
        metadata["query_terms"] = list(draft.terms)
        metadata["selection_score"] = round(draft.score, 6)
    return case.model_copy(update={"metadata": metadata})


def dedupe_cases_by_query(cases: list[RetrievalCase]) -> list[RetrievalCase]:
    selected_by_query: dict[str, RetrievalCase] = {}
    counts: dict[str, int] = {}
    for case in cases:
        key = normalize_query_text(case.query).lower()
        existing = selected_by_query.get(key)
        if existing is None:
            selected_by_query[key] = case
            counts[key] = 1
            continue
        counts[key] += 1
        selected_by_query[key] = merge_retrieval_cases(existing, case, counts[key])
    return list(selected_by_query.values())


def validate_casegen_options(
    query_mode: str,
    selection_strategy: str,
    min_query_terms: int,
    max_query_terms: int,
) -> None:
    if query_mode not in {"snippet", "salient_terms"}:
        raise ValueError("query_mode must be one of: snippet, salient_terms")
    if selection_strategy not in {"document_order", "salience"}:
        raise ValueError("selection_strategy must be one of: document_order, salience")
    if min_query_terms < 1:
        raise ValueError("min_query_terms must be at least 1")
    if max_query_terms < min_query_terms:
        raise ValueError("max_query_terms must be greater than or equal to min_query_terms")


def asset_priority(asset: VisualAsset) -> int:
    return {
        "map": 0,
        "table": 1,
        "chart": 2,
        "figure": 3,
        "page_image": 4,
    }.get(str(asset.kind), 5)
