from __future__ import annotations

import hashlib
import re

from chunking_docs.models import ChunkKind, DocumentChunk

BOUNDARY_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s+|제\d+[장절]\b|\d+[.)]\s+|[-•]\s+|\[[A-Z]+ page \d+)")
SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\"')\]\}]*")
WHITESPACE_RE = re.compile(r"\s+")


def semantic_subchunks(
    chunks: list[DocumentChunk],
    max_chars: int = 1600,
    overlap_chars: int = 180,
    min_chars: int = 180,
) -> list[DocumentChunk]:
    results: list[DocumentChunk] = []
    for chunk in chunks:
        parts = split_text(chunk.text, max_chars=max_chars, overlap_chars=overlap_chars)
        if len(parts) <= 1:
            results.append(chunk)
            continue

        for index, part in enumerate(parts):
            if len(part.strip()) < min_chars and results:
                previous = results[-1]
                if previous.metadata.get("parent_chunk_id") == chunk.chunk_id:
                    merged_text = previous.text.rstrip() + "\n\n" + part.strip()
                    results[-1] = previous.model_copy(update={"text": merged_text})
                    continue

            metadata = {
                **chunk.metadata,
                "parent_chunk_id": chunk.chunk_id,
                "subchunk_index": index,
                "subchunk_count": len(parts),
                "chunking_strategy": "semantic_subchunks",
            }
            results.append(
                chunk.model_copy(
                    update={
                        "chunk_id": subchunk_id(chunk.chunk_id, index),
                        "kind": ChunkKind.TEXT if chunk.kind == ChunkKind.PAGE_SUMMARY else chunk.kind,
                        "text": part,
                        "metadata": metadata,
                    }
                )
            )
    return results


def split_text(text: str, max_chars: int = 1600, overlap_chars: int = 180) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(normalized) <= max_chars:
        return [normalized] if normalized else []

    blocks = paragraph_blocks(normalized)
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = join_blocks(current, block)
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = overlap_tail(current, overlap_chars)
        if len(block) > max_chars:
            chunks.extend(hard_split(block, max_chars=max_chars, overlap_chars=overlap_chars))
            current = ""
        else:
            current = join_blocks(current, block)
    if current:
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def paragraph_blocks(text: str) -> list[str]:
    rough_blocks = re.split(r"\n\s*\n", text)
    blocks: list[str] = []
    for rough in rough_blocks:
        rough = rough.strip()
        if not rough:
            continue
        starts = [match.start() for match in BOUNDARY_RE.finditer(rough)]
        if len(starts) <= 1:
            blocks.append(rough)
            continue
        starts.append(len(rough))
        for start, end in zip(starts, starts[1:]):
            block = rough[start:end].strip()
            if block:
                blocks.append(block)
    return blocks


def hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks = []
    effective_overlap = max(0, min(overlap_chars, max_chars // 2))
    start = 0
    while start < len(text):
        limit = min(start + max_chars, len(text))
        end = best_split_end(text, start=start, limit=limit)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        next_start = overlap_start(text, end=end, overlap_chars=effective_overlap)
        start = next_start if next_start > start else end
    return chunks


def best_split_end(text: str, start: int, limit: int) -> int:
    if limit >= len(text):
        return len(text)
    window_start = start + max(1, int((limit - start) * 0.55))
    for pattern in (SENTENCE_END_RE, "\n", WHITESPACE_RE):
        end = last_boundary_end(text, pattern, window_start=window_start, limit=limit)
        if end is not None and end > start:
            return end
    return limit


def last_boundary_end(
    text: str,
    pattern: str | re.Pattern[str],
    window_start: int,
    limit: int,
) -> int | None:
    if isinstance(pattern, str):
        position = text.rfind(pattern, window_start, limit)
        return position + len(pattern) if position >= 0 else None

    matches = [match.end() for match in pattern.finditer(text, window_start, limit)]
    return matches[-1] if matches else None


def overlap_start(text: str, end: int, overlap_chars: int) -> int:
    if overlap_chars <= 0:
        return end
    lower = max(0, end - overlap_chars)
    if lower == 0:
        return 0
    candidates = []
    previous_window_start = max(0, lower - max(40, overlap_chars))
    previous = last_boundary_end(
        text,
        WHITESPACE_RE,
        window_start=previous_window_start,
        limit=lower,
    )
    if previous is not None and previous > 0:
        candidates.append(previous)
    next_match = WHITESPACE_RE.search(text, lower, end)
    if next_match is not None and next_match.end() < end:
        candidates.append(next_match.end())
    if not candidates:
        return lower
    return min(candidates, key=lambda candidate: abs(candidate - lower))


def join_blocks(left: str, right: str) -> str:
    if not left:
        return right.strip()
    return left.rstrip() + "\n\n" + right.strip()


def overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    start = overlap_start(text, end=len(text), overlap_chars=overlap_chars)
    return text[start:].strip()


def subchunk_id(parent_chunk_id: str, index: int) -> str:
    raw = f"{parent_chunk_id}:semantic:{index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]
