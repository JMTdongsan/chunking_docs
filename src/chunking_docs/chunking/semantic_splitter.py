from __future__ import annotations

import hashlib
import re

from chunking_docs.models import ChunkKind, DocumentChunk

BOUNDARY_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s+|제\d+[장절]\b|\d+[.)]\s+|[-•]\s+|\[[A-Z]+ page \d+)")


def semantic_subchunks(
    chunks: list[DocumentChunk],
    max_chars: int = 1600,
    overlap_chars: int = 180,
    min_chars: int = 180,
) -> list[DocumentChunk]:
    results: list[DocumentChunk] = []
    for chunk in chunks:
        parts = split_text(chunk.text, max_chars=max_chars, overlap_chars=overlap_chars)
        if len(parts) == 1 and len(parts[0]) < min_chars:
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
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def join_blocks(left: str, right: str) -> str:
    if not left:
        return right.strip()
    return left.rstrip() + "\n\n" + right.strip()


def overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    return text[-overlap_chars:].strip()


def subchunk_id(parent_chunk_id: str, index: int) -> str:
    raw = f"{parent_chunk_id}:semantic:{index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]
