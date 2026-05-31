# chunking_docs

`chunking_docs` is a Python library and CLI for preparing complex PDF documents for RAG systems. It focuses on chunking strategy, multimodal document assets, hybrid retrieval artifacts, and storage-ready outputs rather than on a single target document.

## What It Does

- Profiles PDF pages and detects degraded or missing text layers.
- Creates page-level starter chunks and optional semantic subchunks.
- Accepts an external section map so document-specific structure stays outside the library.
- Renders visual assets for pages that need OCR, VLM summaries, or image embeddings.
- Plans and runs prioritized OCR/VLM jobs for visual-heavy pages.
- Produces dense text, dense image, caption, and BM25 lexical artifacts.
- Builds graph triple candidates from section metadata and visual annotations.
- Exports Qdrant upsert records and PostgreSQL-ready normalized rows.
- Evaluates chunking quality and retrieval hit rate with reusable benchmark cases.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional integrations:

```bash
pip install -e ".[qdrant]"              # Qdrant client
pip install -e ".[postgres]"            # PostgreSQL writer
pip install -e ".[embeddings,vision]"   # SentenceTransformer, CLIP, VLM backends
```

## Basic Pipeline

```bash
chunking-docs download "https://example.com/document.pdf" data/raw/document.pdf
chunking-docs profile data/raw/document.pdf --output-dir outputs/profile
chunking-docs package data/raw/document.pdf --output-dir outputs/package
```

The package directory contains:

- `manifest.json`: document and processing metadata
- `pages.jsonl`: page profiles and text quality signals
- `chunks.jsonl`: chunk records
- `assets.jsonl`: rendered visual asset records
- `triples.jsonl`: graph triple candidates
- `bm25_tokens.json`: lexical search token manifest
- `qdrant_*_records.jsonl`: Qdrant upsert records
- `qdrant_collection.json`: named-vector collection configuration

## Document Structure

Document-specific structure should be supplied as data, not hardcoded in the library.

```jsonl
{"page_start":1,"page_end":5,"chapter":"Overview"}
{"page_start":6,"page_end":20,"section":{"chapter":"Strategy","section":"Mobility"}}
```

Use it with:

```bash
chunking-docs package data/raw/document.pdf \
  --output-dir outputs/package \
  --section-map examples/section_map.jsonl
```

## Visual Processing

For documents with maps, diagrams, charts, scans, or broken text layers, plan OCR/VLM work first:

```bash
chunking-docs plan-visual-jobs --package-dir outputs/package
```

Run a small batch:

```bash
chunking-docs run-visual-jobs \
  --package-dir outputs/package \
  --jobs outputs/package/visual_jobs.jsonl \
  --ocr tesseract \
  --vlm hf \
  --vlm-model <local-or-huggingface-vlm-model> \
  --limit 10 \
  --apply
```

The command writes `visual_annotations.jsonl` and `visual_job_results.jsonl`. With `--apply`, annotations are merged into `assets.jsonl`, `chunks.jsonl`, `triples.jsonl`, BM25, and Qdrant record files.

## Embeddings

The default package command writes deterministic hashing vectors so the pipeline can be tested without downloading models. Rebuild model-backed records with:

```bash
chunking-docs embed-package \
  --package-dir outputs/package \
  --text-backend sentence-transformers \
  --text-model BAAI/bge-m3 \
  --caption-backend same-as-text \
  --image-backend clip \
  --image-model openai/clip-vit-large-patch14 \
  --device cuda
```

This regenerates Qdrant text, caption, and image records using the selected model dimensions.

## Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d
chunking-docs qdrant-upsert-package --package-dir outputs/package
```

Without Docker, validate the upsert path with qdrant-client local mode:

```bash
chunking-docs qdrant-upsert-package --package-dir outputs/package --location ':memory:'
```

## PostgreSQL

PostgreSQL is intended for source metadata, page profiles, chunks, assets, and graph triples. Vector search is handled by Qdrant by default.

```bash
chunking-docs postgres-rows --package-dir outputs/package
chunking-docs postgres-upsert "postgresql://user:password@localhost:5432/chunking_docs" \
  --package-dir outputs/package
```

## Evaluation

Correct execution is not enough for a chunking library. Use evaluation commands to check whether the chunking strategy is useful for retrieval.

```bash
chunking-docs audit-package --package-dir outputs/package
chunking-docs eval-chunking --package-dir outputs/package --cases examples/retrieval_cases.jsonl
chunking-docs eval-retrieval examples/retrieval_cases.jsonl --package-dir outputs/package --top-k 5
```

`eval-chunking` reports page coverage, chunk size distribution, section coverage, visual asset linkage, visual annotation coverage, retrieval recall@k, MRR, failed queries, and an aggregate quality score. Retrieval cases are JSONL:

```jsonl
{"query":"policy corridor near river","expected_pages":[12],"graph_expand":true}
{"query":"capital investment table","expected_chunk_ids":["chunk-id"]}
```

For portfolio or production use, maintain benchmark cases for each document family and compare chunking strategies before changing defaults.

## Chunking Strategy Experiments

Generate alternate chunk files without overwriting the package baseline:

```bash
chunking-docs build-chunk-strategy \
  --package-dir outputs/package \
  --strategy semantic \
  --output outputs/package/chunks.semantic.jsonl

chunking-docs build-chunk-strategy \
  --package-dir outputs/package \
  --strategy multimodal \
  --output outputs/package/chunks.multimodal.jsonl
```

Compare candidates with the same retrieval cases:

```bash
chunking-docs compare-chunking \
  --package-dir outputs/package \
  --candidate baseline=outputs/package/chunks.jsonl \
  --candidate semantic=outputs/package/chunks.semantic.jsonl \
  --candidate multimodal=outputs/package/chunks.multimodal.jsonl \
  --cases examples/retrieval_cases.jsonl
```

The `multimodal` strategy keeps semantic text chunks and adds visual asset text chunks from captions, OCR, and VLM summaries. This makes maps, tables, charts, and figures retrievable even when the PDF text layer is weak. Comparison output includes recall@k, MRR, failed queries, chunk size issues, and the best candidate by quality and retrieval behavior.

## Development Checks

```bash
ruff check src tests
pytest
```

## Design Notes

- [Architecture](docs/architecture.md)
