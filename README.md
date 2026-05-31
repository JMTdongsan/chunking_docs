# chunking_docs

`chunking_docs` is a Python library and CLI for preparing complex PDF documents for RAG systems. It focuses on chunking strategy, multimodal document assets, hybrid retrieval artifacts, and storage-ready outputs rather than on a single target document.

## What It Does

- Profiles PDF pages and detects degraded or missing text layers.
- Creates page-level starter chunks and optional semantic subchunks.
- Accepts an external section map so document-specific structure stays outside the library.
- Renders visual assets for pages that need OCR, VLM summaries, or image embeddings.
- Plans and runs prioritized OCR/VLM jobs for visual-heavy pages.
- Produces dense text, dense image, caption, and BM25 lexical artifacts.
- Supports word, character n-gram, and mixed lexical tokenization for languages where whitespace is weak.
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

VLM responses may be plain text or JSON. When JSON includes `title`, `summary`, `key_points`, `visual_elements`, or `triples`, the runner converts those fields into captions, searchable VLM summaries, and graph triple candidates.

Visual job results include OCR language, backend configuration, VLM prompt name, prompt SHA-256, prompt length, latency, output size, parse status, and triple count. This keeps OCR/VLM experiments reproducible without storing document-specific assumptions in the library.

Summarize visual job runs when comparing OCR/VLM backends:

```bash
chunking-docs summarize-visual-results \
  --results outputs/package/visual_job_results.jsonl \
  --output outputs/package/visual_job_summary.json
```

The summary groups completion counts, backend latency, output size, VLM prompt usage, parse status, and extracted triple counts by operation.

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

The package collection config includes payload index definitions for document IDs, chunk IDs, asset IDs, page fields, and section fields. `qdrant-upsert-package`, `qdrant-search-package`, and Qdrant hybrid evaluation create those indexes when the target Qdrant server supports them.

Without Docker, validate the upsert path with qdrant-client local mode:

```bash
chunking-docs qdrant-upsert-package --package-dir outputs/package --location ':memory:'
```

Validate named-vector retrieval with the same local mode:

```bash
chunking-docs qdrant-search-package "policy corridor" \
  --package-dir outputs/package \
  --location ':memory:' \
  --vector-name text_dense \
  --filter page_start<=12 \
  --filter page_end>=12
```

Run hybrid retrieval over Qdrant vectors, BM25, and optional graph expansion:

```bash
chunking-docs qdrant-hybrid-search "policy corridor" \
  --package-dir outputs/package \
  --location ':memory:' \
  --vector-names text_dense,caption_dense \
  --filter kind=text \
  --fusion-weight bm25=1.2 \
  --fusion-weight qdrant:caption_dense=1.4 \
  --graph-expand
```

When `image_dense` is included, use a text query encoder from the same image-text model family:

```bash
chunking-docs qdrant-hybrid-search "map showing station access" \
  --package-dir outputs/package \
  --vector-names text_dense,caption_dense,image_dense \
  --image-query-backend clip \
  --image-query-model openai/clip-vit-large-patch14
```

## RAG Context

Build a citation-ready context bundle from local hybrid retrieval:

```bash
chunking-docs build-rag-context "station access corridor" \
  --package-dir outputs/package \
  --top-k 5 \
  --graph-expand \
  --neighbor-window 1 \
  --output outputs/package/rag_context.json
```

The bundle contains retrieved chunks, optional neighboring chunks, hierarchical evidence chunks, linked visual assets, and graph triples so downstream RAG services can pass structured context to an answer generator.

Use the Qdrant path when validating production retrieval:

```bash
chunking-docs qdrant-rag-context "station access corridor" \
  --package-dir outputs/package \
  --location ':memory:' \
  --vector-names text_dense,caption_dense \
  --top-k 5 \
  --neighbor-window 1 \
  --output outputs/package/rag_context.qdrant.json
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
chunking-docs audit-package --package-dir outputs/package --require-qdrant-records
chunking-docs eval-chunking --package-dir outputs/package --cases examples/retrieval_cases.jsonl
chunking-docs eval-retrieval examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --top-k 5 \
  --repeat 3 \
  --output outputs/package/retrieval_eval.json
chunking-docs eval-qdrant-retrieval examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --location ':memory:' \
  --vector-names text_dense,caption_dense \
  --top-k 5 \
  --repeat 3 \
  --output outputs/package/qdrant_retrieval_eval.json
chunking-docs eval-qdrant-vector-ablation examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --location ':memory:' \
  --modes text,caption,text_caption,text_caption_graph \
  --top-k 5 \
  --repeat 3 \
  --output outputs/package/qdrant_vector_ablation.json
chunking-docs eval-retrieval-ablation examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --modes dense,bm25,hybrid,graph,hybrid_graph \
  --repeat 3 \
  --output outputs/package/retrieval_ablation.json
```

`audit-package` checks structural completeness, orphan triples, remaining OCR/VLM work, Qdrant vector dimensions, required payload fields, and payload index definitions. `eval-chunking` reports page coverage, chunk size distribution, section coverage, visual asset linkage, visual annotation coverage, retrieval recall@k, MRR, failed queries, and an aggregate quality score. `eval-retrieval` also records per-query latency samples plus mean and p95 latency when `--repeat` is greater than one. Retrieval reports include target-specific recall and MRR for page, chunk, visual asset, and graph triple expectations. `eval-qdrant-retrieval` runs the same benchmark cases through Qdrant named vectors, BM25, and optional graph expansion so the production retrieval path can be validated. `eval-qdrant-vector-ablation` compares Qdrant text, visual caption, optional image, and graph-expanded modes on the same cases. `eval-retrieval-ablation` compares dense-only, BM25-only, graph-only, hybrid, and graph-expanded hybrid retrieval so the effect and runtime cost of each retrieval signal is visible. Retrieval cases are JSONL:

Qdrant vector ablation modes include `text`, `caption`, `image`, `text_caption`, `text_image`, `caption_image`, `all`, `text_caption_graph`, and `all_graph`. Image modes require an `image_dense` record file and a compatible image-query encoder.

Hybrid retrieval commands accept repeatable `--fusion-weight source=weight` values. Sources can be exact names such as `qdrant:caption_dense` or families such as `qdrant`, `bm25`, `dense`, and `graph`.

```jsonl
{"query":"policy corridor near river","expected_pages":[12],"graph_expand":true}
{"query":"capital investment table","expected_chunk_ids":["chunk-id"]}
{"query":"map legend for station access","expected_asset_ids":["asset-id"]}
{"query":"district connects to corridor","expected_triple_ids":["triple-id"],"graph_expand":true}
```

For portfolio or production use, maintain benchmark cases for each document family and compare chunking strategies before changing defaults.

## Lexical Tokenization

BM25 uses the `mixed` tokenizer by default. It combines word tokens with CJK character n-grams, which helps retrieve compound terms that may appear without whitespace in PDF text or OCR output.

```bash
chunking-docs search-local "urban renewal plan" \
  --package-dir outputs/package \
  --lexical-tokenizer mixed \
  --ngram-min 2 \
  --ngram-max 4
```

Available tokenizer strategies:

- `word`: regex word tokens
- `char_ngram`: character n-grams
- `mixed`: word tokens plus character n-grams

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

chunking-docs build-chunk-strategy \
  --package-dir outputs/package \
  --strategy hierarchical \
  --output outputs/package/chunks.hierarchical.jsonl
```

Compare candidates with the same retrieval cases:

```bash
chunking-docs compare-chunking \
  --package-dir outputs/package \
  --candidate baseline=outputs/package/chunks.jsonl \
  --candidate semantic=outputs/package/chunks.semantic.jsonl \
  --candidate multimodal=outputs/package/chunks.multimodal.jsonl \
  --candidate hierarchical=outputs/package/chunks.hierarchical.jsonl \
  --collapse-hierarchical \
  --retrieval-repeat 3 \
  --cases examples/retrieval_cases.jsonl
```

The `multimodal` strategy keeps semantic text chunks and adds visual asset text chunks from captions, OCR, and VLM summaries. The `hierarchical` strategy emits coarse parent chunks plus fine child chunks with shared visual context, which supports experiments where broad queries should find a page or section while precise queries should retrieve a smaller evidence span. `--collapse-hierarchical` reports the parent as the final hit while preserving matched child chunks as evidence. Comparison output includes recall@k, MRR, latency, failed queries, chunk size issues, and the best candidate by quality and retrieval behavior.

Run a parameter sweep when choosing defaults:

```bash
chunking-docs sweep-chunking \
  --package-dir outputs/package \
  --strategies semantic,multimodal,hierarchical \
  --max-chars 1000 \
  --max-chars 1600 \
  --overlap-chars 100 \
  --overlap-chars 180 \
  --parent-max-chars 700 \
  --parent-max-chars 900 \
  --visual-context-chars 500 \
  --visual-context-chars 700 \
  --collapse-hierarchical \
  --retrieval-repeat 3 \
  --cases examples/retrieval_cases.jsonl \
  --output outputs/package/chunking_sweep.json
```

The sweep writes candidate chunk files under `outputs/package/chunking_sweep/` and ranks them with the same quality, recall@k, MRR, latency, and failed-query metrics used by `compare-chunking`.

Write a reproducible experiment report for a package:

```bash
chunking-docs write-experiment-report \
  --package-dir outputs/package \
  --candidate baseline=outputs/package/chunks.jsonl \
  --candidate semantic=outputs/package/chunks.semantic.jsonl \
  --candidate multimodal=outputs/package/chunks.multimodal.jsonl \
  --candidate hierarchical=outputs/package/chunks.hierarchical.jsonl \
  --collapse-hierarchical \
  --retrieval-repeat 3 \
  --cases examples/retrieval_cases.jsonl \
  --output outputs/package/experiment_report.json
```

The report records package artifact checksums, JSONL record counts, BM25 tokenizer settings, Qdrant named-vector configuration, chunking quality metrics, retrieval recall@k, MRR, latency, failed queries, and the best candidate by retrieval behavior. This makes chunking changes reviewable and repeatable before new defaults are adopted.

## Development Checks

```bash
ruff check src tests
pytest
```

## Design Notes

- [Architecture](docs/architecture.md)
