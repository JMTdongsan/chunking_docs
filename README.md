# chunking_docs

`chunking_docs` is a Python library and CLI for preparing complex PDF documents for RAG systems. It focuses on chunking strategy, multimodal document assets, hybrid retrieval artifacts, and storage-ready outputs rather than on a single target document.

## What It Does

- Profiles PDF pages and detects degraded or missing text layers.
- Creates page-level starter chunks and optional semantic subchunks.
- Accepts an external section map so document-specific structure stays outside the library.
- Renders visual assets for pages that need OCR, VLM summaries, or image embeddings.
- Extracts structured PDF tables as table chunks and table visual assets.
- Plans and runs prioritized OCR/VLM jobs for visual-heavy pages.
- Produces dense text, dense image, caption, and BM25 lexical artifacts over chunk text plus linked visual text.
- Supports word, character n-gram, and mixed lexical tokenization for languages where whitespace is weak.
- Builds graph triple candidates from section metadata and visual annotations.
- Audits and normalizes graph triples before graph expansion or export.
- Exports Qdrant upsert records with visual asset link payloads and PostgreSQL-ready normalized rows.
- Evaluates chunking quality and retrieval hit rate with reusable benchmark cases.
- Audits public repository files for forbidden text and accidental generated artifacts before publishing.

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
pip install -e ".[ocr]"                 # PaddleOCR backend and engine runtime
```

Check local runtime capabilities before GPU-backed OCR, VLM, embedding, or storage work:

```bash
chunking-docs doctor \
  --require-gpu \
  --require-qdrant \
  --require-embeddings \
  --require-vision \
  --require-ocr \
  --vlm-profile qwen2_5_vl_7b \
  --vlm-memory-margin-ratio 0.1
```

`--vlm-profile` compares the selected Hugging Face VLM profile with visible GPU memory before a long local run. It also records Torch CUDA device names, compute capability, and bfloat16 support when Torch is installed, and `--vlm-memory-margin-ratio` emits a warning when the selected profile only barely fits. Current 7B/8B profiles are marked for 24GB-class GPUs, while the compact Phi-3.5 Vision profile is marked for 12GB-class GPUs.

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
- `embedding_manifest.json`: vector record files, dimensions, counts, and checksums
- `qdrant_*_records.jsonl`: Qdrant upsert records
- `qdrant_collection.json`: named-vector collection configuration

Summarize the package characteristics that should drive chunking, OCR/VLM, graph, and retrieval choices:

```bash
chunking-docs characterize-package \
  --package-dir outputs/package \
  --output outputs/package/document_characteristics.json
```

The report includes observations and processing recommendations for visual annotation, multimodal embeddings, graph signals, and retrieval benchmark coverage. When VLM object metadata is present, it reports object and bbox counts and recommends generating and auditing `visual_object_probe` retrieval cases with visual-only, target-diversity, concentration, and query-strength gates so object detections are evaluated separately from aggregate retrieval scores.

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

Prioritize a GPU/VLM batch for high-value visual evidence:

```bash
chunking-docs plan-visual-jobs \
  --package-dir outputs/package \
  --kind map \
  --kind table \
  --limit 50 \
  --output outputs/package/visual_jobs.priority.jsonl
```

For dense visual pages, create overlapping page tiles before planning jobs:

```bash
chunking-docs build-tile-assets \
  --package-dir outputs/package \
  --pages 10-20 \
  --rows 2 \
  --cols 2 \
  --overlap-ratio 0.08
```

Run a small batch:

```bash
chunking-docs run-visual-jobs \
  --package-dir outputs/package \
  --jobs outputs/package/visual_jobs.jsonl \
  --ocr paddleocr \
  --ocr-model-lang korean \
  --ocr-device cpu \
  --ocr-min-confidence 0.3 \
  --vlm hf \
  --vlm-profile qwen2_5_vl_7b \
  --vlm-device-map auto \
  --vlm-torch-dtype bfloat16 \
  --vlm-max-new-tokens 768 \
  --limit 10 \
  --apply
```

The command writes `visual_annotations.jsonl` and `visual_job_results.jsonl`. With `--apply`, annotations are merged into `assets.jsonl`, `chunks.jsonl`, `triples.jsonl`, and BM25. Chunk updates follow visual asset links from both `asset_ids` and `asset:` source refs. Run `embed-package` after applying annotations to refresh dense, caption, image, and Qdrant record artifacts with the intended embedding models.
PaddleOCR uses a CPU device by default because the standard `paddlepaddle` wheel is CPU-only in many environments. Use `--ocr-device gpu:0` only after `chunking-docs doctor --require-gpu --require-ocr` confirms Paddle CUDA support. `--ocr-enable-mkldnn` can improve CPU throughput after a local smoke test confirms the Paddle runtime is stable with oneDNN enabled.

Use `--vlm-profile` for reproducible Hugging Face VLM experiments. Profiles provide the model id, loader family, dtype, and default generation length for common local VLM families such as Qwen2.5-VL, Qwen2-VL, LLaVA-NeXT, Idefics2, and Phi-3.5 Vision. The `vision` extra installs Transformers, Accelerate, PyTorch, and Torchvision; run `chunking-docs doctor --require-vision --vlm-profile <profile> --vlm-memory-margin-ratio 0.1` before long VLM batches to check memory fit, safety margin, CUDA visibility, and bfloat16 compatibility. Override any profile field with `--vlm-model`, `--vlm-model-class`, `--vlm-device-map`, `--vlm-torch-dtype`, `--vlm-max-new-tokens`, or `--vlm-attn-implementation`.

Create a reusable command plan when comparing several VLM profiles on the same job set:

```bash
chunking-docs plan-vlm-experiments \
  --package-dir outputs/package \
  --jobs outputs/package/visual_jobs.priority.jsonl \
  --profiles qwen2_5_vl_7b,qwen2_vl_7b,llava_next_7b \
  --limit 10 \
  --output outputs/package/vlm_experiment_plan.json
```

Default VLM prompts request a single JSON object with `title`, `summary`, `key_points`, `visual_elements`, `objects`, `entities`, and `triples`. When those fields are present, the runner converts them into captions, searchable VLM summaries, normalized object detections, and graph triple candidates. Object detections can carry attributes, descriptions, locations, bbox coordinates, confidence, and source-field provenance. Entity, visual element, and object fields are also lifted into derived triple candidates and included in visual asset lexical/caption text so useful VLM detections remain searchable even when the model does not emit explicit relationships.
Triples generated from visual annotations include provenance qualifiers such as asset ID, page number, asset kind, annotation source, visual job ID, prompt name, prompt schema version, and prompt hash when available.

Visual job results include OCR language, backend configuration, VLM prompt name, prompt SHA-256, prompt length, latency, output size, parse status, entity count, visual element count, object count, object bbox count, explicit triple count, derived triple count, and total triple count. `--ocr-model-lang`, `--ocr-device`, `--ocr-engine`, `--ocr-min-confidence`, `--ocr-enable-mkldnn`, `--vlm-device-map`, `--vlm-torch-dtype`, `--vlm-max-new-tokens`, and optional `--vlm-attn-implementation` are recorded in backend configuration. This keeps OCR/VLM experiments reproducible without storing document-specific assumptions in the library.

Summarize visual job runs when comparing OCR/VLM backends:

```bash
chunking-docs summarize-visual-results \
  --results outputs/package/visual_job_results.jsonl \
  --output outputs/package/visual_job_summary.json
```

The summary groups completion counts, backend latency, output size, VLM prompt usage, parse status, object detection counts, and extracted triple counts by operation. Compare separate runs when testing multiple OCR/VLM backends on the same job set:

```bash
chunking-docs compare-visual-runs \
  --run vlm_a=outputs/package/visual_job_results.vlm_a.jsonl \
  --run vlm_b=outputs/package/visual_job_results.vlm_b.jsonl \
  --output outputs/package/visual_run_comparison.json \
  --require-same-jobs
```

The comparison ranks OCR/VLM runs by completion rate, annotation coverage, OCR text coverage, VLM summary coverage, JSON parse rate, object coverage, triple density, and total latency. It also reports shared and missing visual job IDs, and `--require-same-jobs` fails the command when runs were produced from different job sets.

Gate a visual run before applying annotations to retrieval artifacts:

```bash
chunking-docs gate-visual-results \
  --results outputs/package/visual_job_results.jsonl \
  --min-completion-rate 0.95 \
  --min-ocr-text-coverage 0.8 \
  --min-vlm-summary-coverage 0.9 \
  --min-vlm-json-parse-rate 0.9 \
  --min-vlm-object-coverage 0.5 \
  --min-object-bbox-coverage 0.5 \
  --max-failed-count 0 \
  --output outputs/package/visual_quality.json
```

After multiple OCR/VLM batches have been applied, gate the final package state directly from `assets.jsonl`:

```bash
chunking-docs summarize-visual-assets \
  --package-dir outputs/package \
  --output outputs/package/visual_asset_summary.json

chunking-docs gate-visual-assets \
  --package-dir outputs/package \
  --min-ocr-text-coverage 0.8 \
  --min-vlm-summary-coverage 0.95 \
  --min-vlm-json-parse-rate 0.95 \
  --min-vlm-object-coverage 0.5 \
  --min-object-bbox-coverage 0.5 \
  --output outputs/package/visual_asset_quality.json
```

This is useful when OCR/VLM work was split across several result files but the retrieval package should be judged by the annotations that are actually present in `assets.jsonl`.

Compare two package directories after applying annotations, changing chunking strategy, or rebuilding embeddings:

```bash
chunking-docs compare-packages \
  outputs/package.baseline \
  outputs/package \
  --output outputs/package/package_delta.json
```

The report shows before/after counts, added and removed IDs, changed chunk/asset/triple IDs, Qdrant record count deltas, and observations such as newly added visual annotations or graph triples.

## Structured Tables

The package command extracts detected PDF tables by default and stores each table as both a `table` chunk and a `table` visual asset. Table text is serialized as Markdown so dense embeddings, BM25, caption vectors, and downstream RAG context can all use the same structured content.
Detected tables with noisy encoded text are skipped so broken PDF text layers do not create misleading table chunks. Use OCR/VLM visual jobs for those pages instead.

Run table extraction on an existing package:

```bash
chunking-docs extract-tables \
  --package-dir outputs/package \
  --pdf data/raw/document.pdf
```

Use `--no-extract-tables` on `package` when table extraction should be handled as a separate experiment.

## Graph Triple Quality

Graph triples can come from section metadata, manual annotations, OCR-adjacent review, or VLM JSON. Normalize and audit them before using graph expansion or exporting a graph view:

```bash
chunking-docs audit-graph-triples \
  --package-dir outputs/package \
  --output outputs/package/graph_triple_quality.json

chunking-docs normalize-graph-triples \
  --package-dir outputs/package \
  --output outputs/package/triples.normalized.jsonl \
  --export-graph
```

Normalization collapses whitespace, canonicalizes predicate names, recomputes stable triple IDs, and can remove semantic duplicates within the same chunk. The audit report counts duplicates, triples that would change under normalization, orphan chunk references, empty fields, invalid confidence values, and normalized predicate frequencies.

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

This regenerates Qdrant text, caption, and image records using the selected model dimensions. It also writes `embedding_manifest.json` so vector files, record counts, dimensions, checksums, backend names, model IDs, devices, and batch sizes can be compared across embedding runs.

## Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d
chunking-docs qdrant-check-collection \
  --package-dir outputs/package \
  --output outputs/package/qdrant_collection_contract.json \
  --allow-missing
chunking-docs qdrant-upsert-package --package-dir outputs/package
```

The package collection config includes payload index definitions for document IDs, chunk IDs, asset IDs, chunking strategy fields, hierarchy links, standalone visual chunk flags, page fields, and section fields. `qdrant-check-collection` validates an existing Qdrant collection against the package named-vector dimensions and payload indexes before upsert. `qdrant-upsert-package`, `qdrant-search-package`, and Qdrant hybrid evaluation create those indexes when the target Qdrant server supports them.

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
  --reranker lexical \
  --rerank-top-k 20 \
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
When Qdrant visual vectors retrieve a linked asset, chunk metadata records compact retrieval references such as asset ID, page, kind, and document ID without copying full payload text into metadata. The referenced asset is also included in the bundle assets so its bounded caption, OCR, and VLM text remain available for answer generation. Graph triples are selected by matched chunk IDs and by visual asset provenance, so VLM-derived triples can follow the asset that triggered retrieval.
Chunk text and visual asset text can be bounded independently with `--max-chars-per-chunk` and `--max-chars-per-asset-text`. Asset metadata records original and context character counts plus truncated fields, which keeps OCR/VLM evidence usable without silently overflowing the answer context.

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

Qdrant query commands validate query encoder dimensions against `qdrant_collection.json` before searching. If a package was embedded with model-backed vectors, use the same text query model at retrieval time:

```bash
chunking-docs qdrant-rag-context "station access corridor" \
  --package-dir outputs/package \
  --location ':memory:' \
  --vector-names text_dense \
  --text-backend sentence-transformers \
  --text-model BAAI/bge-m3 \
  --device cpu \
  --output outputs/package/rag_context.qdrant.json
```

Qdrant search, evaluation, ablation, and RAG context outputs include `query_encoder_details` so each selected vector records the query backend, model, and embedding dimension used for the run.

## PostgreSQL

PostgreSQL is intended for source metadata, page profiles, chunks, BM25 token artifacts, visual asset links, assets, graph triples, and embedding artifact provenance. Vector search is handled by Qdrant by default, while PostgreSQL stores vector file names, dimensions, counts, checksums, backend/model metadata, and collection names so embedding runs remain auditable.

```bash
chunking-docs postgres-schema --output outputs/package/postgres_schema.sql
chunking-docs postgres-check-schema \
  "postgresql://user:password@localhost:5432/chunking_docs" \
  --output outputs/package/postgres_schema_contract.json \
  --apply-schema
chunking-docs postgres-rows --package-dir outputs/package
chunking-docs postgres-upsert "postgresql://user:password@localhost:5432/chunking_docs" \
  --package-dir outputs/package
```

`postgres-schema` writes the SQL contract without opening a database connection. `postgres-check-schema` validates required tables, columns, column types, indexes, and the pgvector extension before metadata rows are upserted. BM25 token rows from `bm25_tokens.json` are stored in `chunk_lexical_tokens` with the tokenizer config and token array, so lexical experiments can be audited or migrated to a PostgreSQL-backed search service later. Chunk-to-asset links are stored in a normalized `chunk_asset_links` table and also preserved in chunk metadata for auditability. Asset-backed graph triples are remapped to an available chunk before PostgreSQL row export while retaining the original chunk ID in qualifiers. Use `--apply-schema` when bootstrapping a new database; omit it when checking an existing schema for drift.

## Ingestion Readiness

Before loading a package into Qdrant, PostgreSQL, or a RAG service, run a combined readiness check:

```bash
chunking-docs ingestion-readiness \
  --package-dir outputs/package \
  --require-visual-annotations \
  --require-visual-quality \
  --min-vlm-summary-coverage 0.95 \
  --min-vlm-json-parse-rate 0.95 \
  --min-visual-text-coverage-ratio 0.8 \
  --min-visual-text-part-coverage-ratio 0.8 \
  --required-vector text_dense \
  --required-vector caption_dense \
  --required-vector image_dense \
  --visual-run-comparison outputs/package/visual_run_comparison.json \
  --require-visual-run-same-jobs \
  --visual-run-best-by-quality qwen2_5_vl_7b \
  --retrieval-cases examples/retrieval_cases.jsonl \
  --min-retrieval-case-group-count case_source:visual_object_probe=4 \
  --min-retrieval-distinct-asset-targets 4 \
  --min-retrieval-case-group-distinct-targets case_source:visual_object_probe:asset=4 \
  --max-retrieval-asset-cases-per-target 3 \
  --min-retrieval-query-terms-per-case 3 \
  --require-visual-only-object-probes \
  --retrieval-evaluation outputs/package/retrieval_eval.json \
  --max-mean-target-rank 3 \
  --max-p95-target-rank 5 \
  --min-retrieval-target-type-coverage asset=0.9 \
  --min-retrieval-target-type-coverage triple=0.9 \
  --min-retrieval-source-family-target-coverage lexical=0.75 \
  --chunking-comparison outputs/package/chunking_comparison.json \
  --baseline-chunking-candidate baseline \
  --min-chunking-visual-text-coverage-ratio 0.8 \
  --min-chunking-visual-text-part-coverage-ratio 0.8 \
  --max-chunking-mean-target-rank 3 \
  --max-chunking-pairwise-mean-target-rank-delta 0 \
  --min-chunking-target-type-coverage asset=0.9 \
  --min-chunking-target-type-coverage triple=0.9 \
  --min-chunking-source-family-target-coverage lexical=0.75 \
  --retrieval-ablation outputs/package/retrieval_ablation.json \
  --retrieval-ablation-mode bm25_visual \
  --retrieval-ablation-baseline-mode bm25_text \
  --min-retrieval-ablation-recall-lift 0.2 \
  --min-retrieval-ablation-pairwise-win-rate 0.55 \
  --min-retrieval-ablation-pairwise-target-coverage-lift 0.02 \
  --max-retrieval-ablation-pairwise-mean-target-rank-delta 0 \
  --max-retrieval-ablation-mean-target-rank 3 \
  --min-retrieval-ablation-target-type-coverage asset=0.9 \
  --min-retrieval-ablation-case-group-target-coverage case_source:visual_lexical_probe=0.7 \
  --qdrant-vector-ablation outputs/package/qdrant_vector_ablation.json \
  --qdrant-vector-mode text_caption \
  --qdrant-vector-baseline-mode text \
  --min-qdrant-vector-pairwise-win-rate 0.55 \
  --min-qdrant-vector-pairwise-target-coverage-lift 0.02 \
  --max-qdrant-vector-pairwise-mean-target-rank-delta 0 \
  --min-qdrant-vector-recall-at-k 0.8 \
  --min-qdrant-vector-target-coverage-at-k 0.75 \
  --max-qdrant-vector-mean-target-rank 3 \
  --min-qdrant-vector-target-type-coverage asset=0.9 \
  --min-qdrant-vector-source-target-coverage qdrant:caption_dense=0.75 \
  --min-qdrant-vector-source-family-target-coverage visual=0.75 \
  --min-qdrant-vector-case-group-target-coverage case_source:visual_object_probe=0.7 \
  --max-qdrant-vector-failed-queries 0 \
  --output outputs/package/ingestion_readiness.json
```

The report combines package audit results, BM25 token manifest validation, required embedding artifacts, required vector-family checks, Qdrant record checks, PostgreSQL row conversion, retrieval case audit, VLM run comparison checks, chunking comparison gates, selected retrieval and Qdrant vector ablation gates, and optional visual or retrieval gates. BM25 validation recomputes asset-enriched lexical text from chunks plus linked captions, OCR text, VLM summaries, and structured VLM metadata, then checks that `bm25_tokens.json` is complete and current before ingestion. `--min-visual-text-coverage-ratio` checks whether linked visual assets have text represented in package chunks, while `--min-visual-text-part-coverage-ratio` checks individual caption, OCR, VLM, object, entity, and visual-element text parts so object metadata is not hidden by a single covered caption. `--required-vector` verifies that selected vector families are present in `qdrant_collection.json`, represented in `embedding_manifest.json`, have non-empty record files, and use consistent dimensions. Retrieval case audit can require metadata group counts such as `case_source:visual_object_probe=4`, distinct target coverage such as `--min-retrieval-distinct-asset-targets 4`, case-group target diversity such as `--min-retrieval-case-group-distinct-targets case_source:visual_object_probe:asset=4`, concentration limits such as `--max-retrieval-asset-cases-per-target 3`, query strength such as `--min-retrieval-query-terms-per-case 3`, and `--require-visual-only-object-probes` so VLM object-detection cases isolate metadata terms instead of disappearing or concentrating on one target. Chunking, retrieval, retrieval ablation, and Qdrant vector gates can all enforce target-type coverage for page, chunk, visual asset, or graph triple expectations, source-family coverage for dense, lexical, graph, or visual evidence, and metadata case-group coverage such as `case_source:visual_object_probe`. Retrieval, retrieval ablation, and Qdrant vector gates can also enforce exact source coverage, for example `--min-retrieval-source-target-coverage qdrant:caption_dense=0.75`, `--min-retrieval-ablation-source-target-coverage bm25=0.75`, or `--min-qdrant-vector-source-target-coverage qdrant:image_dense=0.5`, so a combined family cannot pass unless the intended caption, image, or lexical source contributes. Visual run comparison checks can require the same visual job IDs across candidate VLM runs and confirm the intended profile won by quality or triple density. When `--require-visual-quality` is used without `--visual-results`, readiness evaluates the final OCR/VLM annotations currently stored in `assets.jsonl`.

## Evaluation

Correct execution is not enough for a chunking library. Use evaluation commands to check whether the chunking strategy is useful for retrieval.

```bash
chunking-docs audit-publication . \
  --forbidden-pattern "internal project codename" \
  --forbidden-pattern "private source file name" \
  --output outputs/publication_audit.json
chunking-docs audit-package --package-dir outputs/package
chunking-docs audit-package --package-dir outputs/package --require-qdrant-records
chunking-docs audit-retrieval-cases examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --min-case-count 20 \
  --min-page-cases 8 \
  --min-asset-cases 4 \
  --min-distinct-asset-targets 4 \
  --min-case-group-distinct-targets case_source:visual_object_probe:asset=4 \
  --max-asset-cases-per-target 3 \
  --min-query-terms-per-case 3 \
  --min-case-group-count case_source:visual_object_probe=4 \
  --require-visual-only-object-probes \
  --output outputs/package/retrieval_case_audit.json
chunking-docs eval-chunking --package-dir outputs/package --cases examples/retrieval_cases.jsonl
chunking-docs eval-retrieval examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --top-k 5 \
  --repeat 3 \
  --reranker lexical \
  --output outputs/package/retrieval_eval.json
chunking-docs diagnose-retrieval outputs/package/retrieval_eval.json \
  --output outputs/package/retrieval_diagnostics.json
chunking-docs gate-retrieval outputs/package/retrieval_eval.json \
  --min-recall-at-k 0.8 \
  --min-target-coverage-at-k 0.75 \
  --min-target-ndcg-at-k 0.7 \
  --max-mean-target-rank 3 \
  --max-p95-target-rank 5 \
  --min-target-type-coverage asset=0.9 \
  --min-target-type-coverage triple=0.9 \
  --min-source-family-target-coverage lexical=0.75 \
  --min-source-target-coverage bm25=0.75 \
  --min-chunk-strategy-target-coverage visual_asset_text=0.7 \
  --min-retrieval-role-target-coverage child=0.7 \
  --min-case-group-target-coverage case_source:visual_lexical_probe=0.7 \
  --max-p95-latency-ms 100 \
  --output outputs/package/retrieval_gate.json
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
chunking-docs gate-qdrant-vector-ablation outputs/package/qdrant_vector_ablation.json \
  --mode text_caption \
  --baseline-mode text \
  --min-recall-at-k 0.8 \
  --min-target-coverage-at-k 0.75 \
  --min-target-ndcg-at-k 0.7 \
  --max-mean-target-rank 3 \
  --min-pairwise-win-rate 0.55 \
  --min-pairwise-target-coverage-lift 0.02 \
  --max-pairwise-mean-target-rank-delta 0 \
  --max-pairwise-target-rank-delta-ci-high 0 \
  --min-target-type-coverage asset=0.9 \
  --min-source-family-target-coverage visual=0.75 \
  --min-source-target-coverage qdrant:caption_dense=0.75 \
  --min-case-group-target-coverage case_source:visual_object_probe=0.7 \
  --max-failed-queries 0 \
  --require-best-by-recall \
  --output outputs/package/qdrant_vector_ablation_gate.json
chunking-docs eval-retrieval-ablation examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --modes dense,bm25_text,bm25_visual,hybrid_text,hybrid_visual,graph,hybrid_graph \
  --repeat 3 \
  --output outputs/package/retrieval_ablation.json
chunking-docs gate-retrieval-ablation outputs/package/retrieval_ablation.json \
  --mode bm25_visual \
  --baseline-mode bm25_text \
  --min-recall-lift 0.2 \
  --max-mean-target-rank 3 \
  --min-pairwise-win-rate 0.55 \
  --min-pairwise-target-coverage-lift 0.02 \
  --max-pairwise-mean-target-rank-delta 0 \
  --max-pairwise-target-rank-delta-ci-high 0 \
  --min-target-type-coverage asset=0.9 \
  --min-source-family-target-coverage lexical=0.75 \
  --min-source-target-coverage bm25=0.75 \
  --min-case-group-target-coverage case_source:visual_lexical_probe=0.7 \
  --output outputs/package/retrieval_ablation_gate.json
chunking-docs compare-packages outputs/package.baseline outputs/package \
  --output outputs/package/package_delta.json
```

`audit-publication` scans public repository files for forbidden text patterns, blocked artifact extensions such as PDFs or images, oversized files, and required `.gitignore` rules for generated data. `audit-package` checks structural completeness, orphan triples, remaining OCR/VLM work, Qdrant vector dimensions, required payload fields, payload index definitions, embedding manifest record counts, dimensions, bytes, checksums, whether configured text/image/caption vector records cover the expected chunk and visual asset IDs, and whether text/caption Qdrant payload text is current after chunk or visual annotation changes. `audit-retrieval-cases` verifies that benchmark queries are not empty or TODO placeholders, expected page/chunk/asset/triple targets exist in the package, duplicate queries stay within the configured limit, target families and distinct target IDs have enough coverage, case-group target diversity is sufficient, case concentration per target stays within configured limits, queries have enough distinct terms, required metadata groups such as `case_source:visual_object_probe` are present, and visual object probes can be required to use visual-only object terms. `eval-chunking` reports page coverage, chunk size distribution, section coverage, visual asset linkage, visual annotation coverage, linked visual text coverage at asset and text-part levels, standalone visual text chunk counts, retrieval recall@k, MRR, target coverage@k, target nDCG@k, precision@k, failed queries, and an aggregate quality score. `eval-retrieval` also records per-query latency samples plus mean and p95 latency when `--repeat` is greater than one. `diagnose-retrieval` groups failed or partially covered queries by reasons such as no hits, missing target type, low target nDCG@k, or low precision@k. `gate-retrieval` turns retrieval metrics into pass/fail checks for absolute floors, target rank limits, target-type coverage, source-family target coverage, exact source target coverage, chunking-strategy coverage, retrieval-role coverage, case metadata group coverage, and optional baseline regression limits such as recall drop or latency ratio. Retrieval reports include target-specific recall, MRR, target nDCG@k, source-family and exact source contribution metrics, chunking-strategy and retrieval-role contribution metrics, case group metrics for metadata such as `case_source` and `query_mode`, rank metrics that penalize missing targets as `top_k + 1`, and coverage for page, chunk, visual asset, and graph triple expectations. Visual vector hits are credited for graph triple targets when triples carry visual asset provenance, so VLM-derived relationships can be measured through caption or image retrieval. `eval-qdrant-retrieval` runs the same benchmark cases through Qdrant named vectors, BM25, and optional graph expansion so the production retrieval path can be validated. `eval-qdrant-vector-ablation` compares Qdrant text, visual caption, optional image, and graph-expanded modes on the same cases. `gate-qdrant-vector-ablation` turns a selected Qdrant vector mode into a pass/fail benchmark gate for recall, target coverage, target nDCG, target rank limits, precision, failed-query count, latency, target-type coverage, source-family and exact source target coverage, metadata case-group coverage, strategy/role contribution metrics, and optional best-mode requirements. `eval-retrieval-ablation` compares dense-only, BM25-only, graph-only, hybrid, graph-expanded hybrid, and text-only versus visual-asset-enriched lexical modes so the effect and runtime cost of each retrieval signal is visible. Retrieval and Qdrant vector ablation reports also include case-group best-mode summaries plus query-paired candidate-vs-baseline win rates, metric deltas, confidence intervals, and rank metrics that penalize missing targets as `top_k + 1`, making it clear which signal wins on subsets such as VLM object probes rather than only on the aggregate benchmark. `gate-retrieval-ablation` turns a selected ablation mode into a pass/fail gate using absolute thresholds, baseline lift, target rank limits, target-type coverage, source-family and exact source coverage, metadata case-group coverage, strategy/role contribution metrics, best-mode requirements, latency limits, and query-paired baseline metrics when a baseline mode is supplied. Retrieval cases are JSONL:

Pairwise ablation gates can also cap first-relevant-rank and mean-target-rank deltas, including bootstrap CI high bounds, so a candidate cannot pass by improving aggregate recall while pushing expected evidence deeper than the baseline.

Qdrant vector ablation modes include `text`, `caption`, `image`, `text_caption`, `text_image`, `caption_image`, `all`, `text_caption_graph`, and `all_graph`. Image modes require an `image_dense` record file and a compatible image-query encoder.

Hybrid retrieval commands accept repeatable `--fusion-weight source=weight` values. Sources can be exact names such as `qdrant:caption_dense` or families such as `qdrant`, `bm25`, `dense`, and `graph`. Graph hits score exact subject, predicate, and object phrase matches above loose token overlap, which helps graph-style benchmark queries find the intended evidence chunk. When a triple carries visual asset provenance, graph retrieval can also resolve the chunk linked to that asset. Use `--reranker lexical` for dependency-free overlap reranking, or `--reranker cross-encoder --reranker-model <model>` when the embeddings extra is installed.

Generate a benchmark skeleton from existing package targets, then edit the queries for the document family:

```bash
chunking-docs generate-retrieval-cases \
  --package-dir outputs/package \
  --chunks outputs/package/chunks.multimodal.jsonl \
  --query-mode salient_terms \
  --selection-strategy salience \
  --visual-probe-limit 20 \
  --object-probe-limit 20 \
  --include-todo \
  --output outputs/package/retrieval_cases.skeleton.jsonl
```

`--chunks` can point at a candidate chunk file so the same benchmark drafting logic can be run against semantic, multimodal, or hierarchical candidates. `--query-mode snippet` drafts queries from source text snippets. `--query-mode salient_terms` drafts harder keyword-style queries from document-frequency-weighted terms, and `--selection-strategy salience` prioritizes targets with more distinctive text. `--visual-probe-limit` adds asset-targeted probe cases whose query terms come from linked visual captions, OCR text, or VLM summaries after removing terms already present in the linked non-visual chunk text; these cases are useful for measuring whether visual text actually improves retrieval. `--object-probe-limit` adds separate `visual_object_probe` cases from structured VLM objects, detections, regions, attributes, locations, and bbox-bearing metadata so object detection output can be measured as its own case family. Object probes default to terms that are not already present in linked non-visual chunk text, which makes them better at isolating VLM object value; generated `Visual context:` blocks and `visual_asset_text` chunks do not suppress those probe terms. Use `--no-object-probe-visual-only` only when broad object-label coverage is preferred. Triple cases include visual asset targets when triples carry asset provenance, so generated benchmarks can measure graph and visual retrieval paths together. Duplicate query strings are merged by default so repeated tables, section labels, or graph triples become one case with multiple acceptable targets; use `--no-dedupe-queries` only when auditing duplicate behavior. Treat generated cases as reviewable drafts before using them as a benchmark gate.

```jsonl
{"query":"policy corridor near river","expected_pages":[12],"graph_expand":true}
{"query":"capital investment table","expected_chunk_ids":["chunk-id"]}
{"query":"map legend for station access","expected_asset_ids":["asset-id"]}
{"query":"district connects to corridor","expected_triple_ids":["triple-id"],"graph_expand":true}
```

For portfolio or production use, maintain benchmark cases for each document family and compare chunking strategies before changing defaults.

Compare a new retrieval run against a saved baseline with regression limits:

```bash
chunking-docs gate-retrieval outputs/package/retrieval_eval.json \
  --baseline outputs/package/retrieval_eval.baseline.json \
  --max-recall-drop 0.05 \
  --max-target-coverage-drop 0.05 \
  --max-target-ndcg-drop 0.05 \
  --max-mean-latency-ratio 1.5
```

## Lexical Tokenization

BM25 uses the `mixed` tokenizer by default. It combines word tokens with CJK character n-grams, which helps retrieve compound terms that may appear without whitespace in PDF text or OCR output. The lexical corpus includes chunk text plus visual asset captions, OCR text, and VLM summaries linked through `asset_ids` or `asset:` source refs, so visual-only labels can still recover their parent chunks. The package writes `bm25_tokens.json`, and PostgreSQL row export mirrors those tokens into `chunk_lexical_tokens` for reproducible lexical scoring and future search-service migration.

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
  --cases examples/retrieval_cases.jsonl \
  --output outputs/package/chunking_comparison.json
```

Gate the selected candidate before adopting it as the default strategy:

```bash
chunking-docs gate-chunking-comparison outputs/package/chunking_comparison.json \
  --baseline-candidate baseline \
  --require-retrieval \
  --min-recall-at-k 0.8 \
  --min-target-coverage-at-k 0.75 \
  --min-target-ndcg-at-k 0.7 \
  --min-precision-at-k 0.4 \
  --max-mean-target-rank 3 \
  --min-visual-text-coverage-ratio 0.8 \
  --min-target-type-coverage asset=0.9 \
  --min-target-type-coverage triple=0.9 \
  --min-source-family-target-coverage lexical=0.75 \
  --min-case-group-target-coverage case_source:visual_lexical_probe=0.7 \
  --min-visual-text-part-coverage-ratio 0.8 \
  --max-failed-queries 0 \
  --max-recall-drop 0.05 \
  --max-target-coverage-drop 0.05 \
  --max-target-ndcg-drop 0.05 \
  --min-pairwise-win-rate 0.55 \
  --min-pairwise-target-ndcg-lift 0.02 \
  --max-pairwise-mean-target-rank-delta 0 \
  --max-pairwise-target-rank-delta-ci-high 0 \
  --min-pairwise-target-ndcg-ci-low 0.0 \
  --max-mean-latency-ratio 1.5 \
  --output outputs/package/chunking_comparison_gate.json
```

The `multimodal` strategy keeps semantic text chunks, appends bounded visual context from linked captions, OCR, VLM summaries, and structured VLM metadata, and adds separate visual asset text chunks. Visual links are resolved from both `asset_ids` and `asset:` source refs, so annotations can remain provenance-oriented while still contributing to embedding text. Text-bearing visual assets without a linked parent chunk are emitted as standalone visual chunks instead of being dropped. The `hierarchical` strategy emits coarse parent chunks plus fine child chunks with shared visual context, which supports experiments where broad queries should find a page or section while precise queries should retrieve a smaller evidence span. `--collapse-hierarchical` reports the parent as the final hit while preserving matched child chunks as evidence. Comparison output includes recall@k, MRR, target coverage@k, target nDCG@k, precision@k, target rank metrics, target-type coverage, source-family target coverage, chunking-strategy coverage, retrieval-role coverage, linked visual text asset coverage, linked visual text part coverage, latency, failed queries, chunk size issues, query-paired baseline deltas, paired bootstrap confidence intervals, and the best candidate by quality and retrieval behavior. Pairwise gate options such as `--min-pairwise-win-rate`, `--min-pairwise-target-coverage-lift`, `--min-pairwise-target-ndcg-lift`, `--max-pairwise-mean-target-rank-delta`, `--max-pairwise-target-rank-delta-ci-high`, and `--max-pairwise-mean-latency-delta-ms` help distinguish broad aggregate gains from stable wins on the same benchmark queries.

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
  --selection-min-retrieval-recall-at-k 0.8 \
  --selection-min-target-coverage-at-k 0.75 \
  --selection-min-target-type-coverage asset=0.9 \
  --selection-min-case-group-target-coverage case_source:visual_object_probe=0.7 \
  --selection-max-mean-target-rank 3 \
  --selection-max-mean-latency-ms 150 \
  --selection-max-total-chunk-chars 1000000 \
  --cases examples/retrieval_cases.jsonl \
  --output outputs/package/chunking_sweep.json
```

The sweep writes candidate chunk files under `outputs/package/chunking_sweep/` and ranks them with the same quality, recall@k, MRR, target coverage@k, target nDCG@k, precision@k, target rank, target-type coverage, source-family target coverage, chunking-strategy coverage, retrieval-role coverage, linked visual text coverage, latency, and failed-query metrics used by `compare-chunking`. It also emits a `selection` block with weighted scores, eligibility failures, eligible counts, a recommendation, and a Pareto front so a strategy that improves retrieval can be checked against hard recall, target-rank, latency, visual-text, target-type, source-family, case-group, chunk-count, chunk-length, standalone visual chunk, and embedding text-volume constraints before becoming the default. Pareto dominance treats total chunk text, mean and p95 chunk length, embedding text volume, and standalone visual chunk count as cost axes, so a candidate with slightly better retrieval does not hide a much larger embedding payload.

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

The report records package artifact checksums, JSONL record counts, BM25 tokenizer settings, Qdrant named-vector configuration, readiness, evaluation, audit, gate artifact variants, visual run comparison summaries, top-level and component-level validation pass/fail summaries, chunking quality metrics, linked visual text coverage, retrieval recall@k, MRR, target coverage@k, target nDCG@k, target rank, precision@k, latency, failed queries, case-group metrics such as visual object probe coverage, paired confidence metrics from chunking gates, and the best candidate by retrieval behavior. This makes chunking changes reviewable and repeatable before new defaults are adopted.

## Development Checks

```bash
ruff check src tests
pytest
```

## Design Notes

- [Architecture](docs/architecture.md)
