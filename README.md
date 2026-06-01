# chunking_docs

`chunking_docs` is a Python library and CLI for preparing complex PDF documents for RAG systems. It focuses on chunking strategy, multimodal document assets, hybrid retrieval artifacts, and storage-ready outputs rather than on any one source PDF.

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

`--vlm-profile` compares the selected Hugging Face VLM profile with visible GPU memory before a long local run. It also records Torch CUDA device names, compute capability, CUDA version, compiled architecture targets, and bfloat16 support when Torch is installed. For GPU-backed embedding or VLM runs, `doctor` checks that the Torch CUDA build includes an architecture target for the visible GPU, which catches unsupported builds on newer cards before batch work starts. `--vlm-memory-margin-ratio` emits a warning when the selected profile only barely fits. Current 7B/8B profiles are marked for 24GB-class GPUs, while the compact Phi-3.5 Vision profile is marked for 12GB-class GPUs.

## Basic Pipeline

```bash
chunking-docs download "https://example.com/document.pdf" data/raw/document.pdf
chunking-docs profile data/raw/document.pdf --output-dir outputs/profile
chunking-docs package data/raw/document.pdf --output-dir outputs/package
```

The package directory contains:

- `manifest.json`: document metadata plus reproducibility metadata such as source-file checksum, package settings, tokenizer config, and profile summary
- `pages.jsonl`: page profiles and language-neutral text quality signals such as control-character ratio, readable-character ratio, and quality reasons
- `chunks.jsonl`: chunk records
- `assets.jsonl`: rendered visual asset records
- `triples.jsonl`: graph triple candidates
- `bm25_tokens.json`: lexical search token manifest
- `embedding_manifest.json`: vector record files, dimensions, counts, and checksums
- `qdrant_*_records.jsonl`: Qdrant upsert records
- `qdrant_collection.json`: named-vector collection configuration

For an existing package, refresh reproducibility metadata before readiness checks:

```bash
chunking-docs refresh-package-indexes --package-dir outputs/package
chunking-docs refresh-package-metadata --package-dir outputs/package
```

`refresh-package-indexes` rebuilds `bm25_tokens.json` from the current chunks plus linked visual captions, OCR text, VLM summaries, object metadata, entities, and visual elements. By default it removes stale Qdrant record files, `qdrant_collection.json`, and `embedding_manifest.json` so outdated vectors are not ingested after chunk or visual annotation changes. Use `embed-package` afterward for model-backed text, caption, object, image, and triple vectors, or `--rebuild-dry-run-embeddings` for deterministic local test vectors. `refresh-package-metadata` then updates `manifest.json` with the current source checksum, profile summary, package config, tokenizer config, table count, and inferred embedding mode without rebuilding chunks.

Summarize the package characteristics that should drive chunking, OCR/VLM, graph, and retrieval choices:

```bash
chunking-docs characterize-package \
  --package-dir outputs/package \
  --output outputs/package/document_characteristics.json
chunking-docs plan-ingestion-workflow \
  --package-dir outputs/package \
  --retrieval-cases examples/retrieval_cases.jsonl \
  --vlm-profiles qwen2_5_vl_7b,qwen2_vl_7b,llava_next_7b \
  --output outputs/package/ingestion_workflow_plan.json
```

The report includes observations and processing recommendations for visual annotation, multimodal embeddings, graph signals, and retrieval benchmark coverage. `plan-ingestion-workflow` turns those recommendations into an ordered command plan: runtime checks, characterization, page tiling, OCR/VLM jobs, VLM profile experiments, profile-specific runtime doctor outputs, VLM experiment output gates, profile-specific visual result files, visual run comparison gates, annotation application, retrieval case generation, chunking comparison, index refresh, embedding rebuilds, Qdrant retrieval config export, final RAG context gates, metadata refresh, and final ingestion readiness with the runtime report attached. Workflow plans only include OCR/VLM job commands when the package still has pending visual work; already attempted OCR assets and already structured VLM summaries are not planned again, while final readiness can still gate the applied visual quality. Recommended chunking plans include retrieval-quality, embedding-volume, and latency-efficiency gates so a larger multimodal strategy has to prove downstream search value before it is applied. Dense visual pages are reported as tile candidates with a ready `build-tile-assets` command so maps, tables, and diagrams can be processed as overlapping crops before OCR/VLM evaluation. When rendered assets, VLM object or visual-element metadata, or graph triples are present without matching `image_dense`, `object_dense`, or `triple_dense` records, the report recommends rebuilding those vector families before ablation. When both object and triple vectors are recommended, the workflow exports the Qdrant config with the adaptive route preset and then evaluates the same routed config in retrieval, RAG context, and readiness gates. When rendered visual assets are present, it recommends generating and gating `visual_image_probe` cases so `qdrant:image_dense` contribution is measured separately from caption and object vectors. When VLM object or visual-element metadata is present, it reports object, visual-feature, and bbox counts and recommends generating and auditing `visual_object_probe` retrieval cases with visual-only, target-diversity, concentration, and query-strength gates so object detections and visual elements are evaluated separately from aggregate retrieval scores.

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

Tile jobs keep parent asset, tile index, tile grid, text-quality, and visual-complexity metadata, and are prioritized ahead of their full-page parent when OCR/VLM batches are limited.

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

The command writes `visual_annotations.jsonl` and `visual_job_results.jsonl`. With `--apply`, annotations are merged into `assets.jsonl`, `chunks.jsonl`, `triples.jsonl`, and BM25. Chunk updates follow visual asset links from both `asset_ids` and `asset:` source refs. If an annotation contains structured VLM metadata such as entities, visual elements, or objects but no explicit relationships, the same derived triple rules populate `triples.jsonl` with visual asset provenance. Visual text chunks preserve compact asset scope, tile parent/grid, text-quality, and OCR/VLM work metadata so `text_dense` records can be filtered consistently with caption, object, and image records. Run `embed-package` after applying annotations to refresh dense, caption, object, image, and Qdrant record artifacts with the intended embedding models.
PaddleOCR uses a CPU device by default because the standard `paddlepaddle` wheel is CPU-only in many environments. Use `--ocr-device gpu:0` only after `chunking-docs doctor --require-ocr-gpu` confirms Paddle CUDA support. `--require-ocr` checks OCR dependencies without requiring Paddle CUDA, which is useful for CPU OCR plus GPU VLM runs. `--ocr-enable-mkldnn` can improve CPU throughput after a local smoke test confirms the Paddle runtime is stable with oneDNN enabled.

Use `--vlm-profile` for reproducible Hugging Face VLM experiments. Profiles provide the model id, loader family, dtype, and default generation length for common local VLM families such as Qwen2.5-VL, Qwen2-VL, LLaVA-NeXT, Idefics2, and Phi-3.5 Vision. The `vision` extra installs Transformers, Accelerate, PyTorch, and Torchvision; run `chunking-docs doctor --require-vision --vlm-profile <profile> --vlm-memory-margin-ratio 0.1` before long VLM batches to check memory fit, safety margin, CUDA visibility, and bfloat16 compatibility. Override any profile field with `--vlm-model`, `--vlm-model-class`, `--vlm-device-map`, `--vlm-torch-dtype`, `--vlm-max-new-tokens`, or `--vlm-attn-implementation`.

Create a reusable command plan when comparing several VLM profiles on the same job set:

```bash
chunking-docs plan-vlm-experiments \
  --package-dir outputs/package \
  --jobs outputs/package/visual_jobs.priority.jsonl \
  --profiles qwen2_5_vl_7b,qwen2_vl_7b,llava_next_7b \
  --limit 10 \
  --batch-size 5 \
  --output outputs/package/vlm_experiment_plan.json
```

The plan records the selected job count, OCR/VLM operation counts, asset-kind mix, page span, skipped jobs from the limit, per-profile `doctor` commands and `runtime_doctor.<profile>.json` outputs, per-batch offsets and limits, per-profile merge commands, batch comparison commands, and an upper bound for generated VLM tokens. Generated recipes require OCR runtime checks only when the selected job set actually contains OCR work; VLM-only batches use `--ocr none` so a missing OCR engine does not block visual-summary experiments. Use those fields to size the first GPU run before executing the generated `run-visual-jobs` commands. `merge-visual-results` combines batch result files into the run-level `visual_job_results.<profile>.jsonl` used by `compare-visual-runs`, preferring completed results over skipped offset/limit placeholders for the same job ID.

Gate a VLM experiment plan when you want to distinguish a planned experiment from one that has produced runtime and visual outputs:

```bash
chunking-docs gate-vlm-experiment-plan \
  --plan outputs/package/vlm_experiment_plan.json \
  --require-doctor-outputs \
  --require-results \
  --min-completed-result-profiles 2 \
  --require-same-result-jobs \
  --output outputs/package/vlm_experiment_plan_gate.json
```

The gate checks profile and recipe counts, declared `runtime_doctor.<profile>.json` outputs, passed doctor reports, visual result files, completed result profiles, optional annotation files, and whether existing profile result files share the same visual job IDs.

Default VLM prompts request a single JSON object with `title`, `summary`, `key_points`, `visual_elements`, `objects`, `entities`, and `triples`. The prompt is specialized by asset kind: maps emphasize regions, boundaries, corridors, and legends; tables emphasize headers, units, rows, and highlighted cells; charts emphasize axes, units, legends, trends, and extrema; figures emphasize components, flow direction, labels, and visible objects. When those fields are present, the runner converts them into captions, searchable VLM summaries, normalized object detections, and graph triple candidates. Object detections can carry attributes, descriptions, locations, bbox coordinates, confidence, and source-field provenance. Normalized bbox coordinates are converted into coarse spatial labels such as `upper left` for caption text, triple text, and object-probe retrieval cases. Entity, visual element, and object fields are also lifted into derived triple candidates and included in visual asset lexical/caption text so useful VLM detections remain searchable even when the model does not emit explicit relationships.
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
  --retrieval-eval vlm_a=outputs/package/retrieval_eval.vlm_a.json \
  --retrieval-eval vlm_b=outputs/package/retrieval_eval.vlm_b.json \
  --output outputs/package/visual_run_comparison.json \
  --require-same-jobs
```

The comparison ranks OCR/VLM runs by completion rate, annotation coverage, OCR text coverage, VLM summary coverage, JSON parse rate, object coverage, triple density, and total latency. When matching `--retrieval-eval name=...` files are provided, it also records retrieval hit rate, recall, MRR, target coverage, nDCG, precision, visual-object-probe coverage, and `best_by_retrieval` so VLM choices can be judged by downstream search behavior. It also reports shared and missing visual job IDs, and `--require-same-jobs` fails the command when runs were produced from different job sets.

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

chunking-docs repair-visual-triples \
  --package-dir outputs/package \
  --in-place \
  --export-graph

chunking-docs repair-visual-text \
  --package-dir outputs/package \
  --in-place
```

Normalization collapses whitespace, canonicalizes predicate names, recomputes stable triple IDs, and can remove semantic duplicates within the same chunk. `repair-visual-triples` uses structured metadata already stored in `assets.jsonl` to add missing VLM-derived entity, visual-element, and object triples with asset/page/prompt provenance, or to merge asset IDs into an equivalent existing triple. `repair-visual-text` appends missing linked asset text parts, including entities and visual elements, to `chunks.jsonl` so dense chunk vectors and chunk-level context include the same visual evidence that BM25 and caption vectors see. When either repair command changes package text or triples in place, rebuild model-backed embeddings afterward. The audit report counts duplicates, triples that would change under normalization, orphan chunk references, empty fields, invalid confidence values, and normalized predicate frequencies.
When `--export-graph` is used, graph nodes include degree, direction, predicate, document, and chunk provenance metadata, while `graph_summary.json` records connectivity, predicate counts, document counts, and top-degree nodes for browsing and retrieval-signal review.

## Embeddings

The default package command writes deterministic hashing vectors so the pipeline can be tested without downloading models. Rebuild model-backed records with:

```bash
chunking-docs embed-package \
  --package-dir outputs/package \
  --text-backend sentence-transformers \
  --text-model BAAI/bge-m3 \
  --caption-backend same-as-text \
  --object-backend same-as-caption \
  --image-backend clip \
  --image-model openai/clip-vit-large-patch14 \
  --triple-backend same-as-text \
  --device cuda
```

This regenerates Qdrant text, caption, object, image, and graph triple records using the selected model dimensions. `object_dense` embeds one VLM/OCR-derived visual object, region, or visual element per record, preserving labels, attributes, descriptions, locations, bbox regions, confidence, source field, feature type, and the parent asset ID so object-detection and visual-element terms can be measured separately from broad captions. `triple_dense` embeds normalized `subject predicate object` text plus selected visual qualifiers such as evidence, attributes, locations, bbox-derived regions, and source fields so graph relationships can be evaluated as a vector source as well as through symbolic graph expansion. Triple records keep `record_kind=graph_triple` while copying resolved chunk page, kind, section, strategy, and asset-link payload fields, so chunk-level filters still work when triple vectors participate in hybrid search. It also writes `embedding_manifest.json` so vector files, record counts, dimensions, checksums, backend names, model IDs, devices, and batch sizes can be compared across embedding runs.

## Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d
chunking-docs qdrant-check-collection \
  --package-dir outputs/package \
  --output outputs/package/qdrant_collection_contract.json \
  --allow-missing
chunking-docs qdrant-upsert-package --package-dir outputs/package
```

The package collection config includes payload index definitions for document IDs, chunk IDs, asset IDs, object IDs, chunking strategy fields, hierarchy links, standalone visual chunk flags, text-quality signals, OCR/VLM work flags, visual asset scope, object labels, bbox regions, visual feature types, tile parent/grid fields, page fields, and section fields. `qdrant-check-collection` validates an existing Qdrant collection against the package named-vector dimensions, payload index fields, and payload index schemas before upsert. `qdrant-upsert-package`, `qdrant-search-package`, and Qdrant hybrid evaluation create those indexes when the target Qdrant server supports them.

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
  --vector-names text_dense,caption_dense,object_dense,triple_dense \
  --filter kind=text \
  --fusion-weight bm25=1.2 \
  --fusion-weight qdrant:caption_dense=1.4 \
  --fusion-weight qdrant:object_dense=1.2 \
  --fusion-weight qdrant:triple_dense=1.1 \
  --reranker lexical \
  --rerank-top-k 20 \
  --graph-expand
```

Qdrant hybrid search, Qdrant retrieval evaluation, and exported-config context commands default to `--text-backend auto` and `--image-query-backend auto`. When `embedding_manifest.json` is present, text-like vectors such as `text_dense`, `caption_dense`, `object_dense`, and `triple_dense` use the recorded text embedding backend and model, while `image_dense` uses the recorded CLIP text-side encoder when the image vectors were built with CLIP. Explicit `--text-backend`, `--text-model`, `--image-query-backend`, and `--image-query-model` flags still override the inferred settings.

When `image_dense` is included without a usable manifest, choose a text query encoder from the same image-text model family:

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
When Qdrant visual vectors retrieve a linked asset, chunk metadata records compact retrieval references such as asset ID, page, kind, and document ID without copying full payload text into metadata. The referenced asset is also included in the bundle assets so its bounded caption, OCR, and VLM text remain available for answer generation. Graph triples are selected by matched chunk IDs, visual asset provenance, and explicit `triple_id` payloads from graph-vector hits, so VLM-derived triples can follow the asset or relationship that triggered retrieval.
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

Use an exported retrieval config when the RAG service should share the same vector names, fusion weights, graph expansion, hierarchy collapse, and tokenizer settings that passed benchmark evaluation:

```bash
chunking-docs qdrant-rag-context-config outputs/package/qdrant_retrieval_config.json \
  "station access corridor" \
  --package-dir outputs/package \
  --location ':memory:' \
  --output outputs/package/rag_context.config.json
```

Evaluate the final context bundles produced by an exported config against retrieval cases:

```bash
chunking-docs eval-qdrant-rag-context-config outputs/package/qdrant_retrieval_config.json \
  examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --location ':memory:' \
  --contexts-output outputs/package/rag_context.config.cases.jsonl \
  --output outputs/package/qdrant_rag_context_config_eval.json
chunking-docs gate-rag-context outputs/package/qdrant_rag_context_config_eval.json \
  --min-target-coverage 0.8 \
  --min-target-type-coverage asset=0.75 \
  --min-target-type-coverage triple=0.7 \
  --max-excluded-target-hit-rate 0 \
  --max-mean-context-char-count 12000 \
  --output outputs/package/qdrant_rag_context_gate.json
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

PostgreSQL is intended for source metadata, page profiles, chunks, BM25 token artifacts, visual asset links, assets, normalized VLM object rows, graph triples, embedding artifact provenance, and pgvector-compatible embedding record rows. Vector search is handled by Qdrant by default, while PostgreSQL stores vector file names, dimensions, counts, checksums, backend/model metadata, collection names, payloads, and target IDs so embedding runs remain auditable or migratable.

```bash
docker compose -f docker-compose.postgres.yml up -d

chunking-docs postgres-schema --output outputs/package/postgres_schema.sql
chunking-docs postgres-check-schema \
  "postgresql://chunking_docs:chunking_docs@localhost:5432/chunking_docs" \
  --output outputs/package/postgres_schema_contract.json \
  --apply-schema
chunking-docs postgres-rows --package-dir outputs/package
chunking-docs postgres-upsert "postgresql://chunking_docs:chunking_docs@localhost:5432/chunking_docs" \
  --package-dir outputs/package
```

`docker-compose.postgres.yml` starts a local PostgreSQL 16 service with pgvector enabled and stores local data under `.local/postgres`, which is ignored by Git. Override `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`, or `POSTGRES_DATA_PATH` when a different local database or port is needed. `postgres-schema` writes the SQL contract without opening a database connection. `postgres-check-schema` validates required tables, columns, column types, indexes, and the pgvector extension before metadata rows are upserted. BM25 token rows from `bm25_tokens.json` are stored in `chunk_lexical_tokens` with the tokenizer config and token array, so lexical experiments can be audited or migrated to a PostgreSQL-backed search service later. Chunk-to-asset links are stored in a normalized `chunk_asset_links` table and also preserved in chunk metadata for auditability. Structured VLM object, region, and visual-element metadata from visual assets is exported to `visual_objects` with label, bbox region, attributes, description, confidence, object text, source field, feature type, and asset/page provenance so object-detection probes can be joined without parsing asset JSON. Embedding artifact rows preserve Qdrant payload index fields and schemas alongside vector file metadata, which keeps filter contracts auditable after Qdrant artifacts are loaded or migrated. `embedding_records` mirrors Qdrant JSONL records into PostgreSQL with `vector`, `dimension`, `payload`, `target_kind`, and `target_id` columns; no ANN index is created by default because named vector families can use different dimensions, but the rows can be filtered, audited, or re-indexed per family in a deployment database. Asset-backed graph triples are remapped to an available chunk before PostgreSQL row export while retaining the original chunk ID in qualifiers. Use `--apply-schema` when bootstrapping a new database; omit it when checking an existing schema for drift.

## Ingestion Readiness

Before loading a package into Qdrant, PostgreSQL, or a RAG service, run a combined readiness check:

```bash
chunking-docs ingestion-readiness \
  --package-dir outputs/package \
  --runtime-report outputs/package/runtime_doctor.json \
  --require-runtime-report \
  --require-visual-annotations \
  --require-visual-quality \
  --min-vlm-summary-coverage 0.95 \
  --min-vlm-json-parse-rate 0.95 \
  --min-visual-text-coverage-ratio 0.8 \
  --min-visual-text-part-coverage-ratio 0.8 \
  --require-visual-derived-triples \
  --require-derived-vector-coverage \
  --required-vector text_dense \
  --required-vector caption_dense \
  --required-vector object_dense \
  --required-vector image_dense \
  --required-vector triple_dense \
  --visual-run-comparison outputs/package/visual_run_comparison.json \
  --require-visual-run-same-jobs \
  --visual-run-best-by-quality qwen2_5_vl_7b \
  --retrieval-cases examples/retrieval_cases.jsonl \
  --min-retrieval-case-group-count case_source:visual_image_probe=4 \
  --min-retrieval-case-group-count case_source:visual_object_probe=4 \
  --min-retrieval-distinct-asset-targets 4 \
  --min-retrieval-case-group-distinct-targets case_source:visual_image_probe:asset=4 \
  --min-retrieval-case-group-distinct-targets case_source:visual_object_probe:asset=4 \
  --max-retrieval-asset-cases-per-target 3 \
  --max-retrieval-expected-targets-per-case 5 \
  --min-retrieval-query-terms-per-case 3 \
  --require-visual-only-object-probes \
  --retrieval-evaluation outputs/package/retrieval_eval.json \
  --max-mean-target-rank 3 \
  --max-p95-target-rank 5 \
  --min-result-stability-rate 1.0 \
  --min-retrieval-target-type-coverage asset=0.9 \
  --min-retrieval-target-type-coverage triple=0.9 \
  --min-retrieval-source-family-target-coverage lexical=0.75 \
  --min-retrieval-case-group-source-target-coverage retrieval_route:graph_triple:qdrant:triple_dense=0.7 \
  --min-retrieval-case-group-source-target-coverage retrieval_route:visual_object:qdrant:object_dense=0.3 \
  --chunking-comparison outputs/package/chunking_comparison.json \
  --baseline-chunking-candidate baseline \
  --min-chunking-visual-text-coverage-ratio 0.8 \
  --min-chunking-visual-text-part-coverage-ratio 0.8 \
  --max-chunking-total-chunk-chars 1000000 \
  --min-chunking-retrieval-score-per-embedding-kchar 0.0008 \
  --min-chunking-retrieval-score-per-mean-latency-ms 0.0005 \
  --min-chunking-target-coverage-per-p95-latency-ms 0.0005 \
  --max-chunking-mean-target-rank 3 \
  --min-chunking-result-stability-rate 1.0 \
  --max-chunking-pairwise-mean-target-rank-delta 0 \
  --min-chunking-target-type-coverage asset=0.9 \
  --min-chunking-target-type-coverage triple=0.9 \
  --min-chunking-source-target-coverage bm25=0.75 \
  --min-chunking-source-family-target-coverage lexical=0.75 \
  --min-chunking-case-group-source-target-coverage case_source:visual_lexical_probe:bm25=0.7 \
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
  --min-retrieval-ablation-case-group-source-target-coverage case_source:visual_lexical_probe:bm25=0.7 \
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
  --min-qdrant-vector-case-group-source-target-coverage case_source:visual_object_probe:qdrant:caption_dense=0.5 \
  --max-qdrant-vector-failed-queries 0 \
  --qdrant-reranker-ablation outputs/package/qdrant_reranker_ablation.json \
  --qdrant-reranker-mode lexical \
  --qdrant-reranker-baseline-mode none \
  --min-qdrant-reranker-pairwise-win-rate 0.55 \
  --min-qdrant-reranker-pairwise-target-ndcg-lift 0.02 \
  --max-qdrant-reranker-pairwise-mean-target-rank-delta 0 \
  --max-qdrant-reranker-pairwise-mean-latency-delta-ms 75 \
  --min-qdrant-reranker-target-coverage-at-k 0.75 \
  --min-qdrant-reranker-target-ndcg-at-k 0.7 \
  --max-qdrant-reranker-mean-target-rank 3 \
  --min-qdrant-reranker-source-target-coverage rerank:lexical=0.75 \
  --min-qdrant-reranker-case-group-target-coverage case_source:visual_object_probe=0.7 \
  --min-qdrant-reranker-case-group-source-target-coverage case_source:visual_object_probe:rerank:lexical=0.7 \
  --min-qdrant-reranker-case-group-source-family-target-coverage case_source:visual_object_probe:reranker=0.7 \
  --max-qdrant-reranker-failed-queries 0 \
  --qdrant-retrieval-config outputs/package/qdrant_retrieval_config.json \
  --require-qdrant-retrieval-config \
  --retrieval-evaluation outputs/package/qdrant_retrieval_config_eval.json \
  --require-retrieval-evaluation \
  --min-target-coverage-at-k 0.8 \
  --min-target-ndcg-at-k 0.7 \
  --max-retrieval-failed-queries 3 \
  --max-p95-latency-ms 250 \
  --rag-context-evaluation outputs/package/qdrant_rag_context_config_eval.json \
  --min-rag-context-target-coverage 0.8 \
  --min-rag-context-target-type-coverage asset=0.75 \
  --min-rag-context-target-type-coverage triple=0.7 \
  --min-rag-context-case-group-target-coverage case_source:visual_object_probe=0.7 \
  --max-rag-context-excluded-target-hit-rate 0 \
  --max-rag-context-mean-context-char-count 12000 \
  --output outputs/package/ingestion_readiness.json
```

The report combines package audit results, runtime doctor validation, source checksum and package-config validation, BM25 token manifest validation, required embedding artifacts, required vector-family checks, Qdrant record checks, PostgreSQL row conversion, retrieval case audit, VLM run comparison checks, chunking comparison gates, selected retrieval and Qdrant vector/reranker ablation gates, exported Qdrant retrieval config validation, final RAG context gates, and optional visual or retrieval gates. `--runtime-report` attaches a `doctor` output to readiness so GPU visibility, Torch CUDA architecture support, bfloat16 support, VLM profile memory fit, and optional dependency checks are part of the same final ingestion gate. Reproducibility validation checks `manifest.json` source-file metadata, package generation settings, and tokenizer consistency between `package_config.lexical_tokenizer` and `bm25_tokens.json`. BM25 validation recomputes asset-enriched lexical text from chunks plus linked captions, OCR text, VLM summaries, and structured VLM metadata, then checks that `bm25_tokens.json` is complete and current before ingestion. `--min-visual-text-coverage-ratio` checks whether linked visual assets have text represented in package chunks, while `--min-visual-text-part-coverage-ratio` checks individual caption, OCR, VLM, object, entity, and visual-element text parts so object metadata is not hidden by a single covered caption. `--require-visual-derived-triples` fails readiness when VLM-derived object, entity, or visual-element metadata is missing asset-provenance graph triples, which keeps graph retrieval aligned with visual metadata. `--require-derived-vector-coverage` fails readiness when source data implies `text_dense`, visual text implies `caption_dense`, rendered visual assets imply `image_dense`, structured VLM objects or visual elements imply `object_dense`, or graph triples imply `triple_dense`, but the matching Qdrant collection, embedding manifest, or record file is missing. The same component reports missing derived vector families, generic rebuild commands, and Qdrant vector ablation modes so text, visual text, image, object, and graph signals can be regenerated and measured consistently. `--required-vector` verifies that selected vector families are present in `qdrant_collection.json`, represented in `embedding_manifest.json`, have non-empty record files, and use consistent dimensions. `--qdrant-retrieval-config` verifies that the service config points at available BM25 and Qdrant artifacts, uses named vectors present in `qdrant_collection.json`, has query encoders for those vectors, and matches supplied `eval-qdrant-retrieval-config` or `eval-qdrant-rag-context-config` metadata for collection, vectors, graph expansion, hierarchy collapse, fusion weights, reranker source and candidate depth, tokenizer settings, top-k, and selected sweep candidate. `--retrieval-evaluation` can point at the exported config evaluation so readiness also enforces retrieval target coverage, nDCG, failed-query, and latency gates for the exact service settings before the final RAG context gate runs. Retrieval case audit can require metadata group counts such as `case_source:visual_image_probe=4` or `case_source:visual_object_probe=4`, distinct target coverage such as `--min-retrieval-distinct-asset-targets 4`, case-group target diversity such as `--min-retrieval-case-group-distinct-targets case_source:visual_image_probe:asset=4`, concentration limits such as `--max-retrieval-asset-cases-per-target 3`, per-case target ceilings such as `--max-retrieval-expected-targets-per-case 5`, query strength such as `--min-retrieval-query-terms-per-case 3`, target-text leakage ceilings such as `--max-retrieval-target-query-overlap-ratio 0.9` and `--max-retrieval-target-query-overlap-terms 5`, and `--require-visual-only-object-probes` so VLM object-detection cases isolate metadata terms instead of disappearing or concentrating on one target. Chunking, retrieval, retrieval ablation, Qdrant vector, Qdrant reranker, and final RAG context gates can all enforce target-type coverage for page, chunk, visual asset, or graph triple expectations and metadata case-group coverage such as `case_source:visual_image_probe` or `case_source:visual_object_probe`. Retrieval, chunking, retrieval ablation, Qdrant vector, and Qdrant reranker gates can also enforce source-family, exact source, case-group source, and case-group source-family coverage, for example `--min-retrieval-source-target-coverage qdrant:caption_dense=0.75`, `--min-retrieval-ablation-source-target-coverage bm25=0.75`, `--min-qdrant-vector-source-target-coverage qdrant:image_dense=0.5`, or `--min-qdrant-reranker-case-group-source-target-coverage case_source:visual_object_probe:rerank:lexical=0.7`, so a combined family cannot pass unless the intended caption, image, reranker, or lexical source contributes to the intended benchmark subset. Route-specific source gates such as `--min-retrieval-case-group-source-target-coverage retrieval_route:visual_object:qdrant:object_dense=0.3` verify that an adaptive route is being satisfied by the expected vector source, not only by another source in the fused result set. RAG context readiness can also cap hard-negative leakage and context size with options such as `--max-rag-context-excluded-target-hit-rate` and `--max-rag-context-mean-context-char-count`, so an answer generator is not handed the wrong evidence or an oversized bundle. Visual run comparison checks can require the same visual job IDs across candidate VLM runs and confirm the intended profile won by quality or triple density. When `--require-visual-quality` is used without `--visual-results`, readiness evaluates the final OCR/VLM annotations currently stored in `assets.jsonl`.

## Evaluation

Correct execution is not enough for a chunking library. Use evaluation commands to check whether the chunking strategy is useful for retrieval.

```bash
chunking-docs audit-publication . \
  --forbidden-pattern "<confidential-term>" \
  --forbidden-pattern "<private-filename>" \
  --output outputs/publication_audit.json
chunking-docs audit-package --package-dir outputs/package
chunking-docs audit-package --package-dir outputs/package --require-qdrant-records
chunking-docs audit-package --package-dir outputs/package --require-visual-derived-triples
chunking-docs audit-retrieval-cases examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --min-case-count 20 \
  --min-page-cases 8 \
  --min-asset-cases 4 \
  --min-distinct-asset-targets 4 \
  --min-case-group-distinct-targets case_source:visual_object_probe:asset=4 \
  --max-asset-cases-per-target 3 \
  --min-query-terms-per-case 3 \
  --max-target-query-overlap-ratio 0.9 \
  --max-target-query-overlap-terms 5 \
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
  --min-case-count 30 \
  --min-expected-target-count 60 \
  --min-recall-at-k 0.8 \
  --min-target-coverage-at-k 0.75 \
  --min-target-ndcg-at-k 0.7 \
  --max-failed-queries 3 \
  --max-mean-target-rank 3 \
  --max-p95-target-rank 5 \
  --min-target-type-coverage asset=0.9 \
  --min-target-type-coverage triple=0.9 \
  --min-source-target-coverage bm25=0.75 \
  --min-source-family-target-coverage lexical=0.75 \
  --min-source-target-coverage bm25=0.75 \
  --max-source-family-excluded-target-hit-rate visual=0.0 \
  --max-source-excluded-target-hit-rate qdrant:image_dense=0.0 \
  --min-chunk-strategy-target-coverage visual_asset_text=0.7 \
  --min-retrieval-role-target-coverage child=0.7 \
  --max-chunk-strategy-excluded-target-hit-rate visual_asset_text=0.0 \
  --max-retrieval-role-excluded-target-hit-rate child=0.0 \
  --min-case-group-target-coverage case_source:visual_lexical_probe=0.7 \
  --min-case-group-source-target-coverage case_source:visual_object_probe:qdrant:object_dense=0.3 \
  --min-case-group-source-family-target-coverage case_source:visual_object_probe:visual=0.3 \
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
  --modes text,caption,image,text_image,caption_image,all,text_caption_graph \
  --image-query-backend clip \
  --image-query-model openai/clip-vit-large-patch14 \
  --top-k 5 \
  --repeat 3 \
  --output outputs/package/qdrant_vector_ablation.json
chunking-docs eval-qdrant-reranker-ablation examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --location ':memory:' \
  --vector-names text_dense,caption_dense,object_dense \
  --fusion-weight qdrant:text_dense=1.0 \
  --fusion-weight qdrant:caption_dense=1.2 \
  --fusion-weight qdrant:object_dense=0.5 \
  --fusion-weight bm25=1.5 \
  --modes none,lexical \
  --rerank-top-k 20 \
  --top-k 5 \
  --repeat 3 \
  --output outputs/package/qdrant_reranker_ablation.json
chunking-docs sweep-qdrant-fusion examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --location ':memory:' \
  --vector-names text_dense,caption_dense,object_dense,image_dense,triple_dense \
  --image-query-backend clip \
  --image-query-model openai/clip-vit-large-patch14 \
  --weight-grid bm25=0.8,1.0,1.2 \
  --weight-grid qdrant:caption_dense=0.8,1.0,1.2 \
  --weight-grid qdrant:object_dense=0.5,1.0,1.5 \
  --weight-grid qdrant:image_dense=0.0,0.25,0.5 \
  --fixed-fusion-weight qdrant:text_dense=1.0 \
  --fixed-fusion-weight qdrant:triple_dense=0.5 \
  --min-target-coverage-at-k 0.8 \
  --min-target-ndcg-at-k 0.7 \
  --max-failed-queries 3 \
  --max-p95-latency-ms 250 \
  --max-source-family-excluded-target-hit-rate visual=0.0 \
  --max-source-excluded-target-hit-rate qdrant:image_dense=0.0 \
  --max-chunk-strategy-excluded-target-hit-rate visual_asset_text=0.0 \
  --max-retrieval-role-excluded-target-hit-rate child=0.0 \
  --source-family-excluded-target-hit-penalty 1.0 \
  --chunk-strategy-excluded-target-hit-penalty 1.0 \
  --reranker lexical \
  --rerank-top-k 20 \
  --pairwise-top-k 10 \
  --output outputs/package/qdrant_fusion_sweep.json
chunking-docs export-qdrant-retrieval-config outputs/package/qdrant_fusion_sweep.json \
  --case-group case_source:visual_object_probe \
  --output outputs/package/qdrant_retrieval_config.json
chunking-docs eval-qdrant-retrieval-config outputs/package/qdrant_retrieval_config.json \
  examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --location ':memory:' \
  --repeat 3 \
  --output outputs/package/qdrant_retrieval_config_eval.json
chunking-docs qdrant-rag-context-config outputs/package/qdrant_retrieval_config.json \
  "station access corridor" \
  --package-dir outputs/package \
  --location ':memory:' \
  --output outputs/package/rag_context.config.json
chunking-docs eval-qdrant-rag-context-config outputs/package/qdrant_retrieval_config.json \
  examples/retrieval_cases.jsonl \
  --package-dir outputs/package \
  --location ':memory:' \
  --contexts-output outputs/package/rag_context.config.cases.jsonl \
  --output outputs/package/qdrant_rag_context_config_eval.json
chunking-docs gate-rag-context outputs/package/qdrant_rag_context_config_eval.json \
  --min-target-coverage 0.8 \
  --min-target-type-coverage asset=0.75 \
  --min-target-type-coverage triple=0.7 \
  --max-excluded-target-hit-rate 0 \
  --max-mean-context-char-count 12000 \
  --output outputs/package/qdrant_rag_context_gate.json
chunking-docs gate-qdrant-vector-ablation outputs/package/qdrant_vector_ablation.json \
  --mode caption_image \
  --baseline-mode caption \
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
  --min-source-target-coverage qdrant:image_dense=0.5 \
  --max-source-family-excluded-target-hit-rate visual=0.0 \
  --max-source-excluded-target-hit-rate qdrant:image_dense=0.0 \
  --max-chunk-strategy-excluded-target-hit-rate visual_asset_text=0.0 \
  --max-retrieval-role-excluded-target-hit-rate child=0.0 \
  --min-case-group-target-coverage case_source:visual_image_probe=0.7 \
  --min-case-group-source-target-coverage case_source:visual_image_probe:qdrant:image_dense=0.5 \
  --min-case-group-source-family-target-coverage case_source:visual_image_probe:visual=0.5 \
  --max-failed-queries 0 \
  --require-best-by-recall \
  --output outputs/package/qdrant_vector_ablation_gate.json
chunking-docs gate-qdrant-reranker-ablation outputs/package/qdrant_reranker_ablation.json \
  --mode lexical \
  --baseline-mode none \
  --min-recall-at-k 0.8 \
  --min-target-coverage-at-k 0.75 \
  --min-target-ndcg-at-k 0.7 \
  --max-mean-target-rank 3 \
  --min-pairwise-win-rate 0.55 \
  --min-pairwise-target-coverage-lift 0.02 \
  --min-pairwise-target-ndcg-lift 0.02 \
  --max-pairwise-mean-target-rank-delta 0 \
  --max-pairwise-mean-latency-delta-ms 75 \
  --min-target-type-coverage asset=0.9 \
  --min-source-target-coverage rerank:lexical=0.75 \
  --min-case-group-target-coverage case_source:visual_object_probe=0.7 \
  --min-case-group-source-target-coverage case_source:visual_object_probe:rerank:lexical=0.7 \
  --min-case-group-source-family-target-coverage case_source:visual_object_probe:reranker=0.7 \
  --max-failed-queries 0 \
  --require-best-by-target-ndcg \
  --output outputs/package/qdrant_reranker_ablation_gate.json
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
  --max-source-family-excluded-target-hit-rate lexical=0.0 \
  --max-source-excluded-target-hit-rate bm25=0.0 \
  --max-chunk-strategy-excluded-target-hit-rate visual_asset_text=0.0 \
  --max-retrieval-role-excluded-target-hit-rate child=0.0 \
  --min-case-group-target-coverage case_source:visual_lexical_probe=0.7 \
  --output outputs/package/retrieval_ablation_gate.json
chunking-docs compare-packages outputs/package.baseline outputs/package \
  --output outputs/package/package_delta.json
```

`audit-publication` scans public repository files for forbidden text patterns, blocked artifact extensions such as PDFs or images, oversized files, and required `.gitignore` rules for generated data. `audit-package` checks structural completeness, orphan triples, remaining OCR/VLM work, Qdrant vector dimensions, required payload fields, payload index definitions, embedding manifest record counts, dimensions, bytes, checksums, whether configured text/image/caption/object/triple vector records cover the expected chunk, visual asset, visual object, and graph triple IDs, and whether text, caption, object, OCR, VLM, and structured visual metadata payload fields are current after chunk or visual annotation changes. Use `repair-visual-triples` when that audit reports missing VLM-derived visual triples but the needed entities, visual elements, or objects are already present in `assets.jsonl`. `audit-retrieval-cases` verifies that benchmark queries are not empty or TODO placeholders, expected and excluded page/chunk/asset/triple targets exist in the package, duplicate queries stay within the configured limit, target families and distinct target IDs have enough coverage, case-group target diversity is sufficient, case concentration per target and expected target count per case stay within configured limits, queries have enough distinct terms, query terms do not overlap expected target text above optional ratio and term-count leakage thresholds, required metadata groups such as `case_source:visual_object_probe` are present, and visual object probes can be required to use visual-only object terms. `eval-chunking` reports page coverage, chunk size distribution, section coverage, visual asset linkage, visual annotation coverage, linked visual text coverage at asset and text-part levels, standalone visual text chunk counts, retrieval recall@k, MRR, target coverage@k, target nDCG@k, precision@k, retrieval-per-embedding-cost metrics, retrieval-per-latency metrics, failed queries, and an aggregate quality score. `eval-retrieval` also records per-query latency samples plus mean and p95 latency when `--repeat` is greater than one. `diagnose-retrieval` groups failed or partially covered queries by reasons such as no hits, missing target type, hard-negative excluded target hits, low target nDCG@k, or low precision@k; it also counts the exact sources and source families that retrieved excluded targets and preserves case metadata so failures can be broken down by groups such as `case_source:visual_object_probe` or `query_mode:salient_terms`. `gate-retrieval` turns retrieval metrics into pass/fail checks for benchmark size, expected target count, passed and failed query counts, absolute floors, target rank limits, target-type coverage, source-family target coverage, exact source target coverage, chunking-strategy coverage, retrieval-role coverage, hard-negative excluded-target hit rates, case metadata group coverage, and optional baseline regression limits such as recall drop or latency ratio. Retrieval reports include target-specific recall, MRR, target nDCG@k, source-family and exact source contribution metrics, chunking-strategy and retrieval-role contribution metrics, hard-negative excluded-target metrics, case group metrics for metadata such as `case_source` and `query_mode`, rank metrics that penalize missing targets as `top_k + 1`, and coverage for page, chunk, visual asset, and graph triple expectations. Visual and triple vector hits are credited for graph triple targets when triples carry visual asset provenance, so VLM-derived relationships can be measured through caption, object, image, symbolic graph, or triple-vector retrieval. `eval-qdrant-retrieval` runs the same benchmark cases through Qdrant named vectors, BM25, and optional graph expansion so the production retrieval path can be validated. `eval-qdrant-rag-context-config` evaluates the final context bundles generated from an exported Qdrant retrieval config, including page, chunk, visual asset, graph triple, hard-negative, context-size, latency, and case-group metrics. `gate-rag-context` turns that final context evaluation into pass/fail checks for benchmark size, target coverage, target-type coverage, case-group coverage, hard-negative leakage, context length, context item counts, and latency. `eval-qdrant-vector-ablation` compares Qdrant text, visual caption, object, optional image, triple-vector, and graph-expanded modes on the same cases. `eval-qdrant-reranker-ablation` keeps the same Qdrant vectors, BM25 weights, graph setting, and benchmark cases while comparing no reranker against lexical or cross-encoder reranking, so ranking lift and latency cost are measured separately from chunking and fusion-weight changes. `sweep-qdrant-fusion` evaluates a grid of source weights for Qdrant vectors, BM25, and graph expansion, then recommends the eligible candidate with the best retrieval score so visual or graph signals can be added only when they improve the measured benchmark; candidates can be rejected or penalized when they retrieve hard-negative excluded targets. `gate-qdrant-vector-ablation` turns a selected Qdrant vector mode into a pass/fail benchmark gate for recall, target coverage, target nDCG, target rank limits, precision, failed-query count, latency, hard-negative excluded-target hit rates, source-family and exact-source excluded-target hit rates, target-type coverage, source-family and exact source target coverage, metadata case-group coverage, strategy/role contribution metrics, and optional best-mode requirements. `gate-qdrant-reranker-ablation` turns a selected Qdrant reranker mode into the same style of pass/fail gate, including pairwise lift against a baseline mode, rank-delta ceilings, source contribution checks for rerank sources, and latency-delta ceilings so reranking must prove its value rather than only adding cost. `eval-retrieval-ablation` compares dense-only, BM25-only, graph-only, hybrid, graph-expanded hybrid, and text-only versus visual-asset-enriched lexical modes so the effect and runtime cost of each retrieval signal is visible. Retrieval, Qdrant vector, and Qdrant reranker ablation reports include case-group best-mode summaries plus query-paired candidate-vs-baseline win rates, metric deltas, confidence intervals, and rank metrics that penalize missing targets as `top_k + 1`, making it clear which signal wins on subsets such as VLM object probes rather than only on the aggregate benchmark. `gate-retrieval-ablation` turns a selected ablation mode into a pass/fail gate using absolute thresholds, baseline lift, target rank limits, target-type coverage, source-family and exact source coverage, metadata case-group coverage, hard-negative excluded-target hit rates, source-family and exact-source excluded-target hit rates, strategy/role contribution metrics, best-mode requirements, latency limits, and query-paired baseline metrics when a baseline mode is supplied. Retrieval cases are JSONL:

Source, source-family, chunking-strategy, and retrieval-role metrics all include excluded-target counts and rates, so retrieval and ablation gates can reject a candidate when the wrong evidence is concentrated in a specific vector source, chunking strategy, or hierarchy role.

When retrieval is evaluated with `--repeat`, each case also records whether its top-k result signature stayed stable across repeats. `result_stability_rate` and `unstable_result_count` can be used by retrieval, chunking, and readiness gates so latency tests do not hide nondeterministic ranking behavior.

Pairwise ablation gates can also cap first-relevant-rank and mean-target-rank deltas, including bootstrap CI high bounds, so a candidate cannot pass by improving aggregate recall while pushing expected evidence deeper than the baseline.

Qdrant vector ablation modes include `text`, `caption`, `object`, `image`, `triple`, `text_caption`, `text_object`, `caption_object`, `text_triple`, `text_image`, `caption_image`, `all`, `all_with_object`, `all_with_triple`, `all_with_object_triple`, `text_caption_graph`, `text_object_graph`, `text_triple_graph`, `all_graph`, `all_with_object_graph`, `all_with_triple_graph`, and `all_with_object_triple_graph`. Object and triple modes use the text query encoder. Image modes require an `image_dense` record file and a compatible image-query encoder.

Hybrid retrieval commands accept repeatable `--fusion-weight source=weight` values. Sources can be exact names such as `qdrant:caption_dense` or families such as `qdrant`, `bm25`, `dense`, and `graph`. `sweep-qdrant-fusion` accepts repeatable `--weight-grid source=v1,v2` values plus `--fixed-fusion-weight` defaults and records every candidate's recall, target coverage, nDCG, MRR, mean and p95 latency, failed queries, hard-negative excluded target hit rates, source/family/strategy/role contamination maxima, eligibility failures, and selection score. Use `--max-mean-latency-ms`, `--max-p95-latency-ms`, `--max-excluded-target-hit-rate`, `--max-excluded-query-hit-rate`, `--max-excluded-hit-query-count`, `--max-source-excluded-target-hit-rate`, `--max-source-family-excluded-target-hit-rate`, `--max-chunk-strategy-excluded-target-hit-rate`, or `--max-retrieval-role-excluded-target-hit-rate` to reject candidates that are too slow or retrieve similar-but-wrong targets. Tune `--latency-weight`, `--p95-latency-weight`, `--excluded-query-hit-penalty`, `--excluded-target-hit-penalty`, `--source-excluded-target-hit-penalty`, `--source-family-excluded-target-hit-penalty`, `--chunk-strategy-excluded-target-hit-penalty`, or `--retrieval-role-excluded-target-hit-penalty` to down-rank slow or leaky candidates even when they meet the hard gate. The sweep also records query-paired win rates, metric deltas, rank deltas, latency deltas, and bootstrap confidence intervals among the top ranked candidates controlled by `--pairwise-top-k`, so a candidate can be reviewed by shared-query behavior instead of aggregate averages only. It also reports case-group recommendations, so subsets such as `case_source:visual_object_probe` or `case_source:visual_image_probe` can show a different best weight profile from the aggregate benchmark. `--reranker lexical` applies dependency-free overlap reranking to the fused candidate set, and `--reranker cross-encoder --reranker-model <model>` uses a model-backed reranker when the embeddings extra is installed; the selected reranker and `--rerank-top-k` depth are stored in the sweep metadata. `export-qdrant-retrieval-config` converts the selected global or case-group recommendation into a service-ready JSON config with collection name, package/BM25 artifact paths, vector names, Qdrant/BM25/graph fusion weights, top-k, query encoders, tokenizer settings, reranker settings, aggregate selection metrics, case-group metrics, and selected-vs-baseline pairwise evidence. `eval-qdrant-retrieval-config`, `qdrant-rag-context-config`, and `eval-qdrant-rag-context-config` reload the same reranker settings so the benchmark and service context path match the exported configuration without restating vector names, fusion weights, graph expansion, hierarchy collapse, tokenizer options, or rerank depth. Graph hits score exact subject, predicate, and object phrase matches above loose token overlap, which helps graph-style benchmark queries find the intended evidence chunk. When a triple carries visual asset provenance, graph retrieval can also resolve the chunk linked to that asset.

Exported Qdrant configs can include adaptive routes so specialized vectors are used only when the query asks for them. Add `--route-preset adaptive` to `export-qdrant-retrieval-config` to route visual/object terms such as color, symbols, legends, maps, charts, images, and photos to `object_dense`; relation/evidence terms such as cause, effect, goal, strategy, and graph to `text_dense` plus `triple_dense` with graph expansion; and all other queries to the selected default retrieval profile. `plan-ingestion-workflow` adds this preset automatically when the package recommendations include both object and triple vector families, so the exported service config, config-based retrieval evaluation, RAG context evaluation, and final readiness gate all validate the same routed profile. Config-based retrieval commands prepare the union of vectors required by the default profile and all routes, record per-query route decisions, expose `retrieval_route` and `retrieval_route_reason` case-group metrics for per-route gates, and keep reranker settings shared across routes.

```bash
chunking-docs export-qdrant-retrieval-config outputs/package/qdrant_fusion_sweep.json \
  --route-preset adaptive \
  --output outputs/package/qdrant_retrieval_config.routed.json
chunking-docs eval-qdrant-retrieval-config outputs/package/qdrant_retrieval_config.routed.json \
  outputs/package/retrieval_cases.jsonl \
  --location :memory: \
  --output outputs/package/qdrant_retrieval_config.routed_eval.json
```

Generate a benchmark skeleton from existing package targets, then edit the queries for the document family:

```bash
chunking-docs generate-retrieval-cases \
  --package-dir outputs/package \
  --chunks outputs/package/chunks.multimodal.jsonl \
  --query-mode question \
  --selection-strategy salience \
  --max-query-terms 3 \
  --max-target-query-overlap-ratio 0.9 \
  --max-target-query-overlap-terms 5 \
  --max-asset-cases-per-target 3 \
  --hard-negative-limit 1 \
  --visual-probe-limit 20 \
  --image-probe-limit 20 \
  --object-probe-limit 20 \
  --include-todo \
  --output outputs/package/retrieval_cases.skeleton.jsonl
```

`--chunks` can point at a candidate chunk file so the same benchmark drafting logic can be run against semantic, multimodal, object-aware, or hierarchical candidates. `--query-mode snippet` drafts queries from source text snippets. `--query-mode salient_terms` drafts harder keyword-style queries from document-frequency-weighted terms. `--query-mode question` wraps salient terms in a short natural-language question, which reduces direct target-text overlap while preserving the distinctive evidence terms needed for retrieval. `--selection-strategy salience` prioritizes targets with more distinctive text. `--max-page-cases-per-target`, `--max-chunk-cases-per-target`, `--max-asset-cases-per-target`, and `--max-triple-cases-per-target` bound target concentration during generation so one repeated page, asset, chunk, or relationship cannot dominate a benchmark draft. `--hard-negative-limit` attaches same-kind similar-but-wrong page, chunk, asset, or triple targets to generated cases through `excluded_*` fields; `--hard-negative-min-overlap-terms` controls how much target-text overlap is required before a candidate is considered a hard negative. `--visual-probe-limit` adds asset-targeted probe cases whose query terms come from linked visual captions, OCR text, or VLM summaries after removing terms already present in the linked non-visual chunk text; these cases are useful for measuring whether visual text actually improves retrieval. `--image-probe-limit` adds `visual_image_probe` cases for rendered visual assets and tags them with `target_vector=image_dense`, so Qdrant exact-source gates such as `qdrant:image_dense=0.5` can measure whether image vectors contribute evidence separately from caption or object vectors. `--object-probe-limit` adds separate `visual_object_probe` cases from structured VLM objects, detections, regions, visual elements, attributes, locations, and bbox-bearing metadata so object and visual-element output can be measured as its own case family. Object probes default to terms that are not already present in linked non-visual chunk text, which makes them better at isolating VLM object value; generated `Visual context:` blocks and `visual_asset_text` chunks do not suppress those probe terms. Use `--no-object-probe-visual-only` only when broad object-label coverage is preferred. Triple cases include visual asset targets when triples carry asset provenance, so generated benchmarks can measure graph and visual retrieval paths together. Duplicate query strings are merged by default so repeated tables, section labels, or graph triples become one case with multiple acceptable targets; use `--no-dedupe-queries` only when auditing duplicate behavior. Treat generated cases as reviewable drafts before using them as a benchmark gate.

```jsonl
{"query":"policy corridor near river","expected_pages":[12],"graph_expand":true}
{"query":"capital investment table","expected_chunk_ids":["chunk-id"]}
{"query":"map legend for station access","expected_asset_ids":["asset-id"],"excluded_asset_ids":["similar-but-wrong-asset-id"]}
{"query":"district connects to corridor","expected_triple_ids":["triple-id"],"graph_expand":true}
```

Use `excluded_pages`, `excluded_chunk_ids`, `excluded_asset_ids`, or `excluded_triple_ids` for hard-negative cases where a visually or lexically similar target must not appear in the top-k results. `eval-retrieval` reports excluded target hit counts and rates, including exact-source and source-family contamination metrics, and `gate-retrieval` can enforce them with `--max-excluded-target-hit-rate`, `--max-excluded-query-hit-rate`, `--max-excluded-hit-query-count`, `--max-source-excluded-target-hit-rate`, and `--max-source-family-excluded-target-hit-rate`.

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

BM25 uses the `mixed` tokenizer by default. It combines word tokens with CJK character n-grams, which helps retrieve compound terms that may appear without whitespace in PDF text or OCR output. Tokens keep repeated term frequencies by default so BM25 can distinguish a chunk that emphasizes a concept from one that only mentions it once; set `deduplicate=True` on `LexicalTokenizerConfig` only when compact manifests are more important than frequency-sensitive ranking. The lexical corpus includes chunk text plus visual asset captions, OCR text, and VLM summaries linked through `asset_ids` or `asset:` source refs, so visual-only labels can still recover their parent chunks. The package writes `bm25_tokens.json`, and PostgreSQL row export mirrors those tokens into `chunk_lexical_tokens` for reproducible lexical scoring and future search-service migration.

CLI commands that build or evaluate lexical artifacts expose the same option as `--deduplicate-tokens`.

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
  --strategy object_aware \
  --output outputs/package/chunks.object_aware.jsonl

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
  --candidate object_aware=outputs/package/chunks.object_aware.jsonl \
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
  --min-case-group-source-target-coverage case_source:visual_lexical_probe:bm25=0.7 \
  --min-visual-text-part-coverage-ratio 0.8 \
  --max-total-chunk-chars 1000000 \
  --min-retrieval-score-per-embedding-kchar 0.0008 \
  --min-retrieval-score-per-mean-latency-ms 0.0005 \
  --min-target-coverage-per-p95-latency-ms 0.0005 \
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

The `multimodal` strategy keeps semantic text chunks, appends bounded visual context from linked captions, OCR, VLM summaries, and structured VLM metadata, and adds separate visual asset text chunks. Visual links are resolved from both `asset_ids` and `asset:` source refs, so annotations can remain provenance-oriented while still contributing to embedding text. Text-bearing visual assets without a linked parent chunk are emitted as standalone visual chunks instead of being dropped. The `object_aware` strategy keeps the multimodal chunks and adds one `visual_object_text` chunk per normalized VLM object, detected region, or visual element, preserving object ID, label, attributes, bbox region, source field, feature type, parent chunk, and asset provenance so object probes can be measured through BM25 and dense text retrieval as well as `object_dense`. The `hierarchical` strategy emits coarse parent chunks plus fine child chunks with shared visual context, which supports experiments where broad queries should find a page or section while precise queries should retrieve a smaller evidence span. `--collapse-hierarchical` reports the parent as the final hit while preserving matched child chunks as evidence. Comparison output includes recall@k, MRR, target coverage@k, target nDCG@k, precision@k, target rank metrics, target-type coverage, exact-source target coverage, source-family target coverage, case-group exact-source and source-family target coverage, chunking-strategy coverage, retrieval-role coverage, linked visual text asset coverage, linked visual text part coverage, visual object chunk count, total chunk text, embedding text volume, retrieval-per-embedding-cost metrics, retrieval-per-latency metrics, latency, failed queries, chunk size issues, query-paired baseline deltas, paired bootstrap confidence intervals, and the best candidate by quality and retrieval behavior. Pairwise gate options such as `--min-pairwise-win-rate`, `--min-pairwise-target-coverage-lift`, `--min-pairwise-target-ndcg-lift`, `--max-pairwise-mean-target-rank-delta`, `--max-pairwise-target-rank-delta-ci-high`, and `--max-pairwise-mean-latency-delta-ms` help distinguish broad aggregate gains from stable wins on the same benchmark queries. Source contribution gates such as `--min-source-target-coverage`, `--min-source-family-target-coverage`, `--min-case-group-source-target-coverage`, and `--min-case-group-source-family-target-coverage` can require a selected chunking strategy to preserve lexical, dense, visual, or graph evidence for specific benchmark groups. Cost gates such as `--max-total-chunk-chars`, `--max-embedding-text-kchars`, `--min-retrieval-score-per-embedding-kchar`, `--min-retrieval-score-per-mean-latency-ms`, and `--min-target-coverage-per-p95-latency-ms` keep a larger or slower chunking strategy from winning merely by sending much more text to the embedding/index path or taking longer to retrieve.

Run a parameter sweep when choosing defaults:

```bash
chunking-docs sweep-chunking \
  --package-dir outputs/package \
  --strategies semantic,multimodal,object_aware,hierarchical \
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
  --selection-min-visual-text-part-coverage-ratio 0.8 \
  --selection-min-target-type-coverage asset=0.9 \
  --selection-min-case-group-target-coverage case_source:visual_object_probe=0.7 \
  --selection-max-mean-target-rank 3 \
  --selection-max-mean-latency-ms 150 \
  --selection-max-p95-latency-ms 250 \
  --selection-max-total-chunk-chars 1000000 \
  --selection-min-target-coverage-per-embedding-kchar 0.0008 \
  --selection-min-retrieval-score-per-p95-latency-ms 0.002 \
  --selection-max-visual-object-chunk-count 200 \
  --cases examples/retrieval_cases.jsonl \
  --output outputs/package/chunking_sweep.json
```

Apply the recommended candidate after reviewing the sweep and gates:

```bash
chunking-docs apply-chunking-sweep \
  --package-dir outputs/package \
  --report outputs/package/chunking_sweep.json
```

The sweep writes candidate chunk files under `outputs/package/chunking_sweep/` and ranks them with the same quality, recall@k, MRR, target coverage@k, target nDCG@k, precision@k, target rank, target-type coverage, source-family target coverage, chunking-strategy coverage, retrieval-role coverage, linked visual text asset and part coverage, latency, and failed-query metrics used by `compare-chunking`. It also emits a `selection` block with weighted scores, eligibility failures, eligible counts, a recommendation, and a Pareto front so a strategy that improves retrieval can be checked against hard recall, target-rank, mean/p95 latency, visual-text asset coverage, visual-text part coverage, target-type, source-family, case-group, chunk-count, chunk-length, standalone visual chunk, embedding text-volume, retrieval-per-embedding-cost, and retrieval-per-latency constraints before becoming the default. Pareto dominance treats mean and p95 latency, total chunk text, mean and p95 chunk length, embedding text volume, and standalone visual chunk count as cost axes, so a candidate with slightly better retrieval does not hide a much larger or slower embedding payload. `apply-chunking-sweep` promotes the selected candidate into `chunks.jsonl`, remaps triples to available chunk IDs, backs up the previous chunk and triple files, updates package chunking metadata, rebuilds BM25 tokens, and clears stale Qdrant record, collection, and embedding manifest artifacts by default. Run `embed-package` afterward for real dense/vector artifacts, or pass `--rebuild-dry-run-embeddings` only when deterministic hashing artifacts are enough for a dry run.

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

The report records source-file checksum, package settings, package artifact checksums, JSONL record counts, BM25 tokenizer settings, Qdrant named-vector configuration, VLM experiment plan summaries, runtime doctor variants, exported Qdrant retrieval config variants, Qdrant fusion case-group recommendations, readiness, evaluation, audit, gate artifact variants, visual run comparison summaries, chunking sweep recommendations, top-level and component-level validation pass/fail summaries, chunking quality metrics, linked visual text coverage, retrieval recall@k, MRR, target coverage@k, target nDCG@k, target rank, precision@k, latency, failed queries, diagnostics reason counts, case-group diagnostics such as visual object probe failures, paired confidence metrics from chunking gates, retrieval-per-embedding-cost metrics, retrieval-per-latency metrics, and the best candidate by retrieval behavior. This makes chunking changes reviewable and repeatable before new defaults are adopted.

## Development Checks

```bash
ruff check src tests
pytest
```

## Design Notes

- [Architecture](docs/architecture.md)
