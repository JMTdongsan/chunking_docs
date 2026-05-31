# Architecture

`chunking_docs` prepares complex PDFs for retrieval-augmented generation. The package keeps document-specific assumptions in external data files so the core library can be reused across technical reports, manuals, scanned PDFs, and visual-heavy PDFs.

## Pipeline

0. **Runtime Preflight**
   - Inspect optional dependencies for Qdrant, PostgreSQL, text embeddings, OCR, and VLM backends.
   - Detect visible NVIDIA GPUs and Torch CUDA availability when GPU-backed runs are required.
   - Fail early when requested runtime capabilities are missing.

1. **Document Intake**
   - Download or load a PDF.
   - Generate a stable `doc_id` from file content.
   - Store source metadata and local path.

2. **Page Profiling**
   - Measure page size, text length, text blocks, image blocks, embedded images, and drawing count.
   - Classify the text layer as `good`, `degraded`, or `empty`.
   - Use the profile to decide which pages need OCR, VLM summaries, or visual embeddings.
   - Summarize package characteristics and recommended next processing steps for chunking, OCR/VLM, graph, embeddings, and retrieval benchmarks.

3. **Section Mapping**
   - Accept optional section ranges as JSON or JSONL.
   - Attach `chapter`, `section`, `subsection`, and `issue` metadata to chunks.
   - Keep section maps outside package code because each document family has different structure.

4. **Starter Chunking**
   - Create page-level chunks as the first stable unit.
   - Good text-layer pages receive cleaned PDF text.
   - Degraded or empty text-layer pages receive an explicit placeholder that marks OCR/VLM requirements.

5. **Visual Assets**
   - Render pages that have weak text layers or visual density.
   - Optionally render overlapping page tiles for dense maps, tables, and diagrams.
   - Store each rendered image in `assets.jsonl` with page, kind, path, and processing hints.
   - Link visual assets back to chunks through `asset_ids`.

6. **Structured Tables**
   - Detect PDF tables with the document text/layout layer when available.
   - Reject table text with excessive control-character noise from broken encodings.
   - Store each detected table as a `table` chunk with Markdown table text.
   - Store each detected table as a `table` visual asset with bbox, rendered clip, and caption text.
   - Keep table extraction generic so document-specific schemas remain external data.

7. **OCR/VLM Job Planning**
   - Build `visual_jobs.jsonl` from missing OCR/VLM annotations.
   - Prioritize maps, tables, charts, figures, and pages with empty text.
   - Filter jobs by page range or asset kind to run bounded OCR/VLM batches.
   - Run jobs in bounded batches and store `visual_job_results.jsonl` plus `visual_annotations.jsonl`.
   - Parse structured VLM JSON into captions, summaries, metadata, and triple candidates.
   - Attach asset, page, job, source, and prompt provenance to visual-derived triples.
   - Record OCR language, backend configuration, VLM prompt name, prompt hash, latency, output size, parse status, and triple count.
   - Summarize visual job results by status, backend latency, output size, VLM prompt usage, parse status, and triple count.
   - Compare multiple OCR/VLM runs by completion, annotation coverage, parse rate, triple density, and latency.
   - Write VLM experiment plans so several profiles can be run against the same visual job set and compared afterward.
   - Gate visual runs by completion rate, OCR text coverage, VLM summary coverage, JSON parse rate, triple density, and failure counts.
   - Apply annotations back into chunks, assets, graph triples, BM25, and Qdrant records.
   - Compare before/after package directories to verify how annotations changed chunks, assets, graph triples, and vector records.

8. **Semantic Splitting**
   - Split long annotated chunks into subchunks using paragraph and line boundaries.
   - Preserve original chunk IDs when a chunk does not need splitting.
   - Remap triples to available child chunks when splitting changes IDs.

9. **Strategy Variants**
   - `page`: baseline page chunks with optional context prefix.
   - `semantic`: boundary-aware subchunks for long text.
   - `multimodal`: semantic chunks with bounded linked visual context plus visual asset text chunks from captions, OCR, and VLM summaries.
   - `hierarchical`: coarse parent chunks plus fine child chunks that share page, section, and visual context.
   - `compare-chunking` evaluates candidate files with the same benchmark cases.
   - `sweep-chunking` generates a strategy and parameter grid, writes candidate chunk files, and ranks the results.

10. **Embedding Artifacts**
   - `text_dense`: chunk text, OCR text, and VLM summaries.
   - `caption_dense`: asset caption, OCR, and VLM summary text.
   - `image_dense`: rendered page or visual asset image.
   - Default hashing embedders make the pipeline testable without model downloads.
   - `embed-package` regenerates artifacts with model-backed text and image embedders.
   - `embedding_manifest.json` records vector files, dimensions, counts, and checksums.

11. **Lexical Search**
    - BM25 is generated from chunk text.
    - Lexical search protects exact matches for names, identifiers, dates, codes, and policy terms.
    - Tokenization is configurable as `word`, `char_ngram`, or `mixed`.
    - The default `mixed` tokenizer adds CJK character n-grams so compound terms without whitespace remain retrievable.
    - Dense and lexical results are combined with Reciprocal Rank Fusion in the local evaluator.

12. **Graph Triples**
    - Section metadata creates baseline graph relationships.
    - OCR/VLM JSON or external annotations can add `subject, predicate, object` triples.
    - Triple normalization canonicalizes labels, predicate names, and stable IDs.
    - Triple audit flags duplicates, orphan chunk references, empty fields, and invalid confidence values.
    - Graph terms are used for query expansion and relationship browsing.

13. **Storage**
    - Qdrant stores named vectors and payloads.
    - PostgreSQL stores normalized document, page, chunk, asset, triple, and embedding artifact metadata.
    - BM25 can remain as a local manifest or be replaced by a dedicated lexical search service.

14. **Ingestion Readiness**
    - Combine package audit, required artifact checks, Qdrant record validation, and PostgreSQL row conversion.
    - Optionally include retrieval case audit, visual quality gates, chunking comparison gates, and retrieval quality gates before a package is loaded into serving systems.
    - Emit a single pass/fail report for CI, portfolio review, or deployment handoff.

15. **RAG Context Assembly**
    - Convert retrieval hits into a structured context bundle.
    - Support both local hybrid retrieval and Qdrant hybrid retrieval.
    - Include hit chunks, optional neighboring chunks, hierarchical evidence chunks, linked visual assets, and graph triples.
    - Keep page ranges, section labels, source refs, scores, and retrieval sources available for citation.

## Package Files

`chunking-docs package` writes a local processing package:

- `manifest.json`
- `pages.jsonl`
- `chunks.jsonl`
- `assets.jsonl`
- `triples.jsonl`
- `bm25_tokens.json`
- `embedding_manifest.json`
- `qdrant_text_records.jsonl`
- `qdrant_image_records.jsonl`
- `qdrant_caption_records.jsonl`
- `qdrant_collection.json`

Additional processing commands may create:

- `visual_jobs.jsonl`
- `visual_job_results.jsonl`
- `visual_job_summary.json`
- `visual_run_comparison.json`
- `visual_quality.json`
- `visual_annotations.jsonl`
- `assets.tiled.jsonl`
- `chunks.tiled.jsonl`
- `triples.normalized.jsonl`
- `graph_triple_quality.json`
- `chunks.split.jsonl`
- `chunks.semantic.jsonl`
- `chunks.multimodal.jsonl`
- `chunks.hierarchical.jsonl`
- `chunking_comparison.json`
- `chunking_comparison_gate.json`
- `chunking_sweep.json`
- `chunking_sweep/chunks.*.jsonl`
- `graph_nodes.jsonl`
- `graph_edges.jsonl`
- `experiment_report.json`
- `document_characteristics.json`
- `ingestion_readiness.json`
- `package_delta.json`
- `postgres_schema_contract.json`
- `qdrant_collection_contract.json`
- `qdrant_retrieval_eval.json`
- `qdrant_vector_ablation.json`
- `qdrant_vector_ablation_gate.json`
- `retrieval_case_audit.json`
- `retrieval_diagnostics.json`
- `retrieval_gate.json`
- `retrieval_ablation.json`
- `retrieval_cases.skeleton.jsonl`
- `rag_context.json`
- `rag_context.qdrant.json`

## Qdrant Design

The default collection is `document_chunks`, but callers can choose another collection name.

Named vectors:

- `text_dense`
- `image_dense`
- `caption_dense`

Payload fields include document ID, chunk ID, asset ID, page range, asset kind, section metadata, source references, and text fields needed for answer citation.

The package writes payload index definitions with field schemas. Qdrant ingestion and package query commands apply those definitions so metadata filters such as document ID, asset ID, page, and section remain efficient on server-backed collections.

`qdrant-check-collection` compares a live or local Qdrant collection against `qdrant_collection.json` before upsert. It detects missing named vectors, vector dimension mismatches, and missing payload indexes, which is especially important when embedding models or vector dimensions change between experiments.

Qdrant search commands accept repeatable payload filters using exact and range forms such as `kind=map`, `page_no=12`, `page_start<=12`, and `page_end>=12`.

The Qdrant adapter supports both ingestion and named-vector querying. `qdrant-search-package` can upsert a package into qdrant-client local mode and immediately query `text_dense` or `caption_dense`, which keeps retrieval checks reproducible without requiring a running server.

`qdrant-hybrid-search` queries Qdrant named vectors, BM25, and optional graph expansion, then fuses results with Reciprocal Rank Fusion. Caption vector hits from visual assets are mapped back to their parent chunks so text and visual evidence can be ranked together. Optional reranking can reorder fused candidates with lexical overlap or a sentence-transformers CrossEncoder.

Image vectors may use a different embedding space from text vectors. When querying `image_dense`, the searcher can use a per-vector query encoder, such as CLIP text features for CLIP image embeddings, while continuing to use the document text embedder for `text_dense` and `caption_dense`.

Qdrant query paths validate query encoder dimensions against the package collection contract before search. This catches mismatches between package-time embedding models and retrieval-time query encoders before Qdrant executes vector math, and it surfaces the package vector notes so operators can choose the matching model or rebuild the package. Search, evaluation, ablation, and RAG context outputs also record per-vector query encoder details for reproducibility.

Hierarchical chunk files can be searched with parent collapse enabled. In that mode, dense, BM25, graph, or Qdrant hits against fine child chunks are grouped under the coarse parent chunk while the matched child IDs remain attached as evidence. Retrieval evaluation also treats `source_chunk_id` and `parent_chunk_id` metadata as aliases for expected chunk targets, so benchmark cases and graph triples attached to the original chunk IDs remain valid across semantic, multimodal, and hierarchical candidates. This keeps answer context broad enough for citation while preserving the precise span that triggered retrieval.

## PostgreSQL Design

PostgreSQL is used for provenance and relational queries, not as the default vector store.

Tables:

- `documents`
- `pages`
- `chunks`
- `assets`
- `triples`
- `embedding_artifacts`

The writer upserts in dependency order: documents, pages, chunks, assets, triples, embedding artifacts. The `embedding_artifacts` table stores vector file names, dimensions, counts, checksums, Qdrant collection names, and payload index metadata from `embedding_manifest.json`; vector values remain in Qdrant record files and Qdrant itself.

`postgres-check-schema` validates the live PostgreSQL schema before upsert. It checks required tables, columns, column types, and the pgvector extension so metadata ingestion failures are caught before batch writes.

## Retrieval Evaluation

Chunking changes should be judged by retrieval behavior, not only by successful execution.

Recommended checks:

- `audit-package`: structural completeness, orphan checks, OCR/VLM gaps, Qdrant vector dimensions, required payload fields, payload index definitions, and embedding manifest count/checksum consistency.
- `qdrant-check-collection`: live Qdrant collection contract validation for named-vector dimensions and payload indexes.
- `postgres-check-schema`: live PostgreSQL schema contract validation for required extensions, tables, columns, and column types.
- `eval-chunking`: page coverage, chunk size distribution, section coverage, visual linkage, annotation coverage, retrieval recall@k, MRR, target coverage@k, target nDCG@k, precision@k, latency, failed queries, and aggregate quality score.
- `audit-retrieval-cases`: benchmark case validation for empty or TODO queries, unknown page/chunk/asset/triple targets, duplicate queries, graph-expansion hints, and target-family coverage.
- `eval-retrieval`: focused top-k retrieval benchmark cases with optional repeated latency sampling, target-specific page/chunk/asset/triple metrics, and source-family contribution metrics.
- `generate-retrieval-cases`: benchmark draft generation from package pages, candidate chunk files, visual assets, and graph triples, with snippet or document-frequency-weighted salient-term query modes.
- `diagnose-retrieval`: failure, partial-coverage, low-ranking, and low-precision analysis for retrieval evaluation JSON outputs.
- `eval-qdrant-retrieval`: the same benchmark cases against Qdrant named vectors plus BM25 and optional graph expansion.
- `eval-qdrant-vector-ablation`: Qdrant text, visual caption, optional image, and graph-expanded vector comparison on the same cases.
- `gate-qdrant-vector-ablation`: pass/fail checks for a selected Qdrant vector mode using recall, target coverage, target nDCG, precision, failed-query count, latency, target-type coverage, source-family target coverage, and optional best-mode requirements.
- `ingestion-readiness`: final pre-ingestion gate that can combine package audit results, storage artifacts, PostgreSQL row conversion, visual quality, retrieval gates, chunking comparison gates with target-type and source-family coverage, and selected Qdrant vector ablation gates.
- `compare-visual-runs`: OCR/VLM run comparison by coverage, structured parse rate, graph triple density, and latency.
- `plan-vlm-experiments`: reproducible profile-by-profile command recipes for running the same visual job set through multiple VLMs.
- `eval-retrieval-ablation`: dense-only, BM25-only, graph-only, hybrid, and graph-expanded hybrid comparison on the same cases, including target coverage@k, target nDCG@k, and latency.
- `gate-retrieval`: pass/fail checks for absolute metric floors, target-type coverage, source-family target coverage, and baseline regression limits such as recall drop, target coverage drop, target nDCG drop, precision drop, and latency ratio.
- `compare-packages`: before/after package comparison for count deltas, changed chunk/asset/triple IDs, Qdrant record count deltas, and annotation-related observations.
- `compare-chunking`: side-by-side strategy comparison by quality score, recall@k, MRR, target coverage@k, target nDCG@k, precision@k, target-type coverage, source-family target coverage, latency, and failed queries.
- `gate-chunking-comparison`: pass/fail checks for selected chunking candidates using quality, page coverage, retrieval floors, target-type coverage, source-family target coverage, failed-query limits, and baseline regression limits.
- `sweep-chunking`: parameter grid generation for max size, overlap, parent size, and multimodal or hierarchical visual context size.
- `write-experiment-report`: reproducible package report with artifact checksums, record counts, tokenizer settings, Qdrant configuration, readiness, evaluation, audit, and gate artifact variants, top-level and component-level validation pass/fail summaries, and candidate comparison metrics.
- Qdrant local mode upsert: validates named vector records and payloads.

Benchmark cases should be maintained per document family. A useful case specifies the query, expected page, chunk, visual asset, graph triple, and whether graph expansion should be enabled.

Tokenizer settings are part of the retrieval experiment. Strategy comparisons should keep the tokenizer fixed unless the experiment is explicitly measuring lexical tokenization.

Fusion weights are also part of the retrieval experiment. Use `--fusion-weight` to tune source families such as `dense`, `bm25`, `graph`, and `qdrant`, or exact sources such as `qdrant:caption_dense`.

Reranking is a separate experiment knob. Keep `--reranker`, `--rerank-top-k`, and the reranker model fixed when comparing chunking strategies unless the experiment is explicitly measuring reranking.

Use repeated retrieval evaluation when comparing strategies whose recall is similar. The latency fields are intended to show whether higher recall comes with an acceptable retrieval cost.

For hierarchical candidates, enable parent collapse during `eval-retrieval`, `compare-chunking`, or `write-experiment-report` when the benchmark expects page-level or parent-level citation behavior.

## Model Strategy

The library exposes interfaces instead of locking in one model:

- OCR: `TesseractOCRBackend` and `PaddleOCRBackend` for multilingual scanned pages.
- VLM: `HuggingFaceVLMBackend` with configurable device map, torch dtype, generation length, and optional attention implementation.
- VLM profiles: named Hugging Face profiles record the model id, loader family, dtype, and generation defaults for reproducible local model comparisons.
- Text dense: `SentenceTransformerTextEmbedder`.
- Image dense: `TransformersImageEmbedder`.

Local GPUs can be used for VLM summaries and image embedding batches. Failed jobs remain visible in job result files so experiments can be retried safely. Prompt hashes and backend configuration fields make model comparisons reproducible without embedding document-specific rules in source code.
