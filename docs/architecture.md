# Architecture

`chunking_docs` prepares complex PDFs for retrieval-augmented generation. The package keeps document-specific assumptions in external data files so the core library can be reused across technical reports, manuals, scanned PDFs, and visual-heavy PDFs.

## Pipeline

1. **Document Intake**
   - Download or load a PDF.
   - Generate a stable `doc_id` from file content.
   - Store source metadata and local path.

2. **Page Profiling**
   - Measure page size, text length, text blocks, image blocks, embedded images, and drawing count.
   - Classify the text layer as `good`, `degraded`, or `empty`.
   - Use the profile to decide which pages need OCR, VLM summaries, or visual embeddings.

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
   - Store each detected table as a `table` chunk with Markdown table text.
   - Store each detected table as a `table` visual asset with bbox, rendered clip, and caption text.
   - Keep table extraction generic so document-specific schemas remain external data.

7. **OCR/VLM Job Planning**
   - Build `visual_jobs.jsonl` from missing OCR/VLM annotations.
   - Prioritize maps, tables, charts, figures, and pages with empty text.
   - Filter jobs by page range or asset kind to run bounded OCR/VLM batches.
   - Run jobs in bounded batches and store `visual_job_results.jsonl` plus `visual_annotations.jsonl`.
   - Parse structured VLM JSON into captions, summaries, metadata, and triple candidates.
   - Record OCR language, backend configuration, VLM prompt name, prompt hash, latency, output size, parse status, and triple count.
   - Summarize visual job results by status, backend latency, output size, VLM prompt usage, parse status, and triple count.
   - Apply annotations back into chunks, assets, graph triples, BM25, and Qdrant records.

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

14. **RAG Context Assembly**
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
- `visual_annotations.jsonl`
- `assets.tiled.jsonl`
- `chunks.tiled.jsonl`
- `triples.normalized.jsonl`
- `graph_triple_quality.json`
- `chunks.split.jsonl`
- `chunks.semantic.jsonl`
- `chunks.multimodal.jsonl`
- `chunks.hierarchical.jsonl`
- `chunking_sweep.json`
- `chunking_sweep/chunks.*.jsonl`
- `graph_nodes.jsonl`
- `graph_edges.jsonl`
- `experiment_report.json`
- `qdrant_retrieval_eval.json`
- `qdrant_vector_ablation.json`
- `retrieval_diagnostics.json`
- `retrieval_gate.json`
- `retrieval_ablation.json`
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

Qdrant search commands accept repeatable payload filters using exact and range forms such as `kind=map`, `page_no=12`, `page_start<=12`, and `page_end>=12`.

The Qdrant adapter supports both ingestion and named-vector querying. `qdrant-search-package` can upsert a package into qdrant-client local mode and immediately query `text_dense` or `caption_dense`, which keeps retrieval checks reproducible without requiring a running server.

`qdrant-hybrid-search` queries Qdrant named vectors, BM25, and optional graph expansion, then fuses results with Reciprocal Rank Fusion. Caption vector hits from visual assets are mapped back to their parent chunks so text and visual evidence can be ranked together. Optional reranking can reorder fused candidates with lexical overlap or a sentence-transformers CrossEncoder.

Image vectors may use a different embedding space from text vectors. When querying `image_dense`, the searcher can use a per-vector query encoder, such as CLIP text features for CLIP image embeddings, while continuing to use the document text embedder for `text_dense` and `caption_dense`.

Hierarchical chunk files can be searched with parent collapse enabled. In that mode, dense, BM25, graph, or Qdrant hits against fine child chunks are grouped under the coarse parent chunk while the matched child IDs remain attached as evidence. This keeps answer context broad enough for citation while preserving the precise span that triggered retrieval.

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

## Retrieval Evaluation

Chunking changes should be judged by retrieval behavior, not only by successful execution.

Recommended checks:

- `audit-package`: structural completeness, orphan checks, OCR/VLM gaps, Qdrant vector dimensions, required payload fields, and payload index definitions.
- `eval-chunking`: page coverage, chunk size distribution, section coverage, visual linkage, annotation coverage, retrieval recall@k, MRR, target coverage@k, target nDCG@k, precision@k, latency, failed queries, and aggregate quality score.
- `eval-retrieval`: focused top-k retrieval benchmark cases with optional repeated latency sampling, target-specific page/chunk/asset/triple metrics, and source-family contribution metrics.
- `diagnose-retrieval`: failure, partial-coverage, low-ranking, and low-precision analysis for retrieval evaluation JSON outputs.
- `eval-qdrant-retrieval`: the same benchmark cases against Qdrant named vectors plus BM25 and optional graph expansion.
- `eval-qdrant-vector-ablation`: Qdrant text, visual caption, optional image, and graph-expanded vector comparison on the same cases.
- `eval-retrieval-ablation`: dense-only, BM25-only, graph-only, hybrid, and graph-expanded hybrid comparison on the same cases, including target coverage@k, target nDCG@k, and latency.
- `gate-retrieval`: pass/fail checks for absolute metric floors and baseline regression limits such as recall drop, target coverage drop, target nDCG drop, precision drop, and latency ratio.
- `compare-chunking`: side-by-side strategy comparison by quality score, recall@k, MRR, target coverage@k, target nDCG@k, precision@k, latency, and failed queries.
- `sweep-chunking`: parameter grid generation for max size, overlap, parent size, and multimodal or hierarchical visual context size.
- `write-experiment-report`: reproducible package report with artifact checksums, record counts, tokenizer settings, Qdrant configuration, and candidate comparison metrics.
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
- Text dense: `SentenceTransformerTextEmbedder`.
- Image dense: `TransformersImageEmbedder`.

Local GPUs can be used for VLM summaries and image embedding batches. Failed jobs remain visible in job result files so experiments can be retried safely. Prompt hashes and backend configuration fields make model comparisons reproducible without embedding document-specific rules in source code.
