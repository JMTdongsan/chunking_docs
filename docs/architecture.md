# Architecture

`chunking_docs` prepares complex PDFs for retrieval-augmented generation. The package keeps document-specific assumptions in external data files so the core library can be reused across planning documents, reports, manuals, scanned PDFs, and visual-heavy PDFs.

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
   - Store each rendered image in `assets.jsonl` with page, kind, path, and processing hints.
   - Link visual assets back to chunks through `asset_ids`.

6. **OCR/VLM Job Planning**
   - Build `visual_jobs.jsonl` from missing OCR/VLM annotations.
   - Prioritize maps, tables, charts, figures, and pages with empty text.
   - Run jobs in bounded batches and store `visual_job_results.jsonl` plus `visual_annotations.jsonl`.
   - Apply annotations back into chunks, assets, graph triples, BM25, and Qdrant records.

7. **Semantic Splitting**
   - Split long annotated chunks into subchunks using paragraph and line boundaries.
   - Preserve original chunk IDs when a chunk does not need splitting.
   - Remap triples to available child chunks when splitting changes IDs.

8. **Strategy Variants**
   - `page`: baseline page chunks with optional context prefix.
   - `semantic`: boundary-aware subchunks for long text.
   - `multimodal`: semantic chunks plus visual asset text chunks from captions, OCR, and VLM summaries.
   - `compare-chunking` evaluates candidate files with the same benchmark cases.

9. **Embedding Artifacts**
   - `text_dense`: chunk text, OCR text, and VLM summaries.
   - `caption_dense`: asset caption, OCR, and VLM summary text.
   - `image_dense`: rendered page or visual asset image.
   - Default hashing embedders make the pipeline testable without model downloads.
   - `embed-package` regenerates artifacts with model-backed text and image embedders.

10. **Lexical Search**
    - BM25 is generated from chunk text.
    - Lexical search protects exact matches for names, identifiers, dates, codes, and policy terms.
    - Tokenization is configurable as `word`, `char_ngram`, or `mixed`.
    - The default `mixed` tokenizer adds CJK character n-grams so compound terms without whitespace remain retrievable.
    - Dense and lexical results are combined with Reciprocal Rank Fusion in the local evaluator.

11. **Graph Triples**
    - Section metadata creates baseline graph relationships.
    - OCR/VLM or external annotations can add `subject, predicate, object` triples.
    - Graph terms are used for query expansion and relationship browsing.

12. **Storage**
    - Qdrant stores named vectors and payloads.
    - PostgreSQL stores normalized document, page, chunk, asset, and triple metadata.
    - BM25 can remain as a local manifest or be replaced by a dedicated lexical search service.

## Package Files

`chunking-docs package` writes a local processing package:

- `manifest.json`
- `pages.jsonl`
- `chunks.jsonl`
- `assets.jsonl`
- `triples.jsonl`
- `bm25_tokens.json`
- `qdrant_text_records.jsonl`
- `qdrant_image_records.jsonl`
- `qdrant_caption_records.jsonl`
- `qdrant_collection.json`

Additional processing commands may create:

- `visual_jobs.jsonl`
- `visual_job_results.jsonl`
- `visual_annotations.jsonl`
- `chunks.split.jsonl`
- `chunks.semantic.jsonl`
- `chunks.multimodal.jsonl`
- `graph_nodes.jsonl`
- `graph_edges.jsonl`

## Qdrant Design

The default collection is `planning_chunks`, but callers can choose another collection name.

Named vectors:

- `text_dense`
- `image_dense`
- `caption_dense`

Payload fields include document ID, chunk ID, asset ID, page range, asset kind, section metadata, source references, and text fields needed for answer citation.

## PostgreSQL Design

PostgreSQL is used for provenance and relational queries, not as the default vector store.

Tables:

- `documents`
- `pages`
- `chunks`
- `assets`
- `triples`

The writer upserts in dependency order: documents, pages, chunks, assets, triples.

## Retrieval Evaluation

Chunking changes should be judged by retrieval behavior, not only by successful execution.

Recommended checks:

- `audit-package`: structural completeness and orphan checks.
- `eval-chunking`: page coverage, chunk size distribution, section coverage, visual linkage, annotation coverage, retrieval recall@k, MRR, failed queries, and aggregate quality score.
- `eval-retrieval`: focused top-k retrieval benchmark cases.
- `compare-chunking`: side-by-side strategy comparison by quality score, recall@k, MRR, and failed queries.
- Qdrant local mode upsert: validates named vector records and payloads.

Benchmark cases should be maintained per document family. A useful case specifies the query, expected page or chunk, and whether graph expansion should be enabled.

Tokenizer settings are part of the retrieval experiment. Strategy comparisons should keep the tokenizer fixed unless the experiment is explicitly measuring lexical tokenization.

## Model Strategy

The library exposes interfaces instead of locking in one model:

- OCR: `TesseractOCRBackend`, with room for PaddleOCR or EasyOCR adapters.
- VLM: `HuggingFaceVLMBackend`.
- Text dense: `SentenceTransformerTextEmbedder`.
- Image dense: `TransformersImageEmbedder`.

Local GPUs can be used for VLM summaries and image embedding batches. Failed jobs remain visible in job result files so experiments can be retried safely.
