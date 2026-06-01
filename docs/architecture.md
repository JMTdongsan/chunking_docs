# Architecture

`chunking_docs` prepares complex PDFs for retrieval-augmented generation. The package keeps document-specific assumptions in external data files so the core library can be reused across technical reports, manuals, scanned PDFs, and visual-heavy PDFs.

## Pipeline

0. **Runtime Preflight**
   - Inspect optional dependencies for Qdrant, PostgreSQL, text embeddings, OCR, and VLM backends.
   - Detect visible NVIDIA GPUs, Torch CUDA availability, device names, compute capability, CUDA version, compiled architecture targets, and bfloat16 support when GPU-backed runs are required.
   - Check that the Torch CUDA build includes an architecture target for the visible GPU before GPU-backed embedding or VLM batches.
   - Fail early when requested runtime capabilities are missing, including VLM profile GPU-memory fit, and warn when the configured VLM memory safety margin is not met.
   - Audit public repository files for forbidden text patterns, blocked generated artifact extensions, oversized files, and required `.gitignore` protections before publishing.

1. **Document Intake**
   - Download or load a PDF.
   - Generate a stable `doc_id` from file content.
   - Store source metadata and local path.
   - Record source file name, byte count, and SHA-256 in `manifest.json` for reproducible packages.

2. **Page Profiling**
   - Measure page size, text length, text blocks, image blocks, embedded images, and drawing count.
   - Classify the text layer as `good`, `degraded`, or `empty` using language-neutral control-character and readable-character signals.
   - Use the profile to decide which pages need OCR, VLM summaries, or visual embeddings.
   - Identify dense visual pages that should be rendered as overlapping tiles before OCR/VLM processing.
   - Summarize package characteristics and recommended next processing steps for chunking, OCR/VLM, VLM object probes, graph, embeddings, and retrieval benchmarks, including missing object or triple vector families when the source metadata exists.

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
   - Preserve tile parent, grid, and text-quality metadata in jobs, and prioritize tile crops for dense pages when batch limits are used.
   - Filter jobs by page range or asset kind to run bounded OCR/VLM batches.
   - Run jobs in bounded batches and store `visual_job_results.jsonl` plus `visual_annotations.jsonl`.
   - Parse structured VLM JSON into captions, summaries, metadata, normalized object detections, and triple candidates.
   - Preserve object attributes, descriptions, locations, bbox coordinates, confidence, and source-field provenance for retrieval text and graph qualifiers.
   - Lift entities, visual elements, and detected objects into derived graph triples when explicit relationships are missing.
   - Attach asset, page, job, source, and prompt provenance to visual-derived triples.
   - Record OCR language, backend configuration, VLM prompt name, prompt hash, latency, output size, parse status, object counts, bbox counts, and triple counts.
   - Summarize visual job results by status, backend latency, output size, VLM prompt usage, parse status, object counts, and triple count.
   - Compare multiple OCR/VLM runs by completion, annotation coverage, parse rate, object coverage, triple density, latency, and whether the same visual job IDs were used.
   - Write VLM experiment plans so several profiles can be run against the same visual job set and compared afterward, including selected job counts, operation mix, per-profile runtime checks, and generation-token upper bounds for local GPU sizing.
   - Gate visual runs by completion rate, OCR text coverage, VLM summary coverage, JSON parse rate, object coverage, bbox coverage, triple density, and failure counts.
   - Apply annotations back into chunks, assets, graph triples, BM25, and Qdrant records through direct asset links and `asset:` source refs.
   - Preserve compact asset scope, tile parent/grid, text-quality, and OCR/VLM work metadata on visual text chunks so text, caption, and image vectors share the same filter contract.
   - Compare before/after package directories to verify how annotations changed chunks, assets, graph triples, and vector records.

8. **Semantic Splitting**
   - Split long annotated chunks into subchunks using paragraph, report outline, list, sentence, line, and word boundaries.
   - Preserve original chunk IDs when a chunk does not need splitting.
   - Remap triples to available child chunks when splitting changes IDs.

9. **Strategy Variants**
   - `page`: baseline page chunks with optional context prefix.
   - `semantic`: boundary-aware subchunks for long text.
   - `multimodal`: semantic chunks with bounded linked visual context plus visual asset text chunks from captions, OCR, VLM summaries, and structured VLM metadata. Visual links can come from `asset_ids` or `asset:` source refs; text-bearing assets without a linked parent become standalone visual chunks.
   - `object_aware`: multimodal chunks plus one visual object text chunk per normalized VLM object or detected region, preserving object ID, label, attributes, location, bbox region, source field, asset provenance, and parent chunk linkage.
   - `hierarchical`: coarse parent chunks plus fine child chunks that share page, section, and visual context resolved from the same asset provenance, with standalone visual chunks for unlinked asset text.
   - `compare-chunking` evaluates candidate files with the same benchmark cases.
   - `sweep-chunking` generates a strategy and parameter grid, writes candidate chunk files, ranks the results, and reports eligibility-filtered recommendations plus Pareto-efficient candidates across retrieval quality, target-type coverage, case-group coverage, target rank, latency, chunk-count cost, embedding text volume, and visual-object chunk cost.

10. **Embedding Artifacts**
   - `text_dense`: chunk text, OCR text, VLM summaries, and any visual context included by the selected strategy.
   - `caption_dense`: asset caption, OCR, VLM summary text, and structured VLM metadata.
   - `object_dense`: one detected visual object or region per record, including label, attributes, description, location, bbox region, confidence, source field, and parent asset provenance.
   - `image_dense`: rendered page or visual asset image.
   - `triple_dense`: graph triple text built from subject, predicate, object, and selected provenance hints, with resolved chunk payload fields for page, kind, section, and strategy filters.
   - Default hashing embedders make the pipeline testable without model downloads.
   - `embed-package` regenerates artifacts with model-backed text and image embedders.
   - `embedding_manifest.json` records vector files, dimensions, counts, checksums, backend names, model IDs, devices, and batch sizes.

11. **Lexical Search**
    - BM25 is generated from chunk text plus visual asset captions, OCR text, VLM summaries, and structured VLM metadata linked through `asset_ids` or `asset:` source refs.
    - Lexical search protects exact matches for names, identifiers, dates, codes, and policy terms.
    - Tokenization is configurable as `word`, `char_ngram`, or `mixed`.
    - The default `mixed` tokenizer adds CJK character n-grams so compound terms without whitespace remain retrievable, while preserving repeated term frequencies for BM25 ranking.
    - Dense and lexical results are combined with Reciprocal Rank Fusion in the local evaluator.

12. **Package Reproducibility**
    - `manifest.json` records source-file checksum, base chunking strategy, render zoom, section-map count, table extraction setting, tokenizer config, embedding mode, table count, and profile summary.
    - `embedding_manifest.json` records vector files, dimensions, counts, checksums, backend names, model IDs, devices, and batch sizes.
    - Readiness and experiment reports can validate these artifacts before Qdrant or PostgreSQL ingestion.

13. **Graph Triples**
    - Section metadata creates baseline graph relationships.
    - OCR/VLM JSON or external annotations can add `subject, predicate, object` triples.
    - Triple normalization canonicalizes labels, predicate names, and stable IDs.
    - Triple audit flags duplicates, orphan chunk references, empty fields, invalid confidence values, and optional gaps between VLM-derived visual metadata and asset-provenance graph triples.
    - Graph terms are used for query expansion and relationship browsing.

14. **Storage**
    - Qdrant stores named vectors and payloads.
    - Text vector payloads preserve chunk IDs, page ranges, source refs, and visual asset link IDs for filtering and downstream context assembly.
    - PostgreSQL stores normalized document, page, chunk, asset, visual object, triple, and embedding artifact metadata.
    - BM25 can remain as a local manifest or be replaced by a dedicated lexical search service.

15. **Ingestion Readiness**
    - Combine package audit, required artifact checks, Qdrant record validation, and PostgreSQL row conversion.
    - Optionally include retrieval case audit, visual quality gates, chunking comparison gates, and retrieval quality gates before a package is loaded into serving systems.
    - Emit a single pass/fail report for CI, portfolio review, or deployment handoff.

16. **RAG Context Assembly**
    - Convert retrieval hits into a structured context bundle.
    - Support both local hybrid retrieval and Qdrant hybrid retrieval.
    - Include hit chunks, optional neighboring chunks, hierarchical evidence chunks, linked visual assets, and graph triples.
    - Keep page ranges, section labels, source refs, scores, and retrieval sources available for citation.
    - Preserve compact retrieval payload references for visual vector hits so answer generators can tell which asset triggered the parent chunk, and include those retrieved assets in the bundle even when the parent chunk did not list them directly.
    - Select graph triples by matched chunk IDs, visual asset provenance, and explicit graph-vector payload triple IDs so VLM-derived relationships remain attached to retrieved visual or relationship evidence.
    - Bound chunk text and visual asset text separately, while recording original and context character counts plus truncated OCR/VLM fields.

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
- `qdrant_triple_records.jsonl`
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
- `chunks.object_aware.jsonl`
- `chunks.hierarchical.jsonl`
- `chunking_comparison.json`
- `chunking_comparison_gate.json`
- `chunking_sweep.json`
- `chunking_sweep/chunks.*.jsonl`
- `graph_nodes.jsonl`
- `graph_edges.jsonl`
- `graph_summary.json`
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
- `object_dense`
- `triple_dense`

Payload fields include document ID, chunk ID, asset ID, object ID, triple ID, page range, asset kind, graph predicate, object label, bbox region, chunking strategy, retrieval role, parent and source chunk links, section metadata, source references, standalone visual chunk flags, and text fields needed for answer citation.

The package writes payload index definitions with field schemas. Qdrant ingestion and package query commands apply those definitions so metadata filters such as document ID, asset ID, object ID, object label, bbox region, page, section, chunking strategy, hierarchy role, text quality, OCR/VLM work flags, visual asset scope, tile parent/grid fields, and standalone visual assets remain efficient on server-backed collections.

`qdrant-check-collection` compares a live or local Qdrant collection against `qdrant_collection.json` before upsert. It detects missing named vectors, vector dimension mismatches, and missing payload indexes, which is especially important when embedding models or vector dimensions change between experiments.

Qdrant search commands accept repeatable payload filters using exact and range forms such as `kind=map`, `page_no=12`, `page_start<=12`, and `page_end>=12`.

The Qdrant adapter supports both ingestion and named-vector querying. `qdrant-search-package` can upsert a package into qdrant-client local mode and immediately query `text_dense`, `caption_dense`, or `object_dense`, which keeps retrieval checks reproducible without requiring a running server.

`qdrant-hybrid-search` queries Qdrant named vectors, BM25, and optional graph expansion, then fuses results with Reciprocal Rank Fusion. Caption and object vector hits from visual assets and triple vector hits from graph records are mapped back to their parent chunks so text, visual evidence, and relationship evidence can be ranked together. Optional reranking can reorder fused candidates with lexical overlap or a sentence-transformers CrossEncoder.
Graph expansion resolves triples through chunk IDs, chunk aliases, and visual asset provenance so VLM-derived relationships can recover chunks linked to the source visual evidence.

Image vectors may use a different embedding space from text vectors. When querying `image_dense`, the searcher can use a per-vector query encoder, such as CLIP text features for CLIP image embeddings, while continuing to use the document text embedder for `text_dense`, `caption_dense`, and `object_dense`.

Qdrant query paths validate query encoder dimensions against the package collection contract before search. This catches mismatches between package-time embedding models and retrieval-time query encoders before Qdrant executes vector math, and it surfaces the package vector notes so operators can choose the matching model or rebuild the package. Search, evaluation, ablation, and RAG context outputs also record per-vector query encoder details for reproducibility.

Hierarchical chunk files can be searched with parent collapse enabled. In that mode, dense, BM25, graph, or Qdrant hits against fine child chunks are grouped under the coarse parent chunk while the matched child IDs remain attached as evidence. Retrieval evaluation also treats `source_chunk_id` and `parent_chunk_id` metadata as aliases for expected chunk targets, so benchmark cases and graph triples attached to the original chunk IDs remain valid across semantic, multimodal, and hierarchical candidates. This keeps answer context broad enough for citation while preserving the precise span that triggered retrieval.

## PostgreSQL Design

PostgreSQL is used for provenance and relational queries, not as the default vector store.

Tables:

- `documents`
- `pages`
- `chunks`
- `chunk_lexical_tokens`
- `assets`
- `visual_objects`
- `chunk_asset_links`
- `triples`
- `embedding_artifacts`

The writer upserts in dependency order: documents, pages, chunks, BM25 token rows, assets, visual objects, chunk-to-asset links, triples, embedding artifacts. `chunk_lexical_tokens` mirrors `bm25_tokens.json` with tokenizer configuration, token counts, and token arrays, which keeps lexical retrieval experiments reproducible and leaves room for a PostgreSQL-backed lexical service later. Chunk metadata preserves visual asset IDs derived from both direct asset links and `asset:` source refs, and `chunk_asset_links` stores the same relationships in normalized rows for joins and audits. `visual_objects` normalizes structured VLM object metadata from assets into object-level rows with labels, bbox regions, attributes, descriptions, confidence, object text, and asset/page provenance so object-detection probes can be joined and audited without parsing nested asset JSON. Asset-backed graph triples are remapped to an available chunk before PostgreSQL rows are written, preserving the original chunk ID in qualifiers so VLM-derived triples satisfy relational foreign keys without losing provenance. The `embedding_artifacts` table stores vector file names, dimensions, counts, checksums, Qdrant collection names, backend/model metadata, and payload index metadata from `embedding_manifest.json`; vector values remain in Qdrant record files and Qdrant itself.

`postgres-schema` exports the SQL contract for review or migration tooling without requiring a live database. `postgres-check-schema` validates the live PostgreSQL schema before upsert. It checks required tables, columns, column types, relational/search indexes, and the pgvector extension so metadata ingestion failures and slow-path schema drift are caught before batch writes.

## Retrieval Evaluation

Chunking changes should be judged by retrieval behavior, not only by successful execution.

Recommended checks:

- `audit-publication`: public repository scan for forbidden text, accidental binary/document artifacts, oversized files, and required generated-artifact ignore patterns.
- `audit-package`: structural completeness, orphan checks, OCR/VLM gaps, optional VLM-derived visual triple coverage, Qdrant vector dimensions, required payload fields, payload index definitions, text/caption/object/image payload freshness, and embedding manifest count/checksum consistency.
- `qdrant-check-collection`: live Qdrant collection contract validation for named-vector dimensions and payload indexes.
- `postgres-schema`: offline PostgreSQL SQL contract export for review or migration tooling.
- `postgres-check-schema`: live PostgreSQL schema contract validation for required extensions, tables, columns, column types, and indexes.
- `eval-chunking`: page coverage, chunk size distribution, section coverage, visual linkage, annotation coverage, retrieval recall@k, MRR, target coverage@k, target nDCG@k, precision@k, latency, failed queries, and aggregate quality score.
- `audit-retrieval-cases`: benchmark case validation for empty or TODO queries, weak short queries, unknown page/chunk/asset/triple targets, duplicate queries, graph-expansion hints, target-family coverage, distinct target coverage, case-group distinct target coverage, per-target concentration limits, required case metadata group counts such as visual object probes, and optional enforcement that object probes use visual-only VLM/object terms.
- `eval-retrieval`: focused top-k retrieval benchmark cases with optional repeated latency and result-stability sampling, target-specific page/chunk/asset/triple metrics, visual asset provenance matching for triple targets, source-family contribution metrics, chunking-strategy or retrieval-role contribution metrics, and case metadata group metrics such as `case_source` or `query_mode`.
- `generate-retrieval-cases`: benchmark draft generation from package pages, candidate chunk files, visual assets, graph triples, optional visual lexical probes, and VLM object-detection probes, with snippet or document-frequency-weighted salient-term query modes, visual-only object probe terms by default, and visual asset targets for asset-provenance triples.
- `diagnose-retrieval`: failure, partial-coverage, low-ranking, and low-precision analysis for retrieval evaluation JSON outputs.
- `eval-qdrant-retrieval`: the same benchmark cases against Qdrant named vectors plus BM25 and optional graph expansion.
- `eval-qdrant-vector-ablation`: Qdrant text, visual caption, visual object, optional image, and graph-expanded vector comparison on the same cases, including case-group best-mode summaries, query-paired rank deltas, and candidate-vs-baseline comparisons for benchmark subsets such as visual object probes.
- `gate-qdrant-vector-ablation`: pass/fail checks for a selected Qdrant vector mode using recall, target coverage, target nDCG, target rank limits, precision, failed-query count, latency, target-type coverage, source-family and exact source target coverage, case metadata group coverage, optional best-mode requirements, pairwise rank-delta ceilings, and query-paired baseline thresholds when a baseline mode is supplied.
- `ingestion-readiness`: final pre-ingestion gate that can combine package audit results, source checksum/package-config/tokenizer provenance checks, BM25 token manifest freshness for asset-enriched lexical text, linked visual text coverage at asset and text-part levels in package chunks, VLM-derived visual triple coverage, storage artifacts, required vector-family checks, PostgreSQL row conversion, retrieval case metadata group coverage, visual quality, VLM run comparison checks, retrieval gates, chunking comparison gates, retrieval ablation gates, and selected Qdrant vector ablation gates, including exact source coverage for selected retrieval, ablation, and Qdrant vector sources.
- `compare-visual-runs`: OCR/VLM run comparison by coverage, structured parse rate, object detection coverage, graph triple density, latency, optional retrieval evaluation metrics, visual-object-probe target coverage, and shared job-set validation.
- `plan-vlm-experiments`: reproducible profile-by-profile command recipes for running the same visual job set through multiple VLMs.
- `eval-retrieval-ablation`: dense-only, BM25-only, graph-only, hybrid, graph-expanded hybrid, and text-only versus visual-asset-enriched lexical comparison on the same cases, including target coverage@k, target nDCG@k, chunking-strategy coverage, retrieval-role coverage, case metadata group coverage, case-group best-mode summaries, query-paired rank deltas, query-paired comparisons, and latency.
- `gate-retrieval-ablation`: pass/fail checks for a selected retrieval ablation mode using absolute thresholds, baseline lift, target rank limits, target-type coverage, source-family and exact source coverage, case metadata group coverage, strategy/role contribution metrics, best-mode requirements, latency limits, pairwise rank-delta ceilings, and query-paired baseline thresholds.
- `gate-retrieval`: pass/fail checks for benchmark size, expected target count, passed and failed query counts, absolute metric floors, repeated result stability, target-type coverage, source-family and exact source target coverage, chunking-strategy coverage, retrieval-role coverage, case metadata group coverage, and baseline regression limits such as recall drop, target coverage drop, target nDCG drop, precision drop, and latency ratio.
- `compare-packages`: before/after package comparison for count deltas, changed chunk/asset/triple IDs, Qdrant record count deltas, and annotation-related observations.
- `compare-chunking`: side-by-side strategy comparison by quality score, recall@k, MRR, target coverage@k, target nDCG@k, target rank, precision@k, target-type coverage, source-family target coverage, chunking-strategy coverage, retrieval-role coverage, case group coverage, linked visual text asset and part coverage, latency, failed queries, query-paired baseline deltas, and paired bootstrap confidence intervals.
- `gate-chunking-comparison`: pass/fail checks for selected chunking candidates using quality, page coverage, visual text asset/part coverage, retrieval floors, target rank limits, target-type coverage, source-family target coverage, chunking-strategy coverage, retrieval-role coverage, case group coverage, failed-query limits, baseline regression limits, pairwise lift requirements, pairwise rank-delta ceilings, and paired confidence bounds.
- `sweep-chunking`: parameter grid generation for max size, overlap, parent size, and multimodal, object-aware, or hierarchical visual context size, with weighted selection scores, optional hard selection constraints for aggregate metrics, result stability, target types, source families, case groups, chunk text volume, chunk length, standalone visual chunk count, and visual object chunk count, eligibility failures, and a Pareto front that treats latency, rank, result instability, chunk count, chunk text volume, embedding text volume, standalone visual chunk count, and visual object chunk count as cost axes for retrieval quality-versus-cost review.
- `write-experiment-report`: reproducible package report with source-file/package config metadata, artifact checksums, record counts, tokenizer settings, Qdrant configuration, readiness, evaluation, audit, gate artifact variants, visual run comparison summaries, top-level and component-level validation pass/fail summaries, linked visual text asset/part coverage, target rank metrics, case metadata group metrics, paired confidence metrics, and candidate comparison metrics.
- Qdrant local mode upsert: validates named vector records and payloads.

Benchmark cases should be maintained per document family. A useful case specifies the query, expected page, chunk, visual asset, graph triple, and whether graph expansion should be enabled.

Tokenizer settings are part of the retrieval experiment. Strategy comparisons should keep the tokenizer fixed unless the experiment is explicitly measuring lexical tokenization.

Fusion weights are also part of the retrieval experiment. Use `--fusion-weight` to tune source families such as `dense`, `bm25`, `graph`, and `qdrant`, or exact sources such as `qdrant:caption_dense`. Exact source target coverage gates can verify that a specific source, for example `qdrant:image_dense`, contributes evidence inside a broader combined mode.

Reranking is a separate experiment knob. Keep `--reranker`, `--rerank-top-k`, and the reranker model fixed when comparing chunking strategies unless the experiment is explicitly measuring reranking.

Rank gates are separate from recall gates. `gate-retrieval`, `gate-chunking-comparison`, `gate-retrieval-ablation`, and `gate-qdrant-vector-ablation` can cap mean or p95 first relevant rank and target rank, with missing expected targets counted as `top_k + 1`, so a run that technically retrieves the right evidence but buries it below stronger candidates can still fail.

Use repeated retrieval evaluation when comparing strategies whose recall is similar. The latency fields are intended to show whether higher recall comes with an acceptable retrieval cost. Pairwise comparison metrics are computed on shared benchmark queries, so they should be used with a stable, reviewed retrieval-case set.

For hierarchical candidates, enable parent collapse during `eval-retrieval`, `compare-chunking`, or `write-experiment-report` when the benchmark expects page-level or parent-level citation behavior.

## Model Strategy

The library exposes interfaces instead of locking in one model:

- OCR: `TesseractOCRBackend` and `PaddleOCRBackend` for multilingual scanned pages.
- VLM: `HuggingFaceVLMBackend` with configurable device map, torch dtype, generation length, optional attention implementation, and profile-level GPU memory, memory-margin, and bfloat16 compatibility checks through `doctor --vlm-profile`.
- VLM profiles: named Hugging Face profiles record the model id, loader family, dtype, and generation defaults for reproducible local model comparisons.
- Text dense: `SentenceTransformerTextEmbedder`.
- Image dense: `TransformersImageEmbedder`.

Local GPUs can be used for VLM summaries and image embedding batches. Failed jobs remain visible in job result files so experiments can be retried safely. Prompt hashes and backend configuration fields make model comparisons reproducible without embedding document-specific rules in source code.
