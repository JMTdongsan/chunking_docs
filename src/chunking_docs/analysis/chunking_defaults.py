from __future__ import annotations

CHUNKING_SWEEP_SELECTION_ARGS = [
    "--selection-min-target-coverage-at-k 0.75",
    "--selection-min-target-ndcg-at-k 0.7",
    "--selection-min-retrieval-score-per-embedding-kchar 0.0003",
    "--selection-min-retrieval-score-per-mean-latency-ms 0.0005",
    "--selection-min-target-coverage-per-p95-latency-ms 0.0005",
    "--selection-max-mean-target-rank 3",
    "--selection-min-visual-text-coverage-ratio 0.8",
    "--selection-min-visual-text-part-coverage-ratio 0.8",
]

CHUNKING_COMPARISON_GATE_ARGS = [
    "--require-retrieval",
    "--min-target-coverage-at-k 0.75",
    "--min-target-ndcg-at-k 0.7",
    "--max-mean-target-rank 3",
    "--min-visual-text-coverage-ratio 0.8",
    "--min-visual-text-part-coverage-ratio 0.8",
    "--max-total-chunk-chars 3000000",
    "--min-retrieval-score-per-embedding-kchar 0.0003",
    "--min-retrieval-score-per-mean-latency-ms 0.0005",
    "--min-target-coverage-per-p95-latency-ms 0.0005",
    "--max-failed-queries 30",
]

CHUNKING_READINESS_GATE_ARGS = [
    "--chunking-comparison {chunking_comparison}",
    "--require-chunking-comparison",
    "--min-chunking-target-coverage-at-k 0.75",
    "--min-chunking-target-ndcg-at-k 0.7",
    "--max-chunking-mean-target-rank 3",
    "--min-chunking-visual-text-coverage-ratio 0.8",
    "--min-chunking-visual-text-part-coverage-ratio 0.8",
    "--max-chunking-total-chunk-chars 3000000",
    "--min-chunking-retrieval-score-per-embedding-kchar 0.0003",
    "--min-chunking-retrieval-score-per-mean-latency-ms 0.0005",
    "--min-chunking-target-coverage-per-p95-latency-ms 0.0005",
    "--max-chunking-failed-queries 30",
]
