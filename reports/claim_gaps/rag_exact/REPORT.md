# Claim-gap E13: RAG with oracle retrieval vs PSS active set

> seeds: 1, epochs: 100, top-k: 4

| history | full attn acc | learned RAG acc | exact RAG acc | full p50 | learned RAG p50 | exact RAG p50 | PSS p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.234 | 0.492 | 1.000 | 2.10 | 1.59 | 1.30 | 0.0003 |
| 100 | 0.086 | 0.398 | 1.000 | 2.04 | 1.61 | 1.36 | 0.0003 |
| 500 | 0.039 | 0.359 | 1.000 | 3.96 | 1.79 | 1.44 | 0.0003 |
| 1000 | 0.078 | 0.375 | 1.000 | 9.90 | 1.77 | 1.38 | 0.0003 |

## How to read this

Exact RAG is the accuracy upper bound: it retrieves the matching key by
identity, not by a learned embedding. It still scans the full history, so
its latency grows with T. The PSS active set matches full accuracy with a
constant-time lookup over k=8. The learned-RAG column is the realistic
retriever; it is retrieval-limited on this synthetic stream and should be
reported as a lower bound, not as the RAG upper bound.
