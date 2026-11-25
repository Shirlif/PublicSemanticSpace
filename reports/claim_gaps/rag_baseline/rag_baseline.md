# Retrieval-augmented transformer baseline

- top-k: 4

| history | full attention acc | RAG acc | full p50 | RAG p50 | PSS p50 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.156 | 0.492 | 1.76 ms | 1.81 ms | 0.0003 ms |
| 100 | 0.062 | 0.398 | 2.27 ms | 1.85 ms | 0.0003 ms |
| 500 | 0.070 | 0.359 | 4.24 ms | 1.89 ms | 0.0003 ms |
| 1000 | 0.086 | 0.375 | 9.97 ms | 2.06 ms | 0.0003 ms |

RAG scans the full history for top-k keys, then attends only to retrieved pairs.
PSS searches only its active set k=8.

## Caveat

This learned-retrieval RAG is retrieval-limited on the synthetic stream: the
shared key embeddings collide at long histories, so accuracy stays around
0.375. Treat it as a lower bound, not as the RAG upper bound. The oracle-key
retrieval variant is E13 (`reports/claim_gaps/rag_exact/REPORT.md`), which
reaches 1.000 and still pays an O(history) scan.
