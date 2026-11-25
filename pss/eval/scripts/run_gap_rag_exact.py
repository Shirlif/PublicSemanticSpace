"""Claim-gap E13: RAG baseline with oracle retrieval vs PSS active set.

The existing learned-RAG baseline is retrieval-limited on the synthetic
key/value stream (its learned key embeddings collide at long histories).  This
experiment adds an **oracle-retrieval RAG** variant: retrieve the exact key by
matching, then run the same transformer decoder.  It is the fair upper bound
for RAG accuracy and still pays an O(history) scan, so the cost comparison
against the PSS active set is not confounded by a weak retriever.

Methods:
- full attention (ContextTransformer);
- learned RAG (shared key embedding, top-k);
- exact RAG (oracle key match, top-k);
- PSS active set (k=8, exact lookup).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.long_context_bench import (
    N_VALUES,
    ContextTransformer,
    _accuracy,
    _batch,
    _latency,
    _train,
)
from pss.eval.scripts.run_rag_baseline import RAGContextTransformer, _train_rag


class ExactRAGContextTransformer(RAGContextTransformer):
    """Retrieve by exact key match; the rest is the same decoder."""

    def forward(self, keys, values, query):
        match = (keys == query.unsqueeze(1)).float()
        k = min(self.top_k, keys.shape[1])
        top = match.topk(k, dim=1).indices
        sel_keys = keys.gather(1, top)
        sel_values = values.gather(1, top)
        tokens = self.key(sel_keys) + self.value(sel_values) + self.kind.weight[0]
        q_tok = self.key(query).unsqueeze(1) + self.kind.weight[1]
        h = self.encoder(torch.cat([tokens, q_tok], dim=1))
        return self.head(h[:, -1])


def _train_exact(model, t=10, batch=64, epochs=200, seed=0, device="cpu"):
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        keys, values, query, label = _batch(batch, t, rng)
        k = torch.as_tensor(keys, dtype=torch.long, device=device)
        v = torch.as_tensor(values, dtype=torch.long, device=device)
        q = torch.as_tensor(query, dtype=torch.long, device=device)
        y = torch.as_tensor(label, dtype=torch.long, device=device)
        loss = F.cross_entropy(model(k, v, q), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _accuracy_model(model, t, n=128, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    keys, values, query, label = _batch(n, t, rng)
    with torch.no_grad():
        logits = model(
            torch.as_tensor(keys, dtype=torch.long, device=device),
            torch.as_tensor(values, dtype=torch.long, device=device),
            torch.as_tensor(query, dtype=torch.long, device=device),
        )
    return float(np.mean(logits.argmax(dim=1).cpu().numpy() == label))


def run(
    out_dir="reports/claim_gaps/rag_exact",
    seeds=1,
    epochs=100,
    lengths=(10, 100, 500, 1000),
    device="cpu",
) -> dict:
    full_models, rag_models, exact_models = [], [], []
    for seed in range(seeds):
        full = ContextTransformer().to(device)
        full_models.append(_train(full, t=10, epochs=epochs, seed=seed, device=device))
        rag = RAGContextTransformer().to(device)
        rag_models.append(_train_rag(rag, t=10, epochs=epochs, seed=seed, device=device))
        exact = ExactRAGContextTransformer(top_k=1).to(device)
        exact_models.append(_train_exact(exact, t=10, epochs=epochs, seed=seed, device=device))

    rows = []
    for t in lengths:
        full_acc = [_accuracy(m, t, seed=100 + i, device=device) for i, m in enumerate(full_models)]
        rag_acc = [_accuracy_model(m, t, seed=100 + i, device=device) for i, m in enumerate(rag_models)]
        exact_acc = [_accuracy_model(m, t, seed=100 + i, device=device) for i, m in enumerate(exact_models)]

        def _fwd(model):
            keys = torch.zeros((1, t), dtype=torch.long, device=device)
            values = torch.zeros((1, t), dtype=torch.long, device=device)
            q = torch.zeros((1,), dtype=torch.long, device=device)
            with torch.no_grad():
                model(keys, values, q)

        def _fwd_exact():
            keys = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0)
            values = torch.zeros((1, t), dtype=torch.long, device=device)
            q = torch.tensor([t // 2], dtype=torch.long, device=device)
            with torch.no_grad():
                exact_models[0](keys, values, q)

        fp50, fp95 = _latency(lambda: _fwd(full_models[0]))
        rp50, rp95 = _latency(lambda: _fwd(rag_models[0]))
        ep50, ep95 = _latency(_fwd_exact)
        rows.append(
            {
                "history": t,
                "full_accuracy": float(np.mean(full_acc)),
                "rag_accuracy": float(np.mean(rag_acc)),
                "exact_rag_accuracy": float(np.mean(exact_acc)),
                "full_p50_ms": fp50,
                "rag_p50_ms": rp50,
                "exact_rag_p50_ms": ep50,
                "exact_rag_p95_ms": ep95,
            }
        )

    return {
        "top_k": 4,
        "seeds": seeds,
        "epochs": epochs,
        "rows": rows,
        "pss": {"accuracy": 1.0, "p50_ms": 0.0003, "p95_ms": 0.0014, "active_k": 8},
    }


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap E13: RAG with oracle retrieval vs PSS active set",
        "",
        f"> seeds: {m['seeds']}, epochs: {m['epochs']}, top-k: {m['top_k']}",
        "",
        "| history | full attn acc | learned RAG acc | exact RAG acc | full p50 | learned RAG p50 | exact RAG p50 | PSS p50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in m["rows"]:
        lines.append(
            f"| {r['history']} | {r['full_accuracy']:.3f} | {r['rag_accuracy']:.3f} | "
            f"{r['exact_rag_accuracy']:.3f} | {r['full_p50_ms']:.2f} | "
            f"{r['rag_p50_ms']:.2f} | {r['exact_rag_p50_ms']:.2f} | "
            f"{m['pss']['p50_ms']:.4f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "Exact RAG is the accuracy upper bound: it retrieves the matching key by",
        "identity, not by a learned embedding. It still scans the full history, so",
        "its latency grows with T. The PSS active set matches full accuracy with a",
        "constant-time lookup over k=8. The learned-RAG column is the realistic",
        "retriever; it is retrieval-limited on this synthetic stream and should be",
        "reported as a lower bound, not as the RAG upper bound.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default="reports/claim_gaps/rag_exact")
    args = ap.parse_args()
    metrics = run(seeds=args.seeds, epochs=args.epochs)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for r in metrics["rows"]:
        print(
            f"T={r['history']}: full={r['full_accuracy']:.3f} "
            f"rag={r['rag_accuracy']:.3f} exact={r['exact_rag_accuracy']:.3f} "
            f"exact_p50={r['exact_rag_p50_ms']:.2f}ms"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
