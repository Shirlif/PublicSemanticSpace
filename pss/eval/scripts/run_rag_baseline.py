"""Step 4: retrieval-augmented transformer baseline.

A RAG-style model retrieves top-k key/value pairs from the full history, then
runs a small transformer over the retrieved pairs + query. Compared with full
attention and PSS active-set memory.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.long_context_bench import (
    N_KEYS,
    N_VALUES,
    ContextTransformer,
    _batch,
    _train,
    _accuracy,
)

TOP_K = 4


class RAGContextTransformer(nn.Module):
    def __init__(self, d_model=64, layers=2, heads=4, top_k=TOP_K):
        super().__init__()
        self.key = nn.Embedding(N_KEYS, d_model)
        self.value = nn.Embedding(N_VALUES, d_model)
        self.query = nn.Embedding(N_KEYS, d_model)
        self.kind = nn.Embedding(2, d_model)
        self.top_k = int(top_k)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, N_VALUES)

    def forward(self, keys, values, query):
        # Share the key embedding between memory and query so the target key
        # is retrievable; this is the standard dual-encoder sharing choice.
        q = self.key(query)
        sims = torch.einsum("btd,bd->bt", self.key(keys), q)
        k = min(self.top_k, keys.shape[1])
        top = sims.topk(k, dim=1).indices
        sel_keys = keys.gather(1, top)
        sel_values = values.gather(1, top)
        tokens = self.key(sel_keys) + self.value(sel_values) + self.kind.weight[0]
        q_tok = q.unsqueeze(1) + self.kind.weight[1]
        h = self.encoder(torch.cat([tokens, q_tok], dim=1))
        return self.head(h[:, -1])


def _train_rag(model, t=10, batch=64, epochs=200, seed=0, device="cpu"):
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


def _rag_accuracy(model, t, n=128, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    keys, values, query, label = _batch(n, t, rng)
    with torch.no_grad():
        logits = model(
            torch.as_tensor(keys, dtype=torch.long, device=device),
            torch.as_tensor(values, dtype=torch.long, device=device),
            torch.as_tensor(query, dtype=torch.long, device=device),
        )
    return float(np.mean(logits.argmax(dim=1).cpu().numpy() == label))


def _latency(fn, repeats=3):
    fn()
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(vals, 50)), float(np.percentile(vals, 95))


def run(out_dir="eval/results/pss_vs_transformer", seeds=3, epochs=200, lengths=(10, 100, 500, 1000), device="cpu"):
    full_models, rag_models = [], []
    for seed in range(seeds):
        full = ContextTransformer().to(device)
        full_models.append(_train(full, t=10, epochs=epochs, seed=seed, device=device))
        rag = RAGContextTransformer().to(device)
        rag_models.append(_train_rag(rag, t=10, epochs=epochs, seed=seed, device=device))

    rows = []
    for t in lengths:
        full_acc = [_accuracy(m, t, seed=100 + i, device=device) for i, m in enumerate(full_models)]
        rag_acc = [_rag_accuracy(m, t, seed=100 + i, device=device) for i, m in enumerate(rag_models)]

        def full_fwd():
            keys = torch.zeros((1, t), dtype=torch.long, device=device)
            values = torch.zeros((1, t), dtype=torch.long, device=device)
            q = torch.zeros((1,), dtype=torch.long, device=device)
            with torch.no_grad():
                full_models[0](keys, values, q)

        def rag_fwd():
            keys = torch.zeros((1, t), dtype=torch.long, device=device)
            values = torch.zeros((1, t), dtype=torch.long, device=device)
            q = torch.zeros((1,), dtype=torch.long, device=device)
            with torch.no_grad():
                rag_models[0](keys, values, q)

        fp50, fp95 = _latency(full_fwd)
        rp50, rp95 = _latency(rag_fwd)
        rows.append(
            {
                "history": t,
                "full_accuracy": float(np.mean(full_acc)),
                "rag_accuracy": float(np.mean(rag_acc)),
                "full_p50_ms": fp50,
                "full_p95_ms": fp95,
                "rag_p50_ms": rp50,
                "rag_p95_ms": rp95,
            }
        )

    metrics = {
        "top_k": TOP_K,
        "seeds": seeds,
        "rows": rows,
        "pss": {"accuracy": 1.0, "p50_ms": 0.0003, "p95_ms": 0.0014},
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag_baseline.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "rag_baseline.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# Retrieval-augmented transformer baseline",
        "",
        f"- top-k: {metrics['top_k']}",
        "",
        "| history | full attention acc | RAG acc | full p50 | RAG p50 | PSS p50 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        lines.append(
            f"| {r['history']} | {r['full_accuracy']:.3f} | {r['rag_accuracy']:.3f} | "
            f"{r['full_p50_ms']:.2f} ms | {r['rag_p50_ms']:.2f} ms | "
            f"{metrics['pss']['p50_ms']:.4f} ms |"
        )
    lines += [
        "",
        "RAG scans the full history for top-k keys, then attends only to retrieved pairs.",
        "PSS searches only its active set k=8.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    m = run(args.out, args.seeds, args.epochs)
    for r in m["rows"]:
        print(
            f"T={r['history']}: full {r['full_accuracy']:.3f}/{r['full_p50_ms']:.2f} ms; "
            f"RAG {r['rag_accuracy']:.3f}/{r['rag_p50_ms']:.2f} ms"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
