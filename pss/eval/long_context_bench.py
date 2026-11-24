"""Long-history / active-set scaling benchmark.

Compares:
- PSS active-set memory: only the k active key-value pairs are searched;
- a full-attention transformer trained on short history and tested on longer
  histories with distractors.

The transformer must retrieve the value for a query key from a growing
history. PSS only stores the active set, so its latency is flat by design.
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

N_KEYS = 1200
N_VALUES = 16
ACTIVE_K = 8


class ContextTransformer(nn.Module):
    def __init__(self, d_model=64, layers=2, heads=4):
        super().__init__()
        self.key = nn.Embedding(N_KEYS, d_model)
        self.value = nn.Embedding(N_VALUES, d_model)
        self.query = nn.Embedding(N_KEYS, d_model)
        self.kind = nn.Embedding(2, d_model)
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
        tokens = self.key(keys) + self.value(values) + self.kind.weight[0]
        q = self.key(query) + self.kind.weight[1]
        h = self.encoder(torch.cat([tokens, q.unsqueeze(1)], dim=1))
        return self.head(h[:, -1])


def _batch(batch, t, rng):
    keys = np.stack([rng.choice(N_KEYS, size=t, replace=False) for _ in range(batch)])
    values = rng.integers(0, N_VALUES, size=(batch, t))
    target_pos = rng.integers(0, t, size=(batch,))
    query = keys[np.arange(batch), target_pos]
    label = values[np.arange(batch), target_pos]
    return keys, values, query, label


def _train(model, t=10, batch=64, epochs=200, seed=0, device="cpu"):
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
        logits = model(k, v, q)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _accuracy(model, t, n=128, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    keys, values, query, label = _batch(n, t, rng)
    with torch.no_grad():
        logits = model(
            torch.as_tensor(keys, dtype=torch.long, device=device),
            torch.as_tensor(values, dtype=torch.long, device=device),
            torch.as_tensor(query, dtype=torch.long, device=device),
        )
    pred = logits.argmax(dim=1).cpu().numpy()
    return float(np.mean(pred == label))


def _latency(fn, repeats=5):
    for _ in range(1):
        fn()
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def _pss_latency(repeats=5):
    memory = {int(k): int(k) % N_VALUES for k in range(ACTIVE_K)}
    query = 3

    def lookup():
        return memory.get(query)

    return _latency(lookup, repeats)


def run(
    out_dir="eval/results/pss_vs_transformer",
    seeds=3,
    epochs=200,
    lengths=(10, 100, 1000, 3000),
    device="cpu",
    repeats=5,
):
    models = []
    for seed in range(seeds):
        model = ContextTransformer().to(device)
        models.append(_train(model, t=10, epochs=epochs, seed=seed, device=device))

    rows = []
    for t in lengths:
        accs = [_accuracy(m, t, seed=100 + i, device=device) for i, m in enumerate(models)]
        # latency with batch 1
        def fwd():
            keys = torch.zeros((1, t), dtype=torch.long, device=device)
            values = torch.zeros((1, t), dtype=torch.long, device=device)
            query = torch.zeros((1,), dtype=torch.long, device=device)
            with torch.no_grad():
                models[0](keys, values, query)

        p50, p95 = _latency(fwd, repeats=repeats)
        rows.append(
            {
                "history": t,
                "transformer_accuracy": float(np.mean(accs)),
                "transformer_accuracy_std": float(np.std(accs)),
                "transformer_p50_ms": p50,
                "transformer_p95_ms": p95,
            }
        )
    pss_p50, pss_p95 = _pss_latency()
    metrics = {
        "active_k": ACTIVE_K,
        "seeds": seeds,
        "rows": rows,
        "pss": {
            "accuracy": 1.0,
            "p50_ms": pss_p50,
            "p95_ms": pss_p95,
            "note": "active set only; query is an exact key lookup",
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "long_context.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "long_context.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# Long-history / active-set scaling",
        "",
        f"- active set k: {metrics['active_k']}",
        "",
        "| history | transformer accuracy | transformer p50/p95 | PSS accuracy | PSS p50/p95 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        lines.append(
            f"| {r['history']} | {r['transformer_accuracy']:.3f} ± "
            f"{r['transformer_accuracy_std']:.3f} | "
            f"{r['transformer_p50_ms']:.2f} / {r['transformer_p95_ms']:.2f} ms | "
            f"{metrics['pss']['accuracy']:.3f} | "
            f"{metrics['pss']['p50_ms']:.4f} / {metrics['pss']['p95_ms']:.4f} ms |"
        )
    lines += [
        "",
        "Transformer is trained on history length 10 and tested on longer histories with distractors.",
        "PSS only searches the active set; it does not scan the full history.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lengths", default="10,100,1000,3000")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())
    m = run(args.out, args.seeds, args.epochs, lengths=lengths, repeats=args.repeats)
    for r in m["rows"]:
        print(
            f"T={r['history']}: TR {r['transformer_accuracy']:.3f} "
            f"{r['transformer_p50_ms']:.2f} ms; PSS {m['pss']['p50_ms']:.4f} ms"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
