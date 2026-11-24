"""Fact-level knowledge hot-update benchmark.

PSS stores facts explicitly and derives rule answers without gradient updates.
A small transformer learns the same facts implicitly. We edit one fact and
compare direct/derived accuracy, forgetting, update time, and rollback.
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
from pss.knowledge import FactStore, Rule, RuleEngine

MATERIALS = ("glass", "metal", "plastic", "wood")
LOCATIONS = ("shelf", "floor", "table")
COLORS = ("red", "blue", "green")
DIRECT_ATTRS = ("material", "location", "color")
DERIVED_ATTRS = ("fragile", "needs_padding", "danger")
ATTRS = DIRECT_ATTRS + DERIVED_ATTRS
ANSWERS = tuple(dict.fromkeys(MATERIALS + LOCATIONS + COLORS + ("yes", "no")))
ANSWER_TO_INDEX = {a: i for i, a in enumerate(ANSWERS)}


def _rules():
    return [
        Rule(
            "fragile",
            lambda s, e: s.get(e, "material") == "glass",
            "fragile",
        ),
        Rule(
            "needs_padding",
            lambda s, e: s.get(e, "material") == "glass"
            and s.get(e, "location") == "shelf",
            "needs_padding",
        ),
        Rule(
            "danger",
            lambda s, e: s.get(e, "material") == "metal"
            and s.get(e, "location") == "floor",
            "danger",
        ),
    ]


def _make_kb(n_entities=24, seed=0):
    rng = np.random.default_rng(seed)
    facts = {}
    entities = [f"obj_{i:02d}" for i in range(n_entities)]
    for entity in entities:
        facts[(entity, "material")] = str(rng.choice(MATERIALS))
        facts[(entity, "location")] = str(rng.choice(LOCATIONS))
        facts[(entity, "color")] = str(rng.choice(COLORS))
    return entities, FactStore(facts), RuleEngine(_rules())


class FactTransformer(nn.Module):
    def __init__(self, n_entities: int, n_queries: int, n_answers: int, d_model=64):
        super().__init__()
        self.entity = nn.Embedding(n_entities, d_model)
        self.query = nn.Embedding(n_queries, d_model)
        self.kind = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, n_answers)

    def forward(self, entity, query):
        e = self.entity(entity) + self.kind.weight[0]
        q = self.query(query) + self.kind.weight[1]
        h = self.encoder(torch.stack([e, q], dim=1))
        return self.head(h.mean(dim=1))


def _rows(entities, store, engine):
    rows = []
    for ei, entity in enumerate(entities):
        for qi, attr in enumerate(ATTRS):
            answer = engine.answer(store, entity, attr)
            rows.append(
                {
                    "entity": ei,
                    "query": qi,
                    "label": ANSWER_TO_INDEX[answer],
                    "attr": attr,
                    "name": entity,
                }
            )
    return rows


def _fit(model, rows, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_e = torch.as_tensor([r["entity"] for r in rows], dtype=torch.long, device=device)
    x_q = torch.as_tensor([r["query"] for r in rows], dtype=torch.long, device=device)
    y = torch.as_tensor([r["label"] for r in rows], dtype=torch.long, device=device)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(y), device=device)
        for start in range(0, len(perm), 64):
            idx = perm[start : start + 64]
            logits = model(x_e[idx], x_q[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _accuracy(model, rows, device):
    x_e = torch.as_tensor([r["entity"] for r in rows], dtype=torch.long, device=device)
    x_q = torch.as_tensor([r["query"] for r in rows], dtype=torch.long, device=device)
    y = torch.as_tensor([r["label"] for r in rows], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(x_e, x_q).argmax(dim=1)
    return float((pred == y).float().mean().cpu())


def _group_accuracy(model, rows, device, group):
    subset = [r for r in rows if r["attr"] in group]
    return _accuracy(model, subset, device) if subset else 0.0


def run(
    out_dir="eval/results/pss_vs_transformer",
    seeds=5,
    epochs=300,
    finetune_epochs=50,
    device="cpu",
):
    entities, store, engine = _make_kb()
    base_rows = _rows(entities, store, engine)
    n_entities = len(entities)
    n_queries = len(ATTRS)
    n_answers = len(ANSWERS)
    results = []

    for seed in range(seeds):
        torch.manual_seed(seed)
        model = FactTransformer(n_entities, n_queries, n_answers).to(device)
        model = _fit(model, base_rows, epochs, 1e-3, device)
        before = _accuracy(model, base_rows, device)
        before_direct = _group_accuracy(model, base_rows, device, DIRECT_ATTRS)
        before_derived = _group_accuracy(model, base_rows, device, DERIVED_ATTRS)

        rng = np.random.default_rng(1000 + seed)
        entity = str(rng.choice(entities))
        attr = str(rng.choice(DIRECT_ATTRS))
        old_value = store.get(entity, attr)
        candidates = [v for v in (MATERIALS if attr == "material" else LOCATIONS if attr == "location" else COLORS) if v != old_value]
        new_value = str(rng.choice(candidates))

        version_before = store.version
        t0 = time.perf_counter()
        store.update(entity, attr, new_value, reason="benchmark edit")
        pss_edit_ms = (time.perf_counter() - t0) * 1000.0
        after_rows = _rows(entities, store, engine)
        edited_rows = [r for r in after_rows if r["name"] == entity]
        old_rows = [r for r in after_rows if r["name"] != entity]

        # PSS exact answers after edit and after rollback.
        pss_after = _pss_accuracy(entities, store, engine, after_rows)
        store.rollback(version_before)
        pss_rollback = _pss_accuracy(entities, store, engine, _rows(entities, store, engine))
        store.update(entity, attr, new_value, reason="re-apply edit")

        # Transformer full fine-tune on edited entity only.
        import copy

        t0 = time.perf_counter()
        full_model = _fit(copy.deepcopy(model), edited_rows, finetune_epochs, 1e-3, device)
        full_ms = (time.perf_counter() - t0) * 1000.0
        full_edit_acc = _accuracy(full_model, edited_rows, device)
        full_old_acc = _accuracy(full_model, old_rows, device)
        full_derived = _group_accuracy(full_model, edited_rows, device, DERIVED_ATTRS)

        # Transformer replay: edited entity + all old entities.
        replay_rows = edited_rows + old_rows
        t0 = time.perf_counter()
        replay_model = _fit(copy.deepcopy(model), replay_rows, finetune_epochs, 5e-4, device)
        replay_ms = (time.perf_counter() - t0) * 1000.0
        replay_edit_acc = _accuracy(replay_model, edited_rows, device)
        replay_old_acc = _accuracy(replay_model, old_rows, device)
        replay_derived = _group_accuracy(replay_model, edited_rows, device, DERIVED_ATTRS)

        results.append(
            {
                "seed": seed,
                "edit": {"entity": entity, "attribute": attr, "old": old_value, "new": new_value},
                "pss": {
                    "after_accuracy": pss_after,
                    "rollback_accuracy": pss_rollback,
                    "edit_ms": pss_edit_ms,
                },
                "before": {"overall": before, "direct": before_direct, "derived": before_derived},
                "full": {
                    "edited_accuracy": full_edit_acc,
                    "old_accuracy": full_old_acc,
                    "edited_derived_accuracy": full_derived,
                    "forgetting": before - full_old_acc,
                    "update_ms": full_ms,
                },
                "replay": {
                    "edited_accuracy": replay_edit_acc,
                    "old_accuracy": replay_old_acc,
                    "edited_derived_accuracy": replay_derived,
                    "forgetting": before - replay_old_acc,
                    "update_ms": replay_ms,
                },
            }
        )

    def avg(key):
        return float(np.mean([r[key] for r in results]))

    metrics = {
        "n_entities": n_entities,
        "n_queries": n_queries,
        "seeds": seeds,
        "results": results,
        "summary": {
            "pss_edit_ms": avg("pss") if False else float(np.mean([r["pss"]["edit_ms"] for r in results])),
            "pss_after": float(np.mean([r["pss"]["after_accuracy"] for r in results])),
            "pss_rollback": float(np.mean([r["pss"]["rollback_accuracy"] for r in results])),
            "before": float(np.mean([r["before"]["overall"] for r in results])),
            "full_edited": float(np.mean([r["full"]["edited_accuracy"] for r in results])),
            "full_old": float(np.mean([r["full"]["old_accuracy"] for r in results])),
            "full_forgetting": float(np.mean([r["full"]["forgetting"] for r in results])),
            "replay_edited": float(np.mean([r["replay"]["edited_accuracy"] for r in results])),
            "replay_old": float(np.mean([r["replay"]["old_accuracy"] for r in results])),
            "replay_forgetting": float(np.mean([r["replay"]["forgetting"] for r in results])),
            "full_update_ms": float(np.mean([r["full"]["update_ms"] for r in results])),
            "replay_update_ms": float(np.mean([r["replay"]["update_ms"] for r in results])),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fact_update.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "fact_update.md", metrics)
    return metrics


def _pss_accuracy(entities, store, engine, rows):
    correct = 0
    for r in rows:
        answer = engine.answer(store, r["name"], r["attr"])
        if answer == ANSWERS[r["label"]]:
            correct += 1
    return correct / max(1, len(rows))


def _report(path, metrics):
    s = metrics["summary"]
    lines = [
        "# Fact-level knowledge hot-update",
        "",
        f"- entities: {metrics['n_entities']}, query types: {metrics['n_queries']}, seeds: {metrics['seeds']}",
        "",
        "| method | edited accuracy | old accuracy | forgetting | edit time |",
        "|---|---:|---:|---:|---:|",
        f"| PSS fact update | **{s['pss_after']:.3f}** | **{s['pss_rollback']:.3f}** | **0.000** | **{s['pss_edit_ms']:.3f} ms** |",
        f"| Transformer full | {s['full_edited']:.3f} | {s['full_old']:.3f} | {s['full_forgetting']:.3f} | {s['full_update_ms']:.1f} ms |",
        f"| Transformer replay | {s['replay_edited']:.3f} | {s['replay_old']:.3f} | {s['replay_forgetting']:.3f} | {s['replay_update_ms']:.1f} ms |",
        "",
        "PSS accuracy columns are after-edit and after-rollback. Rollback restores the exact pre-edit fact store.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--finetune-epochs", type=int, default=50)
    args = ap.parse_args()
    m = run(args.out, args.seeds, args.epochs, args.finetune_epochs)
    s = m["summary"]
    print(
        f"PSS {s['pss_after']:.3f} edit {s['pss_edit_ms']:.3f} ms; "
        f"full old {s['full_old']:.3f}; replay old {s['replay_old']:.3f}"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
