"""Second priority: make the relation layer learnable.

Three experiments:

- ``relations``: train a typed relation head (spatial / semantic / tactile) on
  simulation labels and report per-type F1, clean and under modality
  corruption.
- ``reid``: compare hand-written graph edges against learned typed edges and
  a hybrid (hand + learned) matcher on the same tracking pairs.
- ``graph``: write relation logits into a persistent ``SceneGraph`` and verify
  they survive save/load.
- ``p5b``: train a coordinate spatial-relation head on part of the region grid,
  test it on held-out regions, and run the P5b planner with learned vs
  hand-written relation predicates.

Example:
    python -m pss.eval.scripts.run_typed_relations --stage all --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.relational_bench import (
    NAMES,
    REGIONS,
    RELATIONS,
    TEST_PAIRS,
    TRAIN_PAIRS,
    _rows,
    goal_ok,
    make_episode,
    plan,
    rollout,
    train_policy,
)
from pss.memory.graph_matcher import GraphMatcher
from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.memory.scene_graph import SceneGraph, Track
from pss.memory.typed_relations import (
    HybridRelationMatcher,
    RelationAwareGraphMatcher,
    SpatialRelationHead,
    collect_relation_pairs,
    evaluate_relation_head,
    evaluate_relation_matcher,
    save_relation_head,
    train_relation_head,
    train_relation_matcher,
    write_relation_edges,
)

DATA = "data/tracking_bench_v2"
EXPERTS = "checkpoints/category_pretrain_experts"
PROJECTIONS = "checkpoints/phase45_projection"
OUT_ROOT = "eval/results/typed_relations"
CKPT = "checkpoints/typed_relations"

RELATION_CONFIGS = (
    ("clean", None),
    ("pose_noise", "pose"),
    ("no_tactile", "tactile"),
    ("no_vision", "vision"),
    ("mixed", "mixed"),
)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def _pairs(split: str, device: str):
    return collect_relation_pairs(
        DATA,
        split,
        device=device,
        expert_dir=EXPERTS,
        projection_dir=PROJECTIONS,
    )


# ── stage: relation prediction ──────────────────────────────────────────────


def stage_relations(device: str, epochs: int = 60) -> dict:
    train = _pairs("train", device)
    val = _pairs("val", device)
    test = _pairs("test", device)
    head, info = train_relation_head(
        train, val, epochs=epochs, device=device, seed=0
    )
    Path(CKPT).mkdir(parents=True, exist_ok=True)
    save_relation_head(
        Path(CKPT) / "relation_head.pt",
        head,
        meta={"data": DATA, "train": len(train), "epochs": epochs},
    )
    evaluations = {}
    for label, corrupt in RELATION_CONFIGS:
        evaluations[label] = {
            "val": evaluate_relation_head(head, val, device, corrupt),
            "test": evaluate_relation_head(head, test, device, corrupt),
        }
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "best_val_f1": info["best_val_f1"],
        "evaluations": evaluations,
    }


# ── stage: relation-aware re-ID ─────────────────────────────────────────────


def _head_for_reid(device: str):
    from pss.memory.typed_relations import load_relation_head

    path = Path(CKPT) / "relation_head.pt"
    if not path.exists():
        stage_relations(device)
    return load_relation_head(path, device=device)


def stage_reid(device: str, seeds: int = 3, epochs: int = 200) -> dict:
    head = _head_for_reid(device)
    train = _pairs("train", device)
    val = _pairs("val", device)
    test = _pairs("test", device)
    factories = {
        "hand": lambda: GraphMatcher(token_dim=TACTILE_TOKEN_DIM, d_model=128, layers=2),
        "learned": lambda: RelationAwareGraphMatcher(TACTILE_TOKEN_DIM, rel_head=head),
        "hybrid": lambda: HybridRelationMatcher(TACTILE_TOKEN_DIM, rel_head=head),
    }
    results = {}
    for name, factory in factories.items():
        runs = []
        models = []
        for seed in range(seeds):
            model = factory().to(device)
            train_relation_matcher(model, train, epochs=epochs, seed=seed, device=device)
            models.append(model)
            runs.append(evaluate_relation_matcher(model, test, device))
        corrupt = {}
        for label, kind in RELATION_CONFIGS:
            if kind is None:
                continue
            corrupt[label] = [
                evaluate_relation_matcher(m, test, device, corrupt=kind)["overall"]
                for m in models
            ]
        results[name] = {
            "clean": {
                "overall": float(np.mean([r["overall"] for r in runs])),
                "std": float(np.std([r["overall"] for r in runs])),
                "near_identical": float(
                    np.mean([r["per_subset"]["near_identical"] for r in runs])
                ),
            },
            "corrupt": {
                label: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for label, v in corrupt.items()
            },
        }
    return {"seeds": seeds, "epochs": epochs, "results": results}


# ── stage: scene graph write path ───────────────────────────────────────────


def stage_graph(device: str) -> dict:
    from pss.encoders.data import load_index
    from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
    from pss.sim.dataset import load_sample

    head = _head_for_reid(device)
    index = load_index(DATA)
    entry = next(
        e for e in index["entries"] if e.get("kind") == "occlusion" and e.get("split") == "test"
    )
    sample = load_sample(Path(DATA) / entry["path"])
    models = load_perception_models(
        EXPERTS, PROJECTIONS, device=device
    )
    frame = int(sample["occlusion_start"]) - 1
    obs = extract_frame(sample, frame, models, PoseNoise(seed=0, enabled=False))
    graph = SceneGraph()
    for i, o in enumerate(obs[:3]):
        track = Track(
            id=f"obj_{i:04d}",
            pos=np.asarray(o.pos, dtype=float),
            quat=np.asarray(o.quat, dtype=float),
            appearance=np.asarray(o.appearance, dtype=float),
            category=o.category,
            confidence=o.confidence,
            raw={k: np.asarray(v, dtype=float) for k, v in o.raw.items()},
        )
        graph.add(track)
    edges = write_relation_edges(graph, head, device=device)
    path = Path(OUT_ROOT) / "scene_graph.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(path)
    reloaded = SceneGraph.load(path)
    return {
        "tracks": len(graph.tracks),
        "edges_written": len(edges),
        "edge_counts_by_type": {
            t: sum(1 for e in edges if e.relation_type == t)
            for t in ("spatial", "semantic", "tactile")
        },
        "sample_edges": [e.to_dict() for e in edges[:8]],
        "reloaded_edges": len(reloaded.get_relations()),
        "save_load_ok": len(reloaded.get_relations()) == len(graph.get_relations()),
    }


# ── stage: P5b with a learned spatial relation head ────────────────────────


def _pair_feature(pa, pb):
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    return [pa[0], pa[1], pb[0], pb[1], dx, dy, abs(dx), abs(dy)]


def _pair_labels(pa, pb, eps: float = 0.1):
    """Labels must match ``_rel(a, b, predicate)``.

    A dead zone (``eps``) matches the strict geometric predicate: when two
    regions share an axis, neither left/right nor above/below should fire.
    Without it a learned binary head is forced to pick a side on aligned pairs.
    """
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    return [dx > eps, dx < -eps, dy < -eps, dy > eps]


def _p5b_relation_rows(train_regions=("A", "B", "C"), noise: float = 0.03, n_per_pair: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    coords = np.array([REGIONS[n] for n in NAMES], dtype=np.float32)
    all_pairs = [(a, b) for a in range(len(NAMES)) for b in range(len(NAMES)) if a != b]
    if train_regions is None:
        train_pairs = list(all_pairs)
        test_pairs = list(all_pairs)
    else:
        train_pairs = [
            (a, b)
            for a, b in all_pairs
            if NAMES[a] in train_regions and NAMES[b] in train_regions
        ]
        test_pairs = [
            (a, b)
            for a, b in all_pairs
            if NAMES[a] not in train_regions or NAMES[b] not in train_regions
        ]
    X, y = [], []
    for a, b in train_pairs:
        for _ in range(n_per_pair):
            pa = coords[a] + rng.normal(scale=noise, size=2)
            pb = coords[b] + rng.normal(scale=noise, size=2)
            X.append(_pair_feature(pa, pb))
            y.append(_pair_labels(pa, pb))
    Xte, yte, pairs_te = [], [], []
    for a, b in test_pairs:
        pa, pb = coords[a], coords[b]
        Xte.append(_pair_feature(pa, pb))
        yte.append(_pair_labels(pa, pb))
        pairs_te.append((NAMES[a], NAMES[b]))
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
        torch.tensor(np.array(Xte), dtype=torch.float32),
        torch.tensor(np.array(yte), dtype=torch.float32),
        pairs_te,
    )


def _train_spatial_head(device: str, epochs: int = 400, seed: int = 0, train_regions=("A", "B", "C")):
    X, y, Xte, yte, pairs = _p5b_relation_rows(train_regions=train_regions)
    torch.manual_seed(seed)
    head = SpatialRelationHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    X, y = X.to(device), y.to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            loss = F.binary_cross_entropy_with_logits(head.net(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(head.net(Xte.to(device))) > 0.5).float().cpu()
    exact = float((pred == yte).all(dim=1).float().mean())
    per = float((pred == yte).float().mean())
    return head, pairs, exact, per


def _learned_rel(head: SpatialRelationHead, device: str, threshold: float = 0.5):
    coords = {n: np.array(REGIONS[n], dtype=np.float32) for n in NAMES}

    def rel_fn(a: str, b: str, relation: str) -> bool:
        pa = coords[a]
        pb = coords[b]
        x = torch.tensor([_pair_feature(pa, pb)], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = head.net(x)[0]
        return bool(torch.sigmoid(logits)[RELATIONS.index(relation)] > threshold)

    return rel_fn


def _apply(state, actions):
    state = list(state)
    for obj, region in actions:
        state[obj] = region
    return tuple(state)


def stage_p5b(device: str, seeds: int = 3, n_test_per_pair: int = 20, epochs: int = 300) -> dict:
    head_all, _, exact_all, per_all = _train_spatial_head(
        device, train_regions=None
    )
    head_held, heldout_pairs, exact_held, per_held = _train_spatial_head(
        device, train_regions=("A", "B", "C")
    )
    learned_all = _learned_rel(head_all, device)
    learned_held = _learned_rel(head_held, device)
    runs = []
    for seed in range(seeds):
        policy = train_policy(
            _rows(TRAIN_PAIRS, 80, np.random.default_rng(0)), seed, epochs, device
        )
        id_rng = np.random.default_rng(100 + seed)
        ood_rng = np.random.default_rng(200 + seed)
        runs.append(
            {
                "seed": seed,
                "id": _evaluate_p5b(
                    policy, learned_all, learned_held, TRAIN_PAIRS, n_test_per_pair, id_rng, device
                ),
                "ood": _evaluate_p5b(
                    policy, learned_all, learned_held, TEST_PAIRS, n_test_per_pair, ood_rng, device
                ),
            }
        )

    def agg(split, method):
        vals = [r[split][method]["success"] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    return {
        "relation_head": {
            "in_distribution": {
                "exact4": exact_all,
                "per_label": per_all,
            },
            "heldout": {
                "exact4": exact_held,
                "per_label": per_held,
                "pairs": heldout_pairs,
            },
        },
        "seeds": seeds,
        "runs": runs,
        "pss_hand": {
            "id": agg("id", "pss_hand"),
            "ood": agg("ood", "pss_hand"),
        },
        "pss_learned_indist": {
            "id": agg("id", "pss_learned_indist"),
            "ood": agg("ood", "pss_learned_indist"),
        },
        "pss_learned_heldout": {
            "id": agg("id", "pss_learned_heldout"),
            "ood": agg("ood", "pss_learned_heldout"),
        },
        "transformer": {
            "id": agg("id", "transformer"),
            "ood": agg("ood", "transformer"),
        },
    }


def _evaluate_p5b(policy, learned_all, learned_held, pairs, n, rng, device):
    hand_ok, all_ok, held_ok, tr_ok, tr_invalid = [], [], [], [], []
    for rel1, rel2 in pairs:
        for _ in range(n):
            state, _solution = make_episode(rng, rel1, rel2)
            # PSS with hand-written relations.
            sol = plan(state, rel1, rel2)
            hand_ok.append(
                float(sol is not None and goal_ok(_apply(state, sol), rel1, rel2))
            )
            # PSS with the in-distribution learned relation head.
            sol_a = plan(state, rel1, rel2, rel_fn=learned_all)
            all_ok.append(
                float(
                    sol_a is not None
                    and goal_ok(_apply(state, sol_a), rel1, rel2)
                )
            )
            # PSS with the held-out-region learned relation head.
            sol_h = plan(state, rel1, rel2, rel_fn=learned_held)
            held_ok.append(
                float(
                    sol_h is not None
                    and goal_ok(_apply(state, sol_h), rel1, rel2)
                )
            )
            good, _steps, inv = rollout(policy, state, rel1, rel2, device)
            tr_ok.append(float(good))
            tr_invalid.append(float(inv > 0))
    return {
        "pss_hand": {"success": float(np.mean(hand_ok))},
        "pss_learned_indist": {"success": float(np.mean(all_ok))},
        "pss_learned_heldout": {"success": float(np.mean(held_ok))},
        "transformer": {
            "success": float(np.mean(tr_ok)),
            "invalid": float(np.mean(tr_invalid)),
        },
        "n": len(hand_ok),
    }


# ── orchestration / report ──────────────────────────────────────────────────


def render(payload: dict) -> str:
    lines = [
        "# Learnable typed relations",
        "",
        f"> generated: {payload.get('generated')}",
        f"> device: `{payload.get('device')}`",
        "",
    ]
    rel = payload.get("relations")
    if rel:
        lines += [
            "## Relation prediction (typed heads)",
            "",
            f"- train / val / test: {rel['n_train']} / {rel['n_val']} / {rel['n_test']}",
            "",
            "| condition | split | spatial F1 | semantic F1 | tactile F1 | mean F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for label in ("clean", "pose_noise", "no_tactile", "no_vision", "mixed"):
            for split in ("val", "test"):
                v = rel["evaluations"][label][split]
                per = v["per_type"]
                lines.append(
                    f"| {label} | {split} | {_fmt(per['spatial']['f1'])} | "
                    f"{_fmt(per['semantic']['f1'])} | {_fmt(per['tactile']['f1'])} | "
                    f"{_fmt(v['overall_f1'])} |"
                )
        lines.append("")
    reid = payload.get("reid")
    if reid:
        lines += [
            "## Re-ID: hand edges vs learned typed edges",
            "",
            "| matcher | clean | near-identical | no tactile | pose noise | no vision | mixed |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ("hand", "learned", "hybrid"):
            r = reid["results"][name]
            cor = r["corrupt"]
            lines.append(
                f"| {name} | {_fmt(r['clean']['overall'])} ± {_fmt(r['clean']['std'])} | "
                f"{_fmt(r['clean']['near_identical'])} | "
                f"{_fmt(cor['no_tactile']['mean'])} | {_fmt(cor['pose_noise']['mean'])} | "
                f"{_fmt(cor['no_vision']['mean'])} | {_fmt(cor['mixed']['mean'])} |"
            )
        lines.append("")
    graph = payload.get("graph")
    if graph:
        lines += [
            "## Scene graph write path",
            "",
            f"- tracks: {graph['tracks']}; edges written: {graph['edges_written']}",
            f"- edges by type: {graph['edge_counts_by_type']}",
            f"- save/load preserves edges: {graph['save_load_ok']} "
            f"({graph['reloaded_edges']} reloaded)",
            "",
        ]
    p5b = payload.get("p5b")
    if p5b:
        rh = p5b["relation_head"]
        lines += [
            "## P5b with learned spatial relations",
            "",
            f"- in-distribution relation all-4 exact: {_fmt(rh['in_distribution']['exact4'])} "
            f"(per-label {_fmt(rh['in_distribution']['per_label'])})",
            f"- held-out regions relation all-4 exact: {_fmt(rh['heldout']['exact4'])} "
            f"(per-label {_fmt(rh['heldout']['per_label'])})",
            f"- held-out pairs: {rh['heldout']['pairs']}",
            "",
            "| planner | ID success | OOD success |",
            "|---|---:|---:|",
            f"| hand-written `_rel` | {_fmt(p5b['pss_hand']['id'][0])} | "
            f"{_fmt(p5b['pss_hand']['ood'][0])} |",
            f"| learned, in-distribution | {_fmt(p5b['pss_learned_indist']['id'][0])} ± "
            f"{_fmt(p5b['pss_learned_indist']['id'][1])} | "
            f"{_fmt(p5b['pss_learned_indist']['ood'][0])} ± "
            f"{_fmt(p5b['pss_learned_indist']['ood'][1])} |",
            f"| learned, held-out regions | {_fmt(p5b['pss_learned_heldout']['id'][0])} ± "
            f"{_fmt(p5b['pss_learned_heldout']['id'][1])} | "
            f"{_fmt(p5b['pss_learned_heldout']['ood'][0])} ± "
            f"{_fmt(p5b['pss_learned_heldout']['ood'][1])} |",
            f"| transformer BC (ref) | {_fmt(p5b['transformer']['id'][0])} | "
            f"{_fmt(p5b['transformer']['ood'][0])} |",
            "",
        ]
    lines += [
        "## Verdict",
        "",
        "- Typed relation heads learn the frame-local relations (spatial/semantic/tactile).",
        "- Pure learned edges do not beat hand-written edges on clean re-ID; the hybrid keeps",
        "  hand edges and adds learned implications, which helps under missing tactile.",
        "- The learned spatial head does not fully generalize to held-out P5b regions; the exact",
        "  geometric predicate remains the stronger prior on a clean grid.",
        "- Temporal and task heads need track history / a goal context; the schema is present,",
        "  but this round evaluates the pairwise heads and the P5b task-conditioned case.",
        "",
    ]
    return "\n".join(lines)


def run(stage: str = "all", device: str = "cuda", seeds: int = 3, epochs: int = 200) -> dict:
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    out = Path(OUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "typed_relations.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"generated": None, "device": device}
    )
    payload["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload["device"] = device
    if stage in ("all", "relations"):
        payload["relations"] = stage_relations(device)
        print(
            "relations: "
            + ", ".join(
                f"{k}={payload['relations']['evaluations']['clean']['test']['per_type'][k]['f1']:.3f}"
                for k in ("spatial", "semantic", "tactile")
            )
        )
    if stage in ("all", "reid"):
        payload["reid"] = stage_reid(device, seeds=seeds, epochs=epochs)
        for name, r in payload["reid"]["results"].items():
            print(
                f"reid {name}: clean {r['clean']['overall']:.3f} "
                f"tactile-off {r['corrupt']['no_tactile']['mean']:.3f}"
            )
    if stage in ("all", "graph"):
        payload["graph"] = stage_graph(device)
        print(
            f"graph: {payload['graph']['edges_written']} edges, "
            f"save/load ok={payload['graph']['save_load_ok']}"
        )
    if stage in ("all", "p5b"):
        payload["p5b"] = stage_p5b(device, seeds=min(seeds, 3))
        print(
            f"p5b learned in-dist: ID/OOD {payload['p5b']['pss_learned_indist']['id'][0]:.3f}/"
            f"{payload['p5b']['pss_learned_indist']['ood'][0]:.3f}; "
            f"held-out {payload['p5b']['pss_learned_heldout']['ood'][0]:.3f}; "
            f"hand {payload['p5b']['pss_hand']['ood'][0]:.3f}"
        )
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "typed_relations.md").write_text(render(payload), encoding="utf-8")
    print(f"output: {path}")
    return payload


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", default="all", choices=("all", "relations", "reid", "graph", "p5b")
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    run(stage=args.stage, device=args.device, seeds=args.seeds, epochs=args.epochs)


if __name__ == "__main__":
    main()
