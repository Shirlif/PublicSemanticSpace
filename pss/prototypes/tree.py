"""Hierarchical prototype tree (Phase 5).

Built offline from category prototypes by agglomerative clustering on fused
per-modality similarity. Leaves are categories; internal nodes store
per-modality centroids of their children. The topology is data-driven; the
"semantic review" is an offline gate in the real pipeline, not in this code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .bank import PrototypeBank, l2


def _fused_similarity(bank: PrototypeBank) -> np.ndarray:
    sims = [bank.protos[m] @ bank.protos[m].T for m in bank.protos]
    return np.mean(sims, axis=0)


def _node_score(query: dict[str, np.ndarray], node: dict) -> float:
    sims = [
        float(np.dot(l2(query[m]), l2(node["protos"][m])))
        for m in query
        if m in node["protos"]
    ]
    return float(np.mean(sims)) if sims else -1.0


def build_tree(bank: PrototypeBank, method: str = "average") -> dict:
    n = len(bank.categories)
    nodes: dict[str, dict] = {}
    for i, category in enumerate(bank.categories):
        support = sum(bank.stats[category][m]["sample_count"] for m in bank.stats[category])
        nodes[f"leaf_{i}"] = {
            "node_id": f"leaf_{i}",
            "category": category,
            "children": [],
            "parent": None,
            "protos": {m: bank.protos[m][i].astype(np.float64) for m in bank.protos},
            "support": int(support),
            "distance": 0.0,
        }

    if n == 1:
        root = "leaf_0"
    else:
        sim = _fused_similarity(bank)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method=method)
        for k, row in enumerate(Z):
            a, b, d, _count = int(row[0]), int(row[1]), float(row[2]), int(row[3])
            aid = f"leaf_{a}" if a < n else f"node_{a}"
            bid = f"leaf_{b}" if b < n else f"node_{b}"
            na, nb = nodes[aid], nodes[bid]
            nid = f"node_{n + k}"
            protos = {m: l2(na["protos"][m] + nb["protos"][m]) for m in bank.protos}
            nodes[nid] = {
                "node_id": nid,
                "category": None,
                "children": [aid, bid],
                "parent": None,
                "protos": protos,
                "support": na["support"] + nb["support"],
                "distance": d,
            }
            na["parent"] = nid
            nb["parent"] = nid
        root = f"node_{2 * n - 2}"

    return {"root": root, "nodes": nodes, "version": f"{bank.version}_tree"}


def tree_candidates(
    query: dict[str, np.ndarray],
    tree: dict,
    top_n: int = 2,
    threshold: float = -1.0,
) -> list[str]:
    """Confidence-driven dynamic top-N descent from the root."""
    nodes = tree["nodes"]

    def descend(node_id: str) -> list[str]:
        node = nodes[node_id]
        if node["category"] is not None:
            return [node["category"]]
        children = [nodes[c] for c in node["children"]]
        scored = sorted(((_node_score(query, c), c) for c in children), key=lambda x: -x[0])
        keep = [c for s, c in scored if s >= threshold][:top_n]
        if not keep:
            keep = [scored[0][1]]
        out: list[str] = []
        for child in keep:
            out.extend(descend(child["node_id"]))
        return out

    out: list[str] = []
    seen: set[str] = set()
    for category in descend(tree["root"]):
        if category not in seen:
            seen.add(category)
            out.append(category)
    return out


def tree_to_json(tree: dict) -> dict:
    payload = {"root": tree["root"], "version": tree.get("version", ""), "nodes": {}}
    for nid, node in tree["nodes"].items():
        payload["nodes"][nid] = {k: v for k, v in node.items() if k != "protos"}
        payload["nodes"][nid]["protos"] = {
            m: np.asarray(v).tolist() for m, v in node["protos"].items()
        }
    return payload


def save_tree(path: str | Path, tree: dict) -> None:
    Path(path).write_text(
        json.dumps(tree_to_json(tree), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_tree(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for node in payload["nodes"].values():
        node["protos"] = {
            m: np.asarray(v, dtype=np.float64) for m, v in node["protos"].items()
        }
    return payload
