"""Object recognition demo (rebuilt from the 11.19 document, section 6.2-6.5).

Pipeline: hand-defined attributes -> random orthogonal "projection" -> Hadamard
slot binding -> per-object bound vector -> concept retrieval + attribute query.

This is a **skeleton validator**, not a learned system:
- attribute bases are hand-defined, not extracted from pixels;
- the projection is a random orthogonal matrix, not a contrastively trained MLP;
- only single-modality scoring is used here. The summed object vector is fine
  for similarity scoring (9.2.3 route B) but must never be used to fuse
  modalities.

Run: ``python -m pss.demos.semantic_space_demo``
"""

from __future__ import annotations

import numpy as np

from pss._console import ensure_utf8
from pss.binding import bind, cosine, make_slots, slot_orthogonality, unbind

DIM = 128
N_SLOTS = 4

ATTRS = [
    "颜色_红色",
    "颜色_蓝色",
    "形状_弧形曲面",
    "形状_直线棱角",
    "距离_近距离(50cm)",
    "距离_中距离(1m)",
    "纹理_光滑",
    "硬度_硬质",
]

CONCEPTS = {
    "苹果": ["颜色_红色", "形状_弧形曲面", "距离_近距离(50cm)", "纹理_光滑", "硬度_硬质"],
    "杯子": ["颜色_蓝色", "形状_直线棱角", "距离_中距离(1m)", "纹理_光滑", "硬度_硬质"],
}

COMPARE_ATTRS = [
    "颜色_红色",
    "颜色_蓝色",
    "形状_弧形曲面",
    "形状_直线棱角",
    "距离_近距离(50cm)",
    "距离_中距离(1m)",
]

EXPECTED_WINNER = {
    "颜色_红色": "A",
    "颜色_蓝色": "B",
    "形状_弧形曲面": "A",
    "形状_直线棱角": "B",
    "距离_近距离(50cm)": "A",
    "距离_中距离(1m)": "B",
}


def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def _public_space(seed: int):
    """Build projected attribute bases and Hadamard slots."""
    rng = np.random.default_rng(seed)
    q = _random_orthogonal(DIM, rng)
    raw = {}
    for name in ATTRS:
        v = rng.normal(size=DIM)
        raw[name] = v / np.linalg.norm(v)
    # The projection must be applied to attribute bases too, otherwise they live
    # in a different coordinate frame from the object features (6.2 pitfall).
    proj = {name: q @ v for name, v in raw.items()}
    slots = make_slots("hadamard", DIM, N_SLOTS, seed=seed)
    return proj, slots


def _build_bound(feats, proj, slot) -> np.ndarray:
    vec = np.zeros(DIM)
    for name in feats:
        vec += bind(proj[name], slot)
    return vec


def _prototype(feats, proj) -> np.ndarray:
    vec = np.zeros(DIM)
    for name in feats:
        vec += proj[name]
    return vec


def _bar(score: float, width: int = 10) -> str:
    n = max(0, min(width, int(round(max(0.0, score) * width))))
    return "█" * n


def run(seed: int = 7, verbose: bool = True) -> dict:
    proj, slots = _public_space(seed)
    if verbose:
        print("=== 公共语义空间 Demo（Phase 0 重建）===")
        print("说明：属性基向量人工定义；投影为随机正交矩阵；仅验证骨架链路。")
        print(f"公共维度 D={DIM}，槽数 K={N_SLOTS}，槽方案=Hadamard")
        print(f"槽间最大非对角内积={slot_orthogonality(slots):.2e}")
        print()

    prototypes = {concept: _prototype(feats, proj) for concept, feats in CONCEPTS.items()}

    objects = {}
    for idx, (concept, feats) in enumerate(CONCEPTS.items()):
        objects[f"obj{idx}"] = {
            "concept": concept,
            "features": list(feats),
            "slot_idx": idx,
            "vec": _build_bound(feats, proj, slots[idx]),
        }

    results = {"objects": {}, "comparisons": {}, "all_correct": True}
    scores_by_obj = {}

    for oid, obj in objects.items():
        slot = slots[obj["slot_idx"]]
        feat_sum = unbind(obj["vec"], slot)
        sims = {name: cosine(feat_sum, proto) for name, proto in prototypes.items()}
        ranked = sorted(sims.items(), key=lambda kv: -kv[1])

        attr_scores = {
            name: cosine(obj["vec"], bind(proj[name], slot)) for name in ATTRS
        }
        scores_by_obj[oid] = attr_scores
        ranked_attrs = sorted(attr_scores.items(), key=lambda kv: -kv[1])
        top_attrs = [name for name, _ in ranked_attrs[:5]]

        concept_ok = ranked[0][0] == obj["concept"]
        attrs_ok = set(top_attrs) == set(obj["features"])
        results["all_correct"] = results["all_correct"] and concept_ok and attrs_ok
        results["objects"][oid] = {
            "concept": obj["concept"],
            "top_concept": ranked[0][0],
            "sims": {k: float(v) for k, v in sims.items()},
            "features": list(obj["features"]),
            "top_attrs": top_attrs,
            "attr_scores": {k: float(v) for k, v in attr_scores.items()},
            "concept_ok": concept_ok,
            "attrs_ok": attrs_ok,
        }

        if verbose:
            print(f"[{oid}] 概念识别（{obj['concept']}）")
            print(f"  最可能是: {ranked[0][0]} (相似度={ranked[0][1]:.3f})")
            other = ranked[1]
            print(f"  候选:     {other[0]} ({other[1]:.3f})")
            print("  属性匹配 Top 5:")
            for name, score in ranked_attrs[:5]:
                mark = "✓" if name in obj["features"] else " "
                print(f"    {name:<16} {score:+.3f} {_bar(score)} {mark}")
            print()

    if verbose:
        print("跨客体对比（obj0 苹果 vs obj1 杯子）")

    for name in COMPARE_ATTRS:
        a = scores_by_obj["obj0"][name]
        b = scores_by_obj["obj1"][name]
        winner = "A" if a > b else "B"
        expected = EXPECTED_WINNER[name]
        ok = winner == expected
        results["all_correct"] = results["all_correct"] and ok
        results["comparisons"][name] = {
            "A": float(a),
            "B": float(b),
            "winner": winner,
            "expected": expected,
            "ok": ok,
        }
        if verbose:
            print(
                f"  {name:<16}: A={a:+.3f}, B={b:+.3f} → 物体{winner}更强 "
                f"(Δ={a - b:+.3f}) {'✓' if ok else '✗'}"
            )

    if verbose:
        print()
        print(f"全部判断正确: {'✅' if results['all_correct'] else '❌'}")
        print()
        print("边界（6.5）：无真实感官输入；投影为随机矩阵；未训练跨模态对齐；")
        print("仅验证绑定-解绑、客体识别、属性查询、跨客体对比的骨架逻辑。")

    return results


if __name__ == "__main__":
    ensure_utf8()
    run()
