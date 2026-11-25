"""Relation metrics (T25): per-predicate F1 and consistency violations."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

Triple = tuple[str, str, str]

ASYMMETRIC = ("left_of", "right_of", "in_front_of", "behind", "above", "below", "on_top_of", "inside")
INVERSE = {
    "left_of": "right_of",
    "right_of": "left_of",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "above": "below",
    "below": "above",
}
TRANSITIVE = ("left_of", "right_of", "in_front_of", "behind", "above", "below")


def _predicate_of(triple: tuple) -> str:
    # supports (subj, pred, obj) and (frame, subj, pred, obj)
    return str(triple[-2])


def per_predicate_prf(computed: Iterable[Triple], truth: Iterable[Triple]) -> dict:
    computed = set(computed)
    truth = set(truth)
    predicates = sorted({_predicate_of(t) for t in computed | truth})
    per = {}
    tp_total = fp_total = fn_total = 0
    for pred in predicates:
        c = {t for t in computed if _predicate_of(t) == pred}
        g = {t for t in truth if _predicate_of(t) == pred}
        tp = len(c & g)
        fp = len(c - g)
        fn = len(g - c)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[pred] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro_p = tp_total / (tp_total + fp_total) if tp_total + fp_total else 1.0
    micro_r = tp_total / (tp_total + fn_total) if tp_total + fn_total else 1.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    macro_f1 = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    return {
        "per_predicate": per,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro_f1": macro_f1,
        "predicates": len(per),
    }


def consistency_violations(triples: Iterable[Triple]) -> dict:
    triples = set(triples)
    violations: list[dict] = []

    for subj, pred, obj in triples:
        if pred in ASYMMETRIC and (obj, pred, subj) in triples:
            violations.append({"kind": "asymmetric_both_ways", "pred": pred, "pair": (subj, obj)})
        inverse = INVERSE.get(pred)
        if inverse and (subj, inverse, obj) in triples:
            violations.append({"kind": "inverse_on_same_pair", "pred": pred, "pair": (subj, obj)})

    for subj, pred, obj in triples:
        if pred not in TRANSITIVE:
            continue
        for mid, pred2, obj2 in triples:
            if pred2 != pred or mid != obj:
                continue
            if (subj, pred, obj2) not in triples:
                violations.append(
                    {"kind": "transitivity", "pred": pred, "chain": (subj, obj, obj2)}
                )

    return {
        "violations": violations,
        "violation_count": len(violations),
        "rate": len(violations) / max(1, len(triples)),
    }
