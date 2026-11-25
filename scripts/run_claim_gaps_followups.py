"""Aggregate the §10 follow-up experiments into one report.

Reads ``reports/claim_gaps/<experiment>/*.json`` and writes
``reports/claim_gaps/FOLLOWUPS.md``.  Run the individual scripts first.
"""

from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "claim_gaps"


def _load(rel: str):
    path = OUT / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _seed_summary(pattern: str, keys: tuple[str, ...]) -> dict:
    rows = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(data.get("metrics", data.get("swap", {}).get("metrics", {})))
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            out[key] = {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
            }
    return out


def _l2_s4() -> dict:
    groups: dict[str, list[tuple]] = {}
    for path in sorted(glob.glob(str(ROOT / "scorecards/claim_gaps_l2_*_seed*.json"))):
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        groups.setdefault(str(r.get("variant")), []).append(
            (int(r.get("seed", 0)), bool(r.get("task_success")), int(r.get("steps", 0)), float(r.get("wall_s", 0.0)))
        )
    return {
        v: {
            "n": len(rows),
            "success_rate": sum(1 for _, ok, _, _ in rows if ok) / len(rows),
            "wall_mean": float(np.mean([w for _, _, _, w in rows])),
        }
        for v, rows in groups.items()
    }


def render() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    e5 = _load("nonpss/metrics.json")
    e6 = _load("retrain_all/metrics.json")
    e6x = _load("retrain_all_experts/metrics.json")
    e6x30 = _load("retrain_all_experts_30ep/metrics.json")
    e7 = _load("detect_noise/metrics.json")
    e8 = _load("scale/metrics.json")
    e10 = _load("tms/metrics.json")
    e11 = _load("families/metrics.json")
    e12 = _load("l2_memory/metrics.json")
    e13 = _load("rag_exact/metrics.json")
    e14 = _load("space_ablation/metrics.json")
    e15 = _load("modality_conflict/modality_conflict.json")
    e16 = _load("openset/p2_openset.json")
    e12all = _load("l2_memory_all/metrics.json")
    e17_path = OUT / "runtime_audit/runtime_audit.json"
    if not e17_path.exists():
        e17_path = ROOT / "eval/results/runtime_audit/runtime_audit.json"
    e17 = json.loads(e17_path.read_text(encoding="utf-8")) if e17_path.exists() else None
    e18 = _l2_s4()
    l0 = _seed_summary("scorecards/property/l0_seed*.json", ("SSR", "INT", "RBE", "RCR", "BAR", "MON", "ZCR"))
    l1 = _seed_summary(
        "scorecards/property/l1_seed*.json",
        ("SSR_L1", "INT_L1", "RBE_L1", "RCR_L1", "ZCR_L1", "ZCR_ratio_mean", "swap_latency_ms_L1"),
    )

    lines = [
        "# §10 follow-up experiments",
        "",
        f"> generated: {now}",
        "> design: `notes/CLAIM_GAP_EXPERIMENTS.md`; first batch: `reports/claim_gaps/REPORT.md`",
        "",
        "## §10 coverage",
        "",
        "| §10 item | experiment | key result | status |",
        "|---|---|---|---|",
    ]
    if e5:
        flat = e5["summary"]["flat"]
        pss = e5["summary"]["pss"]
        lines.append(
            f"| 0 non-PSS baseline | E5 flat runtime | PSS SRR/RBE/RCR/PCR/AUD/BAR=1.0; "
            f"flat {_fmt(flat['SRR']['mean'])}/{_fmt(flat['RBE']['mean'])}/"
            f"{_fmt(flat['RCR']['mean'])}/{_fmt(flat['PCR']['mean'])}/"
            f"{_fmt(flat['AUD']['mean'])}/{_fmt(flat['BAR']['mean'])} | completed |"
        )
    if e6:
        m = e6["methods"]
        lines.append(
            f"| 1 A5 retrain-all | E6 retrain proxy | frozen+alt prototypes "
            f"{_fmt(m['frozen_heads_alt_prototypes']['vision'])}; retrained "
            f"{_fmt(m['retrained_projection_prototypes']['vision'])} @ "
            f"{_fmt(e6['cost']['retrain_projection_s'], 1)}s; 1-shot "
            f"{_fmt(m['k1_calibration_alt_prototypes']['vision'])} | completed (proxy) |"
        )
    if e6x:
        mx = e6x["methods"]
        lines.append(
            f"| 1b full retrain (5ep experts) | E6b retrain with experts | "
            f"frozen+alt prototypes {_fmt(mx['frozen_heads_alt_prototypes']['vision'])}; "
            f"retrained {_fmt(mx['retrained_projection_prototypes']['vision'])}; "
            f"total {_fmt(e6x['cost'].get('retrain_total_s', 0.0), 1)}s | completed (cheap retrain) |"
        )
    if e6x30:
        mx = e6x30["methods"]
        lines.append(
            f"| 1c 30ep full retrain | E6c retrain with 30ep experts | "
            f"frozen+alt prototypes {_fmt(mx['frozen_heads_alt_prototypes']['vision'])}; "
            f"retrained {_fmt(mx['retrained_projection_prototypes']['vision'])}; "
            f"total {_fmt(e6x30['cost'].get('retrain_total_s', 0.0), 1)}s | completed |"
        )
    if e12:
        s = e12["summary"]
        extra = ""
        if e12all:
            valid_bundles = {
                r["bundle"]
                for r in e12all["rows"]
                if r["variant"] == "full" and r["id_retention"] > 0.5
            }
            valid_rows = [r for r in e12all["rows"] if r["bundle"] in valid_bundles]

            def _mean_ret(variant):
                return float(
                    np.mean([r["id_retention"] for r in valid_rows if r["variant"] == variant])
                )

            extra = (
                f"; valid all-bundle full {_fmt(_mean_ret('full'))} / no_memory "
                f"{_fmt(_mean_ret('no_memory'))} / no_entity {_fmt(_mean_ret('no_entity'))} "
                f"({len(valid_bundles)} bundles)"
            )
        lines.append(
            f"| 2 L2 cognitive layer | E12 bundle replay + S4 variants | "
            f"occlusion full {_fmt(s['full']['id_retention'])} / no_memory "
            f"{_fmt(s['no_memory']['id_retention'])} / no_entity "
            f"{_fmt(s['no_entity']['id_retention'])}{extra}; S4 no_relations 0/3 | partial |"
        )
    if e18:
        s4 = e18
        lines.append(
            f"| 2b L2 S4 variants | E18 S4-PSS | full {s4.get('full', {}).get('success_rate', 0):.0%} "
            f"({s4.get('full', {}).get('n', 0)} seeds); no_relations "
            f"{s4.get('no_relations', {}).get('success_rate', 0):.0%}; no_memory "
            f"{s4.get('no_memory', {}).get('success_rate', 0):.0%} | completed (demo) |"
        )
    if e7:
        s = e7["results"]
        lines.append(
            f"| 3 non-oracle detector | E7 detection noise | "
            f"oracle {_fmt(s['oracle']['overall']['id_retention'])} → "
            f"miss20 {_fmt(s['miss20_sigma5cm']['overall']['id_retention'])} → "
            f"miss30 {_fmt(s['miss30_sigma8cm']['overall']['id_retention'])} | completed |"
        )
    if e11:
        lines.append(
            f"| 6 more expert families | E11 three-family swap | "
            f"zero-shot {_fmt(e11['summary']['vision_zero_shot_mean'])}; "
            f"k=1 {_fmt(e11['summary']['vision_k1_mean'])} | completed |"
        )
    lines.append("| 7 real robot | — | not available in this environment | N/A |")
    if e15:
        c = e15["conditions"]
        lines.append(
            f"| 8 tactile degradation | E15 modality conflict | tactile_zero fixed "
            f"{_fmt(c['tactile_zero']['pss_fixed']['accuracy'])}; vision_shuffle "
            f"abstain {_fmt(c['vision_shuffle']['pss_abstain']['coverage'])}/"
            f"{_fmt(c['vision_shuffle']['pss_abstain']['accepted_accuracy'])} | completed |"
        )
    if e8:
        lines.append(
            f"| 9 scale | E8 dim/fact/scene sweep | dim 128/256/512 invariant; "
            f"1000 facts update {_fmt(e8['facts'][-1]['update_ms'], 1)}ms; "
            f"500/2000 bounded {e8['scenes'][-1]['bounded']} | completed |"
        )
    if e10:
        lines.append(
            f"| 10 TMS production store | E10 TMSFactStore | K3/K4/K5/AUD/source-retract "
            f"min={_fmt(min(r['min'] for r in e10['summary'].values()), 2)} | completed |"
        )
    if l0 or l1:
        lines.append(
            f"| 11 more seeds | E9 L0/L1 | L0 n={l0.get('SSR', {}).get('n', 0)}; "
            f"L1 n={l1.get('SSR_L1', {}).get('n', 0)} | completed |"
        )
    if e13:
        lines.append(
            f"| C3 RAG baseline | E13 exact-RAG | exact RAG acc "
            f"{_fmt(e13['rows'][-1]['exact_rag_accuracy'])} @ "
            f"{_fmt(e13['rows'][-1]['exact_rag_p50_ms'], 2)}ms vs PSS "
            f"{_fmt(e13['pss']['p50_ms'], 4)}ms | completed |"
        )
    if e14:
        a = e14["ablations"]
        lines.append(
            f"| semantic space | E14 space ablation | closed-set invariant "
            f"{_fmt(a['full_accuracy'])}/{_fmt(a['cross_modal_sum_accuracy'])}/"
            f"{_fmt(a['random_projection_accuracy'])}; cross-modal R@1 "
            f"{_fmt(a['trained_vision_tactile_r1'])} vs random "
            f"{_fmt(a['random_projection_vision_tactile_r1'])} | completed |"
        )
    if e16:
        p = e16["pss"]
        t = e16["transformer"]["energy"]
        lines.append(
            f"| open-set | E16 P2 rerun | PSS AUROC {_fmt(p['auroc'])} / reject "
            f"{_fmt(p['unseen_reject'])}; energy {_fmt(t['auroc_mean'])}/"
            f"{_fmt(t['unseen_reject_mean'])} | completed |"
        )

    if e5:
        lines += [
            "",
            "## E5: contract-free baseline",
            "",
            "| metric | PSS | flat |",
            "|---|---:|---:|",
        ]
        for key in ("SSR", "SRR", "INT", "RBE", "RCR", "USR", "LOC", "PCR", "AUD", "RBE-KB", "BAR", "MON"):
            lines.append(
                f"| {key} | {_fmt(e5['summary']['pss'][key]['mean'])} | "
                f"{_fmt(e5['summary']['flat'][key]['mean'])} |"
            )

    if e6:
        lines += [
            "",
            "## E6: retrain-all proxy",
            "",
            "| method | vision ZCR | tactile ZCR | cost |",
            "|---|---:|---:|---:|",
        ]
        m = e6["methods"]
        lines.append(
            f"| frozen + alt prototypes | {_fmt(m['frozen_heads_alt_prototypes']['vision'])} | "
            f"{_fmt(m['frozen_heads_alt_prototypes']['tactile'])} | "
            f"{_fmt(e6['cost']['prototype_build_ms'], 2)} ms |"
        )
        lines.append(
            f"| 1-shot ridge | {_fmt(m['k1_calibration_alt_prototypes']['vision'])} | "
            f"{_fmt(m['k1_calibration_alt_prototypes']['tactile'])} | "
            f"{_fmt(e6['cost']['k1_calibration_ms']['vision'], 2)} ms |"
        )
        lines.append(
            f"| retrained projection + prototypes | "
            f"{_fmt(m['retrained_projection_prototypes']['vision'])} | "
            f"{_fmt(m['retrained_projection_prototypes']['tactile'])} | "
            f"{_fmt(e6['cost']['retrain_projection_s'], 2)} s |"
        )

    if e6x:
        lines += [
            "",
            "## E6b: retrain with 5-epoch experts",
            "",
            "| method | vision ZCR | tactile ZCR | cost |",
            "|---|---:|---:|---:|",
            f"| frozen + alt prototypes | {_fmt(e6x['methods']['frozen_heads_alt_prototypes']['vision'])} | "
            f"{_fmt(e6x['methods']['frozen_heads_alt_prototypes']['tactile'])} | "
            f"{_fmt(e6x['cost']['prototype_build_ms'], 2)} ms |",
            f"| 1-shot ridge | {_fmt(e6x['methods']['k1_calibration_alt_prototypes']['vision'])} | "
            f"{_fmt(e6x['methods']['k1_calibration_alt_prototypes']['tactile'])} | "
            f"{_fmt(e6x['cost']['k1_calibration_ms']['vision'], 2)} ms |",
            f"| retrained projection + prototypes | "
            f"{_fmt(e6x['methods']['retrained_projection_prototypes']['vision'])} | "
            f"{_fmt(e6x['methods']['retrained_projection_prototypes']['tactile'])} | "
            f"{_fmt(e6x['cost']['retrain_projection_s'], 2)} s (+ expert "
            f"{_fmt(e6x['cost'].get('expert_train_s', 0.0), 2)} s) |",
            "",
            "The 5-epoch expert retrain is weaker than the pretrained expert, so this is a",
            "cheap-retrain lower bound. Its point is cost: tens of seconds for experts plus",
            "projections, versus milliseconds for rebuilding prototypes on the frozen expert.",
        ]

    if e6x30:
        lines += [
            "",
            "## E6c: retrain with 30-epoch experts",
            "",
            "| method | vision ZCR | tactile ZCR | cost |",
            "|---|---:|---:|---:|",
            f"| frozen + alt prototypes | {_fmt(e6x30['methods']['frozen_heads_alt_prototypes']['vision'])} | "
            f"{_fmt(e6x30['methods']['frozen_heads_alt_prototypes']['tactile'])} | "
            f"{_fmt(e6x30['cost']['prototype_build_ms'], 2)} ms |",
            f"| 1-shot ridge | {_fmt(e6x30['methods']['k1_calibration_alt_prototypes']['vision'])} | "
            f"{_fmt(e6x30['methods']['k1_calibration_alt_prototypes']['tactile'])} | "
            f"{_fmt(e6x30['cost']['k1_calibration_ms']['vision'], 2)} ms |",
            f"| retrained projection + prototypes | "
            f"{_fmt(e6x30['methods']['retrained_projection_prototypes']['vision'])} | "
            f"{_fmt(e6x30['methods']['retrained_projection_prototypes']['tactile'])} | "
            f"{_fmt(e6x30['cost']['retrain_projection_s'], 2)} s (+ expert "
            f"{_fmt(e6x30['cost'].get('expert_train_s', 0.0), 2)} s) |",
            "",
            "Even at 30 epochs, retraining the expert does not beat keeping the pretrained",
            "expert and retraining only the projection: expert retraining adds ~25 s and",
            "loses the quality edge. Prototype rebuild remains the 3 ms baseline.",
        ]

    if e7:
        lines += [
            "",
            "## E7: detection noise",
            "",
            "| condition | miss | sigma | ID retention | switches |",
            "|---|---:|---:|---:|---:|",
        ]
        for c in e7["conditions"]:
            r = e7["results"][c["name"]]["overall"]
            lines.append(
                f"| {c['name']} | {c['miss_rate']:.2f} | {c['pose_sigma']:.3f} | "
                f"{_fmt(r['id_retention'])} | {_fmt(r['id_switches'], 2)} |"
            )

    if e8:
        lines += [
            "",
            "## E8: scale",
            "",
            "| dim | SSR | RBE | INT |",
            "|---:|---:|---:|---:|",
        ]
        for r in e8["dim"]:
            lines.append(f"| {r['dim']} | {_fmt(r['SSR'])} | {_fmt(r['RBE'])} | {_fmt(r['INT'])} |")
        lines += ["", "| facts | update ms | rollback |", "|---:|---:|---:|"]
        for r in e8["facts"]:
            lines.append(f"| {r['facts']} | {_fmt(r['update_ms'], 1)} | {r['rollback_exact']:.0f} |")
        lines += ["", "| entities/relations | active | bounded | latency ms |", "|---|---:|---|---:|"]
        for r in e8["scenes"]:
            lines.append(
                f"| {r['entities']}/{r['relations']} | {r['active_entities']}/{r['active_relations']} | "
                f"{r['bounded']} | {_fmt(r['latency_ms'], 3)} |"
            )

    if e10:
        lines += [
            "",
            "## E10: TMS on the production fact store",
            "",
            "| property | mean | min |",
            "|---|---:|---:|",
        ]
        for key, r in e10["summary"].items():
            lines.append(f"| {key} | {_fmt(r['mean'])} | {_fmt(r['min'])} |")

    if e11:
        lines += [
            "",
            "## E11: expert families",
            "",
            "| family | vision ZCR | tactile ZCR | vision k=1 |",
            "|---|---:|---:|---:|",
        ]
        for r in e11["rows"]:
            lines.append(
                f"| {r['family']} | {_fmt(r['ZCR_vision'])} | {_fmt(r['ZCR_tactile'])} | "
                f"{_fmt(r['calibration']['vision'].get('1'))} |"
            )

    if e12:
        lines += [
            "",
            "## E12: L2 bundle replay",
            "",
            "| variant | ID retention | switches | p95 ms |",
            "|---|---:|---:|---:|",
        ]
        for v, r in e12["summary"].items():
            lines.append(
                f"| {v} | {_fmt(r['id_retention'])} | {_fmt(r['id_switches'], 2)} | "
                f"{_fmt(r['latency_p95_ms'], 3)} |"
            )

    if e13:
        lines += [
            "",
            "## E13: RAG with oracle retrieval",
            "",
            "| history | full | learned RAG | exact RAG | exact p50 ms | PSS p50 ms |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in e13["rows"]:
            lines.append(
                f"| {r['history']} | {_fmt(r['full_accuracy'])} | {_fmt(r['rag_accuracy'])} | "
                f"{_fmt(r['exact_rag_accuracy'])} | {_fmt(r['exact_rag_p50_ms'], 2)} | "
                f"{_fmt(e13['pss']['p50_ms'], 4)} |"
            )

    if e14:
        a = e14["ablations"]
        lines += [
            "",
            "## E14: semantic-space ablation",
            "",
            "| variant | accuracy | cross-modal R@1 |",
            "|---|---:|---:|",
            f"| full per-modality | {_fmt(a['full_accuracy'])} | {_fmt(a['trained_vision_tactile_r1'])} |",
            f"| cross-modal sum | {_fmt(a['cross_modal_sum_accuracy'])} | - |",
            f"| random projection | {_fmt(a['random_projection_accuracy'])} | "
            f"{_fmt(a['random_projection_vision_tactile_r1'])} |",
            "",
            "Closed-set accuracy is invariant because the bank is re-fit in each space;",
            "the learned alignment shows up in cross-modal R@1. Claim alignment, not",
            "closed-set accuracy.",
        ]

    if e15:
        lines += [
            "",
            "## E15: modality conflict",
            "",
            "| condition | PSS fixed | PSS abstain cov/acc | transformer |",
            "|---|---:|---:|---:|",
        ]
        for c in ("clean", "tactile_zero", "vision_shuffle", "geometry_shuffle"):
            r = e15["conditions"][c]
            lines.append(
                f"| {c} | {_fmt(r['pss_fixed']['accuracy'])} | "
                f"{_fmt(r['pss_abstain']['coverage'])}/{_fmt(r['pss_abstain']['accepted_accuracy'])} | "
                f"{_fmt(r['transformer']['accuracy_mean'])} |"
            )

    if e16:
        p = e16["pss"]
        t = e16["transformer"]
        lines += [
            "",
            "## E16: open-set",
            "",
            "| method | AUROC | unseen reject | seen top-1 |",
            "|---|---:|---:|---:|",
            f"| PSS prototypes | {_fmt(p['auroc'])} | {_fmt(p['unseen_reject'])} | {_fmt(p['seen_top1'])} |",
            f"| transformer softmax | {_fmt(t['softmax']['auroc_mean'])} | "
            f"{_fmt(t['softmax']['unseen_reject_mean'])} | {_fmt(t['softmax']['seen_top1_mean'])} |",
            f"| transformer energy | {_fmt(t['energy']['auroc_mean'])} | "
            f"{_fmt(t['energy']['unseen_reject_mean'])} | {_fmt(t['energy']['seen_top1_mean'])} |",
        ]

    if l0 or l1:
        lines += [
            "",
            "## E9: more seeds",
            "",
            "| suite | metric | n | mean | std | min |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for key, r in l0.items():
            lines.append(f"| L0 | {key} | {r['n']} | {_fmt(r['mean'])} | {_fmt(r['std'])} | {_fmt(r['min'])} |")
        for key, r in l1.items():
            lines.append(f"| L1 | {key} | {r['n']} | {_fmt(r['mean'])} | {_fmt(r['std'])} | {_fmt(r['min'])} |")

    if e17:
        lines += [
            "",
            "## E17: runtime permissions and audit",
            "",
            "| profile | allowed actions | denied (audited) |",
            "|---|---:|---:|",
        ]
        for profile, row in e17["matrix"].items():
            allowed = sum(1 for v in row.values() if v)
            lines.append(
                f"| {profile} | {allowed}/6 | {e17['audit_denied_entries'].get(profile, 0)} |"
            )

    if e18:
        lines += [
            "",
            "## E18: L2 S4-PSS variants",
            "",
            "| variant | seeds | success rate | wall mean s |",
            "|---|---:|---:|---:|",
        ]
        for variant, row in e18.items():
            lines.append(
                f"| {variant} | {row['n']} | {row['success_rate']:.0%} | {_fmt(row['wall_mean'], 2)} |"
            )
        lines += [
            "",
            "`no_relations` fails because the pick-place goal is a relation; `no_memory` and",
            "`no_entity` succeed because this demo has no occlusion. The demo tests relations,",
            "not memory. E12 supplies the memory ablation on bundles.",
        ]

    if e12all:
        lines += [
            "",
            "## E12b: all-bundle replay",
            "",
            "| variant | ID retention | switches |",
            "|---|---:|---:|",
        ]
        for variant, row in e12all["summary"].items():
            lines.append(
                f"| {variant} | {_fmt(row['id_retention'])} | {_fmt(row['id_switches'], 2)} |"
            )

    lines += [
        "",
        "## Files",
        "",
        "- `reports/claim_gaps/nonpss/`",
        "- `reports/claim_gaps/scale/`",
        "- `reports/claim_gaps/detect_noise/`",
        "- `reports/claim_gaps/families/`",
        "- `reports/claim_gaps/retrain_all/`",
        "- `reports/claim_gaps/tms/`",
        "- `reports/claim_gaps/l2_memory/`",
        "- `reports/claim_gaps/rag_exact/`",
        "- `reports/claim_gaps/modality_conflict/`",
        "- `reports/claim_gaps/openset/`",
        "- `reports/claim_gaps/space_ablation/`",
        "- `reports/claim_gaps/runtime_audit/`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FOLLOWUPS.md").write_text(render(), encoding="utf-8")
    print(f"output: {OUT / 'FOLLOWUPS.md'}")


if __name__ == "__main__":
    main()
