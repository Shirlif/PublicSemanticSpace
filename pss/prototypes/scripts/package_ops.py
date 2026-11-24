"""CLI: Phase 5 prototype package add/rollback/verify.

Examples:
    # full demo: init base, add a category, verify, rollback
    python -m pss.prototypes.scripts.package_ops --action run-all \
        --category sheet --root checkpoints/prototype_packages \
        --out eval/results/phase5_package

    # individual operations
    python -m pss.prototypes.scripts.package_ops --action init
    python -m pss.prototypes.scripts.package_ops --action add --category sheet
    python -m pss.prototypes.scripts.package_ops --action rollback --to-version v1
    python -m pss.prototypes.scripts.package_ops --action verify
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pss._console import ensure_utf8
from pss.prototypes.bank import PrototypeBank
from pss.prototypes.package import (
    add_category,
    init_package,
    load_manager,
    rollback,
    run_package_demo,
    verify,
)


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=["run-all", "init", "add", "rollback", "verify"], required=True
    )
    parser.add_argument("--data", default="data/category_pretrain")
    parser.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    parser.add_argument("--projections", default="checkpoints/phase45_projection")
    parser.add_argument("--prototypes", default="checkpoints/phase45_prototypes")
    parser.add_argument("--bank", default="checkpoints/phase45_prototypes")
    parser.add_argument("--root", default="checkpoints/prototype_packages")
    parser.add_argument("--category", default="sheet")
    parser.add_argument("--split", default="test_category")
    parser.add_argument("--base-version", default="v1")
    parser.add_argument("--new-version", default="v2")
    parser.add_argument("--to-version", default=None)
    parser.add_argument("--out", default="eval/results/phase5_package")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.action == "run-all":
        report = run_package_demo(
            args.data,
            args.experts,
            args.projections,
            args.prototypes,
            args.root,
            args.bank,
            category=args.category,
            base_version=args.base_version,
            new_version=args.new_version,
            split=args.split,
        )
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("PACKAGE_REPORT.md", "package_report.json"):
            source = Path(args.root) / name
            if source.exists():
                shutil.copy2(source, out / name)
        print(f"package demo: {'PASS' if report['ok'] else 'FAIL'}")
        print(f"  added: {report['add'].get('added')} {report['add'].get('category')}")
        print(f"  rollback: {report['rollback'].get('from')} -> {report['rollback'].get('to')}")
        print(f"  no backprop: {report['no_backprop']}")
        print(f"  report: {out / 'PACKAGE_REPORT.md'}")
        return

    if args.action == "init":
        bank = PrototypeBank.load(Path(args.bank) / "prototypes.npz")
        package, created = init_package(bank, args.root, version=args.base_version)
        print(f"init: version={package.version} created={created} categories={len(package.categories)}")
        return

    if args.action == "add":
        result = add_category(
            args.data,
            args.experts,
            args.projections,
            args.prototypes,
            args.root,
            args.category,
            version=args.new_version,
            force=args.force,
            split=args.split,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.action == "rollback":
        result = rollback(args.root, to_version=args.to_version)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    manager = load_manager(args.root)
    print(json.dumps({"active": manager.active, **verify(args.root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
