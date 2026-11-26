"""Runner, report, CLI, and pre-registration tests."""

from __future__ import annotations

import json
from pathlib import Path

from pss_eval.__main__ import main
from pss_eval.fixtures import make_gaussian_stream
from pss_eval.properties import default_properties
from pss_eval.runner import make_default_systems, run, run_suite, write_report


def _small_stream():
    return make_gaussian_stream(n=8, dim=32, seed=0)


def test_default_suite_matches_preregistration():
    report = run_suite(
        make_default_systems(dim=32),
        default_properties(),
        _small_stream(),
        seeds=(0, 1),
    )
    systems = report["summary"]["systems"]

    assert systems["e5-flat"]["statuses"] == {
        "determinism": "pass",
        "reset_exactness": "pass",
        "state_hygiene": "pass",
        "contract": "fail",
    }
    assert systems["transformer-state"]["statuses"] == {
        "determinism": "pass",
        "reset_exactness": "pass",
        "state_hygiene": "pass",
        "contract": "skip",
    }
    assert all(summary["smoke_pass"] for summary in systems.values())
    assert report["preregistration"] == {"ok": True, "mismatches": []}


def test_single_run_report_also_checks_preregistration():
    report = run(
        make_default_systems(dim=32),
        default_properties(),
        _small_stream(),
        seed=0,
    )
    assert report["summary"]["systems"]["e5-flat"]["smoke_pass"] is True
    # ``run`` does not add the preregistration block; ``check_preregistration``
    # accepts the single-run shape directly.
    from pss_eval.preregistration import check_preregistration

    assert check_preregistration(report)["ok"] is True


def test_write_report_roundtrip(tmp_path):
    report = run_suite(
        make_default_systems(dim=32),
        default_properties(),
        _small_stream(),
        seeds=(0,),
    )
    paths = write_report(report, tmp_path)

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["preregistration"]["ok"] is True
    assert payload["summary"]["systems"]["e5-flat"]["statuses"]["contract"] == "fail"

    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "Pre-registration" in markdown
    assert "e5-flat" in markdown
    assert "transformer-state" in markdown


def test_cli_smoke(tmp_path, capsys):
    exit_code = main(
        [
            "--seeds",
            "1",
            "--dim",
            "32",
            "--n",
            "8",
            "--out",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "preregistration: OK" in captured.out
    assert "e5-flat" in captured.out
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "REPORT.md").exists()
