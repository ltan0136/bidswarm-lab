"""scripts/compare_runs.py: replicate-paired comparison of two evaluated runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from bidswarm.experiment import evaluate_run, run_experiment

from .conftest import REPO_ROOT, small_config


def _evaluated(config, root: Path) -> Path:
    run_dir = run_experiment(config, out_root=root, serial=True)
    evaluate_run(run_dir, serial=True)
    return run_dir


def _compare(a: Path, b: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "compare_runs.py"), str(a), str(b)],
                          capture_output=True, text=True, timeout=60, check=False)


def test_compare_runs_pairs_two_proposers_on_shared_scenarios(tmp_path: Path):
    es = _evaluated(small_config(), tmp_path / "es")
    cm = _evaluated(small_config(search={"proposer": "cmaes", "cmaes": {"population": 4}}), tmp_path / "cm")
    out = _compare(cm, es)
    assert out.returncode == 0, out.stderr
    assert "proposer cmaes" in out.stdout and "proposer es" in out.stdout
    rows = [line for line in out.stdout.splitlines() if line.startswith("| single") or line.startswith("| independent")]
    assert len(rows) == 2
    same = _compare(es, es)
    assert "| +0.000 | [+0.000, +0.000] |" in same.stdout  # a run paired with itself


def test_compare_runs_refuses_runs_with_different_scenarios(tmp_path: Path):
    a = _evaluated(small_config(), tmp_path / "a")
    b = _evaluated(small_config(experiment={"seed": 8}), tmp_path / "b")
    out = _compare(a, b)
    assert out.returncode != 0 and "cannot be paired" in out.stderr
