"""Shared fixtures: a tiny toy configuration and pre-computed small runs."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from bidswarm.config import ExperimentConfig
from bidswarm.experiment import evaluate_run, run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# Small enough for sub-second serial runs: T=6 steps x 40 opportunities.
# Budget 16 / 2 train scenarios = 8 candidates; single: 4 rounds, independent (2 workers): 2 rounds.
SMALL_CONFIG: dict[str, Any] = {
    "experiment": {"name": "t_small", "seed": 7, "replicates": 2},
    "environment": {
        "backend": "toy",
        "num_time_steps": 6,
        "opportunities_per_step": 40,
        "budget_range": [10.0, 30.0],
        "cpa_target_range": [6.0, 12.0],
    },
    "splits": {"train": 2, "validation": 2, "test": 2},
    "search": {
        "policy": "dual_control",
        "proposer": "es",
        "modes": ["single", "independent"],
        "independent_workers": 2,
        "candidates_per_round": 2,
        "train_episode_budget": 16,
        "fitness_metric": "score_expected",
    },
    "selection": {"top_k": 2},
}


def small_config_dict(**section_overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of SMALL_CONFIG with per-section key overrides."""
    data = copy.deepcopy(SMALL_CONFIG)
    for section, values in section_overrides.items():
        data.setdefault(section, {}).update(values)
    return data


def small_config(**section_overrides: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.from_dict(small_config_dict(**section_overrides))


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(type(value))


def write_toml(path: Path, data: dict[str, Any]) -> Path:
    """Minimal TOML writer for config dicts (tables of scalars/lists, one level of nesting)."""
    lines: list[str] = []
    for section, table in data.items():
        lines.append(f"[{section}]")
        nested = {k: v for k, v in table.items() if isinstance(v, dict)}
        for key, value in table.items():
            if key not in nested:
                lines.append(f"{key} = {_toml_value(value)}")
        for key, sub in nested.items():
            lines.append(f"[{section}.{key}]")
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in sub.items())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def small_run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small serial run (search + validation only, no test evaluation). Do not modify."""
    return run_experiment(small_config(), out_root=tmp_path_factory.mktemp("runs"), serial=True)


@pytest.fixture(scope="session")
def evaluated_run_dir(small_run_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of ``small_run_dir`` after ``evaluate``. Do not modify."""
    target = tmp_path_factory.mktemp("evaluated") / small_run_dir.name
    shutil.copytree(small_run_dir, target)
    evaluate_run(target, serial=True)
    return target


@pytest.fixture
def run_copy(evaluated_run_dir: Path, tmp_path: Path) -> Path:
    """A private, modifiable copy of the evaluated run."""
    target = tmp_path / evaluated_run_dir.name
    shutil.copytree(evaluated_run_dir, target)
    return target


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
