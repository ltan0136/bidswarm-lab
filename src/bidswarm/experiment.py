"""Experiment pipeline: ``run`` (search + validation selection) and ``evaluate`` (test).

Protocol boundaries enforced here:

* ``run`` never builds test tasks: searchers only see training scenarios, and
  the selection step only sees validation scenarios.
* ``evaluate`` reads the frozen selections from ``selected/`` and evaluates
  them, plus two untuned default baselines, on the test scenarios. Nothing it
  produces is read back by the search.
"""

from __future__ import annotations

import platform
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .artifacts.report import write_report, write_summary
from .artifacts.store import (
    ANYTIME,
    BUDGET,
    CANDIDATES,
    EPISODES,
    MANIFEST,
    SCENARIOS,
    TEST_EPISODES,
    RunStore,
    to_jsonable,
)
from .config import ExperimentConfig
from .environments import INTEGRATION_STATUS, make_environment
from .environments.auctionnet import INTEGRATION_STATUS as AUCTIONNET_STATUS
from .evaluation.budget import BudgetLedger
from .evaluation.runner import EpisodeExecutor, resolve_max_workers
from .evaluation.selection import SELECTION_RULE, candidate_fitness, choose_selected, shortlist_top_k
from .evaluation.splits import SplitSet, build_splits, master_seeds
from .kernel import PolicySpec, default_spec, registered_names, spec_hash
from .search.coordinator import Coordinator, SearchPipeline, TaskCounter, build_pipeline
from .types import (
    PHASE_TEST,
    PHASE_TRAIN,
    PHASE_VALIDATION,
    PURPOSE,
    SCORE_FORMULA,
    ArtifactError,
    EpisodeTask,
)

# Untuned reference policies evaluated next to the selected ones. Not strong baselines.
BASELINES: tuple[tuple[str, PolicySpec], ...] = (
    ("baseline:fixed_rate_default", PolicySpec("fixed_rate", {"ratio": 1.0})),
    ("baseline:dual_control_default", default_spec("dual_control")),
)


def selected_label(mode: str) -> str:
    return f"selected:{mode}"


def selected_relpath(replicate: int, mode: str) -> str:
    return f"selected/{replicate:02d}_{mode}.json"


def _git_info() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if not (root / "src" / "bidswarm").is_dir():
        return {"commit": None, "dirty": None}
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        if head.returncode != 0:
            return {"commit": None, "dirty": None}
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _manifest(
    config: ExperimentConfig, run_dir: Path, config_path: str | None, serial: bool, max_workers: int | None
) -> dict[str, Any]:
    return {
        "bidswarm_version": __version__,
        "run_name": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": config_path,
        "config": config.to_dict(),
        "git": _git_info(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "backend": config.backend,
        "integration_status": INTEGRATION_STATUS[config.backend],
        "auctionnet_integration_status": AUCTIONNET_STATUS,
        "purpose": PURPOSE,
        "score_formula": SCORE_FORMULA,
        "executor": {
            "serial": serial,
            "max_workers": None if serial else resolve_max_workers(max_workers),
            "note": "does not affect results",
        },
        "master_seeds": master_seeds(config.seed, config.replicates),
        "budget_plans": {m: config.budget_plan(m).to_dict() for m in config.search.modes},
        "policy_registry": registered_names(),
    }


def _build_all_splits(config: ExperimentConfig, env: Any) -> list[SplitSet]:
    return [
        build_splits(env, r, ms, config.splits)
        for r, ms in enumerate(master_seeds(config.seed, config.replicates))
    ]


def run_experiment(
    config: ExperimentConfig,
    out_root: str | Path = "runs",
    serial: bool = False,
    max_workers: int | None = None,
    config_path: str | None = None,
) -> Path:
    """Search every (replicate, mode), select on validation, write artifacts. No test episodes."""
    env = make_environment(config.environment)  # unintegrated backends fail here, before any output
    split_sets = _build_all_splits(config, env)
    max_workers = max_workers or config.runtime.max_workers

    store = RunStore.create(out_root, config.name)
    store.write_json(MANIFEST, _manifest(config, store.run_dir, config_path, serial, max_workers))
    store.write_json(SCENARIOS, {"backend": config.backend, "replicates": [s.to_dict() for s in split_sets]})
    for relpath in (CANDIDATES, EPISODES, ANYTIME):
        store.write_text(relpath, "")

    ledger = BudgetLedger()
    pipelines: list[SearchPipeline] = []
    n_val = config.splits.validation
    for s in split_sets:
        for mode in config.search.modes:
            plan = config.budget_plan(mode)
            ledger.plan(s.replicate, mode, PHASE_TRAIN, plan.train_episode_budget)
            ledger.plan(s.replicate, mode, PHASE_VALIDATION, config.selection.top_k * n_val)
            pipelines.append(build_pipeline(config, s.replicate, s.master_seed, mode, s.train))

    counter = TaskCounter()
    with EpisodeExecutor(serial=serial, max_workers=max_workers) as executor:
        Coordinator(config, pipelines, executor, ledger, store, counter).run()
        _validate_and_select(config, pipelines, split_sets, executor, ledger, store, counter)

    for e in ledger.entries():
        if e.phase == PHASE_TRAIN and e.used != e.planned:
            raise RuntimeError(f"training budget mismatch for {e}: used {e.used} != planned {e.planned}")
    store.write_json(BUDGET, ledger.to_dict())
    write_summary(store)
    write_report(store)
    return store.run_dir


def _validate_and_select(
    config: ExperimentConfig,
    pipelines: Sequence[SearchPipeline],
    split_sets: Sequence[SplitSet],
    executor: EpisodeExecutor,
    ledger: BudgetLedger,
    store: RunStore,
    counter: TaskCounter,
) -> None:
    metric = config.search.fitness_metric
    top_k = config.selection.top_k
    tasks: list[EpisodeTask] = []
    shortlists = []
    for p in pipelines:
        shortlist = shortlist_top_k(p.candidates, top_k)
        shortlists.append(shortlist)
        first = len(tasks)
        for c in shortlist:
            for sc in split_sets[p.replicate].validation:
                tasks.append(
                    EpisodeTask(
                        task_index=counter.next(),
                        task_id=f"{PHASE_VALIDATION}/{c['candidate_id']}/{sc['scenario_id']}",
                        phase=PHASE_VALIDATION,
                        replicate=p.replicate,
                        group=p.mode,
                        worker_id=c["worker_id"],
                        candidate_id=c["candidate_id"],
                        spec={"name": c["policy"], "params": c["params"]},
                        spec_hash=c["spec_hash"],
                        scenario=sc,
                        env_config=config.environment,
                        timeout_s=config.runtime.episode_timeout_s,
                    )
                )
        ledger.charge(p.replicate, p.mode, PHASE_VALIDATION, len(tasks) - first)

    results = executor.run(tasks)
    store.append_jsonl(EPISODES, results)
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for res in results:
        by_candidate[res["candidate_id"]].append(res)

    for p, shortlist in zip(pipelines, shortlists):
        scores = {}
        for c in shortlist:
            episodes = by_candidate[c["candidate_id"]]
            ledger.record_outcomes(p.replicate, p.mode, PHASE_VALIDATION, [e["status"] for e in episodes])
            scores[c["candidate_id"]] = candidate_fitness(episodes, metric)
        chosen, rows = choose_selected(shortlist, scores)
        by_id = {c["candidate_id"]: c for c in p.candidates}
        winner = by_id[chosen["candidate_id"]] if chosen else None
        store.write_json(
            selected_relpath(p.replicate, p.mode),
            {
                "replicate": p.replicate,
                "master_seed": p.master_seed,
                "mode": p.mode,
                "status": "selected" if winner else "no_valid_candidate",
                "candidate_id": winner["candidate_id"] if winner else None,
                "spec": {"name": winner["policy"], "params": winner["params"]} if winner else None,
                "spec_hash": winner["spec_hash"] if winner else None,
                "train_fitness": winner["fitness"] if winner else None,
                "validation_score": chosen["validation_score"] if chosen else None,
                "fitness_metric": metric,
                "top_k": top_k,
                "n_candidates": len(p.candidates),
                "n_failed_candidates": sum(c["fitness"] is None for c in p.candidates),
                "validation_scenario_ids": [sc["scenario_id"] for sc in split_sets[p.replicate].validation],
                "shortlist": rows,
                "selection_rule": SELECTION_RULE,
            },
        )


def evaluate_run(
    run_dir: str | Path, serial: bool = False, max_workers: int | None = None, force: bool = False
) -> Path:
    """Evaluate frozen selections and untuned baselines on the test scenarios."""
    store = RunStore.open(run_dir)
    manifest = store.read_json(MANIFEST)
    config = ExperimentConfig.from_dict(manifest["config"])
    env = make_environment(config.environment)
    if manifest["master_seeds"] != master_seeds(config.seed, config.replicates):
        raise ArtifactError("manifest master_seeds do not match the config")
    split_sets = _build_all_splits(config, env)
    recorded = store.read_json(SCENARIOS)["replicates"]
    if to_jsonable([s.to_dict() for s in split_sets]) != recorded:
        raise ArtifactError("scenarios.json does not match the scenarios derived from the config")
    if store.exists(TEST_EPISODES) and not force:
        raise ArtifactError(f"{TEST_EPISODES} already exists; pass --force to re-evaluate")

    ledger = BudgetLedger.from_dict(store.read_json(BUDGET))
    ledger.drop_phase(PHASE_TEST)
    max_workers = max_workers or config.runtime.max_workers
    counter = TaskCounter()
    tasks: list[EpisodeTask] = []
    for s in split_sets:
        entries: list[tuple[str, PolicySpec, str | None]] = []
        for mode in config.search.modes:
            relpath = selected_relpath(s.replicate, mode)
            if not store.exists(relpath):
                raise ArtifactError(f"missing frozen selection {relpath}")
            rec = store.read_json(relpath)
            if rec["status"] != "selected":
                continue
            spec = PolicySpec.from_dict(rec["spec"])
            if spec_hash(spec) != rec["spec_hash"]:
                raise ArtifactError(f"{relpath}: spec_hash does not match the stored spec")
            entries.append((selected_label(mode), spec, rec["candidate_id"]))
        entries.extend((label, spec, None) for label, spec in BASELINES)
        for label, spec, candidate_id in entries:
            ledger.plan(s.replicate, label, PHASE_TEST, len(s.test))
            first = len(tasks)
            for sc in s.test:
                tasks.append(
                    EpisodeTask(
                        task_index=counter.next(),
                        task_id=f"{PHASE_TEST}/r{s.replicate:02d}/{label}/{sc['scenario_id']}",
                        phase=PHASE_TEST,
                        replicate=s.replicate,
                        group=label,
                        worker_id=None,
                        candidate_id=candidate_id,
                        spec=spec.to_dict(),
                        spec_hash=spec_hash(spec),
                        scenario=sc,
                        env_config=config.environment,
                        timeout_s=config.runtime.episode_timeout_s,
                    )
                )
            ledger.charge(s.replicate, label, PHASE_TEST, len(tasks) - first)

    with EpisodeExecutor(serial=serial, max_workers=max_workers) as executor:
        results = executor.run(tasks)
    for res in results:
        ledger.record_outcomes(res["replicate"], res["group"], PHASE_TEST, [res["status"]])
    store.write_jsonl(TEST_EPISODES, results)
    store.write_json(BUDGET, ledger.to_dict())
    write_summary(store)
    write_report(store)
    return store.run_dir
