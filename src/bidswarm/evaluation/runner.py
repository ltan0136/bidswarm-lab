"""Episode task function and the (serial or process-pool) executor.

``run_episode_task`` is a pure, module-level function: its result depends only
on the task (apart from ``wall_time_s``), so it can run in a spawned process
and serial and parallel execution produce identical records.

Timeouts are soft: an episode slower than ``timeout_s`` is flagged
``status = "timeout"`` after it finishes; the process is never killed and the
metrics are kept.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from ..environments import make_environment
from ..environments.base import failed_episode
from ..kernel import PolicySpec, make_policy
from ..types import STATUS_OK, STATUS_TIMEOUT, EpisodeTask


def run_episode_task(task: EpisodeTask) -> dict[str, Any]:
    start = time.perf_counter()
    meta = {
        "task_index": task.task_index,
        "task_id": task.task_id,
        "phase": task.phase,
        "replicate": task.replicate,
        "group": task.group,
        "worker_id": task.worker_id,
        "candidate_id": task.candidate_id,
        "spec_hash": task.spec_hash,
    }
    scenario = task.scenario
    try:
        env = make_environment(task.env_config)
        policy = make_policy(
            PolicySpec.from_dict(task.spec),
            budget=scenario["budget"],
            cpa_target=scenario["cpa_target"],
            category=scenario.get("category"),
            num_time_steps=env.num_time_steps,
        )
    except Exception as exc:
        result = failed_episode(
            scenario,
            str(task.env_config.get("backend")),
            f"setup failed: {type(exc).__name__}: {exc}",
            time.perf_counter() - start,
            None,
        )
    else:
        result = env.run_episode(scenario, policy)
    if result["status"] == STATUS_OK and result["wall_time_s"] > task.timeout_s:
        result["status"] = STATUS_TIMEOUT
    return {**meta, **result}


def resolve_max_workers(max_workers: int | None) -> int:
    """Pool size actually used; Windows caps process pools at 61 workers."""
    return max_workers or min(os.cpu_count() or 1, 61)


class EpisodeExecutor:
    """Runs batches of tasks serially or on a spawn-context process pool.

    Results are always returned sorted by ``task_index``. Use as a context
    manager so the pool is created once per run.
    """

    def __init__(self, serial: bool = False, max_workers: int | None = None) -> None:
        self.serial = serial
        self.max_workers = resolve_max_workers(max_workers)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "EpisodeExecutor":
        if not self.serial:
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=multiprocessing.get_context("spawn")
            )
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def run(self, tasks: Sequence[EpisodeTask]) -> list[dict[str, Any]]:
        if not tasks:
            return []
        if self.serial:
            results = [run_episode_task(t) for t in tasks]
        else:
            if self._pool is None:
                raise RuntimeError("EpisodeExecutor must be entered before running in parallel")
            chunksize = max(1, len(tasks) // (4 * self.max_workers))
            results = list(self._pool.map(run_episode_task, tasks, chunksize=chunksize))
        return sorted(results, key=lambda r: r["task_index"])
