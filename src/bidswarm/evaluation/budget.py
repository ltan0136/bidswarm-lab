"""Budget arithmetic and the episode ledger.

The primary budget unit is the number of training episodes per (replicate,
mode). It is converted into candidates and rounds by exact integer division;
any remainder is a configuration error. The coordinator charges the ledger
statically before dispatching a batch, so ``used`` can never exceed
``planned``. v1 performs no retries (``retries`` is always 0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ..types import PHASES, STATUS_FAILED, STATUS_TIMEOUT, ConfigError


@dataclass(frozen=True)
class TrainingBudgetPlan:
    mode: str
    n_workers: int
    train_episode_budget: int
    n_train_scenarios: int
    candidates_per_round: int
    candidates_total: int
    candidates_per_worker: int
    rounds: int

    @property
    def episodes_per_round(self) -> int:
        return self.n_workers * self.candidates_per_round * self.n_train_scenarios

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["episodes_per_round"] = self.episodes_per_round
        return d


def _exact_div(numerator: int, denominator: int, what: str) -> int:
    if denominator <= 0:
        raise ConfigError(f"{what}: divisor must be positive, got {denominator}")
    if numerator % denominator != 0:
        raise ConfigError(f"{what}: {numerator} is not divisible by {denominator}")
    return numerator // denominator


def plan_training_budget(
    mode: str, train_episode_budget: int, n_train_scenarios: int, n_workers: int, candidates_per_round: int
) -> TrainingBudgetPlan:
    """candidates_total = budget / n_train; per_worker = total / n_workers; rounds = per_worker / k."""
    if train_episode_budget <= 0:
        raise ConfigError("train_episode_budget must be positive")
    total = _exact_div(
        train_episode_budget, n_train_scenarios, "train_episode_budget / number of training scenarios"
    )
    per_worker = _exact_div(total, n_workers, f"candidates_total / workers (mode={mode})")
    rounds = _exact_div(per_worker, candidates_per_round, f"candidates_per_worker / k (mode={mode})")
    return TrainingBudgetPlan(
        mode=mode,
        n_workers=n_workers,
        train_episode_budget=train_episode_budget,
        n_train_scenarios=n_train_scenarios,
        candidates_per_round=candidates_per_round,
        candidates_total=total,
        candidates_per_worker=per_worker,
        rounds=rounds,
    )


class BudgetExceededError(RuntimeError):
    """Raised when a dispatch would exceed the planned episode budget."""


@dataclass
class LedgerEntry:
    replicate: int
    group: str
    phase: str
    planned: int
    used: int = 0
    failed: int = 0
    timeouts: int = 0
    retries: int = 0


class BudgetLedger:
    """Episode accounting per (replicate, group, phase); group is a mode or test label."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, str, str], LedgerEntry] = {}

    def plan(self, replicate: int, group: str, phase: str, planned: int) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        key = (replicate, group, phase)
        if key in self._entries:
            raise ValueError(f"budget already planned for {key}")
        self._entries[key] = LedgerEntry(replicate, group, phase, int(planned))

    def charge(self, replicate: int, group: str, phase: str, n: int) -> None:
        """Deduct ``n`` episodes before dispatch; refuses to exceed the plan."""
        entry = self._entries[(replicate, group, phase)]
        if entry.used + n > entry.planned:
            raise BudgetExceededError(
                f"{phase} budget exceeded for replicate={replicate} group={group}: "
                f"used {entry.used} + {n} > planned {entry.planned}"
            )
        entry.used += n

    def record_outcomes(self, replicate: int, group: str, phase: str, statuses: Iterable[str]) -> None:
        entry = self._entries[(replicate, group, phase)]
        for status in statuses:
            if status == STATUS_FAILED:
                entry.failed += 1
            elif status == STATUS_TIMEOUT:
                entry.timeouts += 1

    def used(self, replicate: int, group: str, phase: str) -> int:
        return self._entries[(replicate, group, phase)].used

    def drop_phase(self, phase: str) -> None:
        self._entries = {k: v for k, v in self._entries.items() if v.phase != phase}

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries.values())

    def to_dict(self) -> dict[str, Any]:
        totals: dict[str, dict[str, int]] = {}
        for e in self._entries.values():
            t = totals.setdefault(e.phase, {"planned": 0, "used": 0, "failed": 0, "timeouts": 0, "retries": 0})
            for k in t:
                t[k] += getattr(e, k)
        return {
            "unit": "episodes",
            "retry_policy": "none (v1)",
            "timeout_policy": "soft: flagged only, process not killed",
            "entries": [asdict(e) for e in self._entries.values()],
            "totals": totals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetLedger":
        ledger = cls()
        for row in data["entries"]:
            e = LedgerEntry(**row)
            ledger._entries[(e.replicate, e.group, e.phase)] = e
        return ledger
