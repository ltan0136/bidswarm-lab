"""Experiment configuration: TOML loading and validation.

Validation is strict: unknown sections or keys, wrong types, out-of-range
values, unknown policies/backends and budgets that do not divide exactly are
all errors, reported before anything runs.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .environments import validate_environment_config
from .evaluation.budget import TrainingBudgetPlan, plan_training_budget
from .kernel import registered_names
from .types import FITNESS_METRICS, MODE_COOPERATIVE, MODE_INDEPENDENT, MODE_SINGLE, MODES, ConfigError

PROPOSERS = ("es", "random")

_SECTIONS = {"experiment", "environment", "splits", "search", "selection", "runtime"}


@dataclass(frozen=True)
class EsConfig:
    sigma_init: float = 0.2
    sigma_min: float = 0.01
    sigma_max: float = 0.5
    success_factor: float = 1.22
    failure_factor: float = 0.82


@dataclass(frozen=True)
class SearchConfig:
    policy: str
    proposer: str
    modes: tuple[str, ...]
    independent_workers: int
    candidates_per_round: int
    train_episode_budget: int
    fitness_metric: str
    es: EsConfig
    exchange_interval: int = 1  # cooperative mode: ring migration every this many rounds


@dataclass(frozen=True)
class SplitConfig:
    train: int
    validation: int
    test: int


@dataclass(frozen=True)
class SelectionConfig:
    top_k: int


@dataclass(frozen=True)
class RuntimeConfig:
    episode_timeout_s: float = 120.0
    max_workers: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    replicates: int
    environment: dict[str, Any]
    splits: SplitConfig
    search: SearchConfig
    selection: SelectionConfig
    runtime: RuntimeConfig

    @property
    def backend(self) -> str:
        return self.environment["backend"]

    def n_workers(self, mode: str) -> int:
        return 1 if mode == MODE_SINGLE else self.search.independent_workers

    def budget_plan(self, mode: str) -> TrainingBudgetPlan:
        return plan_training_budget(
            mode,
            self.search.train_episode_budget,
            self.splits.train,
            self.n_workers(mode),
            self.search.candidates_per_round,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["search"]["modes"] = list(self.search.modes)
        return {
            "experiment": {"name": self.name, "seed": self.seed, "replicates": self.replicates},
            "environment": dict(self.environment),
            "splits": d["splits"],
            "search": d["search"],
            "selection": d["selection"],
            "runtime": {k: v for k, v in d["runtime"].items() if v is not None},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        return _parse(data)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None
    return _parse(data)


# ------------------------------------------------------------------ helpers
def _section(data: Mapping[str, Any], name: str, required: bool = True) -> Mapping[str, Any]:
    if name not in data:
        if required:
            raise ConfigError(f"missing section [{name}]")
        return {}
    sec = data[name]
    if not isinstance(sec, Mapping):
        raise ConfigError(f"[{name}] must be a table")
    return sec


def _check_keys(sec: Mapping[str, Any], name: str, allowed: set[str]) -> None:
    unknown = sorted(set(sec) - allowed)
    if unknown:
        raise ConfigError(f"[{name}] unknown keys: {unknown}")


def _get_int(sec: Mapping[str, Any], where: str, key: str, default: Any = ..., minimum: int | None = None) -> int:
    if key not in sec:
        if default is ...:
            raise ConfigError(f"[{where}] missing key {key!r}")
        return default
    value = sec[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"[{where}] {key} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"[{where}] {key} must be >= {minimum}, got {value}")
    return value


def _get_float(sec: Mapping[str, Any], where: str, key: str, default: float) -> float:
    value = sec.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"[{where}] {key} must be a finite number, got {value!r}")
    return float(value)


def _get_str(sec: Mapping[str, Any], where: str, key: str, default: Any = ...) -> str:
    if key not in sec:
        if default is ...:
            raise ConfigError(f"[{where}] missing key {key!r}")
        return default
    value = sec[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"[{where}] {key} must be a non-empty string, got {value!r}")
    return value


def _parse(data: Mapping[str, Any]) -> ExperimentConfig:
    unknown_sections = sorted(set(data) - _SECTIONS)
    if unknown_sections:
        raise ConfigError(f"unknown config sections: {unknown_sections}")

    exp = _section(data, "experiment")
    _check_keys(exp, "experiment", {"name", "seed", "replicates"})
    name = _get_str(exp, "experiment", "name")
    if not all(c.isalnum() or c in "-_." for c in name):
        raise ConfigError("[experiment] name may only contain letters, digits, '-', '_' and '.'")
    seed = _get_int(exp, "experiment", "seed", minimum=0)
    replicates = _get_int(exp, "experiment", "replicates", minimum=1)

    environment = validate_environment_config(_section(data, "environment"))

    spl = _section(data, "splits")
    _check_keys(spl, "splits", {"train", "validation", "test"})
    splits = SplitConfig(
        train=_get_int(spl, "splits", "train", minimum=1),
        validation=_get_int(spl, "splits", "validation", minimum=1),
        test=_get_int(spl, "splits", "test", minimum=1),
    )

    src = _section(data, "search")
    _check_keys(
        src,
        "search",
        {
            "policy",
            "proposer",
            "modes",
            "independent_workers",
            "candidates_per_round",
            "train_episode_budget",
            "fitness_metric",
            "es",
            "exchange_interval",
        },
    )
    policy = _get_str(src, "search", "policy")
    if policy not in registered_names():
        raise ConfigError(f"[search] unknown policy {policy!r}; registered: {registered_names()}")
    proposer = _get_str(src, "search", "proposer", "es")
    if proposer not in PROPOSERS:
        raise ConfigError(f"[search] unknown proposer {proposer!r}; known: {list(PROPOSERS)}")
    modes_raw = src.get("modes", list(MODES))
    if not isinstance(modes_raw, list) or not modes_raw:
        raise ConfigError("[search] modes must be a non-empty list")
    for m in modes_raw:
        if m not in MODES:
            raise ConfigError(f"[search] unknown mode {m!r}; known: {list(MODES)}")
    if len(set(modes_raw)) != len(modes_raw):
        raise ConfigError("[search] modes must not repeat")
    modes = tuple(m for m in MODES if m in modes_raw)  # canonical order: single, independent, cooperative
    fitness_metric = _get_str(src, "search", "fitness_metric", "score_expected")
    if fitness_metric not in FITNESS_METRICS:
        raise ConfigError(f"[search] fitness_metric must be one of {list(FITNESS_METRICS)}")

    es_sec = _section(src, "es", required=False)
    _check_keys(es_sec, "search.es", set(EsConfig.__dataclass_fields__))
    d = EsConfig()
    es = EsConfig(
        sigma_init=_get_float(es_sec, "search.es", "sigma_init", d.sigma_init),
        sigma_min=_get_float(es_sec, "search.es", "sigma_min", d.sigma_min),
        sigma_max=_get_float(es_sec, "search.es", "sigma_max", d.sigma_max),
        success_factor=_get_float(es_sec, "search.es", "success_factor", d.success_factor),
        failure_factor=_get_float(es_sec, "search.es", "failure_factor", d.failure_factor),
    )
    if not (0 < es.sigma_min <= es.sigma_init <= es.sigma_max <= 1):
        raise ConfigError("[search.es] require 0 < sigma_min <= sigma_init <= sigma_max <= 1")
    if not es.success_factor > 1 or not 0 < es.failure_factor < 1:
        raise ConfigError("[search.es] require success_factor > 1 and 0 < failure_factor < 1")

    search = SearchConfig(
        policy=policy,
        proposer=proposer,
        modes=modes,
        independent_workers=_get_int(src, "search", "independent_workers", 4, minimum=1),
        candidates_per_round=_get_int(src, "search", "candidates_per_round", minimum=1),
        train_episode_budget=_get_int(src, "search", "train_episode_budget", minimum=1),
        fitness_metric=fitness_metric,
        es=es,
        exchange_interval=_get_int(src, "search", "exchange_interval", 1, minimum=1),
    )

    sel = _section(data, "selection")
    _check_keys(sel, "selection", {"top_k"})
    selection = SelectionConfig(top_k=_get_int(sel, "selection", "top_k", 4, minimum=1))

    rt = _section(data, "runtime", required=False)
    _check_keys(rt, "runtime", {"episode_timeout_s", "max_workers"})
    timeout = _get_float(rt, "runtime", "episode_timeout_s", RuntimeConfig.episode_timeout_s)
    if timeout <= 0:
        raise ConfigError("[runtime] episode_timeout_s must be > 0")
    max_workers = _get_int(rt, "runtime", "max_workers", None, minimum=1)
    runtime = RuntimeConfig(episode_timeout_s=timeout, max_workers=max_workers)

    cfg = ExperimentConfig(
        name=name,
        seed=seed,
        replicates=replicates,
        environment=environment,
        splits=splits,
        search=search,
        selection=selection,
        runtime=runtime,
    )
    # Budget divisibility is part of validation, for every configured mode.
    plans = [cfg.budget_plan(m) for m in cfg.search.modes]
    smallest_pool = min(p.candidates_total for p in plans)
    if selection.top_k > smallest_pool:
        raise ConfigError(
            f"[selection] top_k={selection.top_k} exceeds the candidate pool ({smallest_pool})"
        )
    for mode in (MODE_INDEPENDENT, MODE_COOPERATIVE):
        if mode in modes and search.independent_workers < 2:
            raise ConfigError(f"[search] {mode} mode needs independent_workers >= 2")
    return cfg
