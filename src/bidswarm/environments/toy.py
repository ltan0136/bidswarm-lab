"""Toy auto-bidding environment shaped like AuctionNet.

One episode is one delivery period of T time steps with n ad opportunities per
step. All exogenous quantities of step t are drawn from
``np.random.default_rng([scenario_seed, t])`` in a fixed order, independently
of the policy (common random numbers): different policies on the same scenario
face exactly the same traffic, prices and uniforms.

Auction: single slot, second price. A bid strictly above
``least_winning_cost`` wins and pays ``least_winning_cost``. This is a
simplification of the upstream multi-slot mechanism. A win is exposed with
probability ``exposure_prob``; an exposed win converts when its conversion
uniform is below ``p_true = clip(pValue + pValueSigma * z, 0, 1)``.

Budget: if a step's winning cost exceeds the remaining budget, winners are
kept in the order of a pre-drawn per-opportunity key while the cumulative cost
fits; the rest become losses. Spend therefore never exceeds the budget.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, NamedTuple

import numpy as np

from ..evaluation.metrics import auctionnet_score, cpa_within_target
from ..types import (
    BACKEND_TOY,
    PURPOSE,
    SCORE_FORMULA,
    STATUS_OK,
    ConfigError,
)
from .base import Environment, failed_episode

INTEGRATION_STATUS = "builtin"
MAX_TIME_STEPS = 10_000
# Stream key for advertiser sampling; never collides with a step index.
_ADVERTISER_STREAM = 2**31 - 1


@dataclass(frozen=True)
class ToyConfig:
    num_time_steps: int = 48
    opportunities_per_step: int = 300
    budget_range: tuple[float, float] = (400.0, 1200.0)
    cpa_target_range: tuple[float, float] = (6.0, 12.0)
    num_categories: int = 5
    pvalue_beta: tuple[float, float] = (2.0, 98.0)
    sigma_ratio_range: tuple[float, float] = (0.1, 0.5)
    exposure_prob: float = 0.8
    price_scale: float = 0.15
    price_pvalue_elasticity: float = 0.8
    price_noise_sigma: float = 0.5
    price_time_amplitude: float = 0.3

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "ToyConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known - {"backend"})
        if unknown:
            raise ConfigError(f"[environment] unknown keys for backend 'toy': {unknown}")
        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in cfg:
                continue
            raw = cfg[f.name]
            if f.name in ("num_time_steps", "opportunities_per_step", "num_categories"):
                values[f.name] = _as_int(f.name, raw)
            elif isinstance(f.default, tuple):
                values[f.name] = _as_pair(f.name, raw)
            else:
                values[f.name] = _as_float(f.name, raw)
        out = cls(**values)
        out._validate()
        return out

    def _validate(self) -> None:
        def require(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(f"[environment] {msg}")

        require(1 <= self.num_time_steps <= MAX_TIME_STEPS, f"num_time_steps must be in [1, {MAX_TIME_STEPS}]")
        require(self.opportunities_per_step >= 1, "opportunities_per_step must be >= 1")
        require(self.num_categories >= 1, "num_categories must be >= 1")
        for name in ("budget_range", "cpa_target_range"):
            lo, hi = getattr(self, name)
            require(0 < lo <= hi, f"{name} must satisfy 0 < low <= high")
        a, b = self.pvalue_beta
        require(a > 0 and b > 0, "pvalue_beta parameters must be > 0")
        lo, hi = self.sigma_ratio_range
        require(0 <= lo <= hi, "sigma_ratio_range must satisfy 0 <= low <= high")
        require(0 < self.exposure_prob <= 1, "exposure_prob must be in (0, 1]")
        require(self.price_scale > 0, "price_scale must be > 0")
        require(self.price_noise_sigma >= 0, "price_noise_sigma must be >= 0")
        require(0 <= self.price_time_amplitude < 1, "price_time_amplitude must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["backend"] = BACKEND_TOY
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def _as_int(name: str, raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"[environment] {name} must be an integer, got {raw!r}")
    return raw


def _as_float(name: str, raw: Any) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ConfigError(f"[environment] {name} must be a finite number, got {raw!r}")
    return float(raw)


def _as_pair(name: str, raw: Any) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ConfigError(f"[environment] {name} must be a [low, high] pair, got {raw!r}")
    return (_as_float(name, raw[0]), _as_float(name, raw[1]))


class StepTraffic(NamedTuple):
    pvalue: np.ndarray
    sigma: np.ndarray
    p_true: np.ndarray
    least_winning_cost: np.ndarray
    u_exposure: np.ndarray
    u_conversion: np.ndarray
    order_key: np.ndarray


def step_traffic(cfg: ToyConfig, scenario_seed: int, step: int) -> StepTraffic:
    """Exogenous draws of one step; depends only on (config, scenario_seed, step)."""
    rng = np.random.default_rng([scenario_seed, step])
    n = cfg.opportunities_per_step
    a, b = cfg.pvalue_beta
    pvalue = rng.beta(a, b, size=n)
    sigma = pvalue * rng.uniform(cfg.sigma_ratio_range[0], cfg.sigma_ratio_range[1], size=n)
    z = rng.standard_normal(n)
    p_true = np.clip(pvalue + sigma * z, 0.0, 1.0)
    eps = rng.standard_normal(n)
    time_factor = 1.0 + cfg.price_time_amplitude * math.sin(2.0 * math.pi * step / cfg.num_time_steps)
    mean_pvalue = a / (a + b)
    price = (
        cfg.price_scale
        * time_factor
        * np.power(pvalue / mean_pvalue, cfg.price_pvalue_elasticity)
        * np.exp(cfg.price_noise_sigma * eps)
    )
    u_exposure = rng.random(n)
    u_conversion = rng.random(n)
    order_key = rng.random(n)
    return StepTraffic(pvalue, sigma, p_true, price, u_exposure, u_conversion, order_key)


def _tags() -> dict[str, Any]:
    return {"backend": BACKEND_TOY, "purpose": PURPOSE, "score_formula": SCORE_FORMULA}


class ToyEnvironment(Environment):
    backend = BACKEND_TOY
    integration_status = INTEGRATION_STATUS

    def __init__(self, cfg: ToyConfig | Mapping[str, Any] | None = None) -> None:
        if cfg is None:
            cfg = ToyConfig()
        elif not isinstance(cfg, ToyConfig):
            cfg = ToyConfig.from_mapping(cfg)
        self.cfg = cfg
        self.num_time_steps = cfg.num_time_steps

    def make_scenario(self, scenario_id: str, split: str, index: int, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng([seed, _ADVERTISER_STREAM])
        budget = float(rng.uniform(*self.cfg.budget_range))
        cpa_target = float(rng.uniform(*self.cfg.cpa_target_range))
        category = int(rng.integers(0, self.cfg.num_categories))
        return {
            "scenario_id": scenario_id,
            "split": split,
            "index": int(index),
            "seed": int(seed),
            "budget": budget,
            "cpa_target": cpa_target,
            "category": category,
            "backend": BACKEND_TOY,
        }

    def run_episode(self, scenario: dict[str, Any], policy: Any) -> dict[str, Any]:
        cfg = self.cfg
        start = time.perf_counter()
        budget = float(scenario["budget"])
        cpa_target = float(scenario["cpa_target"])
        seed = int(scenario["seed"])
        n = cfg.opportunities_per_step

        h_pinfo: list[np.ndarray] = []
        h_bid: list[np.ndarray] = []
        h_auction: list[np.ndarray] = []
        h_impression: list[np.ndarray] = []
        h_lwc: list[np.ndarray] = []
        spent = 0.0
        conversions = 0
        expected_conversions = 0.0
        wins = 0
        try:
            policy.reset()
            for t in range(cfg.num_time_steps):
                tr = step_traffic(cfg, seed, t)
                raw = policy.bidding(
                    t, tr.pvalue.copy(), tr.sigma.copy(), h_pinfo, h_bid, h_auction, h_impression, h_lwc
                )
                bids = np.asarray(raw, dtype=float)
                if bids.shape != (n,):
                    raise ValueError(f"policy returned bids of shape {bids.shape}, expected {(n,)}")
                if not np.all(np.isfinite(bids)) or np.any(bids < 0):
                    raise ValueError("policy returned non-finite or negative bids")

                win = bids > tr.least_winning_cost
                cost = np.where(win, tr.least_winning_cost, 0.0)
                if spent + float(np.sum(cost)) > budget:
                    win = self._truncate_to_budget(win, cost, tr.order_key, spent, budget)
                    cost = np.where(win, tr.least_winning_cost, 0.0)
                exposed = win & (tr.u_exposure < cfg.exposure_prob)
                converted = exposed & (tr.u_conversion < tr.p_true)

                spent += float(np.sum(cost))
                wins += int(np.sum(win))
                conversions += int(np.sum(converted))
                expected_conversions += float(np.sum(tr.pvalue[exposed]))

                win_f = win.astype(float)
                h_pinfo.append(np.column_stack([tr.pvalue, tr.sigma]))
                h_bid.append(bids.copy())
                h_auction.append(np.column_stack([win_f, win_f, cost]))  # [is_win, slot, cost]
                h_impression.append(np.column_stack([exposed.astype(float), converted.astype(float)]))
                h_lwc.append(tr.least_winning_cost.copy())
        except Exception as exc:  # policy errors become a failed episode
            return failed_episode(
                scenario,
                BACKEND_TOY,
                f"{type(exc).__name__}: {exc}",
                time.perf_counter() - start,
                n * cfg.num_time_steps,
            )

        return {
            **_tags(),
            "scenario_id": scenario["scenario_id"],
            "split": scenario["split"],
            "status": STATUS_OK,
            "error": None,
            "budget": budget,
            "cpa_target": cpa_target,
            "cost": spent,
            "conversions": conversions,
            "expected_conversions": expected_conversions,
            "cpa_realized": spent / conversions if conversions > 0 else None,
            "cpa_expected": spent / expected_conversions if expected_conversions > 0 else None,
            "cpa_defined": conversions > 0,
            "cpa_expected_defined": expected_conversions > 0,
            "score_realized": auctionnet_score(float(conversions), spent, cpa_target),
            "score_expected": auctionnet_score(expected_conversions, spent, cpa_target),
            "budget_utilization": spent / budget,
            "wins": wins,
            "opportunities": n * cfg.num_time_steps,
            "cpa_ok": cpa_within_target(float(conversions), spent, cpa_target),
            "cpa_expected_ok": cpa_within_target(expected_conversions, spent, cpa_target),
            "wall_time_s": time.perf_counter() - start,
        }

    @staticmethod
    def _truncate_to_budget(
        win: np.ndarray, cost: np.ndarray, order_key: np.ndarray, spent: float, budget: float
    ) -> np.ndarray:
        """Keep winners in order-key order while their cumulative cost fits."""
        idx = np.flatnonzero(win)
        order = idx[np.argsort(order_key[idx], kind="stable")]
        n_keep = int(np.sum(np.cumsum(cost[order]) <= budget - spent))
        kept = order[:n_keep]
        # Guard against summation-order rounding so spend never exceeds budget.
        while n_keep > 0 and spent + float(np.sum(cost[kept])) > budget:
            n_keep -= 1
            kept = order[:n_keep]
        out = np.zeros_like(win)
        out[kept] = True
        return out
