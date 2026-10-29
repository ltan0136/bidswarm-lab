"""Portable numeric bidding policies.

This module is part of the portable kernel: it must stay compatible with
Python 3.9 and depend only on numpy and the standard library, because the
policies are meant to run inside the AuctionNet process, which calls
``bidding()`` once per time step.

All templates share one functional form::

    bid = lambda_t * cpa_target * p_adj,   p_adj = max(pValue - rho * pValueSigma, 0)

Spend and expected conversions are accumulated incrementally from the history
arguments: each call only processes history entries that were not seen before,
so an episode of T steps costs O(T) history work instead of O(T^2).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class PolicyError(Exception):
    """Base class for errors raised by kernel policies."""


class PolicyInputError(PolicyError, ValueError):
    """Raised when ``bidding()`` receives malformed or non-finite inputs."""


class PolicyOutputError(PolicyError):
    """Raised when a policy would emit invalid bids (broken internal invariant)."""


class ParamDef(NamedTuple):
    """Box bound and default value of one template parameter."""

    name: str
    low: float
    high: float
    default: float


# Exponent clip for multiplicative controller terms; keeps exp() finite.
_LOG_CLIP = 50.0


def _as_vector(name: str, values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PolicyInputError("%s is not numeric: %s" % (name, exc)) from exc
    if arr.ndim != 1:
        raise PolicyInputError("%s must be 1-D, got shape %s" % (name, arr.shape))
    if not np.all(np.isfinite(arr)):
        raise PolicyInputError("%s contains NaN or infinite values" % name)
    return arr


def _as_matrix(name: str, values: Any, n_cols: int) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PolicyInputError("%s is not numeric: %s" % (name, exc)) from exc
    if arr.size == 0:
        return np.zeros((0, n_cols), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < n_cols:
        raise PolicyInputError(
            "%s must have shape (n, >=%d), got %s" % (name, n_cols, arr.shape)
        )
    if not np.all(np.isfinite(arr)):
        raise PolicyInputError("%s contains NaN or infinite values" % name)
    return arr


class BasePolicy(object):
    """Shared input checking, history accounting and the bid formula.

    Subclasses define ``name``, ``PARAM_DEFS`` and implement ``_ratio(t)``
    (the multiplier lambda_t) and ``_sigma_discount()`` (rho).

    The constructor mirrors AuctionNet's ``BaseBiddingStrategy(budget, name,
    cpa, category)``: advertiser values always come from the caller, never
    from constants in the policy.
    """

    name = "base"
    PARAM_DEFS: Tuple[ParamDef, ...] = ()

    def __init__(
        self,
        budget: float,
        cpa_target: float,
        category: Any = None,
        num_time_steps: int = 48,
        **params: float
    ) -> None:
        budget = float(budget)
        cpa_target = float(cpa_target)
        if not math.isfinite(budget) or budget < 0:
            raise PolicyInputError("budget must be finite and >= 0, got %r" % budget)
        if not math.isfinite(cpa_target) or cpa_target <= 0:
            raise PolicyInputError("cpa_target must be finite and > 0, got %r" % cpa_target)
        if int(num_time_steps) != num_time_steps or int(num_time_steps) < 1:
            raise PolicyInputError("num_time_steps must be a positive integer")
        expected = set(p.name for p in self.PARAM_DEFS)
        if set(params) != expected:
            raise PolicyInputError(
                "%s expects params %s, got %s" % (self.name, sorted(expected), sorted(params))
            )
        self.budget = budget
        self.cpa_target = cpa_target
        self.category = category
        self.num_time_steps = int(num_time_steps)
        self.params = dict((k, float(v)) for k, v in params.items())
        self.reset()

    # ----------------------------------------------------------- parameter space
    @classmethod
    def param_names(cls) -> List[str]:
        return [p.name for p in cls.PARAM_DEFS]

    @classmethod
    def default_params(cls) -> Dict[str, float]:
        return dict((p.name, float(p.default)) for p in cls.PARAM_DEFS)

    @classmethod
    def check_constraints(cls, params: Dict[str, float]) -> Optional[str]:
        """Return an error message if cross-parameter constraints fail."""
        return None

    @classmethod
    def to_normalized(cls, params: Dict[str, float]) -> List[float]:
        """Map raw parameters to the unit cube [0, 1]^d (linear per dimension)."""
        out = []
        for p in cls.PARAM_DEFS:
            u = (float(params[p.name]) - p.low) / (p.high - p.low)
            out.append(min(max(u, 0.0), 1.0))
        return out

    @classmethod
    def from_normalized(cls, u: Sequence[float]) -> Dict[str, float]:
        """Map a point of [0, 1]^d to raw parameters (clipped to the box)."""
        if len(u) != len(cls.PARAM_DEFS):
            raise ValueError(
                "%s expects %d normalized values, got %d" % (cls.name, len(cls.PARAM_DEFS), len(u))
            )
        out = {}
        for p, ui in zip(cls.PARAM_DEFS, u):
            ui = min(max(float(ui), 0.0), 1.0)
            x = p.low + ui * (p.high - p.low)
            out[p.name] = min(max(x, p.low), p.high)
        return out

    # ----------------------------------------------------------- episode state
    def reset(self) -> None:
        """Clear per-episode state. Called by the environment between episodes."""
        self._steps_seen = 0
        self._spent = 0.0
        self._expected_conversions = 0.0
        self._reset_controller()

    def _reset_controller(self) -> None:
        pass

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def expected_conversions(self) -> float:
        return self._expected_conversions

    @property
    def remaining_budget(self) -> float:
        return self.budget - self._spent

    def _consume_history(
        self,
        history_pvalue_info: Sequence[Any],
        history_bid: Sequence[Any],
        history_auction_result: Sequence[Any],
        history_impression_result: Sequence[Any],
        history_least_winning_cost: Sequence[Any],
    ) -> None:
        lengths = set(
            len(h)
            for h in (
                history_pvalue_info,
                history_bid,
                history_auction_result,
                history_impression_result,
                history_least_winning_cost,
            )
        )
        if len(lengths) != 1:
            raise PolicyInputError("history arguments have different lengths: %s" % sorted(lengths))
        n_hist = lengths.pop()
        if n_hist < self._steps_seen:
            # History shrank: a new episode started without reset(). Recount.
            self.reset()
        for i in range(self._steps_seen, n_hist):
            pinfo = _as_matrix("historyPValueInfo[%d]" % i, history_pvalue_info[i], 2)
            auction = _as_matrix("historyAuctionResult[%d]" % i, history_auction_result[i], 3)
            impression = _as_matrix("historyImpressionResult[%d]" % i, history_impression_result[i], 2)
            if np.asarray(history_bid[i]).size:
                _as_vector("historyBid[%d]" % i, history_bid[i])
            if np.asarray(history_least_winning_cost[i]).size:
                _as_vector("historyLeastWinningCost[%d]" % i, history_least_winning_cost[i])
            if not (pinfo.shape[0] == auction.shape[0] == impression.shape[0]):
                raise PolicyInputError("history step %d has inconsistent row counts" % i)
            self._spent += float(np.sum(auction[:, 2]))
            self._expected_conversions += float(
                np.sum(pinfo[:, 0] * auction[:, 0] * impression[:, 0])
            )
        self._steps_seen = n_hist

    # ----------------------------------------------------------- bidding
    def _ratio(self, t: int) -> float:
        raise NotImplementedError

    def _sigma_discount(self) -> float:
        return 0.0

    def bidding(
        self,
        timeStepIndex: int,
        pValues: Any,
        pValueSigmas: Any,
        historyPValueInfo: Sequence[Any],
        historyBid: Sequence[Any],
        historyAuctionResult: Sequence[Any],
        historyImpressionResult: Sequence[Any],
        historyLeastWinningCost: Sequence[Any],
    ) -> np.ndarray:
        """Return one non-negative finite bid per ad opportunity of this step.

        The argument list matches AuctionNet's ``BaseBiddingStrategy.bidding``.
        """
        try:
            t = int(timeStepIndex)
        except (TypeError, ValueError) as exc:
            raise PolicyInputError("timeStepIndex must be an integer, got %r" % (timeStepIndex,)) from exc
        if t != timeStepIndex or t < 0:
            raise PolicyInputError("timeStepIndex must be a non-negative integer, got %r" % (timeStepIndex,))
        p = _as_vector("pValues", pValues)
        s = _as_vector("pValueSigmas", pValueSigmas)
        if p.shape != s.shape:
            raise PolicyInputError("pValues %s and pValueSigmas %s differ in shape" % (p.shape, s.shape))
        self._consume_history(
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )
        # lambda_t is updated on every call so its trajectory does not depend
        # on whether a step happens to have opportunities.
        ratio = float(self._ratio(t))
        if p.size == 0:
            return np.zeros(0, dtype=float)
        if self.remaining_budget <= 0:
            return np.zeros(p.shape, dtype=float)
        p_adj = np.maximum(p - self._sigma_discount() * s, 0.0)
        bids = ratio * self.cpa_target * p_adj
        if bids.shape != p.shape or not np.all(np.isfinite(bids)) or np.any(bids < 0):
            raise PolicyOutputError("%s produced invalid bids" % self.name)
        return bids


class FixedRatePolicy(BasePolicy):
    """Constant multiplier: bid = ratio * cpa_target * pValue (rho = 0).

    Corresponds to the Abid-style baseline shipped with AuctionNet.
    """

    name = "fixed_rate"
    PARAM_DEFS = (ParamDef("ratio", 0.3, 3.0, 1.0),)

    def _ratio(self, t: int) -> float:
        return self.params["ratio"]


class DualControlPolicy(BasePolicy):
    """Multiplier driven by a CPA term and a pacing term (8 parameters).

    lambda_target = base_ratio
                    * exp(-cpa_gain * (cpa_est / cpa_target - 1))
                    * exp(-pace_gain * (spend_frac - plan_frac))
    lambda_t = clip(smoothing * lambda_{t-1} + (1 - smoothing) * lambda_target,
                    lambda_min, lambda_max)

    ``cpa_est`` uses expected conversions (sum of pValue over won and exposed
    opportunities); the CPA term is skipped while that sum is zero.
    ``plan_frac = (t / T) ** pace_shape``. ``lambda_{-1}`` is ``base_ratio``
    clipped to the multiplier bounds.
    """

    name = "dual_control"
    PARAM_DEFS = (
        ParamDef("base_ratio", 0.3, 3.0, 1.0),
        ParamDef("cpa_gain", 0.0, 2.0, 1.0),
        ParamDef("pace_gain", 0.0, 2.0, 1.0),
        ParamDef("pace_shape", 0.5, 2.0, 1.0),
        ParamDef("lambda_min", 0.1, 1.0, 0.2),
        ParamDef("lambda_max", 1.0, 5.0, 3.0),
        ParamDef("sigma_discount", 0.0, 1.5, 0.0),
        ParamDef("smoothing", 0.0, 0.9, 0.5),
    )

    @classmethod
    def check_constraints(cls, params: Dict[str, float]) -> Optional[str]:
        if params["lambda_min"] > params["lambda_max"]:
            return "lambda_min (%r) must be <= lambda_max (%r)" % (
                params["lambda_min"],
                params["lambda_max"],
            )
        return None

    def _reset_controller(self) -> None:
        q = self.params
        self._lambda_prev = min(max(q["base_ratio"], q["lambda_min"]), q["lambda_max"])

    def _ratio(self, t: int) -> float:
        q = self.params
        frac_t = min(max(float(t) / self.num_time_steps, 0.0), 1.0)
        plan_frac = frac_t ** q["pace_shape"]
        spend_frac = self._spent / self.budget if self.budget > 0 else 1.0
        log_target = math.log(q["base_ratio"])
        if self._expected_conversions > 0:
            cpa_est = self._spent / self._expected_conversions
            log_target -= q["cpa_gain"] * (cpa_est / self.cpa_target - 1.0)
        log_target -= q["pace_gain"] * (spend_frac - plan_frac)
        lam_target = math.exp(min(max(log_target, -_LOG_CLIP), _LOG_CLIP))
        lam = q["smoothing"] * self._lambda_prev + (1.0 - q["smoothing"]) * lam_target
        lam = min(max(lam, q["lambda_min"]), q["lambda_max"])
        self._lambda_prev = lam
        return lam

    def _sigma_discount(self) -> float:
        return self.params["sigma_discount"]
