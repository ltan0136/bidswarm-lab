"""Kernel policies and PolicySpec: bounds, shapes, edge cases, hashing, normalization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bidswarm.kernel import (
    REGISTRY,
    DualControlPolicy,
    FixedRatePolicy,
    PolicyInputError,
    PolicySpec,
    PolicySpecError,
    canonical_json,
    default_spec,
    make_policy,
    spec_from_normalized,
    spec_hash,
    spec_to_normalized,
    validate_spec,
)
from bidswarm.kernel import policies as policies_module

EMPTY_HISTORY = ([], [], [], [], [])


def _history_step(n: int, pvalue: np.ndarray, win: np.ndarray, cost: np.ndarray, exposed: np.ndarray):
    sigma = 0.2 * pvalue
    return (
        np.column_stack([pvalue, sigma]),
        np.ones(n),
        np.column_stack([win, win, cost]),
        np.column_stack([exposed, np.zeros(n)]),
        np.full(n, 0.1),
    )


def _append(history, step):
    for h, item in zip(history, step):
        h.append(item)


@pytest.fixture(params=sorted(REGISTRY))
def any_spec(request) -> PolicySpec:
    return default_spec(request.param)


def test_registry_has_both_templates():
    assert set(REGISTRY) == {"fixed_rate", "dual_control"}
    assert len(DualControlPolicy.PARAM_DEFS) == 8
    bounds = {p.name: (p.low, p.high) for p in DualControlPolicy.PARAM_DEFS}
    assert bounds == {
        "base_ratio": (0.3, 3.0),
        "cpa_gain": (0.0, 2.0),
        "pace_gain": (0.0, 2.0),
        "pace_shape": (0.5, 2.0),
        "lambda_min": (0.1, 1.0),
        "lambda_max": (1.0, 5.0),
        "sigma_discount": (0.0, 1.5),
        "smoothing": (0.0, 0.9),
    }
    assert [(p.name, p.low, p.high) for p in FixedRatePolicy.PARAM_DEFS] == [("ratio", 0.3, 3.0)]


def test_bids_shape_finite_nonnegative(any_spec):
    rng = np.random.default_rng(0)
    policy = make_policy(any_spec, budget=100.0, cpa_target=8.0, category=1, num_time_steps=10)
    history = ([], [], [], [], [])
    for t in range(10):
        n = int(rng.integers(1, 60))
        p = rng.beta(2, 50, size=n)
        s = p * rng.uniform(0, 3, size=n)  # large sigma drives p_adj to 0 for some
        bids = policy.bidding(t, p, s, *history)
        assert bids.shape == (n,)
        assert np.all(np.isfinite(bids)) and np.all(bids >= 0)
        win = (rng.random(n) < 0.5).astype(float)
        _append(history, _history_step(n, p, win, win * 0.05, win))


def test_fixed_rate_formula_reads_constructor_values():
    spec = PolicySpec("fixed_rate", {"ratio": 1.5})
    p = np.array([0.01, 0.02, 0.0])
    for cpa in (5.0, 9.0):
        policy = make_policy(spec, budget=50.0, cpa_target=cpa)
        np.testing.assert_allclose(policy.bidding(0, p, p * 0.3, *EMPTY_HISTORY), 1.5 * cpa * p)


def test_dual_control_matches_hand_computation():
    params = {
        "base_ratio": 1.2,
        "cpa_gain": 0.5,
        "pace_gain": 1.0,
        "pace_shape": 2.0,
        "lambda_min": 0.2,
        "lambda_max": 4.0,
        "sigma_discount": 0.5,
        "smoothing": 0.3,
    }
    budget, cpa, T = 40.0, 10.0, 4
    policy = make_policy(PolicySpec("dual_control", params), budget, cpa, num_time_steps=T)
    p = np.array([0.02, 0.05])
    s = np.array([0.01, 0.2])
    p_adj = np.maximum(p - 0.5 * s, 0.0)

    # t = 0, no history: lambda_target = base_ratio; lambda_{-1} = base_ratio.
    np.testing.assert_allclose(policy.bidding(0, p, s, *EMPTY_HISTORY), 1.2 * cpa * p_adj)

    history = ([], [], [], [], [])
    _append(history, _history_step(2, p, np.array([1.0, 1.0]), np.array([3.0, 5.0]), np.array([1.0, 0.0])))
    spent, expected = 8.0, 0.02
    spend_frac, plan_frac = spent / budget, (1 / T) ** 2.0
    lam_target = 1.2 * math.exp(-0.5 * (spent / expected / cpa - 1)) * math.exp(-1.0 * (spend_frac - plan_frac))
    lam = min(max(0.3 * 1.2 + 0.7 * lam_target, 0.2), 4.0)
    np.testing.assert_allclose(policy.bidding(1, p, s, *history), lam * cpa * p_adj)
    assert policy.spent == pytest.approx(spent)
    assert policy.expected_conversions == pytest.approx(expected)


def test_dual_control_skips_cpa_term_without_expected_conversions():
    params = dict(default_spec("dual_control").params, cpa_gain=2.0, pace_gain=0.0, smoothing=0.0)
    policy = make_policy(PolicySpec("dual_control", params), budget=100.0, cpa_target=5.0, num_time_steps=4)
    p = np.array([0.03])
    history = ([], [], [], [], [])
    # Money spent but nothing exposed: expected conversions stay 0, so only base_ratio applies.
    _append(history, _history_step(1, p, np.array([1.0]), np.array([10.0]), np.array([0.0])))
    np.testing.assert_allclose(policy.bidding(1, p, p, *history), params["base_ratio"] * 5.0 * p)


def test_zero_budget_bids_zero(any_spec):
    p = np.array([0.02, 0.04])
    policy = make_policy(any_spec, budget=0.0, cpa_target=8.0)
    np.testing.assert_array_equal(policy.bidding(0, p, p * 0.1, *EMPTY_HISTORY), np.zeros(2))


def test_exhausted_budget_bids_zero(any_spec):
    p = np.array([0.02, 0.04])
    policy = make_policy(any_spec, budget=5.0, cpa_target=8.0, num_time_steps=3)
    history = ([], [], [], [], [])
    _append(history, _history_step(2, p, np.array([1.0, 1.0]), np.array([2.0, 3.0]), np.array([1.0, 1.0])))
    np.testing.assert_array_equal(policy.bidding(1, p, p * 0.1, *history), np.zeros(2))


def test_no_opportunities_returns_empty(any_spec):
    policy = make_policy(any_spec, budget=10.0, cpa_target=8.0)
    out = policy.bidding(0, np.zeros(0), np.zeros(0), *EMPTY_HISTORY)
    assert isinstance(out, np.ndarray) and out.shape == (0,)


@pytest.mark.parametrize("where", ["pvalue", "sigma", "history"])
def test_nan_input_raises_policy_input_error(any_spec, where):
    p = np.array([0.02, 0.04])
    s = p * 0.1
    history = ([], [], [], [], [])
    if where == "pvalue":
        p = np.array([0.02, np.nan])
    elif where == "sigma":
        s = np.array([np.inf, 0.0])
    else:
        _append(history, _history_step(2, p, np.array([1.0, 0.0]), np.array([np.nan, 0.0]), np.zeros(2)))
    policy = make_policy(any_spec, budget=10.0, cpa_target=8.0)
    with pytest.raises(PolicyInputError):
        policy.bidding(len(history[0]), p, s, *history)


def test_shape_mismatch_raises():
    policy = make_policy(default_spec("fixed_rate"), budget=10.0, cpa_target=8.0)
    with pytest.raises(PolicyInputError):
        policy.bidding(0, np.ones(3) * 0.01, np.ones(2) * 0.01, *EMPTY_HISTORY)


def test_history_is_consumed_incrementally(monkeypatch):
    """Each history entry is parsed once per episode: O(T), not O(T^2)."""
    calls = {"n": 0}
    original = policies_module._as_matrix

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(policies_module, "_as_matrix", counting)
    T, n = 30, 5
    policy = make_policy(default_spec("dual_control"), budget=1e6, cpa_target=8.0, num_time_steps=T)
    history = ([], [], [], [], [])
    p = np.full(n, 0.02)
    total_cost = 0.0
    for t in range(T):
        policy.bidding(t, p, p * 0.1, *history)
        assert policy.spent == pytest.approx(total_cost)
        cost = np.full(n, 0.01 * (t + 1))
        total_cost += float(cost.sum())
        _append(history, _history_step(n, p, np.ones(n), cost, np.ones(n)))
    assert calls["n"] == 3 * (T - 1)  # pinfo, auction, impression for each of the T-1 past steps


def test_reset_clears_episode_state():
    policy = make_policy(default_spec("dual_control"), budget=10.0, cpa_target=8.0, num_time_steps=4)
    p = np.array([0.02])
    history = ([], [], [], [], [])
    _append(history, _history_step(1, p, np.array([1.0]), np.array([4.0]), np.array([1.0])))
    first = policy.bidding(1, p, p, *history)
    policy.reset()
    assert policy.spent == 0.0 and policy.expected_conversions == 0.0
    np.testing.assert_array_equal(policy.bidding(1, p, p, *history), first)


# ------------------------------------------------------------------ PolicySpec
def test_spec_validation_errors():
    with pytest.raises(PolicySpecError, match="unknown policy"):
        validate_spec(PolicySpec("nope", {}))
    with pytest.raises(PolicySpecError, match="missing"):
        validate_spec(PolicySpec("fixed_rate", {}))
    with pytest.raises(PolicySpecError, match="unknown"):
        validate_spec(PolicySpec("fixed_rate", {"ratio": 1.0, "extra": 1.0}))
    for bad in (3.5, 0.29, float("nan"), float("inf"), True, "1.0"):
        with pytest.raises(PolicySpecError):
            validate_spec(PolicySpec("fixed_rate", {"ratio": bad}))
    with pytest.raises(PolicySpecError, match="outside bounds"):
        validate_spec(dataclass_replace_param("dual_control", "lambda_max", 5.5))
    with pytest.raises(PolicySpecError):
        PolicySpec.from_dict({"name": "fixed_rate", "params": {"ratio": 1.0}, "code": "print(1)"})
    with pytest.raises(PolicySpecError):
        make_policy(PolicySpec("fixed_rate", {"ratio": 9.0}), budget=1.0, cpa_target=1.0)


def dataclass_replace_param(name: str, key: str, value: float) -> PolicySpec:
    params = dict(default_spec(name).params)
    params[key] = value
    return PolicySpec(name, params)


def test_lambda_bounds_constraint():
    ok = dict(default_spec("dual_control").params, lambda_min=1.0, lambda_max=1.0)
    assert validate_spec(PolicySpec("dual_control", ok)).params["lambda_min"] == 1.0
    bad = dict(ok, lambda_min=1.0, lambda_max=0.9)
    assert "lambda_min" in DualControlPolicy.check_constraints(bad)


def test_spec_hash_is_canonical():
    a = PolicySpec("fixed_rate", {"ratio": 1})
    b = PolicySpec("fixed_rate", {"ratio": 1.0})
    assert spec_hash(a) == spec_hash(b)
    assert canonical_json(a.to_dict()) == '{"name":"fixed_rate","params":{"ratio":1.0}}'
    assert spec_hash(a) != spec_hash(PolicySpec("fixed_rate", {"ratio": 1.0000001}))
    assert len(spec_hash(a)) == 64


def test_normalized_round_trip_and_validity():
    rng = np.random.default_rng(1)
    for name, cls in REGISTRY.items():
        d = len(cls.PARAM_DEFS)
        for u in [np.zeros(d), np.ones(d), *rng.random((200, d))]:
            spec = spec_from_normalized(name, u)  # validates: every unit-cube point is a legal spec
            np.testing.assert_allclose(spec_to_normalized(spec), u, atol=1e-12)
        defaults = default_spec(name)
        back = cls.from_normalized(spec_to_normalized(defaults))
        for k, v in defaults.params.items():
            assert back[k] == pytest.approx(v, abs=1e-12)
