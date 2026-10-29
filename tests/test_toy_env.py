"""Toy environment: CRN, observation shape, auction/budget mechanics, failure handling."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from bidswarm.artifacts.store import dumps_strict
from bidswarm.environments.toy import ToyConfig, ToyEnvironment, step_traffic
from bidswarm.kernel import PolicySpec, make_policy

SMALL_ENV = ToyConfig(num_time_steps=6, opportunities_per_step=40, budget_range=(10.0, 30.0))


def scenario(env: ToyEnvironment, seed: int = 11) -> dict:
    return env.make_scenario(f"s{seed}", "train", 0, seed)


class SpyPolicy:
    """Wraps a kernel policy and records everything bidding() receives."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def reset(self):
        self.inner.reset()

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        self.calls.append(
            {
                "t": timeStepIndex,
                "pValues": pValues.copy(),
                "pValueSigmas": pValueSigmas.copy(),
                "history": [
                    [np.array(x) for x in h]
                    for h in (
                        historyPValueInfo,
                        historyBid,
                        historyAuctionResult,
                        historyImpressionResult,
                        historyLeastWinningCost,
                    )
                ],
            }
        )
        return self.inner.bidding(
            timeStepIndex,
            pValues,
            pValueSigmas,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )


class ConstantPolicy:
    def __init__(self, value: float, n: int):
        self.value, self.n = value, n

    def reset(self):
        pass

    def bidding(self, t, p, s, *history):
        return np.full(self.n, self.value)


def _fixed(sc: dict, ratio: float):
    return make_policy(PolicySpec("fixed_rate", {"ratio": ratio}), sc["budget"], sc["cpa_target"], sc["category"], 6)


def test_common_random_numbers_across_policies():
    env = ToyEnvironment(SMALL_ENV)
    sc = scenario(env)
    low, high = SpyPolicy(_fixed(sc, 0.3)), SpyPolicy(_fixed(sc, 3.0))
    r_low, r_high = env.run_episode(sc, low), env.run_episode(sc, high)
    assert r_low["wins"] != r_high["wins"]  # the policies really behave differently
    assert len(low.calls) == len(high.calls) == SMALL_ENV.num_time_steps
    for a, b in zip(low.calls, high.calls):
        np.testing.assert_array_equal(a["pValues"], b["pValues"])
        np.testing.assert_array_equal(a["pValueSigmas"], b["pValueSigmas"])
        for x, y in zip(a["history"][4], b["history"][4]):  # least winning cost
            np.testing.assert_array_equal(x, y)
    # Same draws as the pure traffic function, independent of any policy.
    for call in low.calls:
        np.testing.assert_array_equal(call["pValues"], step_traffic(SMALL_ENV, sc["seed"], call["t"]).pvalue)


def test_traffic_depends_on_scenario_seed():
    a, b = step_traffic(SMALL_ENV, 1, 0), step_traffic(SMALL_ENV, 2, 0)
    assert not np.array_equal(a.pvalue, b.pvalue)
    np.testing.assert_array_equal(step_traffic(SMALL_ENV, 1, 3).least_winning_cost, step_traffic(SMALL_ENV, 1, 3).least_winning_cost)


def test_observation_is_exactly_the_eight_bidding_arguments():
    params = list(inspect.signature(SpyPolicy.bidding).parameters)[1:]
    assert len(params) == 8
    env = ToyEnvironment(SMALL_ENV)
    sc = scenario(env)
    spy = SpyPolicy(_fixed(sc, 1.0))
    env.run_episode(sc, spy)
    n = SMALL_ENV.opportunities_per_step
    last = spy.calls[-1]
    pinfo, bid, auction, impression, lwc = last["history"]
    assert len(pinfo) == SMALL_ENV.num_time_steps - 1
    assert pinfo[0].shape == (n, 2)
    assert bid[0].shape == (n,)
    assert auction[0].shape == (n, 3)
    assert impression[0].shape == (n, 2)
    assert lwc[0].shape == (n,)
    assert spy.calls[0]["history"] == [[], [], [], [], []]


def test_second_price_and_exposure_semantics():
    env = ToyEnvironment(SMALL_ENV)
    sc = dict(scenario(env), budget=1e9)
    spy = SpyPolicy(_fixed(sc, 1.0))
    result = env.run_episode(sc, spy)
    pinfo, bid, auction, impression, lwc = spy.calls[-1]["history"]
    for t in range(len(bid)):
        win = auction[t][:, 0].astype(bool)
        np.testing.assert_array_equal(win, bid[t] > lwc[t])
        np.testing.assert_array_equal(auction[t][:, 2], np.where(win, lwc[t], 0.0))  # pay the price
        exposed = impression[t][:, 0].astype(bool)
        assert not np.any(exposed & ~win)
        assert not np.any(impression[t][:, 1].astype(bool) & ~exposed)  # conversion needs exposure
    assert result["wins"] > 0


def test_spend_never_exceeds_budget():
    env = ToyEnvironment(SMALL_ENV)
    truncated = 0
    for seed in range(20):
        sc = scenario(env, seed)
        result = env.run_episode(sc, ConstantPolicy(1e6, SMALL_ENV.opportunities_per_step))
        assert result["status"] == "ok"
        assert result["cost"] <= result["budget"]
        assert result["budget_utilization"] <= 1.0
        truncated += result["wins"] < SMALL_ENV.opportunities_per_step * SMALL_ENV.num_time_steps
    assert truncated == 20  # every scenario hit the cap, so the truncation path was exercised


def test_zero_conversions_give_null_cpa_and_strict_json():
    env = ToyEnvironment(SMALL_ENV)
    sc = scenario(env)
    result = env.run_episode(sc, ConstantPolicy(0.0, SMALL_ENV.opportunities_per_step))
    assert result["conversions"] == 0 and result["expected_conversions"] == 0.0
    assert result["cpa_realized"] is None and result["cpa_expected"] is None
    assert result["cpa_defined"] is False and result["cpa_expected_defined"] is False
    assert result["score_realized"] == 0.0 and result["score_expected"] == 0.0
    text = dumps_strict(result)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_outputs_are_tagged():
    env = ToyEnvironment(SMALL_ENV)
    sc = scenario(env)
    result = env.run_episode(sc, _fixed(sc, 1.0))
    assert result["backend"] == "toy"
    assert result["purpose"] == "framework_smoke_test"
    assert result["score_formula"] == "auctionnet_neurips_replica"
    expected_keys = {
        "cost", "budget", "cpa_target", "conversions", "expected_conversions", "cpa_realized",
        "cpa_expected", "cpa_defined", "score_realized", "score_expected", "budget_utilization",
        "wins", "opportunities", "cpa_ok", "status", "wall_time_s",
    }  # fmt: skip
    assert expected_keys <= set(result)


class NaNInjectingPolicy:
    def __init__(self, inner):
        self.inner = inner

    def reset(self):
        self.inner.reset()

    def bidding(self, t, p, s, *history):
        if t == 2:
            p = p.copy()
            p[0] = np.nan
        return self.inner.bidding(t, p, s, *history)


class RaisingPolicy:
    def reset(self):
        pass

    def bidding(self, *args):
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    "make, fragment",
    [
        (lambda sc: NaNInjectingPolicy(_fixed(sc, 1.0)), "PolicyInputError"),
        (lambda sc: RaisingPolicy(), "boom"),
        (lambda sc: ConstantPolicy(-1.0, SMALL_ENV.opportunities_per_step), "negative"),
        (lambda sc: ConstantPolicy(1.0, 3), "shape"),
    ],
)
def test_policy_errors_become_failed_episodes(make, fragment):
    env = ToyEnvironment(SMALL_ENV)
    sc = scenario(env)
    result = env.run_episode(sc, make(sc))
    assert result["status"] == "failed"
    assert fragment in result["error"]
    assert result["score_expected"] is None and result["cost"] is None
    json.loads(dumps_strict(result))


def test_advertiser_sampled_from_config_ranges():
    env = ToyEnvironment(SMALL_ENV)
    for seed in range(50):
        sc = scenario(env, seed)
        assert 10.0 <= sc["budget"] <= 30.0
        assert 6.0 <= sc["cpa_target"] <= 12.0
        assert 0 <= sc["category"] < SMALL_ENV.num_categories
    assert scenario(env, 3) == scenario(env, 3)
