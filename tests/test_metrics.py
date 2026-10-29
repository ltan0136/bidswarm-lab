"""Numerical checks of the replicated AuctionNet score formula."""

from __future__ import annotations

import pytest

from bidswarm.evaluation.metrics import (
    auctionnet_aggregate_score,
    auctionnet_cpa,
    auctionnet_score,
    cpa_within_target,
    mean_or_none,
)


def test_cpa_within_limit_scores_reward():
    # cpa = 50 / 10 = 5 <= C = 8
    assert auctionnet_score(reward=10.0, cost=50.0, cpa_constraint=8.0) == 10.0
    assert cpa_within_target(10.0, 50.0, 8.0)


def test_cpa_over_limit_is_penalised_quadratically():
    # cpa = 200 / 10 = 20 > C = 10 -> 10 * (10 / 20)^2 = 2.5
    assert auctionnet_score(reward=10.0, cost=200.0, cpa_constraint=10.0) == pytest.approx(2.5, rel=1e-9)
    assert not cpa_within_target(10.0, 200.0, 10.0)
    # Exact upstream expression, including the 1e-10 in the denominator.
    cpa = 200.0 / (10.0 + 1e-10)
    assert auctionnet_score(10.0, 200.0, 10.0) == 10.0 * (10.0 / cpa) ** 2


def test_zero_conversions_score_about_zero():
    assert auctionnet_cpa(0.0, 50.0) == pytest.approx(5e11)
    assert auctionnet_score(reward=0.0, cost=50.0, cpa_constraint=8.0) == 0.0
    assert auctionnet_score(reward=0.0, cost=0.0, cpa_constraint=8.0) == 0.0
    # A tiny positive reward with real spend is crushed by the penalty.
    assert auctionnet_score(reward=1e-6, cost=50.0, cpa_constraint=8.0) < 1e-12


def test_aggregate_divides_by_20000():
    assert auctionnet_aggregate_score([10000.0, 30000.0]) == 2.0


def test_mean_or_none():
    assert mean_or_none([1.0, 2.0, 4.0]) == pytest.approx(7 / 3)
    assert mean_or_none([1.0, None]) is None
    assert mean_or_none([]) is None
