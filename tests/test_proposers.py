"""(1+lambda)-ES and random proposers; budget ledger invariants."""

from __future__ import annotations

import numpy as np
import pytest

from bidswarm.config import EsConfig
from bidswarm.evaluation.budget import BudgetExceededError, BudgetLedger
from bidswarm.search.proposers import EsProposer, Observation, RandomProposer


def _observe(proposer, proposals, fitnesses, prefix="c"):
    obs = [
        Observation(f"{prefix}{i}", p.normalized, f) for i, (p, f) in enumerate(zip(proposals, fitnesses))
    ]
    proposer.observe(obs)
    return obs


def test_es_first_round_is_random_then_best_becomes_parent():
    rng = np.random.default_rng(0)
    es = EsProposer(dim=3)
    first = es.propose(4, rng)
    assert len(first) == 4
    assert {p.operator for p in first} == {"init_random"}
    assert all(p.parent_id is None and p.sigma is None and p.info_sources == () for p in first)
    obs = _observe(es, first, [1.0, 5.0, None, 3.0])
    assert es.parent == obs[1]
    assert es.sigma == 0.2  # unchanged by initialisation

    children = es.propose(4, rng)
    assert {p.operator for p in children} == {"gaussian_mutation"}
    assert all(p.parent_id == "c1" and p.sigma == 0.2 for p in children)
    assert all(p.info_sources == ("c0", "c1", "c2", "c3") for p in children)
    assert all(0.0 <= x <= 1.0 for p in children for x in p.normalized)


def test_es_step_size_rule():
    rng = np.random.default_rng(1)
    es = EsProposer(dim=2)
    _observe(es, es.propose(2, rng), [1.0, 2.0], "a")

    _observe(es, es.propose(2, rng), [3.0, 0.5], "b")  # improvement -> replace parent, sigma * 1.22
    assert es.parent.candidate_id == "b0"
    assert es.sigma == pytest.approx(0.2 * 1.22)

    _observe(es, es.propose(2, rng), [3.0, None], "c")  # tie is not an improvement -> sigma * 0.82
    assert es.parent.candidate_id == "b0"
    assert es.sigma == pytest.approx(0.2 * 1.22 * 0.82)

    for i in range(40):  # repeated failures clamp at sigma_min
        _observe(es, es.propose(2, rng), [None, None], f"f{i}-")
    assert es.sigma == pytest.approx(0.01)

    for i in range(40):  # repeated successes clamp at sigma_max
        _observe(es, es.propose(2, rng), [10.0 + i, None], f"s{i}-")
    assert es.sigma == pytest.approx(0.5)


def test_es_reinitialises_when_all_initial_candidates_fail():
    rng = np.random.default_rng(2)
    es = EsProposer(dim=2, es=EsConfig())
    _observe(es, es.propose(3, rng), [None, None, None])
    again = es.propose(3, rng)
    assert {p.operator for p in again} == {"init_random"}


def test_es_requires_alternating_calls():
    es = EsProposer(dim=2)
    rng = np.random.default_rng(3)
    es.propose(1, rng)
    with pytest.raises(RuntimeError):
        es.propose(1, rng)


def test_random_proposer_is_memoryless():
    rng = np.random.default_rng(4)
    rp = RandomProposer(dim=5)
    props = rp.propose(6, rng)
    assert {p.operator for p in props} == {"random"}
    assert all(p.parent_id is None and p.info_sources == () for p in props)
    _observe(rp, props, [1.0] * 6)
    assert all(p.info_sources == () for p in rp.propose(2, rng))


def test_ledger_never_exceeds_plan():
    ledger = BudgetLedger()
    ledger.plan(0, "single", "train", 10)
    ledger.charge(0, "single", "train", 6)
    with pytest.raises(BudgetExceededError):
        ledger.charge(0, "single", "train", 5)
    ledger.charge(0, "single", "train", 4)
    ledger.record_outcomes(0, "single", "train", ["ok", "failed", "timeout", "failed"])
    entry = ledger.entries()[0]
    assert (entry.planned, entry.used, entry.failed, entry.timeouts, entry.retries) == (10, 10, 2, 1, 0)
    assert BudgetLedger.from_dict(ledger.to_dict()).to_dict() == ledger.to_dict()
