"""(1+lambda)-ES and random proposers; budget ledger invariants."""

from __future__ import annotations

import numpy as np
import pytest

from bidswarm.config import CmaesConfig, EsConfig
from bidswarm.evaluation.budget import BudgetExceededError, BudgetLedger
from bidswarm.search.proposers import CmaesProposer, EsProposer, Observation, RandomProposer


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


# --- CMA-ES -----------------------------------------------------------------------------------


def _sphere(target):
    return lambda x: -float(np.sum((np.asarray(x) - target) ** 2))


def test_cmaes_population_defaults_to_standard_size_rounded_to_whole_rounds():
    assert CmaesProposer.default_population(8) == 10
    rng = np.random.default_rng(0)
    cm = CmaesProposer(dim=8)
    cm.propose(2, rng)
    assert cm.population == 10
    cm4 = CmaesProposer(dim=8)
    cm4.propose(4, rng)
    assert cm4.population == 12
    fixed = CmaesProposer(dim=8, cmaes=CmaesConfig(population=6))
    with pytest.raises(ValueError, match="multiple"):
        fixed.propose(4, rng)


def test_cmaes_hands_out_one_generation_across_rounds_then_updates():
    rng = np.random.default_rng(3)
    cm = CmaesProposer(dim=2, cmaes=CmaesConfig(population=4))
    first = cm.propose(2, rng)
    assert {p.operator for p in first} == {"cmaes_sample"}
    assert all(p.parent_id is None and p.sigma == 0.3 and p.info_sources == () for p in first)
    mean_before = cm.mean.copy()
    _observe(cm, first, [1.0, 2.0], "a")
    assert cm.generation == 0 and np.array_equal(cm.mean, mean_before)  # half a generation: no update
    second = cm.propose(2, rng)
    assert all(p.info_sources == ("a0", "a1") for p in second)
    _observe(cm, second, [4.0, 3.0], "b")
    assert cm.generation == 1
    # mu = 2: the new mean recombines the two best samples (b0 > b1), with b0 weighted higher.
    b0, b1 = np.array(second[0].normalized), np.array(second[1].normalized)
    assert np.allclose(cm.mean, 0.75 * b0 + 0.25 * b1, atol=0.1 * np.linalg.norm(b0 - b1) + 1e-12)
    assert np.linalg.norm(cm.mean - b0) < np.linalg.norm(cm.mean - b1)
    assert cm.best.candidate_id == "b0"


def test_cmaes_converges_on_a_smooth_objective_inside_the_box():
    rng = np.random.default_rng(11)
    target = np.array([0.2, 0.7, 0.4, 0.9, 0.1])
    f = _sphere(target)
    cm = CmaesProposer(dim=5)
    for g in range(60):
        proposals = cm.propose(4, rng)
        _observe(cm, proposals, [f(p.normalized) for p in proposals], f"g{g}-")
        assert all(0.0 <= x <= 1.0 for p in proposals for x in p.normalized)
    assert np.linalg.norm(cm.mean - target) < 0.02
    assert cm.best.fitness > -1e-3


def test_cmaes_ranks_failures_last_and_ignores_an_all_failed_generation():
    rng = np.random.default_rng(5)
    cm = CmaesProposer(dim=3, cmaes=CmaesConfig(population=4))
    proposals = cm.propose(4, rng)
    mean_before, sigma_before = cm.mean.copy(), cm.sigma
    _observe(cm, proposals, [None, None, None, None])
    assert cm.generation == 1 and cm.best is None
    assert np.array_equal(cm.mean, mean_before) and cm.sigma == sigma_before

    proposals = cm.propose(4, rng)
    _observe(cm, proposals, [None, 2.0, None, None], "x")
    assert np.allclose(cm.mean, proposals[1].normalized)  # the only valid sample


def test_cmaes_received_elite_becomes_the_next_generation_mean():
    rng = np.random.default_rng(9)
    cm = CmaesProposer(dim=2, cmaes=CmaesConfig(population=4))
    _observe(cm, cm.propose(2, rng), [1.0, 0.5], "a")
    elite = Observation("w1-x", (0.9, 0.1), 5.0)
    cm.receive([elite, Observation("w1-y", (0.5, 0.5), None)])
    assert cm.best == elite
    # The rest of the current generation is still drawn around the old mean ...
    rest = cm.propose(2, rng)
    assert all(p.info_sources == ("a0", "a1", "w1-x", "w1-y") for p in rest)
    _observe(cm, rest, [0.0, 0.0], "b")
    # ... and the next generation is centred on the received elite.
    cm.propose(2, rng)
    assert np.allclose(cm.mean, elite.normalized)


def test_cmaes_ignores_a_received_candidate_that_is_not_better():
    rng = np.random.default_rng(9)
    cm = CmaesProposer(dim=2, cmaes=CmaesConfig(population=2))
    _observe(cm, cm.propose(2, rng), [3.0, 1.0], "a")
    cm.receive([Observation("w1-z", (0.1, 0.1), 2.0)])
    assert cm.best.candidate_id == "a0"
    mean = cm.mean.copy()
    cm.propose(2, rng)
    assert np.array_equal(cm.mean, mean)


def test_cmaes_requires_alternating_calls():
    rng = np.random.default_rng(0)
    cm = CmaesProposer(dim=2)
    with pytest.raises(RuntimeError):
        cm.observe([])
    cm.propose(2, rng)
    with pytest.raises(RuntimeError):
        cm.propose(2, rng)
    with pytest.raises(RuntimeError):
        cm.receive([])
