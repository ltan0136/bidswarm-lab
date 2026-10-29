"""Information exchange (ShareBest elite migration): delivery, adoption, isolation, accounting."""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from bidswarm.experiment import evaluate_run, run_experiment
from bidswarm.search.exchange import EvidenceCard, NoExchange, ShareBest
from bidswarm.search.proposers import EsProposer, Observation, RandomProposer
from bidswarm.search.worker import IsolationError, SearchWorker
from bidswarm.types import ConfigError

from .conftest import read_jsonl, small_config

THREE_MODES = {"modes": ["single", "independent", "exchange"]}


def card(cid: str, worker: int, fitness: float | None, rnd: int = 0, dim: int = 2) -> EvidenceCard:
    return EvidenceCard(cid, worker, rnd, "h" + cid, fitness, tuple([0.5] * dim))


# ------------------------------------------------------------------ policy
def test_share_best_all_topology_delivers_neighbours_best():
    policy = ShareBest(n_workers=3, topology="all")
    out = policy.exchange(0, {0: [card("a0", 0, 1.0), card("a1", 0, 3.0)], 1: [card("b0", 1, 2.0)], 2: [card("c0", 2, None)]})
    assert [c.candidate_id for c in out[0]] == ["b0"]  # worker 2 has no usable card yet
    assert [c.candidate_id for c in out[1]] == ["a1"]
    assert [c.candidate_id for c in out[2]] == ["a1", "b0"]
    # Best-so-far is remembered: a worse card later does not replace it.
    out = policy.exchange(1, {0: [card("a2", 0, 0.5, 1)], 1: [card("b1", 1, 9.0, 1)], 2: [card("c1", 2, 4.0, 1)]})
    assert [c.candidate_id for c in out[0]] == ["b1", "c1"]
    assert [c.candidate_id for c in out[2]] == ["a1", "b1"]


def test_share_best_ring_topology():
    policy = ShareBest(n_workers=3, topology="ring")
    out = policy.exchange(0, {w: [card(f"w{w}", w, float(w))] for w in range(3)})
    assert {w: [c.candidate_id for c in cs] for w, cs in out.items()} == {0: ["w2"], 1: ["w0"], 2: ["w1"]}


def test_share_best_rejects_misattributed_cards_and_bad_settings():
    with pytest.raises(ValueError):
        ShareBest(n_workers=2).exchange(0, {0: [card("x", 1, 1.0)], 1: []})
    with pytest.raises(ValueError):
        ShareBest(n_workers=2, topology="star")
    with pytest.raises(ValueError):
        ShareBest(n_workers=1)
    assert NoExchange().exchange(0, {0: [card("x", 0, 1.0)], 1: []}) == {0: [], 1: []}


# ------------------------------------------------------------------ proposers
def test_es_adopts_strictly_better_foreign_parent_without_touching_sigma():
    rng = np.random.default_rng(0)
    es = EsProposer(dim=2)
    props = es.propose(2, rng)
    es.observe([Observation("own0", props[0].normalized, 5.0), Observation("own1", props[1].normalized, 1.0)])
    assert es.receive([card("f0", 1, 5.0)]) is None  # tie: keep own parent
    assert es.receive([card("f1", 1, 4.0), card("f2", 2, 7.0)]) == "f2"
    assert es.parent.candidate_id == "f2" and es.parent.normalized == (0.5, 0.5)
    assert es.sigma == 0.2
    children = es.propose(2, rng)
    assert all(c.parent_id == "f2" for c in children)
    assert children[0].info_sources == ("own0", "own1", "f0", "f1", "f2")
    with pytest.raises(RuntimeError):
        es.receive([])  # between propose and observe


def test_es_adopts_foreign_parent_when_all_own_candidates_failed():
    es = EsProposer(dim=2)
    props = es.propose(1, np.random.default_rng(1))
    es.observe([Observation("own0", props[0].normalized, None)])
    assert es.receive([card("f0", 1, 2.0)]) == "f0"
    assert {p.operator for p in es.propose(2, np.random.default_rng(2))} == {"gaussian_mutation"}


def test_random_proposer_ignores_evidence():
    rp = RandomProposer(dim=2)
    rp.observe([])
    assert rp.receive([card("f0", 1, 100.0)]) is None
    assert all(p.info_sources == () for p in rp.propose(3, np.random.default_rng(3)))


def test_worker_rejects_own_or_foreign_pipeline_cards():
    w = SearchWorker(0, 7, "exchange", 1, "fixed_rate", EsProposer(dim=1), np.random.default_rng(0))
    with pytest.raises(IsolationError):
        w.receive_evidence([card("r00-exchange-w01-c0000", 1, 1.0, dim=1)])
    with pytest.raises(IsolationError):
        w.receive_evidence([card("r01-exchange-w00-c0000", 0, 1.0, dim=1)])
    assert w.receive_evidence([card("r00-exchange-w00-c0000", 0, 1.0, dim=1)]) == "r00-exchange-w00-c0000"


# ------------------------------------------------------------------ config
def test_exchange_config_validation():
    cfg = small_config(search=THREE_MODES)
    assert cfg.budget_plan("exchange") == dataclasses.replace(cfg.budget_plan("independent"), mode="exchange")
    with pytest.raises(ConfigError, match="unknown policy"):
        small_config(search={**THREE_MODES, "exchange": {"policy": "gossip"}})
    with pytest.raises(ConfigError, match="unknown topology"):
        small_config(search={**THREE_MODES, "exchange": {"topology": "star"}})
    with pytest.raises(ConfigError, match="independent_workers >= 2"):
        small_config(search={"modes": ["single", "exchange"], "independent_workers": 1})
    assert small_config().search.modes == ("single", "independent")  # default modes unchanged


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def exchange_run(tmp_path_factory) -> Path:
    run_dir = run_experiment(small_config(search=THREE_MODES), out_root=tmp_path_factory.mktemp("x"), serial=True)
    evaluate_run(run_dir, serial=True)
    return run_dir


def test_exchange_log_and_adoption(exchange_run: Path):
    cfg = small_config(search=THREE_MODES)
    log = read_jsonl(exchange_run / "exchange.jsonl")
    assert {e["mode"] for e in log} == {"exchange"}
    plan = cfg.budget_plan("exchange")
    assert len(log) == cfg.replicates * plan.rounds * plan.n_workers
    assert sum(e["adopted"] is not None for e in log) > 0
    for e in log:
        assert all(not cid.startswith(f"r{e['replicate']:02d}-exchange-w{e['worker_id']:02d}") for cid in e["delivered"])


def test_exchange_info_sources_only_own_or_delivered(exchange_run: Path):
    delivered = defaultdict(list)  # (replicate, worker) -> [(round, cid)]
    for e in read_jsonl(exchange_run / "exchange.jsonl"):
        delivered[(e["replicate"], e["worker_id"])].extend((e["round"], cid) for cid in e["delivered"])
    candidates = read_jsonl(exchange_run / "candidates.jsonl")
    by_id = {c["candidate_id"]: c for c in candidates}
    foreign_parents = 0
    for c in candidates:
        for src in c["info_sources"]:
            other = by_id[src]
            assert (other["replicate"], other["mode"]) == (c["replicate"], c["mode"])
            if other["worker_id"] != c["worker_id"]:
                assert c["mode"] == "exchange"
                got = {cid for rnd, cid in delivered[(c["replicate"], c["worker_id"])] if rnd < c["round"]}
                assert src in got
        if c["parent_id"] and by_id[c["parent_id"]]["worker_id"] != c["worker_id"]:
            foreign_parents += 1
            assert c["parent_id"] in c["info_sources"]
    assert foreign_parents > 0


def test_exchange_budget_matches_independent(exchange_run: Path):
    budget = json.loads((exchange_run / "budget.json").read_text())
    train = {(e["replicate"], e["group"]): e for e in budget["entries"] if e["phase"] == "train"}
    for (rep, group), e in train.items():
        assert e["used"] == e["planned"] == train[(rep, "independent")]["planned"]
    report = (exchange_run / "report.md").read_text()
    assert "exchange − independent" in report and "exchange − single" in report
    assert "## 9. 信息交换" in report


def test_exchange_serial_equals_parallel(exchange_run: Path, tmp_path: Path):
    parallel = run_experiment(small_config(search=THREE_MODES), out_root=tmp_path, serial=False, max_workers=2)
    for name in ("candidates.jsonl", "exchange.jsonl", "anytime.jsonl"):
        assert (parallel / name).read_text() == (exchange_run / name).read_text(), name
    for p in (parallel / "selected").iterdir():
        assert p.read_text() == (exchange_run / "selected" / p.name).read_text()


def test_random_proposer_exchange_is_inert(tmp_path: Path):
    cfg = small_config(search={**THREE_MODES, "proposer": "random"}, experiment={"replicates": 1})
    run_dir = run_experiment(cfg, out_root=tmp_path, serial=True)
    log = read_jsonl(run_dir / "exchange.jsonl")
    assert log and all(e["adopted"] is None for e in log)
    assert any(e["delivered"] for e in log)  # cards were delivered but not used
    assert all(c["info_sources"] == [] for c in read_jsonl(run_dir / "candidates.jsonl"))
