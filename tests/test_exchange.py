"""Ring migration for cooperative mode, and the isolation checks around it."""

from __future__ import annotations

import numpy as np
import pytest

from bidswarm.config import EsConfig
from bidswarm.search.exchange import EvidenceCard, NoExchange, RingExchange
from bidswarm.search.proposers import EsProposer, Observation, RandomProposer
from bidswarm.search.worker import IsolationError, SearchWorker


def card(worker: int, seq: int, fitness: float | None, round_: int = 0, mode: str = "cooperative") -> EvidenceCard:
    return EvidenceCard(f"r00-{mode}-w{worker:02d}-c{seq:04d}", worker, round_, f"h{worker}{seq}", fitness, (0.5, 0.5))


def test_ring_sends_each_best_so_far_to_the_next_worker():
    ring = RingExchange(interval=1, rounds=3)
    out = ring.exchange(0, {0: [card(0, 0, 1.0), card(0, 1, 3.0)], 1: [card(1, 0, 2.0)], 2: [card(2, 0, None)]})
    assert [c.candidate_id for c in out[1]] == ["r00-cooperative-w00-c0001"]
    assert [c.candidate_id for c in out[2]] == ["r00-cooperative-w01-c0000"]
    assert out[0] == []  # worker 2 has no usable candidate yet, so it sends nothing


def test_ring_respects_interval_skips_last_round_and_never_resends():
    ring = RingExchange(interval=2, rounds=5)
    assert ring.exchange(0, {0: [card(0, 0, 1.0)], 1: [card(1, 0, 2.0)]}) == {0: [], 1: []}
    first = ring.exchange(1, {0: [card(0, 1, 0.5)], 1: [card(1, 1, 0.1)]})
    assert [c.candidate_id for c in first[1]] == ["r00-cooperative-w00-c0000"]  # best so far, not this round's
    assert ring.exchange(2, {0: [], 1: []}) == {0: [], 1: []}
    again = ring.exchange(3, {0: [card(0, 2, 0.9)], 1: [card(1, 2, 5.0)]})
    assert again[1] == []  # w0's best is unchanged and was already delivered
    assert [c.candidate_id for c in again[0]] == ["r00-cooperative-w01-c0002"]
    assert ring.exchange(4, {0: [card(0, 3, 9.0)], 1: []}) == {0: [], 1: []}  # last round


def test_no_exchange_delivers_nothing():
    assert NoExchange().exchange(0, {0: [card(0, 0, 1.0)], 1: []}) == {0: [], 1: []}


def test_es_adopts_only_a_better_foreign_parent_and_records_it():
    rng = np.random.default_rng(0)
    es = EsProposer(dim=2, es=EsConfig())
    es.observe([Observation(f"own{i}", p.normalized, f) for i, (p, f) in enumerate(zip(es.propose(2, rng), [1.0, 2.0]))])
    es.receive([Observation("foreign-worse", (0.1, 0.1), 1.5)])
    assert es.parent.candidate_id == "own1"
    es.receive([Observation("foreign-better", (0.9, 0.9), 4.0)])
    assert es.parent.candidate_id == "foreign-better" and es.sigma == pytest.approx(0.2)
    children = es.propose(2, rng)
    assert all(p.parent_id == "foreign-better" for p in children)
    assert all({"foreign-worse", "foreign-better"} <= set(p.info_sources) for p in children)
    with pytest.raises(RuntimeError):
        es.receive([Observation("late", (0.5, 0.5), 9.0)])


def test_random_proposer_ignores_received_candidates():
    rp = RandomProposer(dim=2)
    rp.receive([Observation("foreign", (0.5, 0.5), 1.0)])
    assert all(p.info_sources == () for p in rp.propose(2, np.random.default_rng(0)))


def _worker(mode: str, worker_id: int = 1) -> SearchWorker:
    return SearchWorker(
        replicate=0,
        master_seed=1,
        mode=mode,
        worker_id=worker_id,
        policy_name="fixed_rate",
        proposer=EsProposer(dim=1),
        rng=np.random.default_rng(0),
        accepts_evidence=mode == "cooperative",
    )


def test_only_cooperative_workers_accept_cards_from_their_own_pipeline():
    with pytest.raises(IsolationError, match="independent mode"):
        _worker("independent").receive_evidence([card(0, 0, 1.0, mode="independent")])
    worker = _worker("cooperative")
    for bad in (
        card(0, 0, 1.0, mode="independent"),  # another mode
        EvidenceCard("r01-cooperative-w00-c0000", 0, 0, "h", 1.0, (0.5,)),  # another replicate
        card(1, 0, 1.0),  # its own candidate
    ):
        with pytest.raises(IsolationError):
            worker.receive_evidence([bad])
    good = EvidenceCard("r00-cooperative-w00-c0000", 0, 0, "h", 1.0, (0.5,))
    worker.receive_evidence([good])
    assert worker.received_ids == [good.candidate_id]
    with pytest.raises(IsolationError, match="twice"):
        worker.receive_evidence([good])
    # The adopted foreign parent may appear in info_sources without tripping the isolation check.
    records = worker.propose(0, 2)
    assert all(r["parent_id"] == good.candidate_id for r in records)
