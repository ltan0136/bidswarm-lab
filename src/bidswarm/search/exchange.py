"""Information exchange between searchers, applied at round boundaries.

The coordinator calls ``ExchangePolicy.exchange`` once per round with the
evidence cards each worker produced that round, and hands every worker the
cards addressed to it. Exchange costs no episodes: it only moves information
that was already paid for.

* ``NoExchange`` delivers nothing (``single`` and ``independent`` modes).
* ``ShareBest`` (``exchange`` mode) is elite migration: each worker receives
  its neighbours' current best-so-far card. Topology ``all`` connects every
  pair of workers; ``ring`` delivers worker w-1's best to worker w.

What a worker does with a card is the proposer's decision (``Proposer.receive``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

TOPOLOGIES = ("all", "ring")
EXCHANGE_POLICIES = ("share_best",)


@dataclass(frozen=True)
class EvidenceCard:
    """Minimal record of one evaluated candidate that can be shared."""

    candidate_id: str
    source_worker: int
    round: int
    spec_hash: str
    fitness: float | None
    normalized: tuple[float, ...]


class ExchangePolicy(ABC):
    name = "base"

    @abstractmethod
    def exchange(
        self, round_index: int, outgoing: Mapping[int, Sequence[EvidenceCard]]
    ) -> dict[int, list[EvidenceCard]]:
        """Map worker_id -> cards produced this round, to worker_id -> cards delivered."""


class NoExchange(ExchangePolicy):
    name = "none"

    def exchange(
        self, round_index: int, outgoing: Mapping[int, Sequence[EvidenceCard]]
    ) -> dict[int, list[EvidenceCard]]:
        return {worker_id: [] for worker_id in outgoing}


class ShareBest(ExchangePolicy):
    """Deliver each neighbour's best-so-far card (null fitness is never shared)."""

    name = "share_best"

    def __init__(self, n_workers: int, topology: str = "all") -> None:
        if topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology {topology!r}; known: {list(TOPOLOGIES)}")
        if n_workers < 2:
            raise ValueError("ShareBest needs at least two workers")
        self.n_workers = n_workers
        self.topology = topology
        self._best: dict[int, EvidenceCard] = {}

    def neighbours(self, worker_id: int) -> list[int]:
        if self.topology == "ring":
            return [(worker_id - 1) % self.n_workers]
        return [w for w in range(self.n_workers) if w != worker_id]

    def exchange(
        self, round_index: int, outgoing: Mapping[int, Sequence[EvidenceCard]]
    ) -> dict[int, list[EvidenceCard]]:
        for worker_id, cards in outgoing.items():
            for card in cards:
                if card.source_worker != worker_id:
                    raise ValueError(f"card {card.candidate_id} does not come from worker {worker_id}")
                best = self._best.get(worker_id)
                if card.fitness is not None and (best is None or card.fitness > best.fitness):
                    self._best[worker_id] = card
        return {
            worker_id: [self._best[n] for n in self.neighbours(worker_id) if n in self._best]
            for worker_id in outgoing
        }


def make_exchange_policy(policy: str | None, n_workers: int, topology: str = "all") -> ExchangePolicy:
    if policy is None or policy == "none":
        return NoExchange()
    if policy == "share_best":
        return ShareBest(n_workers, topology)
    raise ValueError(f"unknown exchange policy {policy!r}; known: {list(EXCHANGE_POLICIES)}")
