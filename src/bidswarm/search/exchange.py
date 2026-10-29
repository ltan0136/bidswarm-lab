"""Information exchange between searchers of one (replicate, mode) pipeline.

The coordinator calls ``ExchangePolicy.exchange`` at every round boundary with
the evidence cards each worker produced that round, and hands each worker the
cards addressed to it. ``NoExchange`` delivers nothing (``single`` and
``independent``). ``RingExchange`` is the ``cooperative`` mode: every
``interval`` rounds each worker sends its best-so-far candidate to the next
worker on a ring (0 -> 1 -> ... -> N-1 -> 0). It is the same elite migration
as swarm-scheduler-lab's cooperative mode. Migration costs no episodes, so all
three modes spend the same training budget.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceCard:
    """One evaluated candidate that can be shared with another searcher."""

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


class RingExchange(ExchangePolicy):
    """Synchronous ring migration of each worker's best-so-far card.

    Migration happens after rounds ``interval - 1, 2 * interval - 1, ...`` but
    never after the last round, where it could not influence any proposal.
    Workers whose candidates have all failed so far send nothing, and a card
    already delivered to the same receiver is not sent again.
    """

    name = "ring"

    def __init__(self, interval: int, rounds: int) -> None:
        if interval < 1:
            raise ValueError("exchange interval must be >= 1")
        self.interval = interval
        self.rounds = rounds
        self._best: dict[int, EvidenceCard] = {}
        self._sent: set[tuple[int, str]] = set()

    def exchange(
        self, round_index: int, outgoing: Mapping[int, Sequence[EvidenceCard]]
    ) -> dict[int, list[EvidenceCard]]:
        for worker_id, cards in outgoing.items():
            for card in cards:  # ties keep the earlier card, like the proposers
                best = self._best.get(worker_id)
                if card.fitness is not None and (best is None or card.fitness > best.fitness):
                    self._best[worker_id] = card
        delivered: dict[int, list[EvidenceCard]] = {worker_id: [] for worker_id in outgoing}
        is_last = round_index >= self.rounds - 1
        if is_last or (round_index + 1) % self.interval != 0:
            return delivered
        ring = sorted(outgoing)
        for position, worker_id in enumerate(ring):
            card = self._best.get(worker_id)
            receiver = ring[(position + 1) % len(ring)]
            if card is not None and (receiver, card.candidate_id) not in self._sent:
                self._sent.add((receiver, card.candidate_id))
                delivered[receiver].append(card)
        return delivered
