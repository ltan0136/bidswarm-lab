"""Information-exchange hook between searchers (interface only in v1).

The coordinator calls ``ExchangePolicy.exchange`` at every round boundary with
the evidence cards each worker produced that round. ``NoExchange`` delivers
nothing, which is what the ``independent`` mode uses. Exchange semantics
(what to share, with whom, how a proposer uses it) are future work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceCard:
    """Minimal record of one evaluated candidate that could be shared."""

    candidate_id: str
    source_worker: int
    round: int
    spec_hash: str
    fitness: float | None


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
