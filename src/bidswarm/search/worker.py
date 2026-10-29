"""A searcher: one proposer, its own RNG, and its own candidate history.

A worker only ever sees its own results, plus - in ``cooperative`` mode - the
evidence cards migrated to it from other workers of the same pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..kernel import get_template, spec_from_normalized, spec_hash
from .exchange import EvidenceCard
from .proposers import Observation, Proposer


class IsolationError(RuntimeError):
    """Raised when a worker is handed results or evidence it must not see."""


class SearchWorker:
    """Turns proposals into candidate records and feeds back only its own results."""

    def __init__(
        self,
        replicate: int,
        master_seed: int,
        mode: str,
        worker_id: int,
        policy_name: str,
        proposer: Proposer,
        rng: np.random.Generator,
        accepts_evidence: bool = False,
    ) -> None:
        if proposer.dim != len(get_template(policy_name).PARAM_DEFS):
            raise ValueError("proposer dimension does not match the policy template")
        self.replicate = replicate
        self.master_seed = master_seed
        self.mode = mode
        self.worker_id = worker_id
        self.policy_name = policy_name
        self.proposer = proposer
        self.rng = rng
        self.accepts_evidence = accepts_evidence
        self._seq = 0
        self._pending: dict[str, dict[str, Any]] = {}
        self.own_ids: list[str] = []
        self.received_ids: list[str] = []
        self.best_fitness: float | None = None
        self.best_candidate_id: str | None = None

    @property
    def pipeline_prefix(self) -> str:
        return f"r{self.replicate:02d}-{self.mode}-"

    @property
    def prefix(self) -> str:
        return f"{self.pipeline_prefix}w{self.worker_id:02d}"

    def propose(self, round_index: int, k: int) -> list[dict[str, Any]]:
        """Ask the proposer for k candidates based on this worker's own history."""
        if self._pending:
            raise RuntimeError("previous round not observed yet")
        records = []
        for proposal in self.proposer.propose(k, self.rng):
            unknown = set(proposal.info_sources) - set(self.own_ids) - set(self.received_ids)
            if unknown:
                raise IsolationError(f"{self.prefix} proposal read foreign candidates: {sorted(unknown)}")
            spec = spec_from_normalized(self.policy_name, proposal.normalized)
            candidate_id = f"{self.prefix}-c{self._seq:04d}"
            self._seq += 1
            record = {
                "candidate_id": candidate_id,
                "replicate": self.replicate,
                "master_seed": self.master_seed,
                "mode": self.mode,
                "worker_id": self.worker_id,
                "round": round_index,
                "parent_id": proposal.parent_id,
                "operator": proposal.operator,
                "policy": spec.name,
                "params": spec.to_dict()["params"],
                "normalized": list(proposal.normalized),
                "sigma": proposal.sigma,
                "info_sources": list(proposal.info_sources),
                "spec_hash": spec_hash(spec),
            }
            self._pending[candidate_id] = record
            records.append(record)
        return records

    def observe(self, results: Sequence[dict[str, Any]]) -> None:
        """Receive fitness for exactly the candidates proposed this round."""
        ids = [r["candidate_id"] for r in results]
        if sorted(ids) != sorted(self._pending):
            raise IsolationError(f"{self.prefix} expected results for {sorted(self._pending)}, got {sorted(ids)}")
        observations = []
        for r in results:
            rec = self._pending[r["candidate_id"]]
            observations.append(Observation(rec["candidate_id"], tuple(rec["normalized"]), r["fitness"]))
            fit = r["fitness"]
            if fit is not None and (self.best_fitness is None or fit > self.best_fitness):
                self.best_fitness = fit
                self.best_candidate_id = rec["candidate_id"]
        self.proposer.observe(observations)
        self.own_ids.extend(ids)
        self._pending = {}

    def receive_evidence(self, cards: Sequence[EvidenceCard]) -> None:
        """Exchange hook: only cooperative workers accept cards, and only from their own pipeline."""
        if not cards:
            return
        if not self.accepts_evidence:
            raise IsolationError(f"{self.prefix} received {len(cards)} evidence cards in {self.mode} mode")
        if self._pending:
            raise RuntimeError("evidence arrived before this round was observed")
        for card in cards:
            if not card.candidate_id.startswith(self.pipeline_prefix) or card.source_worker == self.worker_id:
                raise IsolationError(f"{self.prefix} received a card from outside its pipeline: {card.candidate_id}")
            if card.candidate_id in self.received_ids:
                raise IsolationError(f"{self.prefix} received {card.candidate_id} twice")
        self.received_ids.extend(card.candidate_id for card in cards)
        self.proposer.receive(
            [Observation(card.candidate_id, tuple(card.normalized), card.fitness) for card in cards]
        )
