"""Stateful candidate proposers operating in the normalized space [0, 1]^d.

Interface: ``propose(k, rng)`` returns k proposals, then ``observe(results)``
receives the fitness of exactly those proposals (the proposer's own results).
Between rounds, ``receive(foreign)`` may hand over candidates another searcher
evaluated (``cooperative`` mode only).

* ``EsProposer`` - (1+lambda)-ES with lambda = k and the 1/5th-style step
  rule: sigma *= success_factor after a round with an improving child,
  otherwise sigma *= failure_factor, clamped to [sigma_min, sigma_max].
  A received candidate that beats the current parent replaces it; sigma is
  kept, since it reflects this searcher's own success history.
* ``RandomProposer`` - independent uniform sampling, a control only. Under
  random sampling ``single`` and ``independent`` are equal in distribution, so
  comparing them is only a correctness check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..config import EsConfig
from ..types import OP_GAUSSIAN_MUTATION, OP_INIT_RANDOM, OP_RANDOM


@dataclass(frozen=True)
class Proposal:
    normalized: tuple[float, ...]
    operator: str
    parent_id: str | None
    sigma: float | None
    info_sources: tuple[str, ...]  # candidate IDs whose results shaped this proposal


@dataclass(frozen=True)
class Observation:
    candidate_id: str
    normalized: tuple[float, ...]
    fitness: float | None  # None = failed candidate, never selected


def _best(observations: Sequence[Observation]) -> Observation | None:
    """Highest fitness; ties go to the earliest observation."""
    best = None
    for obs in observations:
        if obs.fitness is not None and (best is None or obs.fitness > best.fitness):
            best = obs
    return best


class Proposer(ABC):
    name = "base"

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self._seen: list[str] = []

    @abstractmethod
    def propose(self, k: int, rng: np.random.Generator) -> list[Proposal]:
        ...

    @abstractmethod
    def observe(self, own_results: Sequence[Observation]) -> None:
        ...

    def receive(self, foreign: Sequence[Observation]) -> None:
        """Candidates evaluated by other searchers. The default ignores them."""

    def _uniform(self, k: int, rng: np.random.Generator) -> list[tuple[float, ...]]:
        return [tuple(float(x) for x in row) for row in rng.random((k, self.dim))]


class RandomProposer(Proposer):
    name = "random"

    def propose(self, k: int, rng: np.random.Generator) -> list[Proposal]:
        return [Proposal(u, OP_RANDOM, None, None, ()) for u in self._uniform(k, rng)]

    def observe(self, own_results: Sequence[Observation]) -> None:
        self._seen.extend(o.candidate_id for o in own_results)


class EsProposer(Proposer):
    name = "es"

    def __init__(self, dim: int, es: EsConfig | None = None) -> None:
        super().__init__(dim)
        self.es = es or EsConfig()
        self.sigma = self.es.sigma_init
        self.parent: Observation | None = None
        self._pending: str | None = None

    def propose(self, k: int, rng: np.random.Generator) -> list[Proposal]:
        if self._pending is not None:
            raise RuntimeError("propose() called twice without observe()")
        sources = tuple(self._seen)
        if self.parent is None:
            # First round, or every earlier candidate failed: (re)initialise.
            self._pending = OP_INIT_RANDOM
            return [Proposal(u, OP_INIT_RANDOM, None, None, sources) for u in self._uniform(k, rng)]
        self._pending = OP_GAUSSIAN_MUTATION
        parent = np.asarray(self.parent.normalized, dtype=float)
        noise = rng.standard_normal((k, self.dim))
        children = np.clip(parent + self.sigma * noise, 0.0, 1.0)
        return [
            Proposal(
                tuple(float(x) for x in row),
                OP_GAUSSIAN_MUTATION,
                self.parent.candidate_id,
                self.sigma,
                sources,
            )
            for row in children
        ]

    def observe(self, own_results: Sequence[Observation]) -> None:
        if self._pending is None:
            raise RuntimeError("observe() called without a pending propose()")
        self._seen.extend(o.candidate_id for o in own_results)
        best = _best(own_results)
        if self._pending == OP_INIT_RANDOM:
            self.parent = best  # may stay None if all failed
        else:
            assert self.parent is not None and self.parent.fitness is not None
            if best is not None and best.fitness > self.parent.fitness:
                self.parent = best
                self.sigma *= self.es.success_factor
            else:
                self.sigma *= self.es.failure_factor
            self.sigma = min(max(self.sigma, self.es.sigma_min), self.es.sigma_max)
        self._pending = None

    def receive(self, foreign: Sequence[Observation]) -> None:
        if self._pending is not None:
            raise RuntimeError("receive() called between propose() and observe()")
        if not foreign:
            return
        # Every received card was compared against the parent, so all of them shaped
        # the next proposals and belong in their info_sources.
        self._seen.extend(o.candidate_id for o in foreign)
        best = _best(foreign)
        if best is not None and (self.parent is None or best.fitness > self.parent.fitness):
            self.parent = best


def make_proposer(name: str, dim: int, es: EsConfig) -> Proposer:
    if name == "es":
        return EsProposer(dim, es)
    if name == "random":
        return RandomProposer(dim)
    raise ValueError(f"unknown proposer {name!r}")
