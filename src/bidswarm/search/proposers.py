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
* ``CmaesProposer`` - (mu/mu_w, lambda)-CMA-ES with the default parameter
  settings of Hansen's tutorial (arXiv:1604.00772). One generation of lambda
  samples is drawn at once and handed out k per round, so lambda must be a
  multiple of k; the distribution is updated when the whole generation has
  been observed. Samples are clipped to the box and the clipped points are
  used in the update. A received candidate that beats this searcher's
  best-so-far becomes the mean of the *next* generation (sigma and the
  covariance are kept), so a generation is never mixed across two means.
* ``RandomProposer`` - independent uniform sampling, a control only. Under
  random sampling ``single`` and ``independent`` are equal in distribution, so
  comparing them is only a correctness check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import math

import numpy as np

from ..config import CmaesConfig, EsConfig
from ..types import OP_CMAES_SAMPLE, OP_GAUSSIAN_MUTATION, OP_INIT_RANDOM, OP_RANDOM


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


class CmaesProposer(Proposer):
    name = "cmaes"

    def __init__(self, dim: int, cmaes: CmaesConfig | None = None) -> None:
        super().__init__(dim)
        self.cfg = cmaes or CmaesConfig()
        self.population: int | None = self.cfg.population or None  # resolved at the first propose()
        self.sigma = self.cfg.sigma_init
        self.mean: np.ndarray | None = None  # drawn uniformly at the first propose()
        self.C = np.eye(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.inv_sqrt_C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.generation = 0
        self.best: Observation | None = None  # this searcher's best-so-far, own or received
        self._injected: Observation | None = None
        self._batch: list[tuple[float, ...]] = []  # samples of the current generation
        self._handed_out = 0
        self._observed: list[Observation] = []
        self._pending = False

    @staticmethod
    def default_population(dim: int) -> int:
        return 4 + int(math.floor(3 * math.log(dim)))

    def _resolve_population(self, k: int) -> int:
        if self.population is None:
            base = self.default_population(self.dim)
            self.population = max(2, k * math.ceil(base / k))
        if self.population % k:
            raise ValueError(f"CMA-ES population {self.population} is not a multiple of k={k}")
        return self.population

    def _new_generation(self, lam: int, rng: np.random.Generator) -> None:
        if self.mean is None:
            self.mean = rng.random(self.dim)
        if self._injected is not None:
            self.mean = np.asarray(self._injected.normalized, dtype=float)
            self._injected = None
        z = rng.standard_normal((lam, self.dim))
        x = self.mean + self.sigma * (z * self.D) @ self.B.T
        self._batch = [tuple(float(v) for v in row) for row in np.clip(x, 0.0, 1.0)]
        self._handed_out = 0
        self._observed = []

    def propose(self, k: int, rng: np.random.Generator) -> list[Proposal]:
        if self._pending:
            raise RuntimeError("propose() called twice without observe()")
        lam = self._resolve_population(k)
        if self._handed_out == len(self._batch):
            self._new_generation(lam, rng)
        samples = self._batch[self._handed_out : self._handed_out + k]
        self._handed_out += k
        self._pending = True
        sources = tuple(self._seen)
        return [Proposal(u, OP_CMAES_SAMPLE, None, self.sigma, sources) for u in samples]

    def observe(self, own_results: Sequence[Observation]) -> None:
        if not self._pending:
            raise RuntimeError("observe() called without a pending propose()")
        self._pending = False
        self._seen.extend(o.candidate_id for o in own_results)
        self._observed.extend(own_results)
        self._track_best(own_results)
        if len(self._observed) == len(self._batch):
            self._update(self._observed)

    def _track_best(self, observations: Sequence[Observation]) -> None:
        best = _best(observations)
        if best is not None and (self.best is None or best.fitness > self.best.fitness):
            self.best = best

    def _update(self, observations: Sequence[Observation]) -> None:
        n = self.dim
        lam = len(observations)
        mu = lam // 2
        weights = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        # Failed candidates (fitness None) rank last; with fewer than mu valid samples only
        # the valid ones are recombined, and with none the distribution is left unchanged.
        valid = sorted((o for o in observations if o.fitness is not None), key=lambda o: -o.fitness)
        selected = valid[:mu]
        self.generation += 1
        if not selected:
            return
        w = weights[: len(selected)] / weights[: len(selected)].sum()
        mueff = 1.0 / float(np.sum(w**2))
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        assert self.mean is not None
        old = self.mean
        xs = np.array([o.normalized for o in selected], dtype=float)
        self.mean = w @ xs
        y_w = (self.mean - old) / self.sigma
        self.ps = (1 - cs) * self.ps + math.sqrt(cs * (2 - cs) * mueff) * (self.inv_sqrt_C @ y_w)
        ps_norm = float(np.linalg.norm(self.ps))
        h_sig = ps_norm / math.sqrt(1 - (1 - cs) ** (2 * self.generation)) / chi_n < 1.4 + 2 / (n + 1)
        self.pc = (1 - cc) * self.pc + h_sig * math.sqrt(cc * (2 - cc) * mueff) * y_w
        steps = (xs - old) / self.sigma
        rank_mu = (steps.T * w) @ steps
        self.C = (
            (1 - c1 - cmu) * self.C
            + c1 * (np.outer(self.pc, self.pc) + (1 - h_sig) * cc * (2 - cc) * self.C)
            + cmu * rank_mu
        )
        self.sigma *= math.exp((cs / damps) * (ps_norm / chi_n - 1))
        self.sigma = min(max(self.sigma, self.cfg.sigma_min), self.cfg.sigma_max)
        self.C = (self.C + self.C.T) / 2
        eigenvalues, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(eigenvalues, 1e-20))
        self.inv_sqrt_C = (self.B / self.D) @ self.B.T

    def receive(self, foreign: Sequence[Observation]) -> None:
        if self._pending:
            raise RuntimeError("receive() called between propose() and observe()")
        if not foreign:
            return
        self._seen.extend(o.candidate_id for o in foreign)
        best = _best(foreign)
        if best is not None and (self.best is None or best.fitness > self.best.fitness):
            self.best = best
            self._injected = best  # takes effect when the next generation is drawn


def make_proposer(name: str, dim: int, es: EsConfig, cmaes: CmaesConfig | None = None) -> Proposer:
    if name == "es":
        return EsProposer(dim, es)
    if name == "cmaes":
        return CmaesProposer(dim, cmaes)
    if name == "random":
        return RandomProposer(dim)
    raise ValueError(f"unknown proposer {name!r}")
