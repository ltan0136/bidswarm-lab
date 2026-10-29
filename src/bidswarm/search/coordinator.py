"""Synchronous round scheduler (plain program, not an LLM).

Each round, for every active (replicate, mode) pipeline:

1. every worker proposes k candidates from its own history;
2. episode tasks are generated in the fixed order
   (replicate, mode, worker, candidate, scenario);
3. the ledger is charged before dispatch, the batch runs on the executor and
   results come back sorted by task index;
4. each worker observes only its own candidates' results; the exchange policy
   is called at the round boundary (deliveries and adoptions are logged to
   exchange.jsonl for exchanging modes); anytime best-so-far is recorded.

All (replicate, mode) pipelines share one batch per round to keep the process
pool busy; pipelines with fewer rounds simply drop out of later batches.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..artifacts.store import ANYTIME, CANDIDATES, EPISODES, EXCHANGE_LOG, RunStore
from ..config import ExperimentConfig
from ..evaluation.budget import BudgetLedger, TrainingBudgetPlan
from ..evaluation.metrics import mean_or_none
from ..evaluation.runner import EpisodeExecutor
from ..evaluation.selection import candidate_fitness
from ..kernel import get_template
from ..types import MODE_EXCHANGE, PHASE_TRAIN, STATUS_FAILED, STATUS_TIMEOUT, USABLE_STATUSES, EpisodeTask
from .exchange import EvidenceCard, ExchangePolicy, NoExchange, make_exchange_policy
from .proposers import make_proposer
from .worker import SearchWorker


class TaskCounter:
    """Global, monotonically increasing task index (defines result order)."""

    def __init__(self, start: int = 0) -> None:
        self.value = start

    def next(self) -> int:
        v = self.value
        self.value += 1
        return v


def proposer_rng(master_seed: int, mode: str, worker_id: int) -> np.random.Generator:
    digest = hashlib.sha256(f"bidswarm-proposer:{master_seed}:{mode}:{worker_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


@dataclass
class SearchPipeline:
    """Search state of one (replicate, mode)."""

    replicate: int
    master_seed: int
    mode: str
    plan: TrainingBudgetPlan
    workers: list[SearchWorker]
    train_scenarios: tuple[dict[str, Any], ...]
    exchange: ExchangePolicy
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def best_so_far(self) -> tuple[float | None, str | None]:
        best, best_id = None, None
        for w in self.workers:
            if w.best_fitness is not None and (best is None or w.best_fitness > best):
                best, best_id = w.best_fitness, w.best_candidate_id
        return best, best_id


def build_pipeline(
    config: ExperimentConfig,
    replicate: int,
    master_seed: int,
    mode: str,
    train_scenarios: Sequence[dict[str, Any]],
) -> SearchPipeline:
    """Same policy space, proposer and hyper-parameters for every mode.

    Modes differ only in the number of workers (single: 1) and in the exchange
    policy (exchange: the configured policy; single/independent: NoExchange).
    """
    plan = config.budget_plan(mode)
    dim = len(get_template(config.search.policy).PARAM_DEFS)
    workers = [
        SearchWorker(
            replicate=replicate,
            master_seed=master_seed,
            mode=mode,
            worker_id=w,
            policy_name=config.search.policy,
            proposer=make_proposer(config.search.proposer, dim, config.search.es),
            rng=proposer_rng(master_seed, mode, w),
        )
        for w in range(plan.n_workers)
    ]
    if mode == MODE_EXCHANGE:
        ex = config.search.exchange
        exchange = make_exchange_policy(ex.policy, plan.n_workers, ex.topology)
    else:
        exchange = NoExchange()
    return SearchPipeline(replicate, master_seed, mode, plan, workers, tuple(train_scenarios), exchange)


class Coordinator:
    def __init__(
        self,
        config: ExperimentConfig,
        pipelines: Sequence[SearchPipeline],
        executor: EpisodeExecutor,
        ledger: BudgetLedger,
        store: RunStore,
        counter: TaskCounter,
    ) -> None:
        self.config = config
        self.pipelines = list(pipelines)
        self.executor = executor
        self.ledger = ledger
        self.store = store
        self.counter = counter
        self.metric = config.search.fitness_metric

    def run(self) -> None:
        k = self.config.search.candidates_per_round
        max_rounds = max(p.plan.rounds for p in self.pipelines)
        for round_index in range(max_rounds):
            active = [p for p in self.pipelines if round_index < p.plan.rounds]

            # 1. proposals, each from the worker's own history only
            proposals = [[(w, w.propose(round_index, k)) for w in p.workers] for p in active]

            # 2. tasks in (replicate, mode, worker, candidate, scenario) order; charge before dispatch
            tasks: list[EpisodeTask] = []
            for p, per_worker in zip(active, proposals):
                first = len(tasks)
                for w, records in per_worker:
                    for rec in records:
                        tasks.extend(self._train_tasks(p, w, rec))
                self.ledger.charge(p.replicate, p.mode, PHASE_TRAIN, len(tasks) - first)

            # 3. execute; results sorted by task index
            results = self.executor.run(tasks)
            self.store.append_jsonl(EPISODES, results)
            by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for res in results:
                by_candidate[res["candidate_id"]].append(res)

            # 4. each worker observes only its own candidates; exchange at the boundary
            anytime: list[dict[str, Any]] = []
            exchange_log: list[dict[str, Any]] = []
            for p, per_worker in zip(active, proposals):
                outgoing: dict[int, list[EvidenceCard]] = {}
                for w, records in per_worker:
                    evaluated = []
                    for rec in records:
                        episodes = by_candidate[rec["candidate_id"]]
                        self.ledger.record_outcomes(
                            p.replicate, p.mode, PHASE_TRAIN, [e["status"] for e in episodes]
                        )
                        evaluated.append(self._evaluated_candidate(rec, episodes))
                    w.observe(evaluated)
                    p.candidates.extend(evaluated)
                    self.store.append_jsonl(CANDIDATES, evaluated)
                    outgoing[w.worker_id] = [
                        EvidenceCard(
                            c["candidate_id"],
                            w.worker_id,
                            round_index,
                            c["spec_hash"],
                            c["fitness"],
                            tuple(c["normalized"]),
                        )
                        for c in evaluated
                    ]
                inbound = p.exchange.exchange(round_index, outgoing)
                for w in p.workers:
                    cards = inbound.get(w.worker_id, [])
                    adopted = w.receive_evidence(cards)
                    if not isinstance(p.exchange, NoExchange):
                        exchange_log.append(
                            {
                                "replicate": p.replicate,
                                "master_seed": p.master_seed,
                                "mode": p.mode,
                                "round": round_index,
                                "policy": p.exchange.name,
                                "worker_id": w.worker_id,
                                "delivered": [c.candidate_id for c in cards],
                                "delivered_fitness": [c.fitness for c in cards],
                                "adopted": adopted,
                                "parent_after": getattr(getattr(w.proposer, "parent", None), "candidate_id", None),
                            }
                        )
                anytime.extend(self._anytime_records(p, round_index, k))
            self.store.append_jsonl(ANYTIME, anytime)
            self.store.append_jsonl(EXCHANGE_LOG, exchange_log)

    def _train_tasks(self, p: SearchPipeline, w: SearchWorker, rec: dict[str, Any]) -> list[EpisodeTask]:
        spec = {"name": rec["policy"], "params": rec["params"]}
        return [
            EpisodeTask(
                task_index=self.counter.next(),
                task_id=f"{PHASE_TRAIN}/{rec['candidate_id']}/{sc['scenario_id']}",
                phase=PHASE_TRAIN,
                replicate=p.replicate,
                group=p.mode,
                worker_id=w.worker_id,
                candidate_id=rec["candidate_id"],
                spec=spec,
                spec_hash=rec["spec_hash"],
                scenario=sc,
                env_config=self.config.environment,
                timeout_s=self.config.runtime.episode_timeout_s,
            )
            for sc in p.train_scenarios
        ]

    def _evaluated_candidate(self, rec: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
        fitness = candidate_fitness(episodes, self.metric)
        usable = all(e["status"] in USABLE_STATUSES for e in episodes)
        return {
            **rec,
            "fitness": fitness,
            "fitness_metric": self.metric,
            "status": "ok" if fitness is not None else STATUS_FAILED,
            "n_episodes": len(episodes),
            "n_failed": sum(e["status"] == STATUS_FAILED for e in episodes),
            "n_timeouts": sum(e["status"] == STATUS_TIMEOUT for e in episodes),
            "train_score_expected": mean_or_none(e["score_expected"] for e in episodes) if usable else None,
            "train_score_realized": mean_or_none(e["score_realized"] for e in episodes) if usable else None,
        }

    def _anytime_records(self, p: SearchPipeline, round_index: int, k: int) -> list[dict[str, Any]]:
        used = self.ledger.used(p.replicate, p.mode, PHASE_TRAIN)
        worker_used = (round_index + 1) * k * len(p.train_scenarios)
        base = {"replicate": p.replicate, "master_seed": p.master_seed, "mode": p.mode, "round": round_index}
        rows = [
            {
                **base,
                "scope": "worker",
                "worker_id": w.worker_id,
                "best_so_far": w.best_fitness,
                "best_candidate_id": w.best_candidate_id,
                "worker_train_episodes_used": worker_used,
                "train_episodes_used": used,
            }
            for w in p.workers
        ]
        best, best_id = p.best_so_far()
        rows.append(
            {
                **base,
                "scope": "mode",
                "worker_id": None,
                "best_so_far": best,
                "best_candidate_id": best_id,
                "worker_train_episodes_used": None,
                "train_episodes_used": used,
            }
        )
        return rows
