"""Shared constants, error types and plain-data task records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BACKEND_TOY = "toy"
BACKEND_AUCTIONNET = "auctionnet"

# Tags attached to every toy output so results cannot be mistaken for AuctionNet.
PURPOSE = "framework_smoke_test"
SCORE_FORMULA = "auctionnet_neurips_replica"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
# Episodes whose metrics are usable. A soft timeout only flags slowness.
USABLE_STATUSES = (STATUS_OK, STATUS_TIMEOUT)

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"
PHASE_TEST = "test"
PHASES = (PHASE_TRAIN, PHASE_VALIDATION, PHASE_TEST)
SPLITS = (PHASE_TRAIN, PHASE_VALIDATION, PHASE_TEST)

MODE_SINGLE = "single"
MODE_INDEPENDENT = "independent"
MODE_EXCHANGE = "exchange"
MODES = (MODE_SINGLE, MODE_INDEPENDENT, MODE_EXCHANGE)
# Paired comparisons reported as (a, b) -> a - b, when both modes are present.
MODE_COMPARISONS = (
    (MODE_INDEPENDENT, MODE_SINGLE),
    (MODE_EXCHANGE, MODE_INDEPENDENT),
    (MODE_EXCHANGE, MODE_SINGLE),
)

OP_INIT_RANDOM = "init_random"
OP_GAUSSIAN_MUTATION = "gaussian_mutation"
OP_RANDOM = "random"

FITNESS_METRICS = ("score_expected", "score_realized")


class ConfigError(ValueError):
    """Raised for invalid experiment configuration."""


class ArtifactError(RuntimeError):
    """Raised when run artifacts are missing or inconsistent."""


@dataclass(frozen=True)
class EpisodeTask:
    """One episode to execute: a policy spec on one scenario.

    Everything needed to run the episode is carried as plain data so the task
    can be pickled into a spawned worker process and executed by a pure
    function (``bidswarm.evaluation.runner.run_episode_task``).
    """

    task_index: int
    task_id: str
    phase: str
    replicate: int
    group: str  # search mode, or a test-time policy label
    worker_id: int | None
    candidate_id: str | None
    spec: dict[str, Any]
    spec_hash: str
    scenario: dict[str, Any]
    env_config: dict[str, Any]
    timeout_s: float
