"""AuctionNet backend placeholder.

The adapter is NOT implemented. Constructing the backend always raises
``BackendNotIntegratedError`` so that ``backend = "auctionnet"`` fails loudly
instead of silently running the toy environment. See
docs/AUCTIONNET_INTEGRATION.md for verified facts, open questions and the plan.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..types import BACKEND_AUCTIONNET, ConfigError
from .base import BackendNotIntegratedError, Environment

INTEGRATION_STATUS = "not_integrated"

BLOCKERS = (
    "no shim/backend exists yet; step 1 (read-only investigation at upstream commit "
    "c20de631a40e24669d1d6a8e2acefd8f492f172d) is done, see docs/AUCTIONNET_INTEGRATION.md",
    "upstream run_test silently falls back to its PID agent if the player import fails; the backend "
    "must inject the player directly and verify spec_hash and call counts",
    "the kernel accumulates spend from historyAuctionResult's cost column, but AuctionNet charges "
    "only exposed impressions (about 30% overstatement measured); fix the kernel and decide whether "
    "the toy backend should also charge on exposure",
    "run_test always starts at episode 0; the backend needs its own tick loop to place train / "
    "validation / test scenarios on disjoint episode indices, checked against run_test",
    "one delivery period takes about 45-50 s and 2.7 GB per process (about 280 periods per hour on "
    "4 cores); the toy-sized protocol (~75,000 periods) is infeasible and must be scaled down",
    "no subprocess smoke test of a kernel policy through a finished backend has been run",
)

_ALLOWED_KEYS = frozenset(
    {"backend", "repo_path", "commit_sha", "evaluation_path", "python_executable", "num_time_steps"}
)


def validate_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Shape-check the placeholder config; does not make the backend usable."""
    unknown = sorted(set(cfg) - _ALLOWED_KEYS)
    if unknown:
        raise ConfigError(f"[environment] unknown keys for backend 'auctionnet': {unknown}")
    path = cfg.get("evaluation_path", "offline")
    if path not in ("offline", "online"):
        raise ConfigError("[environment] evaluation_path must be 'offline' or 'online'")
    return dict(cfg)


class AuctionNetEnvironment(Environment):
    backend = BACKEND_AUCTIONNET
    integration_status = INTEGRATION_STATUS

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        raise BackendNotIntegratedError(BACKEND_AUCTIONNET, INTEGRATION_STATUS, BLOCKERS)

    def make_scenario(self, scenario_id: str, split: str, index: int, seed: int) -> dict[str, Any]:
        raise BackendNotIntegratedError(BACKEND_AUCTIONNET, INTEGRATION_STATUS, BLOCKERS)

    def run_episode(self, scenario: dict[str, Any], policy: Any) -> dict[str, Any]:
        raise BackendNotIntegratedError(BACKEND_AUCTIONNET, INTEGRATION_STATUS, BLOCKERS)
