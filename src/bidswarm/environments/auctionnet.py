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
    "no shim exists yet that subclasses AuctionNet's BaseBiddingStrategy and loads a PolicySpec",
    "interface facts were checked against the upstream main branch, not a pinned commit SHA; "
    "they must be re-verified against the exact commit before adapting",
    "open questions are unresolved: whether tickValue records realized or expected conversions, "
    "time steps per delivery period, whether online simulation needs extra weights/data, "
    "seeding and reset behaviour, auction mechanism details",
    "offline evaluation needs per-opportunity raw data (full pre-generated set is about 80GB); "
    "this project deliberately does not download AuctionNet data or weights",
    "no subprocess smoke test against a real AuctionNet checkout (Python 3.9.12) has been run",
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
