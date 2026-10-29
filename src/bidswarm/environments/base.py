"""Environment interface shared by all backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..types import PURPOSE, SCORE_FORMULA, STATUS_FAILED


class BackendNotIntegratedError(RuntimeError):
    """Raised when a backend exists only as a placeholder.

    Selecting such a backend must fail loudly; callers must never fall back to
    another backend.
    """

    def __init__(self, backend: str, integration_status: str, blockers: Sequence[str]) -> None:
        self.backend = backend
        self.integration_status = integration_status
        self.blockers = tuple(blockers)
        lines = [
            f"backend {backend!r} is not usable: integration_status={integration_status}.",
            "No fallback to another backend is performed. Blockers:",
        ]
        lines.extend(f"  - {b}" for b in self.blockers)
        super().__init__("\n".join(lines))


class Environment(ABC):
    """A backend that turns (scenario, policy) into one episode result.

    ``make_scenario`` samples the advertiser for a scenario seed; ``run_episode``
    plays one delivery period, calling ``policy.bidding()`` once per time step
    with exactly the eight AuctionNet arguments.
    """

    backend: str
    integration_status: str
    num_time_steps: int

    @abstractmethod
    def make_scenario(self, scenario_id: str, split: str, index: int, seed: int) -> dict[str, Any]:
        """Return a JSON-serialisable scenario description."""

    @abstractmethod
    def run_episode(self, scenario: dict[str, Any], policy: Any) -> dict[str, Any]:
        """Play one episode and return its metrics record (never raises for policy errors)."""


def failed_episode(
    scenario: dict[str, Any], backend: str, error: str, wall_time_s: float, opportunities: int | None
) -> dict[str, Any]:
    """Episode record for a failure: every metric is null, never a guessed value."""
    return {
        "backend": backend,
        "purpose": PURPOSE,
        "score_formula": SCORE_FORMULA,
        "scenario_id": scenario.get("scenario_id"),
        "split": scenario.get("split"),
        "status": STATUS_FAILED,
        "error": error,
        "budget": scenario.get("budget"),
        "cpa_target": scenario.get("cpa_target"),
        "cost": None,
        "conversions": None,
        "expected_conversions": None,
        "cpa_realized": None,
        "cpa_expected": None,
        "cpa_defined": False,
        "cpa_expected_defined": False,
        "score_realized": None,
        "score_expected": None,
        "budget_utilization": None,
        "wins": None,
        "opportunities": opportunities,
        "cpa_ok": None,
        "cpa_expected_ok": None,
        "wall_time_s": wall_time_s,
    }
