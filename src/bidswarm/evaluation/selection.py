"""Candidate fitness, top-K shortlist and validation-based selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..types import USABLE_STATUSES
from .metrics import mean_or_none

SELECTION_RULE = (
    "Pool all candidates of the mode; take the top-K by training fitness "
    "(ties: earlier candidate); evaluate them on the validation scenarios; "
    "select the highest mean validation score (ties: higher training fitness, "
    "then better training rank). Candidates with null fitness or any failed "
    "validation episode are never selected."
)


def candidate_fitness(episodes: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    """Mean of ``metric`` over the episodes, or None if any episode is unusable."""
    if not episodes or any(e["status"] not in USABLE_STATUSES for e in episodes):
        return None
    return mean_or_none(e[metric] for e in episodes)


def shortlist_top_k(candidates: Sequence[Mapping[str, Any]], k: int) -> list[Mapping[str, Any]]:
    ranked = [(i, c) for i, c in enumerate(candidates) if c["fitness"] is not None]
    ranked.sort(key=lambda ic: (-ic[1]["fitness"], ic[0]))
    return [c for _, c in ranked[:k]]


def choose_selected(
    shortlist: Sequence[Mapping[str, Any]], validation_scores: Mapping[str, float | None]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (selected row or None, all shortlist rows with their validation scores)."""
    rows = [
        {
            "rank": rank,
            "candidate_id": c["candidate_id"],
            "spec_hash": c["spec_hash"],
            "train_fitness": c["fitness"],
            "validation_score": validation_scores.get(c["candidate_id"]),
        }
        for rank, c in enumerate(shortlist)
    ]
    valid = [r for r in rows if r["validation_score"] is not None]
    if not valid:
        return None, rows
    best = max(valid, key=lambda r: (r["validation_score"], r["train_fitness"], -r["rank"]))
    return best, rows
