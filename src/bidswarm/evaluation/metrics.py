"""Score formula replicating AuctionNet's PlayerAnalysis (NeurIPS competition).

Upstream (simul_bidding_env/Tracker/PlayerAnalysis.py), per (playerindex, episode):

    cpa = cost / (reward + 1e-10)
    score = reward * (C / cpa) ** 2   if cpa > C
    score = reward                    otherwise

and the aggregate is sum(group scores) / 20000. Single toy episodes report the
per-group score WITHOUT the 1/20000 normalisation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

AUCTIONNET_CPA_EPSILON = 1e-10
AUCTIONNET_SCORE_NORMALIZER = 20000.0


def auctionnet_cpa(reward: float, cost: float) -> float:
    return cost / (reward + AUCTIONNET_CPA_EPSILON)


def auctionnet_score(reward: float, cost: float, cpa_constraint: float) -> float:
    """Per-(player, episode) score; approximately 0 when reward is 0."""
    cpa = auctionnet_cpa(reward, cost)
    if cpa > cpa_constraint:
        return reward * (cpa_constraint / cpa) ** 2
    return reward


def auctionnet_aggregate_score(group_scores: Iterable[float]) -> float:
    """Upstream leaderboard aggregate: sum of per-group scores / 20000."""
    return math.fsum(group_scores) / AUCTIONNET_SCORE_NORMALIZER


def cpa_within_target(reward: float, cost: float, cpa_constraint: float) -> bool:
    """True when the upstream formula applies no CPA penalty."""
    return auctionnet_cpa(reward, cost) <= cpa_constraint


def mean_or_none(values: Iterable[float | None]) -> float | None:
    """Arithmetic mean (left-to-right summation), or None if any value is None or empty."""
    vals = list(values)
    if not vals or any(v is None for v in vals):
        return None
    total = 0.0
    for v in vals:
        total += float(v)
    return total / len(vals)
