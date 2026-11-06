"""Real AuctionNet runs; skipped unless the pinned checkout and its Python 3.9 venv are available.

    export BIDSWARM_AUCTIONNET_REPO=/path/to/AuctionNet BIDSWARM_AUCTIONNET_PYTHON=/path/to/an-venv/bin/python
    python -m pytest tests/test_auctionnet_integration.py

Each period takes about 45-50 s; this file runs three.
"""

from __future__ import annotations

import os

import pytest

from bidswarm.environments import auctionnet, make_environment
from bidswarm.kernel import default_spec, make_policy

pytestmark = pytest.mark.skipif(
    not (os.environ.get("BIDSWARM_AUCTIONNET_REPO") and os.environ.get("BIDSWARM_AUCTIONNET_PYTHON")),
    reason="set BIDSWARM_AUCTIONNET_REPO and BIDSWARM_AUCTIONNET_PYTHON to run against real AuctionNet",
)


@pytest.fixture(scope="module")
def env():
    auctionnet._close_runners()
    yield make_environment({
        "backend": "auctionnet", "repo_path": "$BIDSWARM_AUCTIONNET_REPO", "commit_sha": auctionnet.PINNED_COMMIT,
        "python_executable": "$BIDSWARM_AUCTIONNET_PYTHON", "player_index": 0,
    })
    auctionnet._close_runners()


def _run(env, episode_seed: int, name: str):
    sc = env.make_scenario(f"s{episode_seed}", "test", 0, episode_seed)
    policy = make_policy(default_spec(name), sc["budget"], sc["cpa_target"], sc["category"], 48)
    return env.run_episode(sc, policy)


def test_fixed_rate_episode_0_matches_the_step1_probe_and_is_order_independent(env):
    first = _run(env, 0, "fixed_rate")
    other = _run(env, 1, "fixed_rate")
    again = _run(env, 0, "fixed_rate")
    assert first["status"] == other["status"] == "ok"
    # Step-1 probe through upstream run_test (docs/AUCTIONNET_INTEGRATION.md, section 3).
    assert first["conversions"] == 31
    assert first["cost"] == pytest.approx(2511.5159, rel=1e-6)
    assert first["cost"] <= first["budget"] == 2900.0
    assert first["score_realized"] == 31.0
    strip = lambda r: {k: v for k, v in r.items() if k != "wall_time_s"}  # noqa: E731
    assert strip(first) == strip(again)
    assert strip(first) != strip(other)
