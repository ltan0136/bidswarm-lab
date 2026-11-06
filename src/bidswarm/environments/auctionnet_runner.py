"""AuctionNet online-simulation runner: one long-lived subprocess per bidswarm worker.

This file is executed by path with the *upstream* Python 3.9 interpreter (see
docs/AUCTIONNET_INTEGRATION.md). It must stay compatible with Python 3.9 and
must not import the ``bidswarm`` package: it imports upstream modules and
``bidswarm/kernel`` as the top-level package ``kernel`` only.
``tests/test_kernel_compat.py`` checks both rules.

Protocol: JSON lines. The runner writes protocol messages to the original
stdout; everything upstream prints is redirected to stderr. After start-up it
sends one ``{"ready": true, ...}`` message, then answers each request line

    {"op": "episode", "episode": <int>, "spec": <PolicySpec dict>, "spec_hash": <str>}

with ``{"ok": true, ...metrics}`` or ``{"ok": false, "kind": ..., "error": ...}``.

The tick loop reproduces ``run/run_test.py`` (lines 138-213 at the pinned
commit) for a single episode index, reusing upstream's ``get_winner`` and
``adjust_over_cost``. It exists because ``run_test`` always starts at episode
0, and because ``run_test.initialize_player_agent`` silently falls back to a
PID agent when the player cannot be imported; here the player object is
injected directly and checked to be the one that bid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

NUM_TICK = 48


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="bidswarm AuctionNet runner (upstream Python 3.9)")
    ap.add_argument("--upstream", required=True, help="AuctionNet checkout")
    ap.add_argument("--kernel-parent", required=True, help="directory containing bidswarm's kernel/ package")
    ap.add_argument("--commit-sha", required=True, help="commit the checkout must be at")
    ap.add_argument("--player-index", type=int, required=True)
    return ap.parse_args(argv)


def _git_head(path: str) -> str:
    out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


class KernelShim:
    """The player object: owns what run_test-style code reads and writes, forwards bidding()."""

    def __init__(self) -> None:
        self.name = "bidswarm_kernel"
        self.budget: Any = None
        self.cpa: Any = None
        self.category: Any = None
        self.remaining_budget = 0.0
        self.spec: Any = None
        self.policy: Any = None
        self.calls = 0

    def set_spec(self, spec: Any) -> None:
        self.spec = spec

    def reset(self) -> None:
        from kernel import make_policy

        self.policy = None if self.spec is None else make_policy(
            self.spec, float(self.budget), float(self.cpa), int(self.category), NUM_TICK
        )
        if self.policy is not None:
            self.policy.reset()
        self.remaining_budget = float(self.budget)
        self.calls = 0

    def bidding(self, *args: Any) -> Any:
        self.calls += 1
        return self.policy.bidding(*args)


class PolicyFailure(Exception):
    """The bidswarm policy raised or returned invalid bids: a failed episode, not a runner error."""


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.upstream = os.path.abspath(args.upstream)
        head = _git_head(self.upstream)
        if head != args.commit_sha:
            raise RuntimeError("upstream checkout is at %s, expected %s" % (head, args.commit_sha))
        self.commit = head
        os.chdir(self.upstream)  # upstream loads weights and CSVs by relative path
        sys.path.insert(0, os.path.join(self.upstream, "strategy_train_env"))
        sys.path.insert(0, self.upstream)
        sys.path.insert(0, os.path.abspath(args.kernel_parent))

        import numpy as np
        import torch

        torch.set_num_threads(1)
        # Competitors are TorchScript models. With graph-executor optimization on, the first
        # calls in a process run unoptimized (profiling) and later ones fused, so the same
        # episode gave results differing at ~1e-9 depending on what the process ran before.
        # Off, every call uses the same interpreter path and results do not depend on history.
        torch._C._set_graph_executor_optimize(False)
        torch.manual_seed(1)  # as upstream main_test.py
        np.random.seed(1)
        import gin

        gin.parse_config_files_and_bindings(["./config/test.gin"], None)
        import run.run_test as rt
        from simul_bidding_env.Controller.Controller import Controller

        self.np = np
        self.rt = rt
        self.shim = KernelShim()
        self.player_index = int(args.player_index)
        self.controller = Controller(player_index=self.player_index, player_agent=self.shim)
        self.agents = self.controller.agents
        if self.agents[self.player_index].player_agent is not self.shim:
            raise RuntimeError("the injected player is not the agent at player_index")
        if self.controller.num_tick != NUM_TICK:
            raise RuntimeError("expected %d ticks per period, got %s" % (NUM_TICK, self.controller.num_tick))
        self.versions = {
            "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "torchscript_graph_optimize": bool(torch._C._get_graph_executor_optimize()),
        }

    def ready_message(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "upstream_commit": self.commit,
            "player_index": self.player_index,
            "budget": float(self.shim.budget),
            "cpa": float(self.shim.cpa),
            "category": int(self.shim.category),
            "num_tick": NUM_TICK,
            "num_agents": len(self.agents),
            "pv_num": int(self.controller.pv_num),
            "versions": self.versions,
        }

    def run_episode(self, episode: int, spec_dict: Dict[str, Any], expected_hash: str) -> Dict[str, Any]:
        from kernel import PolicySpec, spec_hash

        spec = PolicySpec.from_dict(spec_dict)
        if spec_hash(spec) != expected_hash:
            raise RuntimeError("spec_hash mismatch between client and runner")
        np, rt = self.np, self.rt
        controller, agents, player = self.controller, self.agents, self.player_index
        envs, pv_generator = controller.biddingEnv, controller.pvGenerator
        self.shim.set_spec(spec)
        started = time.perf_counter()

        h_pvalue_infos: List[Any] = []
        h_bids: List[Any] = []
        h_auction: List[Any] = []
        h_impression: List[Any] = []
        h_lwc: List[Any] = []
        tick_rewards: List[float] = []
        tick_costs: List[float] = []
        expected_conversions = 0.0
        wins = 0
        exposures = 0
        opportunities = 0
        expected_calls = 0
        last_active_tick = -1

        controller.reset(episode=episode)
        for tick_index in range(NUM_TICK):
            pv_values = pv_generator.pv_values[tick_index]
            pvalue_sigmas = pv_generator.pValueSigmas[tick_index]
            n_pv = pv_values.shape[0]
            bids = []
            for i, agent in enumerate(agents):
                if agent.remaining_budget >= envs.min_remaining_budget:
                    args = (
                        tick_index, pv_values[:, i], pvalue_sigmas[:, i],
                        [x[i] for x in h_pvalue_infos], [x[i] for x in h_bids],
                        [x[i] for x in h_auction], [x[i] for x in h_impression], h_lwc,
                    )
                    if i == player:
                        expected_calls += 1
                        last_active_tick = tick_index
                        try:
                            b = np.asarray(agent.bidding(*args), dtype=float)
                        except Exception as exc:
                            raise PolicyFailure("%s: %s" % (type(exc).__name__, exc))
                        if b.shape != (n_pv,) or not np.all(np.isfinite(b)) or np.any(b < 0):
                            raise PolicyFailure("policy returned invalid bids (shape %s)" % (b.shape,))
                        bids.append(b)
                    else:
                        bids.append(agent.bidding(*args))
                else:
                    bids.append(np.zeros(n_pv))

            bids = np.array(bids).transpose()
            bids[bids < 0] = 0
            remaining_budget_list = np.array([agent.remaining_budget for agent in agents])

            ratio_max = None
            cost = None
            winner_pit = None
            while ratio_max is None or ratio_max > 0:
                if ratio_max and ratio_max > 0:
                    over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                    rt.adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients, winner_pit)
                (xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit,
                 least_winning_cost_pit, _market_price_pit) = envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)
                real_cost = cost_pit * is_exposed_pit
                cost = real_cost.sum(axis=1)
                reward = conversion_action_pit.sum(axis=1)
                winner_pit = rt.get_winner(slot_pit)
                over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                ratio_max = over_cost_ratio.max()

            for i, agent in enumerate(agents):
                agent.remaining_budget -= cost[i]
            h_bids.append(bids.transpose())
            h_lwc.append(least_winning_cost_pit)
            h_pvalue_infos.append(np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
            h_auction.append(np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
            h_impression.append(np.stack((is_exposed_pit, conversion_action_pit), axis=-1))

            tick_rewards.append(float(reward[player]))
            tick_costs.append(float(cost[player]))
            expected_conversions += float(np.sum(pv_values[:, player] * xi_pit[player] * is_exposed_pit[player]))
            wins += int(np.sum(xi_pit[player]))
            exposures += int(np.sum(is_exposed_pit[player]))
            opportunities += int(n_pv)

        # Sum per-tick values with np.sum, as PlayerAnalysis does (np.sum over the tick column),
        # so totals match upstream to the last bit rather than differing by summation order.
        reward_total = float(np.sum(np.asarray(tick_rewards, dtype=float)))
        cost_total = float(np.sum(np.asarray(tick_costs, dtype=float)))
        # Let the kernel read the final step, then compare its books with what was charged.
        policy = self.shim.policy
        policy._consume_history(
            [x[player] for x in h_pvalue_infos], [x[player] for x in h_bids],
            [x[player] for x in h_auction], [x[player] for x in h_impression], h_lwc,
        )
        return {
            "ok": True,
            "episode": int(episode),
            "reward": reward_total,
            "cost": cost_total,
            "expected_conversions": expected_conversions,
            "wins": wins,
            "exposures": exposures,
            "opportunities": opportunities,
            "budget": float(self.shim.budget),
            "remaining_budget": float(self.shim.remaining_budget),
            "kernel_spent": float(policy.spent),
            "bidding_calls": int(self.shim.calls),
            "expected_bidding_calls": int(expected_calls),
            "last_active_tick": int(last_active_tick),
            "spec_hash": expected_hash,
            "wall_time_s": time.perf_counter() - started,
        }


def main(argv: Optional[List[str]] = None) -> int:
    # Keep the protocol channel clean: upstream prints go to stderr from here on.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(message: Dict[str, Any]) -> None:
        proto.write(json.dumps(message, allow_nan=False) + "\n")
        proto.flush()

    try:
        runner = Runner(_parse_args(argv))
    except Exception as exc:
        send({"ready": False, "error": "%s: %s" % (type(exc).__name__, exc), "traceback": traceback.format_exc()})
        return 1
    send(runner.ready_message())
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request.get("op") == "shutdown":
                break
            if request.get("op") != "episode":
                raise RuntimeError("unknown op %r" % request.get("op"))
            response = runner.run_episode(int(request["episode"]), request["spec"], request["spec_hash"])
        except PolicyFailure as exc:
            response = {"ok": False, "kind": "policy_error", "error": str(exc)}
        except Exception as exc:
            response = {"ok": False, "kind": "runner_error", "error": "%s: %s" % (type(exc).__name__, exc),
                        "traceback": traceback.format_exc()}
        for key, value in list(response.items()):
            if isinstance(value, float) and not math.isfinite(value):
                response[key] = None
        send(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
