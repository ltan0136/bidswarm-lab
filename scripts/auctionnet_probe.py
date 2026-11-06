"""AuctionNet step-1 probe: run one online-simulation period with a chosen player.

This is an investigation tool, NOT the AuctionNet backend (which is still
``not_integrated``). It must run inside the upstream Python 3.9 environment
described in docs/AUCTIONNET_INTEGRATION.md, never inside bidswarm's own venv.

The player is injected by replacing ``run_test.initialize_player_agent``, so a
failure raises instead of upstream's silent fallback to PID. ``KernelShim`` is
a minimal prototype of the step-2 shim: it owns the attributes run_test reads
and writes (``remaining_budget`` included) and forwards ``bidding()`` to a
bidswarm kernel policy loaded from ``bidswarm/kernel`` as a top-level package.

Usage:
    OMP_NUM_THREADS=1 an-venv/bin/python scripts/auctionnet_probe.py \
        --upstream AuctionNet --kernel-parent src/bidswarm \
        --player pid|fixed_rate|dual_control [--player-index N] [--episodes N]

Prints one line ``PROBE_RESULT {json}`` with wall time, peak memory, the
upstream score and, for kernel players, kernel-side spend versus the amount
upstream actually charged.
"""

import argparse
import json
import os
import resource
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--upstream", required=True)
ap.add_argument("--kernel-parent", required=True)
ap.add_argument("--player", default="pid")
ap.add_argument("--player-index", type=int, default=0)
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--jit-optimize", choices=("on", "off"), default="on",
                help="TorchScript graph-executor optimization; upstream default is on, the bidswarm runner uses off")
args = ap.parse_args()

upstream = os.path.abspath(args.upstream)
kernel_parent = os.path.abspath(args.kernel_parent)
os.chdir(upstream)  # upstream loads weights and CSVs by relative path
sys.path.insert(0, os.path.join(upstream, "strategy_train_env"))
sys.path.insert(0, upstream)
sys.path.insert(0, kernel_parent)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch._C._set_graph_executor_optimize(args.jit_optimize == "on")
torch.manual_seed(1)  # as upstream main_test.py
np.random.seed(1)

import gin  # noqa: E402

gin.parse_config_files_and_bindings(["./config/test.gin"], [f"run_test.num_episode = {args.episodes}"])

import run.run_test as rt  # noqa: E402
from simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis  # noqa: E402


class KernelShim:
    """Minimal player: owns the attributes run_test reads and writes, forwards bidding()."""

    def __init__(self, spec):
        self.spec = spec
        self.name = "bidswarm_probe"
        self.budget = self.cpa = self.category = None
        self.remaining_budget = 0.0
        self.policy = None
        self.calls = 0
        self.kernel_spent_total = 0.0
        self.charged_total = 0.0

    def reset(self):
        from kernel import make_policy  # bidswarm/kernel as a top-level package

        if self.policy is not None:  # close the previous episode's books
            self.kernel_spent_total += self.policy.spent
        self.policy = make_policy(self.spec, float(self.budget), float(self.cpa), int(self.category), 48)
        self.policy.reset()
        self.remaining_budget = float(self.budget)

    def bidding(self, *a):
        self.calls += 1
        return self.policy.bidding(*a)


if args.player == "pid":
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    player = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
else:
    from kernel import default_spec

    player = KernelShim(default_spec(args.player))

analysis = PlayerAnalysis("player_analysis")
rt.initialize_player_agent = lambda: player
rt.initialize_player_analysis = lambda: analysis

started = time.time()
result = rt.run_test(player_index=args.player_index)
elapsed = time.time() - started
out = {
    "player": args.player, "player_index": args.player_index, "episodes": args.episodes,
    "jit_optimize": args.jit_optimize,
    "seconds": round(elapsed, 2),
    "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    "score": float(result["score"]), "reward": float(result["reward"]),
    "per_episode": [{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                     for k, v in r.items()} for r in analysis.analysis_res],
}
if isinstance(player, KernelShim):
    out["kernel_calls"] = player.calls
    out["kernel_spent_last_episode"] = player.policy.spent
    out["charged_last_episode"] = float(player.budget) - player.remaining_budget
    out["kernel_expected_conversions_last_episode"] = player.policy.expected_conversions
print("PROBE_RESULT " + json.dumps(out))
