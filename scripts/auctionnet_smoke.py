"""AuctionNet backend smoke test against the real upstream checkout.

Checks, for the FixedRate and DualControl default specs at one player position:

1. The runner's own tick loop reproduces upstream ``run/run_test.py`` exactly:
   per-episode reward (realized conversions) and charged cost for episodes 0
   and 1 equal the values ``run_test`` produces for the same player, both
   single-threaded (``scripts/auctionnet_probe.py`` drives ``run_test``).
2. bidswarm's score equals upstream ``PlayerAnalysis``'s per-episode score.
3. Results do not depend on request order: episode 0 asked again after
   episode 1, and after another spec, is identical.
4. The client's invariants hold on every request (kernel books == charge,
   bidding calls == expected), otherwise it raises.

Runs with bidswarm's own Python; needs BIDSWARM_AUCTIONNET_REPO and
BIDSWARM_AUCTIONNET_PYTHON. Exit code 0 when every check passes.

Usage:
    python scripts/auctionnet_smoke.py [--player-index 0]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bidswarm.environments import auctionnet
from bidswarm.evaluation.metrics import auctionnet_score
from bidswarm.kernel import default_spec, spec_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "auctionnet_probe.py"
POLICIES = ("fixed_rate", "dual_control")


def reference(python: str, upstream: str, name: str, player_index: int) -> list[dict]:
    """Per-episode results of upstream run_test for episodes 0 and 1."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    out = subprocess.run(
        [python, str(PROBE), "--upstream", upstream, "--kernel-parent", str(auctionnet.KERNEL_PARENT),
         "--player", name, "--player-index", str(player_index), "--episodes", "2"],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    line = next(x for x in out.splitlines() if x.startswith("PROBE_RESULT "))
    return json.loads(line[len("PROBE_RESULT "):])["per_episode"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--player-index", type=int, default=0)
    args = ap.parse_args()
    cfg = auctionnet.validate_config({
        "backend": "auctionnet", "repo_path": "$BIDSWARM_AUCTIONNET_REPO", "commit_sha": auctionnet.PINNED_COMMIT,
        "python_executable": "$BIDSWARM_AUCTIONNET_PYTHON", "player_index": args.player_index,
    })
    auctionnet.check_setup(cfg)
    cpa = auctionnet.advertiser(args.player_index)["cpa_target"]

    with ThreadPoolExecutor(max_workers=len(POLICIES)) as pool:
        refs = dict(zip(POLICIES, pool.map(
            lambda n: reference(cfg["python_executable"], cfg["repo_path"], n, args.player_index), POLICIES)))
        runner = auctionnet.RunnerProcess(cfg)
        try:
            got: dict[str, list[dict]] = {}
            for name in POLICIES:
                spec = default_spec(name)
                msg = {"op": "episode", "spec": spec.to_dict(), "spec_hash": spec_hash(spec)}
                got[name] = [runner.request({**msg, "episode": e}) for e in (0, 1, 0)]
            again = runner.request({**msg, "episode": 0, "spec": default_spec(POLICIES[0]).to_dict(),
                                    "spec_hash": spec_hash(default_spec(POLICIES[0]))})
        finally:
            runner.close()

    failures: list[str] = []
    summary = {"upstream_commit": cfg["commit_sha"], "player_index": args.player_index, "policies": {}}
    for name in POLICIES:
        rows = []
        for episode in (0, 1):
            ours, theirs = got[name][episode], refs[name][episode]
            for problem in (
                [] if ours["ok"] else [f"{name} episode {episode}: {ours}"]
            ):
                failures.append(problem)
            if not ours["ok"]:
                continue
            score = auctionnet_score(ours["reward"], ours["cost"], cpa)
            row = {"episode": episode, "reward": ours["reward"], "cost": ours["cost"],
                   "run_test_reward": theirs["reward"], "run_test_cost": theirs["allCost"],
                   "score": score, "player_analysis_score": theirs["score"],
                   "kernel_spent": ours["kernel_spent"], "seconds": round(ours["wall_time_s"], 1)}
            rows.append(row)
            if ours["reward"] != theirs["reward"] or ours["cost"] != theirs["allCost"]:
                failures.append(f"{name} episode {episode}: runner {ours['reward']}/{ours['cost']} "
                                f"!= run_test {theirs['reward']}/{theirs['allCost']}")
            if abs(score - theirs["score"]) > 1e-9 * max(1.0, abs(theirs["score"])):
                failures.append(f"{name} episode {episode}: score {score} != PlayerAnalysis {theirs['score']}")
            if abs(ours["kernel_spent"] - ours["cost"]) > 1e-6 * max(1.0, ours["cost"]):
                failures.append(f"{name} episode {episode}: kernel books {ours['kernel_spent']} != charge {ours['cost']}")
        strip = lambda r: {k: v for k, v in r.items() if k != "wall_time_s"}  # noqa: E731
        if strip(got[name][0]) != strip(got[name][2]):
            failures.append(f"{name}: episode 0 differs when requested again after episode 1")
        summary["policies"][name] = rows
    if strip(again) != strip(got[POLICIES[0]][0]):
        failures.append("episode 0 differs when requested after another spec")
    summary["failures"] = failures
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
