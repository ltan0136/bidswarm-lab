"""Stand-in for environments/auctionnet_runner.py: same protocol, no upstream code.

Runs the real kernel policy against a tiny synthetic market so the bidswarm-side
client can be tested without an AuctionNet checkout. ``STUB_MODE`` selects a
failure to simulate: ok (default), policy_error, runner_error, bad_books,
bad_calls, crash_on_start, not_ready, wrong_budget, exit_after_ready.
"""

import argparse
import json
import os
import sys

import numpy as np

BUDGETS = (2900, 4350, 3000)
CPA = (100, 70, 90)


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--upstream", "--kernel-parent", "--commit-sha"):
        ap.add_argument(name, required=True)
    ap.add_argument("--player-index", type=int, required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.kernel_parent)
    from kernel import PolicySpec, make_policy, spec_hash

    mode = os.environ.get("STUB_MODE", "ok")
    counter_path = os.environ.get("STUB_COUNTER")
    if counter_path:
        with open(counter_path, "a") as fh:
            fh.write("start\n")
    if mode == "crash_on_start":
        return 3
    idx = args.player_index
    budget = BUDGETS[idx] * (2 if mode == "wrong_budget" else 1)
    ready = {"ready": mode != "not_ready", "error": "stub refused" if mode == "not_ready" else None,
             "upstream_commit": args.commit_sha, "player_index": idx, "budget": float(budget),
             "cpa": float(CPA[idx]), "category": idx // 8, "num_tick": 48, "num_agents": 48,
             "pv_num": 100, "versions": {"python": "stub"}}
    print(json.dumps(ready), flush=True)
    if mode == "exit_after_ready":
        return 0
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "shutdown":
            break
        spec = PolicySpec.from_dict(req["spec"])
        assert spec_hash(spec) == req["spec_hash"]
        if mode == "policy_error":
            print(json.dumps({"ok": False, "kind": "policy_error", "error": "PolicyInputError: stub"}), flush=True)
            continue
        if mode == "runner_error":
            print(json.dumps({"ok": False, "kind": "runner_error", "error": "RuntimeError: stub"}), flush=True)
            continue
        policy = make_policy(spec, float(budget), float(CPA[idx]), idx // 8, 48)
        policy.reset()
        rng = np.random.default_rng(req["episode"])
        hp, hb, ha, hi, hl = [], [], [], [], []
        cost = reward = expected = 0.0
        wins = exposures = 0
        for t in range(48):
            p = rng.beta(2, 98, 100)
            s = 0.2 * p
            bids = policy.bidding(t, p, s, hp, hb, ha, hi, hl)
            price = rng.uniform(0.0, 0.5, 100)
            win = (bids > price).astype(float)
            exposed = win * (rng.random(100) < 0.8)
            conv = exposed * (rng.random(100) < p)
            cost += float(np.sum(price * exposed))
            reward += float(np.sum(conv))
            expected += float(np.sum(p * exposed))
            wins += int(win.sum())
            exposures += int(exposed.sum())
            hp.append(np.stack([p, s], axis=1)); hb.append(bids); hl.append(price)
            ha.append(np.stack([win, win, price * win], axis=1)); hi.append(np.stack([exposed, conv], axis=1))
        policy._consume_history(hp, hb, ha, hi, hl)
        out = {"ok": True, "episode": req["episode"], "reward": reward, "cost": cost,
               "expected_conversions": expected, "wins": wins, "exposures": exposures,
               "opportunities": 4800, "budget": float(budget), "remaining_budget": budget - cost,
               "kernel_spent": policy.spent * (1.5 if mode == "bad_books" else 1.0),
               "bidding_calls": 48, "expected_bidding_calls": 47 if mode == "bad_calls" else 48,
               "last_active_tick": 47, "spec_hash": req["spec_hash"], "wall_time_s": 0.01}
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
