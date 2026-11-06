"""Paired comparison of two runs that share seeds and scenarios (e.g. two proposers).

Reads only run artifacts. For every mode present in both runs it prints the
replicate-paired difference A - B of the selected policies' test score and of
the final training best-so-far, with the report's percentile bootstrap CI and
sign-flip permutation p (same resamples and seeds as report.md), plus Holm-
corrected p over the test comparisons. Both runs must have been evaluated.

Usage:
    python scripts/compare_runs.py RUN_A RUN_B
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bidswarm.artifacts.report import paired_bootstrap_ci, paired_sign_flip_pvalue
from bidswarm.artifacts.store import RunStore
from bidswarm.evaluation.selection import candidate_fitness
from bidswarm.types import MODES


class Run:
    def __init__(self, run_dir: str) -> None:
        store = RunStore.open(run_dir)
        self.name = Path(run_dir).name
        manifest = store.read_json("manifest.json")
        search = manifest["config"]["search"]
        self.proposer, self.budget, self.metric = search["proposer"], search["train_episode_budget"], search["fitness_metric"]
        self.modes = [m for m in MODES if m in search["modes"]]
        self.seeds = manifest["master_seeds"]
        self.scenarios = store.read_json("scenarios.json")["replicates"]
        if not store.exists("test_episodes.jsonl"):
            raise SystemExit(f"{self.name}: run `bidswarm evaluate` first")
        grouped = defaultdict(list)
        for e in store.read_jsonl("test_episodes.jsonl"):
            grouped[(e["replicate"], e["group"])].append(e)
        self.test = {k: candidate_fitness(v, self.metric) for k, v in grouped.items()}
        self.train_best: dict[tuple[int, str], tuple[int, float]] = {}
        for a in store.read_jsonl("anytime.jsonl"):
            if a["scope"] == "mode":
                key = (a["replicate"], a["mode"])
                if key not in self.train_best or a["round"] > self.train_best[key][0]:
                    self.train_best[key] = (a["round"], a["best_so_far"])

    def test_scores(self, mode: str) -> np.ndarray:
        values = [self.test.get((r, f"selected:{mode}")) for r in range(len(self.seeds))]
        if any(v is None for v in values):
            raise SystemExit(f"{self.name}: missing or failed test score for {mode}")
        return np.asarray(values, dtype=float)

    def final_train_best(self, mode: str) -> np.ndarray:
        return np.asarray([self.train_best[(r, mode)][1] for r in range(len(self.seeds))], dtype=float)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    args = ap.parse_args()
    a, b = Run(args.run_a), Run(args.run_b)
    if a.seeds != b.seeds or a.scenarios != b.scenarios:
        raise SystemExit("the runs do not share master seeds and scenarios; they cannot be paired")
    if a.metric != b.metric or a.budget != b.budget:
        print(f"warning: metric/budget differ ({a.metric}/{a.budget} vs {b.metric}/{b.budget})", file=sys.stderr)
    modes = [m for m in a.modes if m in b.modes]
    print(f"A = {a.name} (proposer {a.proposer}), B = {b.name} (proposer {b.proposer}); "
          f"R = {len(a.seeds)}, budget {a.budget}, metric {a.metric}\n")
    print("| mode | A test | B test | A − B (test) | 95% CI | p | Holm p | A − B (train best) | 95% CI | p |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows, pvalues = [], []
    for mode in modes:
        ta, tb = a.test_scores(mode), b.test_scores(mode)
        d, dt = ta - tb, a.final_train_best(mode) - b.final_train_best(mode)
        p = paired_sign_flip_pvalue(d)
        rows.append((mode, ta, tb, d, paired_bootstrap_ci(d), p, dt, paired_bootstrap_ci(dt), paired_sign_flip_pvalue(dt)))
        pvalues.append(p)
    for (mode, ta, tb, d, ci, p, dt, cit, pt), ph in zip(rows, holm(pvalues)):
        print(f"| {mode} | {ta.mean():.3f} | {tb.mean():.3f} | {d.mean():+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
              f"{fmt_p(p)} | {fmt_p(ph)} | {dt.mean():+.3f} | [{cit[0]:+.3f}, {cit[1]:+.3f}] | {fmt_p(pt)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
