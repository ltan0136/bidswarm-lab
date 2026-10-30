"""Statistical power and calibration of the BidSwarm Lab comparison protocol.

Reads only run artifacts (selected/*.json, test_episodes.jsonl, anytime.jsonl,
scenarios.json, manifest.json) and prints a Markdown summary:

1. A/A check (random proposer): independent - single must be centred on 0.
2. Calibration: false-positive rate of the report's percentile-bootstrap CI,
   estimated by resampling replicate-level differences of the A/A pool.
3. Power / minimum detectable effect (MDE) for the ES comparison, using the
   shape of the ES paired differences under a shift model.
4. Positive controls: paired effects between runs that share scenarios
   (e.g. more vs less search budget; ES vs random).

Usage:
    python scripts/power_analysis.py --aa RUN_DIR --es RUN_DIR \
        [--power RUN_DIR:MODE_A:MODE_B ...] [--control LABEL=RUN_DIR_HIGH,RUN_DIR_LOW ...] \
        [--prefix-check SHORT_RUN_DIR:LONG_RUN_DIR]

Section 3 analyses ``--es`` for independent - single, plus every ``--power``
entry (e.g. ``RUN:cooperative:independent``).
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from bidswarm.artifacts.report import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    paired_bootstrap_ci,
    paired_sign_flip_pvalue,
)
from bidswarm.artifacts.store import RunStore
from bidswarm.evaluation.selection import candidate_fitness

N_SIM = 2000  # simulated experiments per design size
N_SIM_PERM = 500  # simulated experiments for the (more expensive) permutation-test calibration
PERM_FLIPS = 2000  # sign flips per simulated permutation test
SIM_SEED = 12345
DESIGN_SIZES = (20, 50, 100)
Z_975, Z_80 = 1.959964, 0.841621


class Run:
    def __init__(self, run_dir: str) -> None:
        self.store = RunStore.open(run_dir)
        self.name = Path(run_dir).name
        self.manifest = self.store.read_json("manifest.json")
        self.metric = self.manifest["config"]["search"]["fitness_metric"]
        self.proposer = self.manifest["config"]["search"]["proposer"]
        self.budget = self.manifest["config"]["search"]["train_episode_budget"]
        self.n_rep = len(self.manifest["master_seeds"])
        grouped = defaultdict(list)
        for e in self.store.read_jsonl("test_episodes.jsonl"):
            grouped[(e["replicate"], e["group"])].append(e)
        self.test = {k: candidate_fitness(v, self.metric) for k, v in grouped.items()}
        self.scenarios = self.store.read_json("scenarios.json")["replicates"]
        self.anytime = self.store.read_jsonl("anytime.jsonl")

    def scores(self, label: str) -> np.ndarray:
        vals = [self.test.get((r, label)) for r in range(self.n_rep)]
        if any(v is None for v in vals):
            raise SystemExit(f"{self.name}: missing or failed test score for {label}")
        return np.asarray(vals, dtype=float)

    def final_train_best(self, mode: str) -> np.ndarray:
        best: dict[int, tuple[int, float]] = {}
        for a in self.anytime:
            if a["scope"] == "mode" and a["mode"] == mode:
                prev = best.get(a["replicate"])
                if prev is None or a["round"] > prev[0]:
                    best[a["replicate"]] = (a["round"], a["best_so_far"])
        return np.asarray([best[r][1] for r in range(self.n_rep)], dtype=float)


def same_test_scenarios(a: Run, b: Run) -> bool:
    n = min(a.n_rep, b.n_rep)
    return all(a.scenarios[r]["test"] == b.scenarios[r]["test"] for r in range(n))


def describe(d: np.ndarray) -> str:
    ci = paired_bootstrap_ci(d)
    se = d.std(ddof=1) / math.sqrt(d.size)
    return (
        f"n={d.size}，均值 {d.mean():+.3f}，标准差 {d.std(ddof=1):.3f}，标准误 {se:.3f}，"
        f"95% bootstrap CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    )


def simulate_cis(pool: np.ndarray, size: int, seed: int) -> np.ndarray:
    """CI (report procedure) of N_SIM experiments of `size` replicates drawn from `pool`."""
    rng = np.random.default_rng(seed)
    out = np.empty((N_SIM, 2))
    for s in range(N_SIM):
        out[s] = paired_bootstrap_ci(pool[rng.integers(0, pool.size, size)])
    return out


def permutation_fpr(pool: np.ndarray, size: int, seed: int, alpha: float = 0.05) -> float:
    """Rejection rate of the sign-flip permutation test on experiments drawn from a null pool."""
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(N_SIM_PERM):
        d = pool[rng.integers(0, pool.size, size)]
        hits += paired_sign_flip_pvalue(d, resamples=PERM_FLIPS, seed=seed + s) <= alpha
    return hits / N_SIM_PERM


def rejection_rate(cis: np.ndarray, shift: float = 0.0) -> float:
    lo, hi = cis[:, 0] + shift, cis[:, 1] + shift  # percentile CI is shift-equivariant
    return float(np.mean((lo > 0) | (hi < 0)))


def mde(cis: np.ndarray, target: float = 0.8, step: float = 0.01) -> float:
    shift = 0.0
    while rejection_rate(cis, shift) < target:
        shift += step
    return shift


def fmt_p(p: float) -> str:
    """The smallest attainable Monte Carlo p is 1/(flips+1), so never print 0."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--aa", required=True, help="run dir with proposer=random (A/A pool)")
    ap.add_argument("--es", required=True, help="run dir with proposer=es (same seed/scenarios)")
    ap.add_argument("--power", action="append", default=[], help="RUN_DIR:MODE_A:MODE_B (ES run), extra power analysis")
    ap.add_argument("--control", action="append", default=[], help="LABEL=HIGH_RUN_DIR,LOW_RUN_DIR")
    ap.add_argument("--prefix-check", action="append", default=[], help="SHORT_RUN_DIR:LONG_RUN_DIR")
    args = ap.parse_args()

    aa, es = Run(args.aa), Run(args.es)
    assert aa.proposer == "random" and es.proposer == "es", "wrong proposer for --aa / --es"
    assert aa.metric == es.metric
    print(f"# 统计功效分析（backend=toy，主指标 `{aa.metric}`）\n")
    print(f"- A/A 池：`{aa.name}`（proposer=random，R={aa.n_rep}）")
    print(f"- ES 池：`{es.name}`（proposer=es，R={es.n_rep}）")
    print(f"- CI 方法与报告一致：百分位 bootstrap，{BOOTSTRAP_RESAMPLES} 次重采样，固定种子 {BOOTSTRAP_SEED}")
    print(f"- 模拟：每个设计规模 {N_SIM} 次模拟实验，从池中有放回抽取 replicate 级配对差（种子 {SIM_SEED}）\n")

    for short, long in (s.split(":") for s in args.prefix_check):
        a, b = RunStore.open(short), RunStore.open(long)
        same = all(a.read_json(f) == b.read_json(f) for f in a.selected_files())
        print(f"- 一致性检查：`{Path(short).name}` 的 {len(a.selected_files())} 个 selected 文件"
              f"与 `{Path(long).name}` 中同名文件{'完全一致' if same else '不一致'}")
    print()

    # 1. A/A
    d_aa = aa.scores("selected:independent") - aa.scores("selected:single")
    lo90, hi90 = paired_bootstrap_ci(d_aa, alpha=0.10)
    margin = max(abs(lo90), abs(hi90))
    t_aa = aa.final_train_best("independent") - aa.final_train_best("single")
    print("## 1. A/A 检查（random：single 与 independent 在分布上等价）\n")
    print(f"- 测试分配对差 independent − single：{describe(d_aa)}")
    print(f"- 符号翻转置换检验（均值 = 0）：p = {paired_sign_flip_pvalue(d_aa):.3f}")
    print(f"- 90% CI [{lo90:+.3f}, {hi90:+.3f}]：等价性（TOST 形式）在 ±{margin:.2f} 分的界值内成立")
    print(f"- 训练集最终 best-so-far 配对差：{describe(t_aa)}")
    print()

    # 2. calibration
    pool_aa = d_aa - d_aa.mean()
    print("## 2. 校准：真实差为 0 时的假阳性率（名义 0.05）\n")
    rows = []
    cis_by_size = {}
    for size in DESIGN_SIZES:
        cis_by_size[size] = simulate_cis(pool_aa, size, SIM_SEED + size)
        fpr = rejection_rate(cis_by_size[size])
        mc_se = math.sqrt(fpr * (1 - fpr) / N_SIM)
        perm = permutation_fpr(pool_aa, size, SIM_SEED + 500 + size)
        perm_se = math.sqrt(perm * (1 - perm) / N_SIM_PERM)
        rows.append([size, f"{fpr:.3f} ± {mc_se:.3f}", f"{perm:.3f} ± {perm_se:.3f}"])
    print(table(["R", f"bootstrap CI 不含 0（{N_SIM} 次模拟）", f"置换检验 p ≤ 0.05（{N_SIM_PERM} 次模拟）"], rows))
    print(
        "\n模拟实验从去中心化的 A/A 配对差中有放回抽取，因此真实效应为 0；"
        "± 为蒙特卡洛标准误。"
    )
    print()

    # 3. ES power
    print("## 3. ES 下的功效\n")
    entries = [(es, "independent", "single")]
    for spec in args.power:
        run_dir, a_mode, b_mode = spec.rsplit(":", 2)
        entries.append((Run(run_dir), a_mode, b_mode))
    for k, (run, a_mode, b_mode) in enumerate(entries):
        assert run.proposer == "es", f"{run.name}: --power needs an ES run"
        d = run.scores(f"selected:{a_mode}") - run.scores(f"selected:{b_mode}")
        sd = d.std(ddof=1)
        pool = d - d.mean()
        mean_score = float(np.mean(run.scores(f"selected:{b_mode}")))
        print(f"### {a_mode} − {b_mode}（`{run.name}`）\n")
        print(f"- 观测配对差：{describe(d)}")
        print(f"- 符号翻转置换检验：p = {paired_sign_flip_pvalue(d):.3f}")
        print(f"- {b_mode} 选定策略平均测试分 {mean_score:.3f}（用于把 MDE 换算成相对值）\n")
        rows = []
        for size in DESIGN_SIZES:
            cis = simulate_cis(pool, size, SIM_SEED + 1000 + 100 * k + size)
            m = mde(cis)
            analytic = (Z_975 + Z_80) * sd / math.sqrt(size)
            rows.append(
                [
                    size,
                    f"{rejection_rate(cis):.3f}",
                    *(f"{rejection_rate(cis, s):.2f}" for s in (1.0, 2.0, 3.0, 5.0)),
                    f"{m:.2f}（{100 * m / mean_score:.1f}%）",
                    f"{analytic:.2f}",
                ]
            )
        print(table(["R", "δ=0 拒绝率", "功效 δ=1", "δ=2", "δ=3", "δ=5", "MDE（80% 功效）", "正态近似 MDE"], rows))
        print()
    print("δ 为加在配对差上的真实效应（分）；MDE 为模拟功效首次达到 0.8 的 δ（步长 0.01）。\n")

    # 4. positive controls
    if args.control:
        print("## 4. 阳性对照：共享场景的两组运行之间的配对效应\n")
        rows = []
        for spec in args.control:
            label, dirs = spec.split("=", 1)
            high_dir, low_dir = dirs.split(",")
            high, low = Run(high_dir), Run(low_dir)
            assert same_test_scenarios(high, low), f"{label}: runs do not share test scenarios"
            for group in ("selected:single", "selected:independent"):
                d = high.scores(group) - low.scores(group)
                ci = paired_bootstrap_ci(d)
                power20 = rejection_rate(simulate_cis(d, 20, SIM_SEED + 2000))
                rows.append(
                    [
                        label,
                        group,
                        f"{d.mean():+.3f}",
                        f"{d.std(ddof=1):.3f}",
                        f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                        fmt_p(paired_sign_flip_pvalue(d)),
                        f"{power20:.2f}",
                    ]
                )
        print(table(["对照", "组", "均值差", "差的标准差", f"95% CI（R={high.n_rep}）", "置换 p", "R=20 时检出率"], rows))
        print()


if __name__ == "__main__":
    main()
