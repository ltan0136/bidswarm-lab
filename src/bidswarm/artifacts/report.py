"""Markdown report and summary CSV, built only from run artifacts.

Nothing here re-runs episodes or reads in-memory state: deleting report.md and
calling ``write_report`` again reproduces it byte for byte.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..evaluation.selection import candidate_fitness
from ..types import MODES, PHASES
from .store import (
    ANYTIME,
    BUDGET,
    CANDIDATES,
    EPISODES,
    MANIFEST,
    REPORT,
    SUMMARY,
    TEST_EPISODES,
    RunStore,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261030
BOOTSTRAP_ALPHA = 0.05

SUMMARY_COLUMNS = (
    "replicate",
    "master_seed",
    "policy_group",
    "status",
    "candidate_id",
    "spec_hash",
    "train_fitness",
    "validation_score",
    "test_score_expected",
    "test_score_realized",
    "n_test_episodes",
    "n_test_failed",
)


# ------------------------------------------------------------------ statistics
def _mean(xs: Sequence[float]) -> float | None:
    return math.fsum(xs) / len(xs) if xs else None


def _std(xs: Sequence[float]) -> float | None:
    """Sample standard deviation (ddof=1); None for fewer than two values."""
    if len(xs) < 2:
        return None
    m = math.fsum(xs) / len(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_bootstrap_ci(
    diffs: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean paired difference (resampling replicates)."""
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(resamples, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _fmt(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _mean_std(xs: Sequence[float]) -> str:
    if not xs:
        return "—"
    return f"{_fmt(_mean(xs))} ± {_fmt(_std(xs))}"


# ------------------------------------------------------------------ artifact views
class _RunView:
    def __init__(self, store: RunStore) -> None:
        self.store = store
        self.manifest = store.read_json(MANIFEST)
        self.config = self.manifest["config"]
        self.metric = self.config["search"]["fitness_metric"]
        self.modes = [m for m in MODES if m in self.config["search"]["modes"]]
        self.budget = store.read_json(BUDGET)
        self.candidates = store.read_jsonl(CANDIDATES)
        self.anytime = store.read_jsonl(ANYTIME)
        self.episodes = store.read_jsonl(EPISODES)
        self.selected = [store.read_json(f) for f in store.selected_files()]
        self.has_test = store.exists(TEST_EPISODES)
        self.test_episodes = store.read_jsonl(TEST_EPISODES) if self.has_test else []
        self.test = self._test_scores()

    def _test_scores(self) -> dict[tuple[int, str], dict[str, Any]]:
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for e in self.test_episodes:
            grouped[(e["replicate"], e["group"])].append(e)
        out = {}
        for key, eps in grouped.items():
            out[key] = {
                "score_expected": candidate_fitness(eps, "score_expected"),
                "score_realized": candidate_fitness(eps, "score_realized"),
                "n": len(eps),
                "n_failed": sum(e["status"] == "failed" for e in eps),
                "spec_hash": eps[0]["spec_hash"],
            }
        return out

    def test_labels(self) -> list[str]:
        labels: list[str] = []
        for e in self.test_episodes:
            if e["group"] not in labels:
                labels.append(e["group"])
        return labels

    def baseline_labels(self) -> list[str]:
        return [lab for lab in self.test_labels() if lab.startswith("baseline:")]

    def selection(self, replicate: int, mode: str) -> dict[str, Any] | None:
        for rec in self.selected:
            if rec["replicate"] == replicate and rec["mode"] == mode:
                return rec
        return None

    def test_value(self, replicate: int, label: str, metric: str) -> float | None:
        row = self.test.get((replicate, label))
        return None if row is None else row[metric]


# ------------------------------------------------------------------ summary.csv
def build_summary_rows(store: RunStore) -> list[dict[str, Any]]:
    view = _RunView(store)
    seeds = view.manifest["master_seeds"]
    rows = []
    for r, seed in enumerate(seeds):
        for mode in view.modes:
            rec = view.selection(r, mode)
            label = f"selected:{mode}"
            test = view.test.get((r, label), {})
            rows.append(
                {
                    "replicate": r,
                    "master_seed": seed,
                    "policy_group": label,
                    "status": rec["status"] if rec else "missing",
                    "candidate_id": rec["candidate_id"] if rec else None,
                    "spec_hash": rec["spec_hash"] if rec else None,
                    "train_fitness": rec["train_fitness"] if rec else None,
                    "validation_score": rec["validation_score"] if rec else None,
                    "test_score_expected": test.get("score_expected"),
                    "test_score_realized": test.get("score_realized"),
                    "n_test_episodes": test.get("n"),
                    "n_test_failed": test.get("n_failed"),
                }
            )
        for label in view.baseline_labels():
            test = view.test.get((r, label), {})
            rows.append(
                {
                    "replicate": r,
                    "master_seed": seed,
                    "policy_group": label,
                    "status": "baseline_untuned",
                    "candidate_id": None,
                    "spec_hash": test.get("spec_hash"),
                    "train_fitness": None,
                    "validation_score": None,
                    "test_score_expected": test.get("score_expected"),
                    "test_score_realized": test.get("score_realized"),
                    "n_test_episodes": test.get("n"),
                    "n_test_failed": test.get("n_failed"),
                }
            )
    return rows


def write_summary(store: RunStore) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in build_summary_rows(store):
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    store.write_text(SUMMARY, buf.getvalue())


# ------------------------------------------------------------------ report.md
def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return lines


def _disclaimer(view: _RunView) -> str:
    m = view.manifest
    return (
        f"> **固定声明**：backend={m['backend']}，purpose={m['purpose']}，"
        f"score_formula={m['score_formula']}。本报告的全部数字来自内置 toy 环境，"
        "只用于检查实验框架是否正确运行，**不代表 AuctionNet 上的表现，也不代表任何真实广告收益**。"
        f"AuctionNet 后端状态：integration_status={m['auctionnet_integration_status']}。"
    )


def _section_run_info(view: _RunView) -> list[str]:
    m, c = view.manifest, view.config
    s = c["search"]
    git = m["git"]
    commit = git["commit"] or "无（git 不可用或仓库尚无提交）"
    if git["commit"] and git["dirty"]:
        commit += "（工作区有未提交改动）"
    lines = ["## 1. 运行信息", ""]
    lines += [
        f"- run 目录：`{m['run_name']}`，创建时间（UTC）：{m['created_at_utc']}",
        f"- git commit：{commit}",
        f"- Python {m['python_version']}，numpy {m['numpy_version']}，bidswarm {m['bidswarm_version']}",
        f"- backend={m['backend']}，integration_status={m['integration_status']}",
        f"- replicate 数 R={c['experiment']['replicates']}（master seed {m['master_seeds'][0]} 起连续递增）",
        f"- 模式：{', '.join(view.modes)}；策略模板：{s['policy']}；proposer：{s['proposer']}；"
        f"每轮候选数 k={s['candidates_per_round']}；independent 搜索者数 N={s['independent_workers']}",
        f"- 训练预算：每个 (replicate, mode) {s['train_episode_budget']} 个训练 episode；"
        f"场景数 train/validation/test = {c['splits']['train']}/{c['splits']['validation']}/{c['splits']['test']}；"
        f"top_k={c['selection']['top_k']}",
        f"- 主指标（fitness、验证分、测试分）：`{view.metric}`",
    ]
    plans = m["budget_plans"]
    lines += ["", "预算换算（配置校验时已检查整除）：", ""]
    lines += _table(
        ["模式", "搜索者数", "候选总数", "每个搜索者候选数", "轮数", "每轮 episode 数"],
        [
            [
                mode,
                plans[mode]["n_workers"],
                plans[mode]["candidates_total"],
                plans[mode]["candidates_per_worker"],
                plans[mode]["rounds"],
                plans[mode]["episodes_per_round"],
            ]
            for mode in view.modes
        ],
    )
    return lines


def _section_test(view: _RunView) -> list[str]:
    lines = ["## 2. 测试集得分（各模式选定策略）", ""]
    if not view.has_test:
        return lines + ["尚未执行 `evaluate`，暂无测试结果。`run` 命令按协议不执行测试。"]
    lines.append("每个 replicate 先对测试场景取均值，下表再对 replicate 取均值 ± 样本标准差。")
    lines.append("")
    rows = []
    for mode in view.modes:
        label = f"selected:{mode}"
        exp = [v for r in range(len(view.manifest["master_seeds"])) if (v := view.test_value(r, label, "score_expected")) is not None]
        real = [v for r in range(len(view.manifest["master_seeds"])) if (v := view.test_value(r, label, "score_realized")) is not None]
        rows.append([mode, len(exp), _mean_std(exp), _mean_std(real)])
    return lines + _table(["模式", "n", "score_expected", "score_realized"], rows)


def _paired_diffs(view: _RunView, metric: str) -> list[float]:
    diffs = []
    for r in range(len(view.manifest["master_seeds"])):
        a = view.test_value(r, "selected:independent", metric)
        b = view.test_value(r, "selected:single", metric)
        if a is not None and b is not None:
            diffs.append(a - b)
    return diffs


def _section_paired(view: _RunView) -> list[str]:
    lines = ["## 3. 配对比较：independent − single", ""]
    if not ("single" in view.modes and "independent" in view.modes):
        return lines + ["本次运行未同时包含 single 与 independent，无配对比较。"]
    if not view.has_test:
        return lines + ["尚未执行 `evaluate`，暂无测试结果。"]
    lines.append(
        f"同一 replicate 内两模式共用同一组场景，差值按 replicate 配对。"
        f"置信区间为百分位 bootstrap（对 replicate 重采样 {BOOTSTRAP_RESAMPLES} 次，固定种子 {BOOTSTRAP_SEED}）。"
    )
    lines.append("")
    rows = []
    primary_ci = None
    primary_n = 0
    for metric in (view.metric, *[m for m in ("score_expected", "score_realized") if m != view.metric]):
        diffs = _paired_diffs(view, metric)
        ci = paired_bootstrap_ci(diffs)
        if metric == view.metric:
            primary_ci, primary_n = ci, len(diffs)
        tag = "（主指标）" if metric == view.metric else ""
        rows.append(
            [
                f"{metric}{tag}",
                len(diffs),
                _fmt(_mean(diffs)),
                "—" if ci is None else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]",
            ]
        )
    lines += _table(["指标", "配对数 n", "差值均值", "95% bootstrap CI"], rows)
    lines.append("")
    if primary_ci is None:
        verdict = f"配对样本不足（n={primary_n} < 2），无法给出置信区间。"
    elif primary_ci[0] > 0:
        verdict = "主指标的 95% CI 不含 0 且为正：在本 toy 设置下，independent 的测试分高于 single。"
    elif primary_ci[1] < 0:
        verdict = "主指标的 95% CI 不含 0 且为负：在本 toy 设置下，independent 的测试分低于 single。"
    else:
        verdict = "主指标的 95% CI 包含 0：在本 toy 设置下未观察到 independent 与 single 之间可分辨的差异。"
    lines.append(f"结论（自动生成，仅描述上表）：{verdict}")
    if primary_n < 10:
        lines.append("")
        lines.append(f"注意：配对数只有 {primary_n}，bootstrap 区间很粗糙，不宜据此下结论。")
    if view.config["search"]["proposer"] == "random":
        lines.append("")
        lines.append(
            "注意：proposer=random 时，single 与 independent 的候选在分布上等价，"
            "此比较只是正确性检查，不反映搜索方式的优劣。"
        )
    return lines


def _section_winners_curse(view: _RunView) -> list[str]:
    lines = ["## 4. 验证分与测试分差距（赢家诅咒）", ""]
    if not view.has_test:
        return lines + ["尚未执行 `evaluate`，暂无测试结果。"]
    lines.append(f"对选定策略：训练 fitness、验证分、测试分均为 `{view.metric}`；差距 = 验证分 − 测试分。")
    lines.append("")
    rows = []
    for mode in view.modes:
        train, val, test, gap = [], [], [], []
        for r in range(len(view.manifest["master_seeds"])):
            rec = view.selection(r, mode)
            t = view.test_value(r, f"selected:{mode}", view.metric)
            if rec is None or rec["status"] != "selected" or t is None:
                continue
            train.append(rec["train_fitness"])
            val.append(rec["validation_score"])
            test.append(t)
            gap.append(rec["validation_score"] - t)
        rows.append([mode, len(gap), _mean_std(train), _mean_std(val), _mean_std(test), _mean_std(gap)])
    return lines + _table(["模式", "n", "训练 fitness", "验证分", "测试分", "验证 − 测试"], rows)


def _section_baselines(view: _RunView) -> list[str]:
    lines = ["## 5. 未调优默认基线", ""]
    if not view.has_test:
        return lines + ["尚未执行 `evaluate`，暂无基线结果。"]
    lines.append(
        "以下基线使用模板默认参数，**没有经过任何调优，不是强基线**，只作为同一测试场景上的参照。"
    )
    lines.append("")
    specs = {}
    for e in view.test_episodes:
        specs.setdefault(e["group"], e["spec_hash"])
    rows = []
    n_rep = len(view.manifest["master_seeds"])
    for label in view.baseline_labels():
        exp = [v for r in range(n_rep) if (v := view.test_value(r, label, "score_expected")) is not None]
        real = [v for r in range(n_rep) if (v := view.test_value(r, label, "score_realized")) is not None]
        rows.append([label, f"`{specs[label][:12]}`", len(exp), _mean_std(exp), _mean_std(real)])
    return lines + _table(["基线", "spec_hash 前缀", "n", "score_expected", "score_realized"], rows)


def _section_details(view: _RunView) -> list[str]:
    lines = ["## 6. 各 replicate 选定策略明细", ""]
    rows = []
    for r in range(len(view.manifest["master_seeds"])):
        for mode in view.modes:
            rec = view.selection(r, mode)
            if rec is None:
                rows.append([r, mode, "（缺失）", "—", "—", "—", "—"])
                continue
            rows.append(
                [
                    r,
                    mode,
                    rec["candidate_id"] or rec["status"],
                    f"`{rec['spec_hash'][:12]}`" if rec["spec_hash"] else "—",
                    _fmt(rec["train_fitness"]),
                    _fmt(rec["validation_score"]),
                    _fmt(view.test_value(r, f"selected:{mode}", view.metric)),
                ]
            )
    return lines + _table(["replicate", "模式", "选定候选", "spec_hash 前缀", "训练", "验证", "测试"], rows)


def _section_anytime(view: _RunView) -> list[str]:
    lines = ["## 7. Anytime 检查点（训练 best-so-far fitness）", ""]
    rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in view.anytime:
        if a["scope"] == "mode":
            rows_by_mode[a["mode"]].append(a)
    used_sets = [set(a["train_episodes_used"] for a in rows_by_mode[m]) for m in view.modes]
    checkpoints = sorted(set.intersection(*used_sets)) if used_sets else []
    lines.append(
        "检查点取所有模式都恰好完成一轮的训练 episode 数（各模式轮次粒度的公共点）。"
        "表中为各 replicate 模式整体 best-so-far 的均值 ± 标准差；无有效候选的 replicate 不计入。"
    )
    lines.append("")
    table_rows = []
    for c in checkpoints:
        row: list[Any] = [c]
        for m in view.modes:
            vals = [a["best_so_far"] for a in rows_by_mode[m] if a["train_episodes_used"] == c]
            valid = [v for v in vals if v is not None]
            cell = _mean_std(valid)
            if len(valid) != len(vals):
                cell += f"（{len(vals) - len(valid)} 个无有效候选）"
            row.append(cell)
        table_rows.append(row)
    return lines + _table(["已用训练 episode", *view.modes], table_rows)


def _section_budget(view: _RunView) -> list[str]:
    lines = ["## 8. 预算账本", ""]
    agg: dict[tuple[str, str], dict[str, int]] = {}
    for e in view.budget["entries"]:
        a = agg.setdefault((e["phase"], e["group"]), {"planned": 0, "used": 0, "failed": 0, "timeouts": 0, "retries": 0})
        for k in a:
            a[k] += e[k]
    order = {p: i for i, p in enumerate(PHASES)}
    keys = sorted(agg, key=lambda k: order[k[0]])  # stable: groups keep ledger order
    lines.append("单位：episode；对全部 replicate 求和。v1 不重试（retries 恒为 0）。")
    lines.append("")
    lines += _table(
        ["阶段", "组", "planned", "used", "failed", "timeout", "retries"],
        [[p, g, *(agg[(p, g)][k] for k in ("planned", "used", "failed", "timeouts", "retries"))] for p, g in keys],
    )
    train_ok = all(e["used"] == e["planned"] for e in view.budget["entries"] if e["phase"] == "train")
    over = [e for e in view.budget["entries"] if e["used"] > e["planned"]]
    lines.append("")
    lines.append(f"- 训练阶段 used == planned：{'是' if train_ok else '否'}；used > planned 的条目数：{len(over)}")
    cand_rows = []
    for m in view.modes:
        cs = [c for c in view.candidates if c["mode"] == m]
        cand_rows.append([m, len(cs), sum(c["fitness"] is None for c in cs)])
    lines.append("")
    lines += _table(["模式", "候选数", "失败候选（fitness=null）"], cand_rows)
    errors = Counter(
        e["error"] for e in [*view.episodes, *view.test_episodes] if e["status"] == "failed"
    )
    timeouts = sum(e["status"] == "timeout" for e in [*view.episodes, *view.test_episodes])
    lines.append("")
    lines.append(
        f"- 失败 episode：{sum(errors.values())}；软超时 episode：{timeouts}"
        f"（超时阈值 {view.config['runtime']['episode_timeout_s']} 秒；只做标记，不强杀进程，指标照常保留）"
    )
    for err, n in errors.most_common(5):
        lines.append(f"  - {n} × `{err}`")
    return lines


def _section_artifacts(view: _RunView) -> list[str]:
    files = [
        "manifest.json",
        "scenarios.json",
        "candidates.jsonl",
        "episodes.jsonl",
        "anytime.jsonl",
        "budget.json",
        "selected/*.json",
        "summary.csv",
        "test_episodes.jsonl" + ("" if view.has_test else "（尚未生成）"),
        "report.md",
    ]
    return ["## 9. 产物", "", "本报告只读取以下产物生成（`python -m bidswarm report --run-dir <dir>` 可重建）：", ""] + [
        f"- `{f}`" for f in files
    ]


def build_report(store: RunStore) -> str:
    view = _RunView(store)
    parts = [
        [f"# BidSwarm Lab 实验报告：{view.config['experiment']['name']}", "", _disclaimer(view)],
        _section_run_info(view),
        _section_test(view),
        _section_paired(view),
        _section_winners_curse(view),
        _section_baselines(view),
        _section_details(view),
        _section_anytime(view),
        _section_budget(view),
        _section_artifacts(view),
        [_disclaimer(view)],
    ]
    return "\n\n".join("\n".join(p) for p in parts) + "\n"


def write_report(store: RunStore) -> None:
    store.write_text(REPORT, build_report(store))
