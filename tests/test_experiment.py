"""End-to-end protocol tests on small toy runs.

Covers determinism, serial/parallel equality, budget accounting, split and
searcher isolation, strict JSON, failure accounting, test isolation and
report reconstruction.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from bidswarm.artifacts.store import RunStore
from bidswarm.cli import main
from bidswarm.experiment import evaluate_run, run_experiment
from bidswarm.kernel import ParamDef, registry
from bidswarm.kernel.policies import BasePolicy
from bidswarm.search.worker import SearchWorker
from bidswarm.types import ArtifactError

from .conftest import read_jsonl, small_config, small_config_dict, write_toml

CORE_FILES = ("candidates.jsonl", "anytime.jsonl", "budget.json", "scenarios.json", "summary.csv")


def _episodes_without_timing(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    for r in rows:
        r.pop("wall_time_s")
    return rows


def _assert_same_core_results(a: Path, b: Path) -> None:
    for name in CORE_FILES:
        assert (a / name).read_text() == (b / name).read_text(), name
    assert sorted(p.name for p in (a / "selected").iterdir()) == sorted(p.name for p in (b / "selected").iterdir())
    for p in (a / "selected").iterdir():
        assert p.read_text() == (b / "selected" / p.name).read_text(), p.name
    assert _episodes_without_timing(a / "episodes.jsonl") == _episodes_without_timing(b / "episodes.jsonl")


# ------------------------------------------------------------------ artifacts & test isolation
def test_run_writes_artifacts_but_never_touches_test_split(small_run_dir: Path):
    for name in ("manifest.json", "scenarios.json", "candidates.jsonl", "episodes.jsonl", "anytime.jsonl",
                 "budget.json", "summary.csv", "report.md"):  # fmt: skip
        assert (small_run_dir / name).is_file(), name
    assert not (small_run_dir / "test_episodes.jsonl").exists()
    assert sorted(p.name for p in (small_run_dir / "selected").iterdir()) == [
        "00_independent.json", "00_single.json", "01_independent.json", "01_single.json",
    ]  # fmt: skip

    scenarios = json.loads((small_run_dir / "scenarios.json").read_text())["replicates"]
    test_ids = {sc["scenario_id"] for rep in scenarios for sc in rep["test"]}
    episodes = read_jsonl(small_run_dir / "episodes.jsonl")
    assert {e["phase"] for e in episodes} == {"train", "validation"}
    assert not test_ids & {e["scenario_id"] for e in episodes}
    for e in episodes:
        assert e["split"] == e["phase"]
    assert "尚未执行 `evaluate`" in (small_run_dir / "report.md").read_text()

    manifest = json.loads((small_run_dir / "manifest.json").read_text())
    assert manifest["backend"] == "toy"
    assert manifest["auctionnet_integration_status"] == "not_integrated"
    assert manifest["purpose"] == "framework_smoke_test"
    assert manifest["python_version"] and manifest["numpy_version"]


def test_evaluate_writes_test_episodes_with_baselines(evaluated_run_dir: Path):
    rows = read_jsonl(evaluated_run_dir / "test_episodes.jsonl")
    assert {r["phase"] for r in rows} == {"test"}
    assert {r["split"] for r in rows} == {"test"}
    groups = {r["group"] for r in rows}
    assert groups == {
        "selected:single", "selected:independent",
        "baseline:fixed_rate_default", "baseline:dual_control_default",
    }  # fmt: skip
    report = (evaluated_run_dir / "report.md").read_text()
    assert "不是强基线" in report and "independent − single" in report
    assert "不代表 AuctionNet 上的表现" in report


def test_evaluate_refuses_to_rerun_without_force(run_copy: Path):
    with pytest.raises(ArtifactError, match="already exists"):
        evaluate_run(run_copy, serial=True)
    before = _episodes_without_timing(run_copy / "test_episodes.jsonl")
    evaluate_run(run_copy, serial=True, force=True)
    assert _episodes_without_timing(run_copy / "test_episodes.jsonl") == before


def test_evaluate_rejects_tampered_selection(run_copy: Path):
    path = run_copy / "selected" / "00_single.json"
    rec = json.loads(path.read_text())
    key = next(iter(rec["spec"]["params"]))
    rec["spec"]["params"][key] = rec["spec"]["params"][key] * 0.5
    path.write_text(json.dumps(rec))
    with pytest.raises(ArtifactError, match="spec_hash"):
        evaluate_run(run_copy, serial=True, force=True)


# ------------------------------------------------------------------ determinism
def test_same_seed_gives_identical_core_results(small_run_dir: Path, tmp_path: Path):
    again = run_experiment(small_config(), out_root=tmp_path, serial=True)
    _assert_same_core_results(small_run_dir, again)


def test_serial_and_parallel_results_are_identical(small_run_dir: Path, tmp_path: Path):
    parallel = run_experiment(small_config(), out_root=tmp_path, serial=False, max_workers=2)
    _assert_same_core_results(small_run_dir, parallel)
    assert json.loads((parallel / "manifest.json").read_text())["executor"]["serial"] is False


def test_different_seed_changes_results(small_run_dir: Path, tmp_path: Path):
    other = run_experiment(small_config(experiment={"seed": 8}), out_root=tmp_path, serial=True)
    assert (other / "candidates.jsonl").read_text() != (small_run_dir / "candidates.jsonl").read_text()


# ------------------------------------------------------------------ budget
def test_budget_ledger_and_spend(evaluated_run_dir: Path):
    budget = json.loads((evaluated_run_dir / "budget.json").read_text())
    cfg = small_config()
    for e in budget["entries"]:
        assert e["used"] <= e["planned"]
        assert e["retries"] == 0
        if e["phase"] == "train":
            assert e["planned"] == cfg.search.train_episode_budget
            assert e["used"] == e["planned"]
    counts = defaultdict(int)
    for ep in read_jsonl(evaluated_run_dir / "episodes.jsonl"):
        counts[(ep["replicate"], ep["group"], ep["phase"])] += 1
        if ep["status"] != "failed":
            assert ep["cost"] <= ep["budget"]
    for e in budget["entries"]:
        if e["phase"] != "test":
            assert counts[(e["replicate"], e["group"], e["phase"])] == e["used"]

    candidates = read_jsonl(evaluated_run_dir / "candidates.jsonl")
    per_pipeline = defaultdict(int)
    for c in candidates:
        per_pipeline[(c["replicate"], c["mode"])] += 1
    assert set(per_pipeline.values()) == {cfg.search.train_episode_budget // cfg.splits.train}


def test_anytime_records_per_round(small_run_dir: Path):
    rows = read_jsonl(small_run_dir / "anytime.jsonl")
    mode_rows = [r for r in rows if r["scope"] == "mode"]
    single = [r for r in mode_rows if r["mode"] == "single" and r["replicate"] == 0]
    independent = [r for r in mode_rows if r["mode"] == "independent" and r["replicate"] == 0]
    assert [r["train_episodes_used"] for r in single] == [4, 8, 12, 16]  # k=2 x 2 scenarios per round
    assert [r["train_episodes_used"] for r in independent] == [8, 16]  # 2 workers x k=2 x 2 scenarios
    best = [r["best_so_far"] for r in single]
    assert best == sorted(best)  # best-so-far is monotone
    assert len([r for r in rows if r["scope"] == "worker" and r["mode"] == "independent"]) == 2 * 2 * 2


# ------------------------------------------------------------------ isolation
def test_splits_are_disjoint_and_paired(small_run_dir: Path):
    reps = json.loads((small_run_dir / "scenarios.json").read_text())["replicates"]
    for rep in reps:
        ids = [sc["scenario_id"] for split in ("train", "validation", "test") for sc in rep[split]]
        seeds = [sc["seed"] for split in ("train", "validation", "test") for sc in rep[split]]
        assert len(ids) == len(set(ids)) and len(seeds) == len(set(seeds))
    episodes = read_jsonl(small_run_dir / "episodes.jsonl")
    for r in (0, 1):
        by_mode = defaultdict(set)
        for e in episodes:
            if e["replicate"] == r and e["phase"] == "train":
                by_mode[e["group"]].add(e["scenario_id"])
        assert by_mode["single"] == by_mode["independent"]  # paired design


def test_info_sources_only_reference_own_earlier_candidates(small_run_dir: Path):
    candidates = read_jsonl(small_run_dir / "candidates.jsonl")
    by_id = {c["candidate_id"]: c for c in candidates}
    for c in candidates:
        for src in c["info_sources"]:
            other = by_id[src]
            assert (other["replicate"], other["mode"], other["worker_id"]) == (c["replicate"], c["mode"], c["worker_id"])
            assert other["round"] < c["round"]
        if c["parent_id"] is not None:
            assert c["parent_id"] in c["info_sources"]
            assert c["operator"] == "gaussian_mutation"
        else:
            assert c["operator"] == "init_random"


def test_workers_observe_only_their_own_candidates(monkeypatch, tmp_path: Path):
    seen = []
    original = SearchWorker.observe

    def spy(self, results):
        seen.append((self.prefix, [r["candidate_id"] for r in results]))
        return original(self, results)

    monkeypatch.setattr(SearchWorker, "observe", spy)
    run_experiment(small_config(), out_root=tmp_path, serial=True)
    assert seen
    for prefix, ids in seen:
        assert ids and all(cid.startswith(prefix + "-c") for cid in ids)
    independent = {p for p, _ in seen if "-independent-" in p}
    assert len(independent) == 2 * 2  # 2 replicates x 2 workers, each observed separately


# ------------------------------------------------------------------ failures & strict JSON
class FlakyPolicy(BasePolicy):
    """Test-only template: raises for half of its parameter range."""

    name = "flaky_test_policy"
    PARAM_DEFS = (ParamDef("x", 0.0, 1.0, 0.25),)

    def _ratio(self, t):
        if self.params["x"] > 0.5:
            raise RuntimeError("flaky policy exploded")
        return 1.0


def test_raising_policy_is_failed_and_still_charged(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(registry.REGISTRY, FlakyPolicy.name, FlakyPolicy)
    cfg = small_config(search={"policy": FlakyPolicy.name, "proposer": "random"}, selection={"top_k": 1})
    run_dir = run_experiment(cfg, out_root=tmp_path, serial=True)  # serial: registry patch is in-process
    candidates = read_jsonl(run_dir / "candidates.jsonl")
    failed = [c for c in candidates if c["params"]["x"] > 0.5]
    assert failed, "seeded random proposals should hit the failing region"
    for c in failed:
        assert c["fitness"] is None and c["status"] == "failed" and c["n_failed"] == c["n_episodes"]
    budget = json.loads((run_dir / "budget.json").read_text())
    train = [e for e in budget["entries"] if e["phase"] == "train"]
    assert all(e["used"] == e["planned"] for e in train)
    assert sum(e["failed"] for e in train) == sum(c["n_failed"] for c in candidates) > 0
    failed_ids = {c["candidate_id"] for c in failed}
    for p in (run_dir / "selected").iterdir():
        rec = json.loads(p.read_text())
        assert rec["candidate_id"] not in failed_ids
        assert all(row["candidate_id"] not in failed_ids for row in rec["shortlist"])
    report = (run_dir / "report.md").read_text()
    assert "flaky policy exploded" in report
    _assert_strict_json_dir(run_dir)


def _reject_constant(name: str):
    raise AssertionError(f"non-standard JSON constant {name}")


def _assert_strict_json_dir(run_dir: Path) -> None:
    for path in sorted(run_dir.rglob("*.json*")):
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text, path
        lines = text.splitlines() if path.suffix == ".jsonl" else [text]
        for line in lines:
            if line.strip():
                json.loads(line, parse_constant=_reject_constant)


def test_artifacts_are_strict_json(evaluated_run_dir: Path):
    _assert_strict_json_dir(evaluated_run_dir)
    episodes = read_jsonl(evaluated_run_dir / "episodes.jsonl") + read_jsonl(evaluated_run_dir / "test_episodes.jsonl")
    for e in episodes:
        assert e["backend"] == "toy" and e["purpose"] == "framework_smoke_test"
        assert e["score_formula"] == "auctionnet_neurips_replica"
        if e["status"] == "ok" and e["conversions"] == 0:
            assert e["cpa_realized"] is None and e["cpa_defined"] is False


# ------------------------------------------------------------------ report
def test_report_rebuilds_identically_from_artifacts(run_copy: Path):
    original = (run_copy / "report.md").read_text(encoding="utf-8")
    (run_copy / "report.md").unlink()
    assert main(["report", "--run-dir", str(run_copy)]) == 0
    assert (run_copy / "report.md").read_text(encoding="utf-8") == original


def test_report_numbers_come_from_artifacts(evaluated_run_dir: Path):
    """Recompute the paired difference from test_episodes.jsonl and find it in report.md."""
    rows = read_jsonl(evaluated_run_dir / "test_episodes.jsonl")
    per = defaultdict(list)
    for r in rows:
        per[(r["replicate"], r["group"])].append(r["score_expected"])
    diffs = [
        np.mean(per[(rep, "selected:independent")]) - np.mean(per[(rep, "selected:single")]) for rep in (0, 1)
    ]
    report = (evaluated_run_dir / "report.md").read_text(encoding="utf-8")
    assert f"| score_expected（主指标） | 2 | {np.mean(diffs):.3f} |" in report


# ------------------------------------------------------------------ CLI round trip
def test_cli_run_evaluate_report(tmp_path: Path, capsys):
    config_path = write_toml(tmp_path / "small.toml", small_config_dict())
    out = tmp_path / "runs"
    assert main(["run", "--config", str(config_path), "--serial", "--out", str(out)]) == 0
    (run_dir,) = list(out.iterdir())
    assert run_dir.name.startswith("t_small-")
    assert main(["evaluate", "--run-dir", str(run_dir), "--serial"]) == 0
    assert main(["report", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "test_episodes.jsonl").is_file()
    assert main(["report", "--run-dir", str(tmp_path)]) == 2  # not a run directory
    assert RunStore.open(run_dir).read_json("manifest.json")["config_path"] == str(config_path)


# ------------------------------------------------------------------ soft timeout & split overlap
def test_soft_timeout_flags_but_keeps_metrics(tmp_path: Path):
    cfg = small_config(experiment={"replicates": 1}, runtime={"episode_timeout_s": 1e-9})
    run_dir = run_experiment(cfg, out_root=tmp_path, serial=True)
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    assert {e["status"] for e in episodes} == {"timeout"}
    assert all(e["score_expected"] is not None for e in episodes)  # flagged, not discarded
    assert all(c["fitness"] is not None for c in read_jsonl(run_dir / "candidates.jsonl"))
    budget = json.loads((run_dir / "budget.json").read_text())
    assert sum(e["timeouts"] for e in budget["entries"]) == len(episodes)
    assert budget["timeout_policy"].startswith("soft")


def test_overlapping_splits_are_rejected():
    from bidswarm.evaluation.splits import SplitOverlapError, SplitSet, check_disjoint

    a = {"scenario_id": "a", "seed": 1}
    with pytest.raises(SplitOverlapError, match="seed"):
        check_disjoint(SplitSet(0, 0, train=(a,), validation=({"scenario_id": "b", "seed": 1},), test=()))
    with pytest.raises(SplitOverlapError, match="id"):
        check_disjoint(SplitSet(0, 0, train=(a,), validation=(), test=({"scenario_id": "a", "seed": 2},)))


# ------------------------------------------------------------------ cooperative mode
def test_cooperative_mode_migrates_within_its_pipeline_only(tmp_path: Path):
    config = small_config(search={"modes": ["single", "independent", "cooperative"]})
    run_dir = run_experiment(config, out_root=tmp_path / "a", serial=True)
    deliveries = read_jsonl(run_dir / "exchanges.jsonl")
    candidates = {c["candidate_id"]: c for c in read_jsonl(run_dir / "candidates.jsonl")}

    # 2 rounds per worker: one migration after round 0, none after the last round.
    assert deliveries and {d["round"] for d in deliveries} == {0}
    assert {d["mode"] for d in deliveries} == {"cooperative"}
    received = defaultdict(set)
    for d in deliveries:
        sent = candidates[d["candidate_id"]]
        assert (sent["replicate"], sent["mode"], sent["worker_id"]) == (d["replicate"], "cooperative", d["from_worker"])
        assert d["to_worker"] == (d["from_worker"] + 1) % 2 and d["fitness"] == sent["fitness"]
        received[(d["replicate"], d["to_worker"])].add((d["candidate_id"], d["round"]))

    for c in candidates.values():
        own = (c["replicate"], c["mode"], c["worker_id"])
        for src in c["info_sources"]:
            other = candidates[src]
            if (other["replicate"], other["mode"], other["worker_id"]) == own:
                assert other["round"] < c["round"]
            else:  # only a card migrated to this worker before this round
                assert c["mode"] == "cooperative"
                assert any(cid == src and r < c["round"] for cid, r in received[(c["replicate"], c["worker_id"])])

    # Migration is free: every mode spends exactly the same training budget.
    train = [e for e in json.loads((run_dir / "budget.json").read_text())["entries"] if e["phase"] == "train"]
    assert {e["group"] for e in train} == {"single", "independent", "cooperative"}
    assert {(e["planned"], e["used"]) for e in train} == {(16, 16)}

    rerun = run_experiment(config, out_root=tmp_path / "b", serial=True)
    assert (run_dir / "exchanges.jsonl").read_text() == (rerun / "exchanges.jsonl").read_text()
    assert (run_dir / "candidates.jsonl").read_text() == (rerun / "candidates.jsonl").read_text()

    evaluate_run(run_dir, serial=True)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "### independent − single" in report and "### cooperative − independent" in report
    assert f"本次共 {len(deliveries)} 次送达" in report


def test_independent_runs_record_no_exchanges(small_run_dir: Path):
    assert (small_run_dir / "exchanges.jsonl").read_text() == ""


def test_cooperative_shares_independent_random_streams_until_migration(tmp_path: Path):
    config = small_config(search={"modes": ["independent", "cooperative"]})
    run_dir = run_experiment(config, out_root=tmp_path, serial=True)
    by_key = defaultdict(dict)
    for c in read_jsonl(run_dir / "candidates.jsonl"):
        by_key[(c["replicate"], c["worker_id"], c["round"], c["candidate_id"][-5:])][c["mode"]] = c["normalized"]
    first_round = [v for (_, _, round_, _), v in by_key.items() if round_ == 0]
    assert first_round and all(v["independent"] == v["cooperative"] for v in first_round)


def test_replicate_results_do_not_depend_on_number_of_replicates(small_run_dir: Path, tmp_path: Path):
    """Replicate r is a pure function of (config, master_seed r): a short run is a prefix of a long one."""
    short = run_experiment(small_config(experiment={"replicates": 1}), out_root=tmp_path, serial=True)
    for name in ("00_single.json", "00_independent.json"):
        assert (short / "selected" / name).read_text() == (small_run_dir / "selected" / name).read_text()
    long_candidates = [c for c in read_jsonl(small_run_dir / "candidates.jsonl") if c["replicate"] == 0]
    assert read_jsonl(short / "candidates.jsonl") == long_candidates
