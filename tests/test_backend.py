"""AuctionNet backend: config checks, loud setup failures, and the runner protocol (via a stub).

The real runner needs the pinned upstream checkout and its Python 3.9 venv; it
is exercised by tests/test_auctionnet_integration.py when those are available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bidswarm.cli import main
from bidswarm.config import load_config
from bidswarm.environments import AuctionNetRunnerError, AuctionNetSetupError, auctionnet, make_environment
from bidswarm.environments.toy import ToyEnvironment
from bidswarm.evaluation.splits import SplitOverlapError, SplitSet, check_disjoint
from bidswarm.experiment import evaluate_run, run_experiment
from bidswarm.kernel import default_spec, make_policy

from .conftest import CONFIG_DIR, small_config_dict

STUB = Path(__file__).with_name("auctionnet_stub_runner.py")
PINNED = auctionnet.PINNED_COMMIT


def env_cfg(**overrides):
    cfg = {"backend": "auctionnet", "repo_path": "/nonexistent/AuctionNet", "commit_sha": PINNED,
           "python_executable": sys.executable, "player_index": 1}
    cfg.update(overrides)
    return cfg


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Point the backend at the stub runner and skip the checkout check."""
    counter = tmp_path / "starts.txt"
    monkeypatch.setattr(auctionnet, "RUNNER_PATH", STUB)
    monkeypatch.setattr(auctionnet, "check_setup", lambda cfg: None)
    monkeypatch.setenv("STUB_COUNTER", str(counter))
    auctionnet._close_runners()
    yield counter
    auctionnet._close_runners()


def _episode(env, seed=12345, name="dual_control"):
    sc = env.make_scenario("m1-train-000", "train", 0, seed)
    policy = make_policy(default_spec(name), sc["budget"], sc["cpa_target"], sc["category"], 48)
    return sc, env.run_episode(sc, policy)


# --- configuration and setup -------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"evaluation_path": "offline"}, "not integrated"),
        ({"evaluation_path": "replay"}, "must be 'online'"),
        ({"commit_sha": "abc"}, "40-character"),
        ({"commit_sha": "0" * 40}, "not the verified commit"),
        ({"player_index": 48}, "player_index"),
        ({"player_index": True}, "player_index"),
        ({"num_time_steps": 24}, "num_time_steps"),
        ({"repo_path": ""}, "repo_path"),
        ({"startup_timeout_s": 0}, "startup_timeout_s"),
        ({"weights": "x"}, "unknown keys"),
    ],
)
def test_config_validation(overrides, message):
    with pytest.raises(auctionnet.ConfigError, match=message):
        auctionnet.validate_config(env_cfg(**overrides))


def test_config_defaults():
    cfg = auctionnet.validate_config({k: v for k, v in env_cfg().items() if k != "player_index"})
    assert (cfg["evaluation_path"], cfg["num_time_steps"], cfg["player_index"]) == ("online", 48, 0)


def test_missing_checkout_fails_loudly_and_never_falls_back():
    with pytest.raises(AuctionNetSetupError, match="no AuctionNet checkout"):
        make_environment(env_cfg())


def test_checkout_at_another_commit_is_rejected(tmp_path):
    import subprocess

    repo = tmp_path / "AuctionNet"
    (repo / "run").mkdir(parents=True)
    (repo / "run" / "run_test.py").write_text("")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
                   check=True)
    with pytest.raises(AuctionNetSetupError, match="expected"):
        make_environment(env_cfg(repo_path=str(repo)))


def test_example_config_fails_before_writing(tmp_path: Path, capsys):
    cfg = load_config(CONFIG_DIR / "auctionnet.example.toml")
    with pytest.raises(AuctionNetSetupError):
        run_experiment(cfg, out_root=tmp_path / "runs", serial=True)
    assert not (tmp_path / "runs").exists()
    out = tmp_path / "cli"
    code = main(["run", "--config", str(CONFIG_DIR / "auctionnet.example.toml"), "--serial", "--out", str(out)])
    assert code == 2 and "no AuctionNet checkout" in capsys.readouterr().err
    assert not out.exists()


# --- scenarios ---------------------------------------------------------------------------------


def test_scenarios_fix_the_advertiser_and_map_seeds_to_episodes(stub):
    env = make_environment(env_cfg())
    sc = env.make_scenario("m1-test-003", "test", 3, 2**40 + 17)
    assert sc["episode"] == (2**40 + 17) % auctionnet.EPISODE_SPACE
    assert (sc["budget"], sc["cpa_target"], sc["category"], sc["player_index"]) == (4350.0, 70.0, 0, 1)
    assert sc == env.make_scenario("m1-test-003", "test", 3, 2**40 + 17)


def test_splits_must_not_share_an_episode():
    def sc(split, i, seed, episode):
        return {"scenario_id": f"{split}-{i}", "seed": seed, "episode": episode}

    clash = SplitSet(replicate=0, master_seed=1, train=(sc("train", 0, 1, 5),), validation=(sc("validation", 0, 2, 5),),
                     test=(sc("test", 0, 3, 6),))
    with pytest.raises(SplitOverlapError, match="episode 5"):
        check_disjoint(clash)


# --- runner protocol (stub) --------------------------------------------------------------------


def test_episode_record_matches_the_toy_schema(stub):
    env = make_environment(env_cfg())
    sc, record = _episode(env)
    toy = ToyEnvironment()
    toy_sc = toy.make_scenario("x", "train", 0, 1)
    toy_record = toy.run_episode(toy_sc, make_policy(default_spec("dual_control"), toy_sc["budget"],
                                                     toy_sc["cpa_target"], toy_sc["category"], 48))
    assert set(toy_record) <= set(record)
    assert record["status"] == "ok" and record["backend"] == "auctionnet"
    assert record["purpose"] == "auctionnet_online_simulation"
    assert record["episode"] == sc["episode"] and record["upstream_commit"] == PINNED
    assert 0 < record["cost"] <= record["budget"]
    assert json.dumps(record, allow_nan=False)


def test_one_runner_per_process_is_reused(stub):
    env = make_environment(env_cfg())
    first = _episode(env, seed=1)[1]
    again = _episode(env, seed=1)[1]
    _episode(env, seed=2)
    assert stub.read_text().count("start") == 1
    assert {k: v for k, v in first.items() if k != "wall_time_s"} == {k: v for k, v in again.items() if k != "wall_time_s"}


def test_policy_error_is_a_failed_episode(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "policy_error")
    _, record = _episode(make_environment(env_cfg()))
    assert record["status"] == "failed" and "PolicyInputError" in record["error"]
    assert record["purpose"] == "auctionnet_online_simulation" and record["score_expected"] is None


@pytest.mark.parametrize("mode, message", [
    ("runner_error", "runner error"),
    ("bad_books", "kernel books"),
    ("bad_calls", "player bid"),
])
def test_runner_failures_and_broken_invariants_abort(stub, monkeypatch, mode, message):
    monkeypatch.setenv("STUB_MODE", mode)
    with pytest.raises(AuctionNetRunnerError, match=message):
        _episode(make_environment(env_cfg()))


@pytest.mark.parametrize("mode, error, message", [
    ("crash_on_start", AuctionNetRunnerError, "exited with code 3"),
    ("not_ready", AuctionNetSetupError, "stub refused"),
    ("wrong_budget", AuctionNetSetupError, "expected advertiser"),
])
def test_runner_start_up_is_checked(stub, monkeypatch, mode, error, message):
    monkeypatch.setenv("STUB_MODE", mode)
    with pytest.raises(error, match=message):
        _episode(make_environment(env_cfg()))


def test_runner_that_dies_mid_run_aborts(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "exit_after_ready")
    with pytest.raises(AuctionNetRunnerError, match="exited"):
        _episode(make_environment(env_cfg()))


def test_end_to_end_run_through_the_stub(stub, tmp_path):
    data = small_config_dict(
        experiment={"name": "an_stub", "seed": 3, "replicates": 1},
        search={"modes": ["single"], "train_episode_budget": 8, "candidates_per_round": 2},
        splits={"train": 2, "validation": 2, "test": 2},
        selection={"top_k": 2},
    )
    data["environment"] = env_cfg()
    from bidswarm.config import ExperimentConfig

    run_dir = run_experiment(ExperimentConfig.from_dict(data), out_root=tmp_path, serial=True)
    evaluate_run(run_dir, serial=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["backend"] == "auctionnet" and manifest["purpose"] == "auctionnet_online_simulation"
    assert manifest["integration_status"] == "integrated_online"
    scenarios = json.loads((run_dir / "scenarios.json").read_text())["replicates"][0]
    episodes = [s["episode"] for split in ("train", "validation", "test") for s in scenarios[split]]
    assert len(set(episodes)) == len(episodes)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "AuctionNet 在线仿真" in report and "内置 toy 环境" not in report
    records = [json.loads(line) for line in (run_dir / "test_episodes.jsonl").read_text().splitlines()]
    assert records and all(r["backend"] == "auctionnet" and r["status"] == "ok" for r in records)
