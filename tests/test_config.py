"""Config validation: divisibility, unknown policy/backend/keys, out-of-range values."""

from __future__ import annotations

import dataclasses

import pytest

from bidswarm.config import ExperimentConfig, load_config
from bidswarm.evaluation.budget import plan_training_budget
from bidswarm.types import ConfigError

from .conftest import CONFIG_DIR, small_config, small_config_dict


def test_example_toy_configs_load_and_match_documented_budget():
    smoke = load_config(CONFIG_DIR / "toy_smoke.toml")
    assert smoke.replicates == 3
    single, independent = smoke.budget_plan("single"), smoke.budget_plan("independent")
    # Budget 96, 4 train scenarios, k = 2 -> 24 candidates; single 12 rounds; independent 6 per worker, 3 rounds.
    assert single.candidates_total == independent.candidates_total == 24
    assert (single.n_workers, single.candidates_per_worker, single.rounds) == (1, 24, 12)
    assert (independent.n_workers, independent.candidates_per_worker, independent.rounds) == (4, 6, 3)

    experiment = load_config(CONFIG_DIR / "toy_experiment.toml")
    assert experiment.replicates == 20
    # Same search method and hyper-parameters as the smoke config; only the budget differs.
    assert experiment.search == dataclasses.replace(smoke.search, train_episode_budget=192)


def test_auctionnet_example_config_parses_as_config():
    # Parsing succeeds; running must fail (see test_backend.py).
    cfg = load_config(CONFIG_DIR / "auctionnet.example.toml")
    assert cfg.backend == "auctionnet"


def test_config_round_trips_through_dict():
    cfg = small_config()
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"search": {"train_episode_budget": 15}}, "not divisible"),  # 15 / 2 train scenarios
        ({"search": {"train_episode_budget": 12, "independent_workers": 4}}, "not divisible"),  # 6 / 4 workers
        ({"search": {"train_episode_budget": 24, "candidates_per_round": 4}}, "not divisible"),  # 6 per worker / k=4
    ],
)
def test_budget_must_divide_exactly(overrides, message):
    with pytest.raises(ConfigError, match=message):
        small_config(**overrides)


def test_budget_plan_arithmetic():
    plan = plan_training_budget("independent", 96, 4, 4, 2)
    assert (plan.candidates_total, plan.candidates_per_worker, plan.rounds) == (24, 6, 3)
    assert plan.episodes_per_round == 4 * 2 * 4
    with pytest.raises(ConfigError):
        plan_training_budget("single", 96, 5, 1, 2)


def test_unknown_policy_rejected():
    with pytest.raises(ConfigError, match="unknown policy"):
        small_config(search={"policy": "magic_bidder"})


def test_unknown_backend_rejected():
    with pytest.raises(ConfigError, match="unknown backend"):
        small_config(environment={"backend": "simulator_x"})


def test_unknown_keys_and_sections_rejected():
    with pytest.raises(ConfigError, match="unknown keys"):
        small_config(search={"lr": 0.1})
    with pytest.raises(ConfigError, match="unknown keys"):
        small_config(environment={"num_steps": 48})
    data = small_config_dict()
    data["extras"] = {}
    with pytest.raises(ConfigError, match="unknown config sections"):
        ExperimentConfig.from_dict(data)


@pytest.mark.parametrize(
    "overrides",
    [
        {"environment": {"exposure_prob": 1.5}},
        {"environment": {"budget_range": [30.0, 10.0]}},
        {"environment": {"num_time_steps": 0}},
        {"environment": {"opportunities_per_step": 2.5}},
        {"search": {"es": {"sigma_init": 0.9}}},
        {"search": {"es": {"failure_factor": 1.1}}},
        {"search": {"fitness_metric": "revenue"}},
        {"search": {"proposer": "llm"}},
        {"search": {"modes": ["single", "swarm"]}},
        {"search": {"independent_workers": 1}},
        {"selection": {"top_k": 9}},  # pool has 8 candidates
        {"runtime": {"episode_timeout_s": 0}},
        {"experiment": {"replicates": 0}},
        {"experiment": {"name": "bad/name"}},
    ],
)
def test_out_of_range_values_rejected(overrides):
    with pytest.raises(ConfigError):
        small_config(**overrides)


def test_missing_required_key_rejected():
    data = small_config_dict()
    del data["search"]["train_episode_budget"]
    with pytest.raises(ConfigError, match="missing key"):
        ExperimentConfig.from_dict(data)
