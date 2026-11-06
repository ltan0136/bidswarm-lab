"""AuctionNet backend: the online simulation, run in an upstream Python 3.9 subprocess.

Only the online path (``run/run_test.py`` + ``config/test.gin`` at the pinned
commit) is integrated; the offline path needs ~80 GB of data and is rejected.
See docs/AUCTIONNET_INTEGRATION.md for the verified upstream facts.

Each bidswarm process (the main process when serial, otherwise every spawned
pool worker) starts one long-lived runner subprocess on first use
(``auctionnet_runner.py``, executed with ``python_executable``), so the
upstream models load once per process. One request = one delivery period of
48 ticks for the configured player position and an episode index.

Scenarios: the advertiser is fixed by ``player_index`` (upstream's budget /
CPA / category table, copied below and re-checked against the runner at
start-up); a scenario's seed selects the episode index, which seeds the
traffic. Splits are checked to use disjoint episode indices.

Nothing falls back: a missing checkout, wrong commit, runner crash or
broken invariant raises, and a policy error becomes a failed episode.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import select
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from ..evaluation.metrics import auctionnet_score, cpa_within_target
from ..kernel import PolicySpec, spec_hash
from ..types import BACKEND_AUCTIONNET, PURPOSE_AUCTIONNET, SCORE_FORMULA, STATUS_OK, ConfigError
from .base import Environment, failed_episode

INTEGRATION_STATUS = "integrated_online"
PINNED_COMMIT = "c20de631a40e24669d1d6a8e2acefd8f492f172d"
NUM_TIME_STEPS = 48
EPISODE_SPACE = 2**31  # episode index = scenario seed mod this

# Limitations shown by `bidswarm doctor`; none of them makes the online backend unusable.
LIMITATIONS = (
    "only the online simulation is integrated; evaluation_path = 'offline' needs ~80GB of data and is rejected",
    "one delivery period takes about 45-50 s and 2.7 GB per process (about 280 periods per hour on 4 cores "
    "with 4 workers); a toy-sized protocol (~75,000 periods) is infeasible, use configs/auctionnet_*.toml sizes",
    "the advertiser is fixed by player_index; scenarios differ only in the episode index that seeds traffic",
    "the 47 competitors are upstream's fixed baseline agents, so results may not transfer to other opponents",
)

# Controller.py at the pinned commit: calculate_budget (line 114) and get_cpa_constraints (line 103).
UPSTREAM_BUDGETS = (
    2900, 4350, 3000, 2400, 4800, 2000, 2050, 3500, 4600, 2000, 2800, 2350, 2050, 2900, 4750, 3450,
    2000, 3500, 2200, 2700, 3100, 2100, 4850, 4100, 2000, 4800, 3050, 4250, 2850, 2250, 2000, 3900,
    2000, 3250, 4450, 3550, 2700, 2100, 4650, 2000, 3400, 2650, 2300, 4100, 4800, 4450, 2000, 2050,
)
UPSTREAM_CPA = (
    100, 70, 90, 110, 60, 130, 120, 80, 70, 130, 100, 110, 120, 90, 60, 80,
    130, 80, 110, 100, 90, 120, 60, 70, 120, 60, 90, 70, 100, 110, 130, 80,
    120, 90, 70, 80, 100, 110, 60, 130, 90, 100, 110, 80, 60, 70, 130, 120,
)
AGENTS_PER_CATEGORY = 8  # config/test.gin NUM_AGENT_CATERORY; category = index // 8 (Controller.py:31)

RUNNER_PATH = Path(__file__).with_name("auctionnet_runner.py")
KERNEL_PARENT = Path(__file__).resolve().parents[1]  # src/bidswarm, which contains kernel/

_ALLOWED_KEYS = frozenset(
    {
        "backend", "repo_path", "commit_sha", "evaluation_path", "python_executable",
        "num_time_steps", "player_index", "startup_timeout_s", "request_timeout_s",
    }
)
_SHA = re.compile(r"^[0-9a-f]{40}$")


class AuctionNetSetupError(RuntimeError):
    """The upstream checkout or interpreter is missing or not the pinned one. Never falls back."""


class AuctionNetRunnerError(RuntimeError):
    """The runner subprocess failed (not the policy). Aborts the run."""


def advertiser(player_index: int) -> dict[str, Any]:
    return {
        "budget": float(UPSTREAM_BUDGETS[player_index]),
        "cpa_target": float(UPSTREAM_CPA[player_index]),
        "category": int(player_index // AGENTS_PER_CATEGORY),
    }


def validate_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Check and complete the [environment] table; does not touch the filesystem."""
    unknown = sorted(set(cfg) - _ALLOWED_KEYS)
    if unknown:
        raise ConfigError(f"[environment] unknown keys for backend 'auctionnet': {unknown}")
    out = dict(cfg)
    path = out.setdefault("evaluation_path", "online")
    if path == "offline":
        raise ConfigError(
            "[environment] evaluation_path 'offline' is not integrated (it needs ~80GB of pre-generated data); "
            "use 'online'"
        )
    if path != "online":
        raise ConfigError("[environment] evaluation_path must be 'online'")
    for key in ("repo_path", "python_executable"):
        if not isinstance(out.get(key), str) or not out[key]:
            raise ConfigError(f"[environment] {key} must be a non-empty string")
        # "$BIDSWARM_AUCTIONNET_REPO" style values keep machine paths out of committed configs;
        # the manifest records the expanded path.
        out[key] = os.path.expanduser(os.path.expandvars(out[key]))
    sha = out.get("commit_sha")
    if not isinstance(sha, str) or not _SHA.match(sha):
        raise ConfigError("[environment] commit_sha must be a full 40-character lowercase hex SHA")
    if sha != PINNED_COMMIT:
        raise ConfigError(
            f"[environment] commit_sha {sha} is not the verified commit {PINNED_COMMIT}; "
            "re-run the step-1 checks in docs/AUCTIONNET_INTEGRATION.md before changing it"
        )
    steps = out.setdefault("num_time_steps", NUM_TIME_STEPS)
    if steps != NUM_TIME_STEPS or isinstance(steps, bool):
        raise ConfigError(f"[environment] num_time_steps must be {NUM_TIME_STEPS} (config/test.gin NUM_TICK)")
    index = out.setdefault("player_index", 0)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(UPSTREAM_BUDGETS):
        raise ConfigError(f"[environment] player_index must be an integer in [0, {len(UPSTREAM_BUDGETS) - 1}]")
    for key, default in (("startup_timeout_s", 600.0), ("request_timeout_s", 1800.0)):
        value = out.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ConfigError(f"[environment] {key} must be a positive number")
        out[key] = float(value)
    return out


def check_setup(cfg: Mapping[str, Any]) -> None:
    """Fail loudly unless the checkout is at the pinned commit and the interpreter exists."""
    repo = Path(cfg["repo_path"])
    if not (repo / "run" / "run_test.py").is_file():
        raise AuctionNetSetupError(f"no AuctionNet checkout at {repo} (see docs/AUCTIONNET_INTEGRATION.md)")
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuctionNetSetupError(f"cannot read the commit of {repo}: {exc}") from exc
    if head != cfg["commit_sha"]:
        raise AuctionNetSetupError(f"{repo} is at {head}, expected {cfg['commit_sha']}")
    python = cfg["python_executable"]
    if not Path(python).is_file() and not _which(python):
        raise AuctionNetSetupError(f"python_executable {python!r} not found")


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


class RunnerProcess:
    """One upstream runner subprocess speaking JSON lines."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.stderr = tempfile.NamedTemporaryFile(prefix="bidswarm-auctionnet-", suffix=".log", delete=False)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.proc = subprocess.Popen(
            [
                cfg["python_executable"], str(RUNNER_PATH),
                "--upstream", str(cfg["repo_path"]),
                "--kernel-parent", str(KERNEL_PARENT),
                "--commit-sha", cfg["commit_sha"],
                "--player-index", str(cfg["player_index"]),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True, env=env,
        )
        ready = self._read(float(cfg["startup_timeout_s"]))
        if not ready.get("ready"):
            self.close()
            raise AuctionNetSetupError(f"AuctionNet runner failed to start: {ready.get('error')}")
        expected = advertiser(int(cfg["player_index"]))
        got = {"budget": ready["budget"], "cpa_target": ready["cpa"], "category": ready["category"]}
        if got != expected or ready["upstream_commit"] != cfg["commit_sha"] or ready["num_tick"] != NUM_TIME_STEPS:
            self.close()
            raise AuctionNetSetupError(f"runner reports {ready}, expected advertiser {expected}")
        self.ready = ready

    def _stderr_tail(self, lines: int = 20) -> str:
        try:
            text = Path(self.stderr.name).read_text(errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _read(self, timeout_s: float) -> dict[str, Any]:
        assert self.proc.stdout is not None
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout_s)
        if not ready:
            raise AuctionNetRunnerError(
                f"AuctionNet runner did not answer within {timeout_s:g} s; stderr tail:\n{self._stderr_tail()}"
            )
        line = self.proc.stdout.readline()
        if not line:  # end of output: the runner exited
            try:
                code = self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                code = None
            raise AuctionNetRunnerError(
                f"AuctionNet runner exited with code {code}; stderr tail:\n{self._stderr_tail()}"
            )
        return json.loads(line)

    def request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AuctionNetRunnerError(f"AuctionNet runner is gone: {exc}; stderr tail:\n{self._stderr_tail()}")
        return self._read(float(self.cfg["request_timeout_s"]))

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.stderr.close()
        try:
            os.unlink(self.stderr.name)
        except OSError:
            pass


_RUNNERS: dict[tuple, RunnerProcess] = {}


def _runner_for(cfg: Mapping[str, Any]) -> RunnerProcess:
    key = (cfg["python_executable"], str(cfg["repo_path"]), cfg["commit_sha"], int(cfg["player_index"]))
    runner = _RUNNERS.get(key)
    if runner is None or runner.proc.poll() is not None:
        runner = _RUNNERS[key] = RunnerProcess(cfg)
    return runner


@atexit.register
def _close_runners() -> None:
    for runner in list(_RUNNERS.values()):
        runner.close()
    _RUNNERS.clear()


def _tags() -> dict[str, Any]:
    return {"backend": BACKEND_AUCTIONNET, "purpose": PURPOSE_AUCTIONNET, "score_formula": SCORE_FORMULA}


class AuctionNetEnvironment(Environment):
    backend = BACKEND_AUCTIONNET
    integration_status = INTEGRATION_STATUS
    num_time_steps = NUM_TIME_STEPS

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = validate_config(cfg)
        check_setup(self.cfg)
        self.player_index = int(self.cfg["player_index"])

    def make_scenario(self, scenario_id: str, split: str, index: int, seed: int) -> dict[str, Any]:
        return {
            "scenario_id": scenario_id,
            "split": split,
            "index": int(index),
            "seed": int(seed),
            "episode": int(seed) % EPISODE_SPACE,
            "player_index": self.player_index,
            **advertiser(self.player_index),
            "backend": BACKEND_AUCTIONNET,
        }

    def run_episode(self, scenario: dict[str, Any], policy: Any) -> dict[str, Any]:
        start = time.perf_counter()
        spec = PolicySpec(policy.name, dict(policy.params))
        h = spec_hash(spec)
        response = _runner_for(self.cfg).request(
            {"op": "episode", "episode": int(scenario["episode"]), "spec": spec.to_dict(), "spec_hash": h}
        )
        wall = time.perf_counter() - start
        if not response.get("ok"):
            if response.get("kind") == "policy_error":
                return {
                    **failed_episode(scenario, BACKEND_AUCTIONNET, response["error"], wall, None, PURPOSE_AUCTIONNET),
                    "episode": scenario["episode"],
                }
            raise AuctionNetRunnerError(f"AuctionNet runner error: {response.get('error')}\n{response.get('traceback', '')}")
        problems = []
        if response["spec_hash"] != h:
            problems.append("spec_hash echo differs")
        if response["bidding_calls"] != response["expected_bidding_calls"]:
            problems.append(
                f"player bid {response['bidding_calls']} times, the loop expected {response['expected_bidding_calls']}"
            )
        if abs(response["kernel_spent"] - response["cost"]) > 1e-6 * max(1.0, response["cost"]):
            problems.append(f"kernel books {response['kernel_spent']} differ from the charge {response['cost']}")
        if response["budget"] != scenario["budget"]:
            problems.append("runner budget differs from the scenario")
        if problems:
            raise AuctionNetRunnerError("AuctionNet invariant violated: " + "; ".join(problems))

        budget, cpa_target = float(scenario["budget"]), float(scenario["cpa_target"])
        cost, conversions = float(response["cost"]), float(response["reward"])
        expected = float(response["expected_conversions"])
        return {
            **_tags(),
            "scenario_id": scenario["scenario_id"],
            "split": scenario["split"],
            "status": STATUS_OK,
            "error": None,
            "budget": budget,
            "cpa_target": cpa_target,
            "cost": cost,
            "conversions": int(round(conversions)),
            "expected_conversions": expected,
            "cpa_realized": cost / conversions if conversions > 0 else None,
            "cpa_expected": cost / expected if expected > 0 else None,
            "cpa_defined": conversions > 0,
            "cpa_expected_defined": expected > 0,
            "score_realized": auctionnet_score(conversions, cost, cpa_target),
            "score_expected": auctionnet_score(expected, cost, cpa_target),
            "budget_utilization": cost / budget,
            "wins": int(response["wins"]),
            "opportunities": int(response["opportunities"]),
            "cpa_ok": cpa_within_target(conversions, cost, cpa_target),
            "cpa_expected_ok": cpa_within_target(expected, cost, cpa_target),
            "wall_time_s": wall,
            "episode": int(scenario["episode"]),
            "player_index": self.player_index,
            "exposures": int(response["exposures"]),
            "upstream_commit": self.cfg["commit_sha"],
        }
