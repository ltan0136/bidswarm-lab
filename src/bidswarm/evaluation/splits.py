"""Scenario splits per replicate (paired design).

Within a replicate every mode uses the same train / validation / test
scenarios. Scenario seeds are derived from (master_seed, split, index) by
hashing, and splits are checked to be disjoint in both IDs and seeds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..environments.base import Environment
from ..types import SPLITS


class SplitOverlapError(ValueError):
    """Raised when two splits share a scenario ID or seed."""


def derive_seed(master_seed: int, split: str, index: int) -> int:
    """63-bit scenario seed from sha256("bidswarm-scenario:<master>:<split>:<index>")."""
    digest = hashlib.sha256(f"bidswarm-scenario:{master_seed}:{split}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def scenario_id(master_seed: int, split: str, index: int) -> str:
    return f"m{master_seed}-{split}-{index:03d}"


def master_seeds(base_seed: int, replicates: int) -> list[int]:
    """Replicate r uses master seed base_seed + r."""
    return [base_seed + r for r in range(replicates)]


@dataclass(frozen=True)
class SplitSet:
    replicate: int
    master_seed: int
    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]

    def split(self, name: str) -> tuple[dict[str, Any], ...]:
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicate": self.replicate,
            "master_seed": self.master_seed,
            **{name: list(self.split(name)) for name in SPLITS},
        }


def check_disjoint(split_set: SplitSet) -> None:
    seen_ids: dict[str, str] = {}
    seen_seeds: dict[int, str] = {}
    for name in SPLITS:
        for sc in split_set.split(name):
            sid, seed = sc["scenario_id"], sc["seed"]
            if sid in seen_ids:
                raise SplitOverlapError(f"scenario id {sid} appears in {seen_ids[sid]} and {name}")
            if seed in seen_seeds:
                raise SplitOverlapError(f"scenario seed {seed} appears in {seen_seeds[seed]} and {name}")
            seen_ids[sid] = name
            seen_seeds[seed] = name


def build_splits(env: Environment, replicate: int, master_seed: int, sizes: Any) -> SplitSet:
    """Build and check the scenarios of one replicate. ``sizes`` has train/validation/test counts."""
    parts = {}
    for name in SPLITS:
        parts[name] = tuple(
            env.make_scenario(scenario_id(master_seed, name, i), name, i, derive_seed(master_seed, name, i))
            for i in range(getattr(sizes, name))
        )
    split_set = SplitSet(replicate=replicate, master_seed=master_seed, **parts)
    check_disjoint(split_set)
    return split_set
