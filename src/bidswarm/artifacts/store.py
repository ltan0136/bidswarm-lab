"""Run directory and strict-JSON artifact writing.

All JSON goes through ``dumps_strict``: numpy scalars become Python numbers,
non-finite floats become ``null`` and ``json.dumps(allow_nan=False)`` makes
any NaN/Infinity that slipped through an error. Only the coordinator process
writes artifacts; there is no database.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..types import ArtifactError

MANIFEST = "manifest.json"
SCENARIOS = "scenarios.json"
CANDIDATES = "candidates.jsonl"
EPISODES = "episodes.jsonl"
ANYTIME = "anytime.jsonl"
BUDGET = "budget.json"
SELECTED_DIR = "selected"
SUMMARY = "summary.csv"
TEST_EPISODES = "test_episodes.jsonl"
REPORT = "report.md"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert to JSON-native types; non-finite floats -> None."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__} to strict JSON")


def dumps_strict(obj: Any, indent: int | None = None) -> str:
    return json.dumps(to_jsonable(obj), allow_nan=False, sort_keys=True, ensure_ascii=False, indent=indent)


def utc_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


class RunStore:
    """Read/write access to one run directory ``runs/<name>-<UTC timestamp>/``."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    @classmethod
    def create(cls, out_root: str | Path, name: str, now: datetime | None = None) -> "RunStore":
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        base = f"{name}-{utc_timestamp(now)}"
        candidate = out_root / base
        suffix = 1
        while True:
            try:
                candidate.mkdir()
                break
            except FileExistsError:
                suffix += 1
                candidate = out_root / f"{base}-{suffix}"
        (candidate / SELECTED_DIR).mkdir()
        return cls(candidate)

    @classmethod
    def open(cls, run_dir: str | Path) -> "RunStore":
        store = cls(run_dir)
        if not (store.run_dir / MANIFEST).is_file():
            raise ArtifactError(f"{run_dir} is not a run directory (no {MANIFEST})")
        return store

    def path(self, relpath: str) -> Path:
        return self.run_dir / relpath

    def exists(self, relpath: str) -> bool:
        return self.path(relpath).exists()

    def write_text(self, relpath: str, text: str) -> Path:
        target = self.path(relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, target)
        return target

    def write_json(self, relpath: str, obj: Any) -> Path:
        return self.write_text(relpath, dumps_strict(obj, indent=2) + "\n")

    def write_jsonl(self, relpath: str, records: Iterable[Any]) -> Path:
        return self.write_text(relpath, "".join(dumps_strict(r) + "\n" for r in records))

    def append_jsonl(self, relpath: str, records: Iterable[Any]) -> None:
        lines = "".join(dumps_strict(r) + "\n" for r in records)
        with self.path(relpath).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(lines)

    def read_text(self, relpath: str) -> str:
        try:
            return self.path(relpath).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ArtifactError(f"missing artifact {relpath} in {self.run_dir}") from None

    def read_json(self, relpath: str) -> Any:
        return json.loads(self.read_text(relpath))

    def read_jsonl(self, relpath: str) -> list[Any]:
        return [json.loads(line) for line in self.read_text(relpath).splitlines() if line.strip()]

    def selected_files(self) -> list[str]:
        folder = self.path(SELECTED_DIR)
        if not folder.is_dir():
            return []
        return sorted(f"{SELECTED_DIR}/{p.name}" for p in folder.glob("*.json"))
