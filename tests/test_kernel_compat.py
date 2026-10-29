"""The portable kernel must stay Python 3.9 compatible (numpy + stdlib only)."""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from bidswarm.diagnostics import KERNEL_DIR, check_kernel, check_kernel_file, kernel_files


def test_kernel_has_expected_files():
    assert {p.name for p in kernel_files()} == {"__init__.py", "policies.py", "registry.py", "spec.py"}


@pytest.mark.parametrize("path", kernel_files(), ids=lambda p: p.name)
def test_kernel_file_parses_as_python39(path: Path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))


def test_kernel_static_rules_pass():
    problems = {c.path.name: c.problems for c in check_kernel() if not c.ok}
    assert problems == {}


def test_feature_version_check_is_meaningful():
    with pytest.raises(SyntaxError):
        ast.parse("match x:\n    case 1:\n        pass\n", feature_version=(3, 9))


def test_static_rules_catch_violations(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        '"""doc"""\n'
        "import os\n"
        "import pandas\n"
        "from ..config import ExperimentConfig\n"
        "from dataclasses import dataclass\n"
        "@dataclass(slots=True)\n"
        "class A:\n"
        "    x: int = 0\n"
        "pairs = list(zip([1], [2], strict=True))\n"
    )
    problems = " ".join(check_kernel_file(bad).problems)
    assert "__future__" in problems
    assert "pandas" in problems
    assert "escapes the kernel" in problems
    assert "slots" in problems and "strict" in problems
    assert "'os'" not in problems  # stdlib is allowed


def test_kernel_does_not_import_other_bidswarm_packages():
    for path in kernel_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1, path.name
                assert not (node.module or "").startswith("bidswarm"), path.name
            elif isinstance(node, ast.Import):
                assert all(not a.name.startswith("bidswarm") for a in node.names), path.name


@pytest.mark.skipif(shutil.which("python3.9") is None, reason="no python3.9 interpreter on PATH")
def test_kernel_compiles_on_real_python39():
    files = [str(p) for p in kernel_files()]
    proc = subprocess.run(["python3.9", "-m", "py_compile", *files], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert KERNEL_DIR.is_dir()
