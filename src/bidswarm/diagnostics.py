"""Static checks that the portable kernel stays Python 3.9 compatible.

``ast.parse(..., feature_version=(3, 9))`` rejects syntax newer than 3.9
(e.g. ``match``). On top of that we check rules the parser cannot see:
the ``from __future__ import annotations`` header, the import allow-list
(numpy + stdlib + intra-kernel relative imports) and a few 3.10+ keyword
arguments. Running the files on a real 3.9 interpreter is a separate,
optional check (see tests/test_kernel_compat.py).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parent / "kernel"
ALLOWED_THIRD_PARTY = {"numpy"}
# Keyword arguments that only exist on Python >= 3.10 APIs.
_FORBIDDEN_KWARGS = {("dataclass", "slots"), ("dataclass", "kw_only"), ("zip", "strict")}


@dataclass(frozen=True)
class KernelFileCheck:
    path: Path
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def kernel_files() -> list[Path]:
    return sorted(KERNEL_DIR.glob("*.py"))


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def check_kernel_file(path: Path) -> KernelFileCheck:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path), feature_version=(3, 9))
    except SyntaxError as exc:
        return KernelFileCheck(path, (f"not valid Python 3.9 syntax: {exc}",))
    problems = []

    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]  # module docstring
    first = body[0] if body else None
    if not (
        isinstance(first, ast.ImportFrom)
        and first.module == "__future__"
        and any(a.name == "annotations" for a in first.names)
    ):
        problems.append("first statement must be 'from __future__ import annotations'")

    stdlib = set(sys.stdlib_module_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in stdlib and top not in ALLOWED_THIRD_PARTY:
                    problems.append(f"line {node.lineno}: import of non-allowed module {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                top = (node.module or "").split(".")[0]
                if top != "__future__" and top not in stdlib and top not in ALLOWED_THIRD_PARTY:
                    problems.append(f"line {node.lineno}: import from non-allowed module {node.module!r}")
            elif node.level > 1:
                problems.append(f"line {node.lineno}: relative import escapes the kernel package")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            for kw in node.keywords:
                if (name, kw.arg) in _FORBIDDEN_KWARGS:
                    problems.append(f"line {node.lineno}: {name}({kw.arg}=...) needs Python >= 3.10")
        elif isinstance(node, ast.Match):  # unreachable when feature_version works, kept explicit
            problems.append(f"line {node.lineno}: match statement needs Python >= 3.10")
    return KernelFileCheck(path, tuple(problems))


def check_kernel() -> list[KernelFileCheck]:
    return [check_kernel_file(p) for p in kernel_files()]
