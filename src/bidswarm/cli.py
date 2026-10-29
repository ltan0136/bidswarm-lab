"""Command line: ``python -m bidswarm doctor | run | evaluate | report``."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence

from . import __version__
from .environments import INTEGRATION_STATUS, BackendNotIntegratedError
from .environments import auctionnet
from .evaluation.splits import SplitOverlapError
from .types import ArtifactError, ConfigError

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_ERROR = 2


def _cmd_doctor(_: argparse.Namespace) -> int:
    import numpy as np

    from .diagnostics import KERNEL_DIR, check_kernel

    print(f"bidswarm {__version__}")
    py_ok = sys.version_info >= (3, 11)
    print(f"python: {platform.python_version()} ({sys.executable}) [{'ok' if py_ok else 'need >= 3.11'}]")
    print(f"numpy: {np.__version__}")
    checks = check_kernel()
    kernel_ok = all(c.ok for c in checks)
    print(f"kernel Python 3.9 syntax check: {'ok' if kernel_ok else 'FAILED'} ({len(checks)} files in {KERNEL_DIR})")
    for c in checks:
        print(f"  - {c.path.name}: {'ok' if c.ok else 'FAILED'}")
        for problem in c.problems:
            print(f"      {problem}")
    print(f"backend toy: integration_status={INTEGRATION_STATUS['toy']}")
    print(f"backend auctionnet: integration_status={auctionnet.INTEGRATION_STATUS}")
    for blocker in auctionnet.BLOCKERS:
        print(f"  - {blocker}")
    print("doctor does not install anything.")
    return EXIT_OK if (py_ok and kernel_ok) else EXIT_CHECK_FAILED


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import load_config
    from .experiment import run_experiment

    config = load_config(args.config)
    run_dir = run_experiment(
        config, out_root=args.out, serial=args.serial, max_workers=args.max_workers, config_path=args.config
    )
    print(f"run directory: {run_dir}")
    print("test split NOT evaluated; next: python -m bidswarm evaluate --run-dir", run_dir)
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .experiment import evaluate_run

    run_dir = evaluate_run(args.run_dir, serial=args.serial, max_workers=args.max_workers, force=args.force)
    print(f"test episodes written to {run_dir / 'test_episodes.jsonl'}")
    print(f"report: {run_dir / 'report.md'}")
    return EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    from .artifacts.report import write_report
    from .artifacts.store import REPORT, RunStore

    store = RunStore.open(args.run_dir)
    write_report(store)
    print(f"report: {store.path(REPORT)}")
    return EXIT_OK


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bidswarm", description="BidSwarm Lab experiment framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="print environment and kernel compatibility checks")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("run", help="search + validation selection (does not touch the test split)")
    p.add_argument("--config", required=True, help="path to a TOML config")
    p.add_argument("--serial", action="store_true", help="run episodes in-process")
    p.add_argument("--max-workers", type=_positive_int, default=None, help="process pool size")
    p.add_argument("--out", default="runs", help="output root directory (default: runs)")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("evaluate", help="evaluate frozen selections and baselines on the test split")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--serial", action="store_true")
    p.add_argument("--max-workers", type=_positive_int, default=None)
    p.add_argument("--force", action="store_true", help="overwrite an existing test_episodes.jsonl")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("report", help="rebuild report.md from the run artifacts")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, BackendNotIntegratedError, ArtifactError, SplitOverlapError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
