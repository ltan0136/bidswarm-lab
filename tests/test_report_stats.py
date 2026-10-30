"""Statistical helpers used by the report: bootstrap CI, permutation test, Holm correction."""

from __future__ import annotations

import numpy as np
import pytest

from bidswarm.artifacts.report import (
    PERMUTATION_RESAMPLES,
    _fmt_p,
    _verdict,
    holm_adjust,
    paired_bootstrap_ci,
    paired_sign_flip_pvalue,
)


def test_bootstrap_ci_needs_two_pairs_and_is_shift_equivariant():
    assert paired_bootstrap_ci([1.0]) is None
    d = np.random.default_rng(0).normal(size=30)
    lo, hi = paired_bootstrap_ci(d)
    lo2, hi2 = paired_bootstrap_ci(d + 2.5)
    assert lo < d.mean() < hi
    assert (lo2, hi2) == pytest.approx((lo + 2.5, hi + 2.5))


def test_sign_flip_pvalue_bounds_and_extremes():
    assert paired_sign_flip_pvalue([0.3]) is None
    assert paired_sign_flip_pvalue([0.0, 0.0, 0.0]) == 1.0
    p = paired_sign_flip_pvalue(np.full(40, 1.0))  # every flip but all-plus is smaller
    assert p == pytest.approx(1 / (PERMUTATION_RESAMPLES + 1), abs=2 / PERMUTATION_RESAMPLES)
    assert p > 0
    rng = np.random.default_rng(1)
    null_ps = [paired_sign_flip_pvalue(rng.normal(size=20), resamples=2000, seed=s) for s in range(200)]
    assert 0.01 < np.mean(np.array(null_ps) <= 0.05) < 0.12  # roughly calibrated under a symmetric null


def test_holm_adjust():
    assert holm_adjust([0.25, 0.33, 0.04]) == pytest.approx([0.5, 0.5, 0.12])
    assert holm_adjust([0.01, None, 0.03]) == [pytest.approx(0.02), None, pytest.approx(0.03)]
    assert holm_adjust([0.9, 0.8]) == [1.0, 1.0]  # 2 x 0.8 capped at 1, then kept monotone
    assert holm_adjust([]) == []


def test_pvalue_formatting_never_shows_zero():
    assert _fmt_p(0.00004) == "< 0.001"
    assert _fmt_p(0.25) == "0.250"
    assert _fmt_p(None) == "—"


def test_verdict_respects_holm_correction():
    assert "未观察到 x 与 y 之间可分辨的差异" in _verdict("x", "y", (-1.0, 1.0), 0.4, 100)
    assert "x 的测试分高于 y" in _verdict("x", "y", (0.1, 1.0), 0.01, 100)
    hedged = _verdict("x", "y", (0.1, 1.0), 0.04, 100, holm=0.12, family=3)
    assert "不能确认" in hedged and "0.120" in hedged
    assert "低于" in _verdict("x", "y", (-2.0, -0.5), 0.001, 100, holm=0.003, family=3)
