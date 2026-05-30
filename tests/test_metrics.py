"""Tests for quant_lab.metrics against known, hand-checkable inputs."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from quant_lab import metrics  # noqa: E402


def test_total_return_compounds():
    # (1.1 * 1.1) - 1 = 0.21
    assert metrics.total_return([0.1, 0.1]) == pytest.approx(0.21)


def test_total_return_empty_is_zero():
    assert metrics.total_return([]) == 0.0


def test_sharpe_of_constant_positive_return_is_inf_guarded_to_zero():
    # Zero variance -> we define Sharpe as 0.0 (no risk-adjusted info).
    assert metrics.sharpe_ratio([0.01] * 50) == 0.0


def test_sharpe_known_value():
    # Equal up/down returns -> mean 0 -> Sharpe 0.
    r = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    assert metrics.sharpe_ratio(r) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_sign_matches_mean():
    rng = np.random.default_rng(0)
    pos = rng.normal(0.001, 0.01, size=500)
    assert metrics.sharpe_ratio(pos) > 0


def test_annualized_vol_scales_with_sqrt_time():
    r = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
    expected = np.std(r, ddof=1) * math.sqrt(252)
    assert metrics.annualized_volatility(r) == pytest.approx(expected)


def test_max_drawdown_known():
    # +10% then -50% from peak: equity 1.1 -> 0.55, drawdown = 0.55/1.1 - 1 = -0.5
    r = [0.10, -0.50]
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_monotone_increasing_is_zero():
    assert metrics.max_drawdown([0.01, 0.02, 0.03]) == pytest.approx(0.0)


def test_hit_rate_ignores_zeros():
    # Active returns: +, -, + -> 2/3 positive.
    r = [0.0, 0.01, -0.01, 0.0, 0.02]
    assert metrics.hit_rate(r) == pytest.approx(2.0 / 3.0)


def test_hit_rate_all_zero_is_zero():
    assert metrics.hit_rate([0.0, 0.0]) == 0.0


def test_exposure_fraction_nonzero():
    assert metrics.exposure([0, 1, 1, 0, -1]) == pytest.approx(3.0 / 5.0)


def test_cagr_of_flat_returns_zero():
    assert metrics.cagr([0.0] * 252) == pytest.approx(0.0)


def test_cagr_doubling_in_one_year():
    # 252 daily returns compounding to exactly 2x -> CAGR ~ 100%.
    daily = 2.0 ** (1.0 / 252) - 1.0
    assert metrics.cagr([daily] * 252) == pytest.approx(1.0, rel=1e-6)


def test_summary_keys_with_positions():
    stats = metrics.summary([0.01, -0.01, 0.02], positions=[1, 1, 0])
    assert set(stats) == {
        "total_return",
        "cagr",
        "ann_volatility",
        "sharpe",
        "max_drawdown",
        "hit_rate",
        "exposure",
    }
