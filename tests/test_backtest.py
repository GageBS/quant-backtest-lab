"""Tests for quant_lab.backtest with controlled inputs."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from quant_lab import backtest  # noqa: E402


def _prices(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float, name="close")


def test_flat_signal_yields_zero_pnl():
    prices = _prices([100, 101, 102, 99, 105])
    signal = pd.Series(0.0, index=prices.index)
    res = backtest.run_backtest(prices, signal, cost_bps=5.0)
    assert res.returns.sum() == pytest.approx(0.0)
    assert res.equity.iloc[-1] == pytest.approx(1.0)
    assert res.costs.sum() == pytest.approx(0.0)


def test_constant_long_matches_buy_and_hold_minus_entry_cost():
    prices = _prices([100, 110, 121])  # +10% then +10%
    signal = pd.Series(1.0, index=prices.index)
    res = backtest.run_backtest(prices, signal, cost_bps=0.0)
    # Position is lagged: bar0 flat, then long. Captures the two +10% moves.
    assert res.gross_returns.iloc[1] == pytest.approx(0.10)
    assert res.gross_returns.iloc[2] == pytest.approx(0.10)
    # Buy-and-hold (held from bar 1) compounds 1.1 * 1.1 = 1.21.
    assert res.equity.iloc[-1] == pytest.approx(1.21)


def test_costs_charged_on_turnover():
    prices = _prices([100, 100, 100])  # no price moves -> only cost matters
    # Go long at bar 1 (held from bar 2 onward after lag): one unit of turnover.
    signal = pd.Series([0.0, 1.0, 1.0], index=prices.index)
    res = backtest.run_backtest(prices, signal, cost_bps=10.0)
    # 10 bps = 0.001 charged once when position changes 0 -> 1.
    assert res.costs.sum() == pytest.approx(0.001)
    assert res.returns.sum() == pytest.approx(-0.001)


def test_no_lookahead_first_bar_is_flat():
    prices = _prices([100, 200])
    signal = pd.Series([1.0, 1.0], index=prices.index)
    res = backtest.run_backtest(prices, signal, cost_bps=0.0)
    # First bar's position is the lagged (pre-sample) value -> 0, so no return.
    assert res.positions.iloc[0] == 0.0
    assert res.returns.iloc[0] == pytest.approx(0.0)


def test_negative_cost_rejected():
    prices = _prices([100, 101])
    with pytest.raises(ValueError):
        backtest.run_backtest(prices, pd.Series([1.0, 1.0], index=prices.index), cost_bps=-1.0)


def test_too_short_rejected():
    prices = _prices([100])
    with pytest.raises(ValueError):
        backtest.run_backtest(prices, pd.Series([1.0], index=prices.index))


def test_ma_crossover_is_long_or_flat_only():
    prices = _prices(list(np.linspace(100, 200, 120)))
    sig = backtest.moving_average_crossover_signal(prices, fast=5, slow=20)
    assert set(np.unique(sig.to_numpy())) <= {0.0, 1.0}
    # On a steadily rising series the fast MA ends above the slow MA -> long.
    assert sig.iloc[-1] == 1.0


def test_ma_crossover_validates_lengths():
    prices = _prices([100, 101, 102])
    with pytest.raises(ValueError):
        backtest.moving_average_crossover_signal(prices, fast=20, slow=20)


def test_signal_accepts_numpy_array():
    prices = _prices([100, 110, 121])
    res = backtest.run_backtest(prices, np.array([1.0, 1.0, 1.0]), cost_bps=0.0)
    assert res.equity.iloc[-1] == pytest.approx(1.21)
