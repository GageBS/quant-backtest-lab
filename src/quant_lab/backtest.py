"""A small, vectorized signal backtester.

The backtester takes a price series and a per-bar *position* signal (in
``{-1, 0, +1}`` or any real-valued weight) and produces strategy returns, an
equity curve, and a per-bar trade-cost series. It is deliberately simple and
fully vectorized — no event loop — which makes it fast and easy to reason about.

Key conventions
---------------
- **No look-ahead.** The position decided using information available *at* bar
  ``t`` earns the asset return realized from ``t`` to ``t+1``. We therefore lag
  the signal by one bar before applying it to forward returns.
- **Costs** are charged on *turnover* (the absolute change in position) and are
  expressed in basis points of notional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Container for the output of :func:`run_backtest`.

    Attributes
    ----------
    returns:
        Per-bar net strategy returns (after costs), aligned to the price index.
    gross_returns:
        Per-bar strategy returns *before* costs.
    costs:
        Per-bar cost drag (always >= 0), expressed as a fraction of notional.
    equity:
        Compounded net equity curve, starting at ``1.0``.
    positions:
        The lagged positions actually held each bar (what earned the bar return).
    turnover:
        Per-bar absolute change in position used to charge costs.
    """

    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    equity: pd.Series
    positions: pd.Series
    turnover: pd.Series


def run_backtest(
    prices: pd.Series,
    signal: pd.Series | np.ndarray,
    cost_bps: float = 1.0,
    initial_equity: float = 1.0,
) -> BacktestResult:
    """Backtest a position signal against a price series.

    Parameters
    ----------
    prices:
        Price (close) series. Returns are computed as simple percentage changes.
    signal:
        Desired position per bar, aligned to ``prices``. Values are typically in
        ``{-1, 0, +1}`` but any real weight is accepted. The signal is lagged by
        one bar internally to avoid look-ahead bias.
    cost_bps:
        Per-unit-turnover transaction cost in basis points (1 bp = 0.01%). A round
        trip from flat to fully long and back costs ``2 * cost_bps``.
    initial_equity:
        Starting value of the equity curve.

    Returns
    -------
    BacktestResult
        Net/gross returns, cost drag, equity curve, held positions, and turnover.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if len(prices) < 2:
        raise ValueError("need at least two price observations to backtest")

    prices = prices.astype(float)

    if isinstance(signal, np.ndarray):
        signal = pd.Series(signal, index=prices.index)
    signal = signal.reindex(prices.index).astype(float).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)

    # Lag the signal by one bar: the position chosen at t earns the t->t+1 return.
    held = signal.shift(1).fillna(0.0)

    # Turnover is the absolute change in position relative to the previous bar.
    turnover = held.diff().abs().fillna(held.abs())

    cost_rate = cost_bps / 10_000.0
    costs = turnover * cost_rate

    gross_returns = held * asset_returns
    net_returns = gross_returns - costs

    equity = initial_equity * (1.0 + net_returns).cumprod()

    return BacktestResult(
        returns=net_returns.rename("returns"),
        gross_returns=gross_returns.rename("gross_returns"),
        costs=costs.rename("costs"),
        equity=equity.rename("equity"),
        positions=held.rename("positions"),
        turnover=turnover.rename("turnover"),
    )


def moving_average_crossover_signal(
    prices: pd.Series,
    fast: int = 20,
    slow: int = 50,
) -> pd.Series:
    """Return a long/flat moving-average-crossover signal. **ILLUSTRATIVE ONLY.**

    .. warning::

       This is a textbook placeholder used purely to demonstrate the backtesting
       plumbing. It is **not** a trading strategy and carries no expectation of
       profit. It exists so the demo and walk-forward harness have *some* signal
       to push through the engine.

    The rule: hold a long position (``+1``) when the fast simple moving average is
    above the slow one, otherwise stay flat (``0``).

    Parameters
    ----------
    prices:
        Price (close) series.
    fast, slow:
        Lookback lengths for the fast and slow simple moving averages. ``fast``
        must be strictly less than ``slow``.

    Returns
    -------
    pandas.Series
        Position series in ``{0, +1}`` aligned to ``prices``.
    """
    if fast < 1 or slow < 1:
        raise ValueError("fast and slow must be >= 1")
    if fast >= slow:
        raise ValueError("fast must be strictly less than slow")

    fast_ma = prices.rolling(window=fast, min_periods=fast).mean()
    slow_ma = prices.rolling(window=slow, min_periods=slow).mean()

    signal = (fast_ma > slow_ma).astype(float)
    # Until both moving averages are defined, stay flat.
    signal[slow_ma.isna()] = 0.0
    return signal.rename("signal")
