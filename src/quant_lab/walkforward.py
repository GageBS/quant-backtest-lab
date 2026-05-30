"""A walk-forward evaluation harness.

Walk-forward analysis fits/parameterizes a signal on an in-sample window, then
evaluates it on the *next*, unseen out-of-sample window — repeating as the
window rolls forward through time. This guards against overfitting to a single
historical slice.

The "fit" step here is intentionally trivial: it just constructs the illustrative
moving-average-crossover signal. The point of this module is to demonstrate the
*harness* (folding, out-of-sample backtesting, metric aggregation), not to fit a
real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import BacktestResult, moving_average_crossover_signal, run_backtest
from .data import walk_forward_windows
from .metrics import summary

# A SignalFn maps an in-sample price slice + the full price series to a position
# series aligned to the full price index. Returning the full-length signal lets
# the harness slice out the out-of-sample portion (and lets rolling indicators
# "warm up" using data just before the test window).
SignalFn = Callable[[pd.Series, pd.Series], pd.Series]


def default_demo_signal_fn(fast: int = 20, slow: int = 50) -> SignalFn:
    """Build a SignalFn for the illustrative MA-crossover. **DEMO ONLY.**

    The returned function ignores the in-sample slice for "fitting" (the crossover
    has no learned parameters) and simply computes the crossover over the full
    series so rolling means are warmed up before each test window.
    """

    def _fn(train_prices: pd.Series, full_prices: pd.Series) -> pd.Series:  # noqa: ARG001
        return moving_average_crossover_signal(full_prices, fast=fast, slow=slow)

    return _fn


@dataclass
class FoldResult:
    """Per-fold walk-forward output."""

    fold: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_obs: int
    backtest: BacktestResult
    metrics: dict[str, float]


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward results across all folds."""

    folds: list[FoldResult] = field(default_factory=list)

    @property
    def per_fold_metrics(self) -> pd.DataFrame:
        """Metrics for every fold as a tidy DataFrame (one row per fold)."""
        rows = []
        for f in self.folds:
            row = {"fold": f.fold, "test_start": f.test_start, "n_obs": f.n_obs}
            row.update(f.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def combined_returns(self) -> pd.Series:
        """Concatenated out-of-sample returns across all folds, in time order."""
        if not self.folds:
            return pd.Series(dtype=float, name="returns")
        parts = [f.backtest.returns for f in self.folds]
        combined = pd.concat(parts)
        return combined[~combined.index.duplicated(keep="first")].sort_index()

    def aggregate_metrics(self) -> dict[str, float]:
        """Headline metrics computed on the stitched-together OOS return stream."""
        return summary(self.combined_returns)

    def mean_fold_metrics(self) -> dict[str, float]:
        """Mean of each metric across folds (equal-weighted)."""
        df = self.per_fold_metrics.drop(columns=["fold", "test_start", "n_obs"], errors="ignore")
        if df.empty:
            return {}
        return {col: float(np.nanmean(df[col].to_numpy())) for col in df.columns}


def run_walk_forward(
    prices: pd.Series,
    signal_fn: SignalFn | None = None,
    train_size: int = 252,
    test_size: int = 63,
    step: int | None = None,
    anchored: bool = False,
    cost_bps: float = 1.0,
) -> WalkForwardReport:
    """Run a walk-forward evaluation of ``signal_fn`` over ``prices``.

    Parameters
    ----------
    prices:
        Full price (close) series.
    signal_fn:
        A callable ``(train_prices, full_prices) -> position_series``. Defaults to
        the illustrative MA-crossover via :func:`default_demo_signal_fn`.
    train_size, test_size, step, anchored:
        Walk-forward window geometry; see :func:`quant_lab.data.walk_forward_windows`.
    cost_bps:
        Per-unit-turnover transaction cost in basis points, passed to the backtester.

    Returns
    -------
    WalkForwardReport
        Per-fold backtests/metrics plus aggregation helpers.
    """
    if signal_fn is None:
        signal_fn = default_demo_signal_fn()

    report = WalkForwardReport()
    n = len(prices)

    for win in walk_forward_windows(
        n=n, train_size=train_size, test_size=test_size, step=step, anchored=anchored
    ):
        train_prices = prices.iloc[win.train_start : win.train_end]
        full_signal = signal_fn(train_prices, prices)

        # Out-of-sample slice. We include the last bar of the prior window so the
        # first OOS bar earns a return (pct_change needs a predecessor).
        oos_slice = slice(max(0, win.test_start - 1), win.test_end)
        oos_prices = prices.iloc[oos_slice]
        oos_signal = full_signal.iloc[oos_slice]

        bt = run_backtest(oos_prices, oos_signal, cost_bps=cost_bps)

        # Drop the carried-in predecessor bar so folds don't overlap.
        test_returns = bt.returns.iloc[1:]
        test_positions = bt.positions.iloc[1:]

        metrics = summary(test_returns, positions=test_positions)
        report.folds.append(
            FoldResult(
                fold=win.fold,
                test_start=prices.index[win.test_start],
                test_end=prices.index[win.test_end - 1],
                n_obs=len(test_returns),
                backtest=bt,
                metrics=metrics,
            )
        )

    return report
