"""quant_lab — an educational, vectorized backtesting and evaluation toolkit.

This package provides small, well-documented building blocks for studying the
*engineering* of quantitative research workflows on **synthetic** data:

- ``data``        : seeded synthetic price/OHLC generation and walk-forward splits
- ``backtest``    : a vectorized signal -> returns -> equity backtester with costs
- ``metrics``     : standard performance statistics (CAGR, Sharpe, drawdown, ...)
- ``calibration`` : probabilistic-forecast evaluation (Brier score, reliability)
- ``sizing``      : Kelly and fractional-Kelly position sizing helpers
- ``walkforward`` : a walk-forward evaluation harness

DISCLAIMER
----------
Nothing in this package is a trading strategy or contains any real-world alpha.
All examples operate on synthetic geometric-Brownian-motion data and use a
deliberately trivial moving-average crossover purely to demonstrate the
*plumbing*. Do not use this for trading decisions.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import backtest, calibration, data, metrics, sizing, walkforward

__all__ = [
    "data",
    "backtest",
    "metrics",
    "calibration",
    "sizing",
    "walkforward",
    "__version__",
]
