"""Performance statistics for return and equity series.

All functions are pure and operate on array-likes (``numpy`` arrays, ``pandas``
Series, or lists). Returns are interpreted as *simple* per-period returns unless
stated otherwise. The default annualization factor assumes daily data
(``252`` periods per year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def _as_array(x: np.ndarray | pd.Series | list) -> np.ndarray:
    """Coerce an array-like of returns to a 1-D float ``numpy`` array."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr


def total_return(returns: np.ndarray | pd.Series) -> float:
    """Cumulative compounded return over the full sample.

    ``total_return = prod(1 + r) - 1``.
    """
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def cagr(returns: np.ndarray | pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Compound annual growth rate implied by a return series.

    Computed from the terminal compounded wealth and the number of periods::

        cagr = wealth ** (periods_per_year / n) - 1
    """
    arr = _as_array(returns)
    n = arr.size
    if n == 0:
        return 0.0
    wealth = float(np.prod(1.0 + arr))
    if wealth <= 0:
        # Total loss of capital — annualized growth is -100%.
        return -1.0
    return wealth ** (periods_per_year / n) - 1.0


def annualized_volatility(
    returns: np.ndarray | pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualized standard deviation of returns (sample std, ddof=1)."""
    arr = _as_array(returns)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns:
        Per-period simple returns.
    risk_free_rate:
        *Annual* risk-free rate, converted to a per-period rate internally.
    periods_per_year:
        Annualization factor.

    Returns
    -------
    float
        The annualized Sharpe ratio, or ``0.0`` if volatility is zero or the
        sample is too small.
    """
    arr = _as_array(returns)
    if arr.size < 2:
        return 0.0
    per_period_rf = risk_free_rate / periods_per_year
    excess = arr - per_period_rf
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def equity_curve(returns: np.ndarray | pd.Series, initial: float = 1.0) -> np.ndarray:
    """Compounded equity curve from a return series, starting at ``initial``."""
    arr = _as_array(returns)
    return initial * np.cumprod(1.0 + arr)


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the compounded equity curve.

    Returned as a non-positive fraction (e.g. ``-0.25`` for a 25% drawdown).
    """
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + arr)
    running_peak = np.maximum.accumulate(eq)
    drawdowns = eq / running_peak - 1.0
    return float(drawdowns.min())


def hit_rate(returns: np.ndarray | pd.Series) -> float:
    """Fraction of *active* periods with a strictly positive return.

    Periods with exactly zero return (typically flat/no-position bars) are
    excluded from both numerator and denominator. Returns ``0.0`` if there are
    no active periods.
    """
    arr = _as_array(returns)
    active = arr[arr != 0.0]
    if active.size == 0:
        return 0.0
    return float(np.mean(active > 0.0))


def exposure(positions: np.ndarray | pd.Series) -> float:
    """Fraction of bars during which a (non-zero) position was held."""
    arr = _as_array(positions)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr != 0.0))


def summary(
    returns: np.ndarray | pd.Series,
    positions: np.ndarray | pd.Series | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, float]:
    """Compute a dictionary of headline performance statistics.

    Parameters
    ----------
    returns:
        Per-period net returns of the strategy.
    positions:
        Optional per-bar positions; if supplied, ``exposure`` is included.
    periods_per_year:
        Annualization factor.

    Returns
    -------
    dict
        Keys: ``total_return``, ``cagr``, ``ann_volatility``, ``sharpe``,
        ``max_drawdown``, ``hit_rate`` and (if positions given) ``exposure``.
    """
    stats = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
    }
    if positions is not None:
        stats["exposure"] = exposure(positions)
    return stats
