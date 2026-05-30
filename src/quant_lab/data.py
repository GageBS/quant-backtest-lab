"""Synthetic data generation and dataset splitting utilities.

All price data produced here is *synthetic* — generated from a seeded geometric
Brownian motion (GBM). It carries no real-world information and exists only so
the rest of the toolkit has something deterministic and reproducible to operate
on in tests, demos, and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def generate_gbm_prices(
    n: int = 1_000,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.20,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    seed: int | None = 42,
) -> pd.Series:
    """Generate a synthetic close-price series from geometric Brownian motion.

    The closed-form GBM step is used::

        S_{t+1} = S_t * exp((mu - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * Z)

    Parameters
    ----------
    n:
        Number of price observations to produce (including the initial price).
    s0:
        Initial price level (must be positive).
    mu:
        Annualized drift.
    sigma:
        Annualized volatility (must be non-negative).
    dt:
        Time step in years. Defaults to one trading day.
    seed:
        Seed for the random number generator. Use ``None`` for nondeterminism.

    Returns
    -------
    pandas.Series
        A price series of length ``n`` indexed by a synthetic business-day
        ``DatetimeIndex`` and named ``"close"``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(seed)
    # n - 1 random shocks produce n price points (including s0).
    shocks = rng.standard_normal(n - 1)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * shocks
    log_increments = drift + diffusion

    log_prices = np.concatenate([[np.log(s0)], np.log(s0) + np.cumsum(log_increments)])
    prices = np.exp(log_prices)

    index = pd.date_range(start="2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=index, name="close")


def generate_ohlc(
    n: int = 1_000,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.20,
    intrabar_vol: float = 0.5,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLC frame around a GBM close series.

    The open/high/low are derived from the close path with a small synthetic
    intrabar perturbation so that ``low <= open, close <= high`` holds by
    construction. This is illustrative only — it is **not** a market microstructure
    model.

    Parameters
    ----------
    n, s0, mu, sigma, seed:
        Passed through to :func:`generate_gbm_prices`.
    intrabar_vol:
        Scales the synthetic intrabar range as a fraction of the daily GBM vol.

    Returns
    -------
    pandas.DataFrame
        Columns ``["open", "high", "low", "close"]`` indexed by business days.
    """
    close = generate_gbm_prices(n=n, s0=s0, mu=mu, sigma=sigma, seed=seed)
    rng = np.random.default_rng(None if seed is None else seed + 1)

    daily_sigma = sigma / np.sqrt(TRADING_DAYS_PER_YEAR)
    span = intrabar_vol * daily_sigma * close.to_numpy()

    # Open is previous close (first open == first close); add a small gap noise.
    prev_close = close.shift(1).fillna(close.iloc[0]).to_numpy()
    gap = rng.standard_normal(n) * span * 0.25
    open_ = prev_close + gap

    base = np.maximum(open_, close.to_numpy())
    floor = np.minimum(open_, close.to_numpy())
    high = base + np.abs(rng.standard_normal(n)) * span
    low = floor - np.abs(rng.standard_normal(n)) * span

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close.to_numpy(),
        },
        index=close.index,
    )
    return frame


def train_test_split(
    series: pd.Series | pd.DataFrame,
    train_frac: float = 0.7,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """Split a time-ordered series/frame into contiguous train and test parts.

    No shuffling is performed — ordering is preserved, as required for time
    series. ``train_frac`` controls the fraction of observations in the train set.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    n = len(series)
    cut = int(round(n * train_frac))
    cut = max(1, min(cut, n - 1))
    return series.iloc[:cut], series.iloc[cut:]


@dataclass(frozen=True)
class WalkForwardWindow:
    """A single walk-forward fold expressed as integer index ranges.

    Attributes
    ----------
    fold:
        Zero-based fold number.
    train_start, train_end:
        Half-open ``[train_start, train_end)`` index range for the in-sample slice.
    test_start, test_end:
        Half-open ``[test_start, test_end)`` index range for the out-of-sample slice.
    """

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_windows(
    n: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    anchored: bool = False,
) -> Iterator[WalkForwardWindow]:
    """Yield rolling (or anchored) walk-forward windows over ``n`` observations.

    Parameters
    ----------
    n:
        Total number of observations available.
    train_size:
        Number of in-sample observations per fold (minimum window if anchored).
    test_size:
        Number of out-of-sample observations per fold.
    step:
        Advance between successive folds. Defaults to ``test_size`` (non-overlapping
        test windows).
    anchored:
        If ``True``, the training window always starts at index 0 and grows; if
        ``False`` (default), it rolls forward with a fixed length.

    Yields
    ------
    WalkForwardWindow
        Successive folds until the data is exhausted.
    """
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    step = test_size if step is None else step
    if step < 1:
        raise ValueError("step must be >= 1")

    fold = 0
    test_start = train_size
    while test_start + test_size <= n:
        train_start = 0 if anchored else test_start - train_size
        yield WalkForwardWindow(
            fold=fold,
            train_start=train_start,
            train_end=test_start,
            test_start=test_start,
            test_end=test_start + test_size,
        )
        fold += 1
        test_start += step
