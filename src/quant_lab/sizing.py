"""Kelly and fractional-Kelly position sizing.

These are the standard, textbook Kelly formulas. They are provided as clean,
tested utilities — not as advice. In practice full-Kelly sizing is famously
aggressive and sensitive to estimation error, which is why fractional Kelly is
the more common choice.
"""

from __future__ import annotations

import numpy as np


def kelly_fraction(win_prob: float, payoff: float = 1.0) -> float:
    """Optimal Kelly fraction for a single binary bet.

    For a bet that wins ``payoff`` per unit staked with probability ``p`` and
    loses the unit stake with probability ``1 - p``, the growth-optimal fraction
    of bankroll to wager is::

        f* = (p * b - (1 - p)) / b = p - (1 - p) / b

    where ``b`` is the net odds (``payoff``).

    Parameters
    ----------
    win_prob:
        Probability of winning, in ``[0, 1]``.
    payoff:
        Net odds received on a win per unit staked (``b > 0``). ``payoff=1.0``
        corresponds to even money.

    Returns
    -------
    float
        The Kelly fraction, clipped at ``0.0`` from below (no negative/ short
        bet is returned; a non-positive edge yields ``0.0``).
    """
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError("win_prob must be in [0, 1]")
    if payoff <= 0.0:
        raise ValueError("payoff must be positive")
    f = win_prob - (1.0 - win_prob) / payoff
    return float(max(0.0, f))


def fractional_kelly(win_prob: float, payoff: float = 1.0, fraction: float = 0.5) -> float:
    """Scaled-down Kelly fraction (e.g. half-Kelly with ``fraction=0.5``).

    Multiplies the full :func:`kelly_fraction` by ``fraction`` to trade a small
    amount of growth for a large reduction in variance and drawdown.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return fraction * kelly_fraction(win_prob, payoff)


def probabilities_to_positions(
    probs: np.ndarray,
    payoff: float = 1.0,
    fraction: float = 0.5,
    max_position: float = 1.0,
) -> np.ndarray:
    """Map an array of model win-probabilities to capped position sizes.

    For each probability ``p`` the (fractional) Kelly stake is computed treating
    ``p`` as the probability of an up move. Probabilities below the break-even
    threshold produce a flat (``0.0``) position. The resulting sizes are clipped
    to ``[0, max_position]``.

    Parameters
    ----------
    probs:
        Array of predicted win probabilities in ``[0, 1]``.
    payoff:
        Net odds per unit staked.
    fraction:
        Kelly scaling factor (e.g. ``0.5`` for half-Kelly).
    max_position:
        Upper bound applied to each position size.

    Returns
    -------
    numpy.ndarray
        Position sizes, same shape as ``probs``.
    """
    if max_position <= 0:
        raise ValueError("max_position must be positive")
    p = np.asarray(probs, dtype=float).ravel()
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probs must lie in [0, 1]")
    if payoff <= 0.0:
        raise ValueError("payoff must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")

    raw = fraction * (p - (1.0 - p) / payoff)
    return np.clip(raw, 0.0, max_position)
