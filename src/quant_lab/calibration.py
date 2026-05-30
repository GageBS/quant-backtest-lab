"""Probabilistic-forecast evaluation: Brier score and reliability curves.

When a model emits a *probability* (e.g. "65% chance the next bar is up"), we
want to know whether those probabilities are *calibrated*: of all the times the
model said 65%, did the event actually happen ~65% of the time? This module
provides the Brier score and a reliability-curve (calibration) binning, plus a
matplotlib helper to render a reliability diagram to a file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate(probs: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(outcomes, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError("probs and outcomes must have the same shape")
    if p.size == 0:
        raise ValueError("inputs must be non-empty")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probs must lie in [0, 1]")
    if not np.all(np.isin(np.unique(y), (0.0, 1.0))):
        raise ValueError("outcomes must be binary (0 or 1)")
    return p, y


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    ``brier = mean((p - y)**2)``. Lower is better; a perfect forecaster scores
    ``0.0`` and always predicting ``0.5`` on a balanced target scores ``0.25``.
    """
    p, y = _validate(probs, outcomes)
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class ReliabilityCurve:
    """Binned reliability-curve data.

    Attributes
    ----------
    bin_edges:
        The ``n_bins + 1`` bin edges spanning ``[0, 1]``.
    mean_predicted:
        Mean predicted probability within each populated bin (NaN if empty).
    observed_freq:
        Observed event frequency within each populated bin (NaN if empty).
    counts:
        Number of samples in each bin.
    """

    bin_edges: np.ndarray
    mean_predicted: np.ndarray
    observed_freq: np.ndarray
    counts: np.ndarray


def reliability_curve(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> ReliabilityCurve:
    """Bin predictions and compute observed frequency vs. mean prediction.

    Predictions are grouped into ``n_bins`` equal-width bins over ``[0, 1]``.
    For each bin we report the mean predicted probability and the observed event
    frequency; a well-calibrated model has these roughly equal (points hug the
    diagonal).

    Empty bins yield ``NaN`` for ``mean_predicted`` and ``observed_freq`` and a
    count of ``0``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _validate(probs, outcomes)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each prob to a bin in [0, n_bins-1]; clip so p == 1.0 lands in last bin.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = idx == b
        c = int(np.count_nonzero(mask))
        counts[b] = c
        if c > 0:
            mean_pred[b] = float(np.mean(p[mask]))
            obs_freq[b] = float(np.mean(y[mask]))

    return ReliabilityCurve(
        bin_edges=edges,
        mean_predicted=mean_pred,
        observed_freq=obs_freq,
        counts=counts,
    )


def plot_reliability_diagram(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    out_path: str = "reliability.png",
    title: str = "Reliability diagram",
) -> str:
    """Render a reliability diagram (calibration plot) and save it to ``out_path``.

    The plot shows the perfect-calibration diagonal, the model's binned
    reliability curve, and a small histogram of prediction counts per bin. The
    matplotlib import is local so the rest of the package has no hard dependency
    on a display backend.

    Returns
    -------
    str
        The path the figure was written to.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless, file-only backend
    import matplotlib.pyplot as plt

    curve = reliability_curve(probs, outcomes, n_bins=n_bins)
    bs = brier_score(probs, outcomes)
    populated = curve.counts > 0

    fig, (ax_rel, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(6, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax_rel.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax_rel.plot(
        curve.mean_predicted[populated],
        curve.observed_freq[populated],
        marker="o",
        label=f"Model (Brier={bs:.3f})",
    )
    ax_rel.set_ylabel("Observed frequency")
    ax_rel.set_ylim(-0.02, 1.02)
    ax_rel.set_title(title)
    ax_rel.legend(loc="upper left")
    ax_rel.grid(True, alpha=0.3)

    centers = 0.5 * (curve.bin_edges[:-1] + curve.bin_edges[1:])
    ax_hist.bar(centers, curve.counts, width=1.0 / n_bins * 0.9, color="steelblue")
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(-0.02, 1.02)
    ax_hist.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
