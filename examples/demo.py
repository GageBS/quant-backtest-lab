"""End-to-end demonstration of quant_lab.

Run with::

    python examples/demo.py

This script:

1. Generates a synthetic GBM price series.
2. Runs the *illustrative* moving-average-crossover signal through the
   walk-forward harness and prints a per-fold + aggregate metrics table.
3. Builds a set of synthetic probabilistic forecasts, scores them with the
   Brier score, and saves a reliability diagram to ``examples/reliability.png``.

Everything here is synthetic and educational — there is no real strategy or alpha.
Only numpy, pandas, and matplotlib are required.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Allow running directly from a checkout without installing the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, os.pardir, "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from quant_lab import calibration, data, metrics, sizing, walkforward  # noqa: E402


def _fmt_metrics(stats: dict[str, float]) -> str:
    order = [
        "total_return",
        "cagr",
        "ann_volatility",
        "sharpe",
        "max_drawdown",
        "hit_rate",
        "exposure",
    ]
    parts = []
    for key in order:
        if key in stats:
            parts.append(f"{key:>14}: {stats[key]:+.4f}")
    return "\n".join(parts)


def run_backtest_demo() -> None:
    print("=" * 64)
    print("1) WALK-FORWARD BACKTEST  (illustrative MA-crossover, synthetic data)")
    print("=" * 64)

    prices = data.generate_gbm_prices(n=1_500, s0=100.0, mu=0.06, sigma=0.20, seed=7)
    print(f"Synthetic price series: {len(prices)} bars, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    report = walkforward.run_walk_forward(
        prices,
        train_size=252,
        test_size=63,
        cost_bps=1.0,
    )

    print(f"\nFolds evaluated: {len(report.folds)}")
    per_fold = report.per_fold_metrics
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}", "display.width", 120):
        cols = ["fold", "test_start", "n_obs", "cagr", "sharpe", "max_drawdown", "hit_rate", "exposure"]
        print("\nPer-fold (out-of-sample) metrics:")
        print(per_fold[cols].to_string(index=False))

    print("\nAggregate metrics on stitched out-of-sample return stream:")
    print(_fmt_metrics(report.aggregate_metrics()))

    print("\nNOTE: the MA-crossover is a textbook placeholder to exercise the")
    print("      engine. Any apparent edge is an artifact of one synthetic path.")


def run_calibration_demo() -> None:
    print("\n" + "=" * 64)
    print("2) PROBABILITY CALIBRATION  (synthetic probabilistic forecasts)")
    print("=" * 64)

    rng = np.random.default_rng(123)
    n = 5_000

    # Construct a deliberately *slightly overconfident* forecaster: the true event
    # probability is a shrunk version of the stated probability.
    stated = rng.uniform(0.0, 1.0, size=n)
    true_prob = 0.15 + 0.70 * stated  # compresses toward the middle
    outcomes = (rng.uniform(size=n) < true_prob).astype(float)

    bs = calibration.brier_score(stated, outcomes)
    curve = calibration.reliability_curve(stated, outcomes, n_bins=10)

    print(f"Samples: {n}")
    print(f"Brier score: {bs:.4f}  (lower is better; 0.25 = always-0.5 baseline)")
    print("\nReliability table (bin -> mean predicted vs observed frequency):")
    for i, (mp, of, c) in enumerate(
        zip(curve.mean_predicted, curve.observed_freq, curve.counts)
    ):
        if c > 0:
            print(f"  bin {i:>2}: pred={mp:.3f}  observed={of:.3f}  n={c}")

    out_path = os.path.join(_HERE, "reliability.png")
    saved = calibration.plot_reliability_diagram(
        stated,
        outcomes,
        n_bins=10,
        out_path=out_path,
        title="Synthetic forecaster reliability",
    )
    print(f"\nSaved reliability diagram -> {saved}")


def run_sizing_demo() -> None:
    print("\n" + "=" * 64)
    print("3) KELLY POSITION SIZING  (textbook formulas)")
    print("=" * 64)

    for p in (0.50, 0.55, 0.60, 0.70):
        full = sizing.kelly_fraction(p, payoff=1.0)
        half = sizing.fractional_kelly(p, payoff=1.0, fraction=0.5)
        print(f"  win_prob={p:.2f}  even-money  full-Kelly={full:.3f}  half-Kelly={half:.3f}")

    probs = np.array([0.30, 0.50, 0.55, 0.65, 0.90])
    sizes = sizing.probabilities_to_positions(probs, payoff=1.0, fraction=0.5, max_position=0.25)
    print("\n  Mapping model probabilities -> capped (half-Kelly, max 0.25) positions:")
    for prob, size in zip(probs, sizes):
        print(f"    prob={prob:.2f} -> position={size:.3f}")


def main() -> None:
    run_backtest_demo()
    run_calibration_demo()
    run_sizing_demo()
    print("\nDone. (All data synthetic; no real strategy, parameters, or alpha.)")


if __name__ == "__main__":
    main()
