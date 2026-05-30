# quant-backtest-lab

**A clean, vectorized backtesting and forecast-evaluation toolkit — built on synthetic data, for learning and demonstration.**

`quant-backtest-lab` is a small, well-tested Python library that shows how the
*engineering* pieces of a quantitative research workflow fit together: a fast
vectorized backtester, walk-forward evaluation, standard performance statistics,
probabilistic-forecast calibration, and Kelly position sizing. It runs entirely on
seeded synthetic price data generated in-process — no downloads, no API keys, no
external services.

---

## What this demonstrates

- **Vectorized, event-driven-style backtester** — maps a per-bar position signal
  (`{-1, 0, +1}` or arbitrary weights) to strategy returns and an equity curve,
  with one-bar signal lagging to avoid look-ahead and per-trade transaction costs
  charged on turnover (in basis points).
- **Walk-forward evaluation** — a rolling/anchored harness that "fits" a signal
  in-sample and evaluates it on the next out-of-sample window, then aggregates
  metrics across folds (and stitches the OOS return stream).
- **Performance statistics** — total return, CAGR, annualized Sharpe, max
  drawdown, hit rate, and exposure, implemented as pure, individually testable
  functions.
- **Probability calibration** — Brier score and reliability-curve binning for
  evaluating *probabilistic* forecasts, plus a matplotlib reliability diagram.
- **Kelly position sizing** — full Kelly, fractional (e.g. half) Kelly, and a
  capped mapping from model probabilities to position sizes.

---

## Disclaimer

This is an **educational framework**, not a trading system. It contains **no real
strategy, no proprietary parameters, and no alpha**. All price data is synthetic
(seeded geometric Brownian motion generated in code), and the included
moving-average crossover is a **deliberately trivial, illustrative placeholder**
used only to exercise the backtesting engine. Any apparent "performance" is an
artifact of a single synthetic random path. **Do not use this for real trading
decisions.**

---

## Quickstart

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

# 2. install (editable) with dev extras
pip install -e ".[dev]"

# 3. run the end-to-end demo
python examples/demo.py

# 4. run the test suite
pytest
```

The demo generates synthetic data, runs the illustrative signal through the
walk-forward harness, prints per-fold and aggregate metrics, scores a set of
synthetic probabilistic forecasts with the Brier score, and saves a reliability
diagram to `examples/reliability.png`.

---

## Project tree

```
quant-backtest-lab/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── examples/
│   └── demo.py                 # runnable end-to-end demonstration
├── src/
│   └── quant_lab/
│       ├── __init__.py
│       ├── data.py             # synthetic GBM prices + walk-forward splits
│       ├── backtest.py         # vectorized signal -> returns -> equity engine
│       ├── metrics.py          # CAGR, Sharpe, drawdown, hit rate, exposure
│       ├── calibration.py      # Brier score + reliability diagram
│       ├── sizing.py           # Kelly / fractional-Kelly position sizing
│       └── walkforward.py      # walk-forward evaluation harness
└── tests/
    ├── test_metrics.py
    ├── test_backtest.py
    └── test_sizing.py
```

---

## Design notes

- **No look-ahead by construction.** The backtester lags the signal one bar
  before applying it to forward returns, so a position chosen using information
  available at bar `t` earns the return realized from `t` to `t+1`.
- **Costs on turnover.** Transaction costs are charged on the *absolute change*
  in position (`|Δposition|`) at a configurable basis-point rate, so a flat-to-long
  round trip costs `2 × cost_bps`. A flat signal therefore produces exactly zero
  PnL — a property the tests assert directly.
- **Pure metric functions.** Each statistic is a standalone pure function over an
  array of returns/positions, which keeps them trivially testable against
  hand-computed values (e.g. Sharpe of a constant return, a known max drawdown).
- **Walk-forward over single splits.** Evaluating on rolling out-of-sample windows
  guards against overfitting to one historical slice; the harness reports both
  per-fold metrics and metrics on the stitched OOS stream. Rolling indicators are
  warmed up using data immediately preceding each test window.
- **Calibration as a first-class concern.** A model that emits probabilities is
  only useful if those probabilities are *calibrated*; the reliability curve makes
  miscalibration (e.g. overconfidence) visually obvious, and the Brier score
  summarizes it in one number.
- **Reproducibility.** All randomness is seeded via `numpy.random.default_rng`, so
  the demo and tests are deterministic.

---

## License

MIT © 2026 Gage Summers. See [LICENSE](LICENSE).
