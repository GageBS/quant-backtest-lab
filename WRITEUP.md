# Technical write-up: why this toolkit is built the way it is

> **Disclaimer up front.** This is an **educational framework** running entirely on
> **seeded synthetic data**. The strategy it ships with is a deliberately trivial
> **moving-average crossover placeholder** that exists only to exercise the engine.
> There is **no real alpha, no proprietary parameters, and no claim of an edge**.
> Every number below is an *illustrative* result on one synthetic random path,
> reproduced by running `python examples/demo.py`. Do not read these figures as a
> trading result.

I built `quant-backtest-lab` to show the part of quantitative research that
actually decides whether a result is real: the *engineering* around the idea.
A clever signal is worthless if the harness that evaluates it leaks future
information, ignores costs, overfits to one slice of history, or sizes bets off
probabilities that don't mean what they claim. This write-up walks through the
five design choices that address exactly those failure modes, and quotes the
real output the demo produced on my machine.

## Reproducible backtesting and calibration

The first principle is determinism. All randomness flows through
`numpy.random.default_rng(seed)`, so the demo and the 33-test suite produce the
same numbers on every run and in CI. Reproducibility is not a nicety here — it is
what lets a reviewer trust that a reported metric came from the code rather than
from a lucky seed someone forgot to pin. The same discipline underlies the
calibration work: a probability is a falsifiable claim about long-run frequency,
and you can only check it if the experiment is repeatable.

## No look-ahead by construction

The most common way a backtest lies is by acting on information it would not have
had in real time. I designed the engine so that look-ahead is *structurally*
impossible rather than something you have to remember to avoid. The position
signal is lagged exactly one bar before it touches forward returns: the position
chosen using information available at bar `t` earns the return realized from `t`
to `t+1`. Transaction costs are charged on **turnover** — the absolute change in
position `|Δposition|` at a configurable basis-point rate — so a flat-to-long
round trip costs `2 × cost_bps`, and a strategy that never trades produces exactly
zero PnL. That last property is asserted directly in the tests, which is the kind
of invariant that catches a sign error before it becomes a fake Sharpe ratio.

## The walk-forward harness

A single train/test split tells you almost nothing; it is one draw from a noisy
distribution. The harness instead rolls a fixed in-sample window forward, warms up
the rolling indicators on the data immediately preceding each test window, and
evaluates strictly out-of-sample, then **stitches the OOS return streams** into one
continuous equity curve and reports both per-fold and aggregate metrics. On the
synthetic series (1,500 bars, a 252-bar train / 63-bar test cadence) the demo
evaluated **19 out-of-sample folds**. The per-fold table makes the dispersion
impossible to hide: fold Sharpes ranged from roughly +3.3 to −6.4, and the
stitched aggregate came out to `sharpe: -0.4470`, `cagr: -0.0668`, `total_return:
-0.2800`, `max_drawdown: -0.3336`. That negative aggregate is *the point* — a
textbook MA crossover on a driftless synthetic path has no edge, and the harness
says so loudly instead of letting one good fold flatter the result.

## Why calibration matters for sized betting

If you intend to *size* positions from model probabilities, the probabilities
have to mean what they say. A forecaster can be accurate on average and still be
systematically overconfident at the extremes, and that miscalibration is exactly
what blows up a Kelly-sized book. The demo builds a deliberately overconfident
synthetic forecaster (the true event rate is a shrunk version of the stated
probability) and measures it two ways. The Brier score came out to **0.2098**,
modestly better than the always-0.5 baseline of 0.25. But the single number hides
the structure; the reliability table exposes it. At the top bin the model
predicted **0.951** while the observed frequency was only **0.836**, and at the
bottom bin it predicted **0.050** against an observed **0.176** — textbook
overconfidence, where stated extremes are too extreme. The reliability diagram
(`examples/reliability.png`) makes that S-shaped deviation from the diagonal
obvious at a glance. This is why I treat calibration as a first-class concern
rather than an afterthought: the Brier score tells you *whether* there is a
problem, and the reliability curve tells you *where*.

## Kelly sizing

The last module closes the loop from probability to position. It implements full
Kelly, fractional (e.g. half) Kelly, and a capped mapping from model probabilities
to position sizes. The formulas are textbook: at even money a 55% edge implies a
0.10 full-Kelly stake (0.05 at half-Kelly), and the capped mapper refuses to bet
when the probability is at or below break-even. Fractional Kelly is the practical
default precisely *because* of the calibration point above — if your probabilities
are even slightly overconfident, full Kelly over-bets and the geometric growth
advantage turns into ruin risk. Half-Kelly buys a large margin of safety against
exactly the miscalibration the reliability diagram surfaces.

## What I want a reader to take away

The interesting engineering in quant research is defensive: lag the signal, charge
the costs, walk it forward, check the probabilities, and size conservatively. This
repo is a clean, tested, reproducible demonstration of those habits on data that
is honestly labeled as synthetic — the scaffolding you would drop a *real* signal
into, not the signal itself.
