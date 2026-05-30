"""Tests for quant_lab.sizing Kelly helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from quant_lab import sizing  # noqa: E402


def test_kelly_even_money_known_edge():
    # p=0.6, b=1 -> f* = 0.6 - 0.4 = 0.2
    assert sizing.kelly_fraction(0.6, payoff=1.0) == pytest.approx(0.2)


def test_kelly_fair_coin_even_money_is_zero():
    assert sizing.kelly_fraction(0.5, payoff=1.0) == pytest.approx(0.0)


def test_kelly_no_edge_clipped_to_zero():
    # p=0.4 even money -> negative raw Kelly, clipped to 0.
    assert sizing.kelly_fraction(0.4, payoff=1.0) == 0.0


def test_kelly_with_higher_payoff():
    # p=0.5, b=2 -> f* = 0.5 - 0.5/2 = 0.25
    assert sizing.kelly_fraction(0.5, payoff=2.0) == pytest.approx(0.25)


def test_fractional_kelly_halves():
    full = sizing.kelly_fraction(0.6, payoff=1.0)
    assert sizing.fractional_kelly(0.6, payoff=1.0, fraction=0.5) == pytest.approx(full * 0.5)


def test_kelly_rejects_bad_prob():
    with pytest.raises(ValueError):
        sizing.kelly_fraction(1.5)


def test_kelly_rejects_nonpositive_payoff():
    with pytest.raises(ValueError):
        sizing.kelly_fraction(0.6, payoff=0.0)


def test_probabilities_to_positions_caps_and_floors():
    probs = np.array([0.3, 0.5, 0.9])
    out = sizing.probabilities_to_positions(probs, payoff=1.0, fraction=1.0, max_position=0.5)
    # p=0.3 -> negative -> 0; p=0.5 -> 0; p=0.9 -> raw 0.8 capped to 0.5
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5)


def test_probabilities_to_positions_half_kelly_value():
    # p=0.7 even money: full Kelly = 0.4, half = 0.2 (under the cap).
    out = sizing.probabilities_to_positions(
        np.array([0.7]), payoff=1.0, fraction=0.5, max_position=1.0
    )
    assert out[0] == pytest.approx(0.2)


def test_probabilities_to_positions_rejects_out_of_range():
    with pytest.raises(ValueError):
        sizing.probabilities_to_positions(np.array([1.2]))
