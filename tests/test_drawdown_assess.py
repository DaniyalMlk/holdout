"""Judging a record's drawdown against the law its own parameters imply."""

from __future__ import annotations

import math

import numpy as np
import pytest

from holdout import (
    assess_drawdown,
    drawdown_quantile,
    drawdown_survival,
    expected_maximum_drawdown,
    maximum_drawdown,
)
from holdout.exceptions import ValidationError


def test_the_assessment_agrees_with_the_primitives_it_is_built_from() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0004, 0.01, 252)
    got = assess_drawdown(returns)
    horizon = float(returns.size)
    assert got.periods == 252
    assert got.estimated is True
    assert got.drift == pytest.approx(float(np.mean(returns)))
    assert got.volatility == pytest.approx(float(np.std(returns, ddof=1)))
    assert got.expected == pytest.approx(
        expected_maximum_drawdown(horizon, got.drift, got.volatility)
    )
    assert got.percentile == pytest.approx(
        drawdown_survival(got.observed.depth, horizon, got.drift, got.volatility)
    )
    assert got.exceedance == pytest.approx(1.0 - got.percentile)
    assert got.limit == pytest.approx(drawdown_quantile(0.95, horizon, got.drift, got.volatility))
    assert got.exceedance_bound <= got.exceedance + 1e-13


def test_supplying_the_parameters_marks_the_result_as_not_estimated() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0004, 0.01, 500)
    assert assess_drawdown(returns, drift=0.0004, volatility=0.01).estimated is False
    assert assess_drawdown(returns, drift=0.0004).estimated is True
    assert assess_drawdown(returns, volatility=0.01).estimated is True


def test_the_null_is_invariant_to_how_the_periods_are_labelled() -> None:
    """Nothing is annualised, because the law does not need it to be.

    Scaling every return by a constant scales the drawdown and the volatility
    together, so the percentile does not move.
    """
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0003, 0.012, 300)
    base = assess_drawdown(returns)
    scaled = assess_drawdown(returns * 7.0)
    assert scaled.percentile == pytest.approx(base.percentile, rel=1e-10)
    assert scaled.expected == pytest.approx(7.0 * base.expected, rel=1e-10)


def test_a_record_that_only_went_up_is_reported_as_such() -> None:
    got = assess_drawdown(np.linspace(0.001, 0.002, 50))
    assert got.observed.depth == 0.0
    assert got.percentile == 0.0
    assert got.exceedance == 1.0


def test_a_record_with_no_variation_is_refused() -> None:
    with pytest.raises(ValidationError, match="no variation"):
        assess_drawdown([0.01] * 20, volatility=None)


def test_the_realised_drawdown_sits_just_below_the_predicted_one() -> None:
    """And the gap is the monitoring gap, not a mistake in the law.

    The continuous law predicts the drawdown of the path; the record shows the
    drawdown of 252 marks on it, which is 7% smaller in the mean. The ratio of
    averages lands near 0.84 rather than 0.93 because the drift and volatility
    are re-estimated on each record too.
    """
    rng = np.random.default_rng(11)
    realised, predicted = [], []
    for _ in range(150):
        returns = rng.normal(0.0004, 0.01, 252)
        got = assess_drawdown(returns)
        realised.append(got.observed.depth)
        predicted.append(got.expected)
    ratio = float(np.mean(realised) / np.mean(predicted))
    assert 0.80 < ratio < 0.90, ratio


def test_the_plug_in_exceedance_understates_and_both_causes_are_measured() -> None:
    """Two one-sided effects, separated.

    With the drift and volatility known, a nominal 5% test on 252 daily marks
    rejects 3.4% of the time: the drawdown is read off the marks while the law
    describes the path under them. Estimating the two parameters from the same
    path takes it to 0.27%, a further factor of thirteen, and that is the larger
    effect by far.
    """
    rng = np.random.default_rng(99)
    records = rng.normal(0.0004, 0.01, (6000, 252))
    depths = np.array([maximum_drawdown(row).depth for row in records])
    known = np.array([1.0 - drawdown_survival(d, 252.0, 0.0004, 0.01) for d in depths])
    plug_in = np.array(
        [
            1.0
            - drawdown_survival(
                maximum_drawdown(row).depth,
                252.0,
                float(row.mean()),
                float(row.std(ddof=1)),
            )
            for row in records
        ]
    )
    known_size = float((known <= 0.05).mean())
    plug_in_size = float((plug_in <= 0.05).mean())
    assert 0.028 < known_size < 0.041, known_size
    assert 0.001 < plug_in_size < 0.006, plug_in_size
    assert known_size / plug_in_size > 5.0
    assert float((plug_in <= 0.01).mean()) < 0.003


def test_a_drawdown_far_past_the_null_is_flagged() -> None:
    """A record whose returns are normal but whose drawdown was engineered deep."""
    rng = np.random.default_rng(31)
    returns = rng.normal(0.0, 0.004, 400)
    # One long losing stretch in the middle, then a recovery, so that the
    # overall mean and volatility stay close to the original.
    returns[150:250] -= 0.006
    returns[250:350] += 0.006
    got = assess_drawdown(returns)
    assert got.observed.depth > got.limit
    assert got.exceedance < 0.05
    assert got.percentile > 0.95


def test_the_quoted_figures_for_a_strategy_with_no_edge() -> None:
    """The headline number, stated once so a change to it is visible."""
    # Ten years of daily returns, 15% annual volatility, no edge at all.
    vol = 0.15 / math.sqrt(252)
    horizon = 2520.0
    assert expected_maximum_drawdown(horizon, 0.0, vol) == pytest.approx(0.5944, abs=5e-4)
    assert drawdown_quantile(0.95, horizon, 0.0, vol) == pytest.approx(1.0632, abs=5e-4)
    # With a 1.0 Sharpe ratio the expected worst drawdown is far smaller, but
    # not small: a fifth of the money.
    drift = 0.15 / 252
    assert expected_maximum_drawdown(horizon, drift, vol) == pytest.approx(0.2783, abs=5e-4)
