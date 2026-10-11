"""The expected drawdown, its quantiles, and what discrete marks miss."""

from __future__ import annotations

import math

import numpy as np
import pytest

from holdout import (
    drawdown_exceedance,
    drawdown_quantile,
    drawdown_survival,
    expected_maximum_drawdown,
    final_drawdown_exceedance,
    maximum_drawdown,
    mean_time_to_drawdown,
)
from holdout.exceptions import ValidationError


def test_expected_drawdown_at_zero_drift_is_its_closed_form() -> None:
    """``E[MDD] == sigma sqrt(pi T / 2)`` when there is no edge.

    Lévy's theorem makes the drawdown of a driftless Brownian motion a reflected
    Brownian motion, whose maximum is distributed as the maximum absolute value
    of the original, and ``E[max |B|] == sqrt(pi / 2)`` over a unit interval.
    """
    for horizon, vol in [(1.0, 0.15), (4.0, 0.2), (0.25, 0.3), (10.0, 0.1)]:
        ours = expected_maximum_drawdown(horizon, 0.0, vol)
        exact = vol * math.sqrt(math.pi * horizon / 2.0)
        assert ours == pytest.approx(exact, rel=5e-9)


def test_a_strategy_with_no_edge_still_expects_a_drawdown() -> None:
    # The number worth quoting: 1.2533 standard deviations of the horizon.
    assert expected_maximum_drawdown(1.0, 0.0, 1.0) == pytest.approx(1.2533141373, rel=1e-8)


def test_expected_drawdown_grows_logarithmically_once_there_is_an_edge() -> None:
    """Square root in the horizon without an edge, logarithmic with one.

    The increment per doubling approaches ``sigma**2 / (2 drift) log 2`` from
    below, which is the thing a drawdown limit scaled to the length of a track
    record gets wrong.
    """
    drift, vol = 0.08, 0.15
    limit = vol**2 / (2.0 * drift) * math.log(2.0)
    values = [
        expected_maximum_drawdown(t, drift, vol) for t in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
    ]
    steps = np.diff(values)
    assert np.all(steps > 0.0)
    assert np.all(np.diff(steps) > 0.0)  # rising towards the limit
    assert np.all(steps < limit)
    assert steps[-1] > 0.9 * limit
    # Without an edge the same doublings grow without bound, like sqrt(2).
    flat = [expected_maximum_drawdown(t, 0.0, vol) for t in (5.0, 10.0, 20.0, 40.0)]
    ratios = np.array(flat[1:]) / np.array(flat[:-1])
    assert ratios == pytest.approx(math.sqrt(2.0), rel=1e-9)


def test_expected_drawdown_is_the_integral_of_the_exceedance() -> None:
    """Checked against a coarse but independent trapezoid of the same integrand."""
    for horizon, drift, vol in [(1.0, 0.0, 0.15), (3.0, 0.08, 0.2), (2.0, -0.04, 0.25)]:
        grid = np.linspace(1e-6, 12.0 * vol * math.sqrt(horizon) + abs(drift) * horizon, 20001)
        values = [drawdown_exceedance(float(x), horizon, drift, vol) for x in grid]
        rough = float(np.trapezoid(values, grid)) + grid[0]
        assert expected_maximum_drawdown(horizon, drift, vol) == pytest.approx(rough, rel=1e-6)


def test_quantiles_invert_the_law() -> None:
    for probability in (0.01, 0.25, 0.5, 0.8, 0.95, 0.99, 0.999):
        level = drawdown_quantile(probability, 3.0, 0.08, 0.15)
        assert drawdown_survival(level, 3.0, 0.08, 0.15) == pytest.approx(probability, abs=1e-12)


def test_quantiles_are_increasing_and_straddle_the_mean() -> None:
    args = (5.0, 0.06, 0.16)
    levels = [drawdown_quantile(p, *args) for p in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)]
    assert np.all(np.diff(levels) > 0.0)
    mean = expected_maximum_drawdown(*args)
    assert levels[2] < mean < levels[4]


def test_mean_passage_time_is_exponential_in_the_level() -> None:
    drift, vol = 0.08, 0.15
    # A fifth of the money, at a 0.53 Sharpe ratio: about three years.
    assert 3.0 < mean_time_to_drawdown(0.2, drift, vol) < 3.1
    # Tripling the level does not triple the wait.
    assert mean_time_to_drawdown(0.6, drift, vol) / mean_time_to_drawdown(0.2, drift, vol) > 30.0
    # At zero drift it is the elementary level**2 / variance.
    assert mean_time_to_drawdown(0.3, 0.0, 0.2) == pytest.approx(0.09 / 0.04, rel=1e-14)
    # The growth rate in the level approaches 2 drift / sigma**2 from *above*,
    # and it is worth measuring rather than asserting: at a 20% level the local
    # rate is still 22% over the limit, because the exact mean carries a term
    # linear in the level that the exponential has not yet buried.
    limit = 2.0 * drift / vol**2
    rates = [
        math.log(
            mean_time_to_drawdown(level + 0.05, drift, vol)
            / mean_time_to_drawdown(level - 0.05, drift, vol)
        )
        / 0.1
        for level in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
    ]
    assert np.all(np.diff(rates) < 0.0)
    assert all(rate > limit for rate in rates)
    assert rates[0] / limit > 1.2
    assert rates[-1] == pytest.approx(limit, rel=1e-4)


# --------------------------------------------------------------------------- #
# the end-of-horizon drawdown, as a bound and in the deep tail


def test_final_drawdown_reduces_to_the_folded_normal_at_zero_drift() -> None:
    """Lévy again: with no drift the drawdown at a fixed time is ``|N(0, sigma**2 T)|``."""
    for level, horizon, vol in [(0.1, 1.0, 0.2), (0.4, 2.0, 0.15), (1.0, 0.5, 0.3)]:
        z = level / (vol * math.sqrt(horizon))
        folded = math.erfc(z / math.sqrt(2.0))
        assert final_drawdown_exceedance(level, horizon, 0.0, vol) == pytest.approx(
            folded, rel=1e-14
        )


def test_the_final_drawdown_bounds_the_maximum_everywhere() -> None:
    """A record that ends this far down has been this far down. No exceptions."""
    for level in (0.02, 0.1, 0.3, 0.6, 1.0):
        for drift in (-0.05, 0.0, 0.05, 0.2):
            for vol in (0.1, 0.2, 0.4):
                for horizon in (0.25, 1.0, 8.0):
                    bound = final_drawdown_exceedance(level, horizon, drift, vol)
                    exact = drawdown_exceedance(level, horizon, drift, vol)
                    assert bound <= exact + 1e-13, (level, drift, vol, horizon, bound, exact)


def test_the_bound_is_the_right_order_in_the_tail() -> None:
    """Rising towards about 0.45 of the exact exceedance, not falling away from it."""
    ratios = []
    for level in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        exact = drawdown_exceedance(level, 1.0, 0.1, 0.15)
        ratios.append(final_drawdown_exceedance(level, 1.0, 0.1, 0.15) / exact)
    assert np.all(np.diff(ratios) > 0.0)
    assert 0.35 < ratios[0] < 0.37
    assert 0.44 < ratios[-1] < 0.46


def test_the_bound_still_resolves_where_the_expansion_cannot() -> None:
    # The expansion's exceedance is a complement of a number near one and is
    # zero to round-off here; the closed form is not.
    # What the expansion returns here is the round-off of a sum near one, at
    # 1.1e-16, and not a probability of 1.1e-16.
    assert drawdown_exceedance(2.0, 1.0, 0.1, 0.15) < 1e-15
    assert 1e-45 < final_drawdown_exceedance(2.0, 1.0, 0.1, 0.15) < 1e-43
    assert final_drawdown_exceedance(3.0, 1.0, 0.1, 0.15) > 0.0


def test_the_exponential_reading_of_the_tail_is_wrong_and_by_how_much() -> None:
    """The long-horizon tail is exponential; the deep tail at a fixed horizon is not.

    Treating the first passage as exponential with its own exact mean -- which
    is what "a drawdown this deep arrives once every E[tau] years" amounts to --
    overstates the exceedance, and the overstatement grows as the level deepens
    rather than settling.
    """
    drift, vol, horizon = 0.1, 0.15, 1.0
    errors = []
    for level in (0.3, 0.4, 0.5, 0.6):
        exponential = -math.expm1(-horizon / mean_time_to_drawdown(level, drift, vol))
        errors.append(exponential / drawdown_exceedance(level, horizon, drift, vol))
    assert np.all(np.diff(errors) > 0.0)
    assert errors[-1] > 500.0
    # Whereas at a fixed level and a long horizon it is right: the exceedance
    # approaches one at the stationary rate.
    long_run = [
        drawdown_exceedance(0.3, t, drift, vol)
        / -math.expm1(-t / mean_time_to_drawdown(0.3, drift, vol))
        for t in (50.0, 200.0, 800.0)
    ]
    assert long_run[-1] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------- #
# what daily marks miss


def discrete_maximum_drawdowns(
    horizon: float, drift: float, vol: float, periods: int, draws: int, seed: int
) -> np.ndarray:
    """Maximum drawdowns of a random walk observed exactly ``periods`` times.

    No discretisation error in the *estimator*: the drawdown of a discretely
    observed record is a function of the observations, and this computes it from
    them. The gap against the continuous law is the gap between two different
    contracts, not an error.
    """
    rng = np.random.default_rng(seed)
    step = horizon / periods
    chunk = max(1, 2_000_000 // periods)
    out = np.empty(draws)
    done = 0
    while done < draws:
        rows = min(chunk, draws - done)
        increments = rng.normal(drift * step, vol * math.sqrt(step), (rows, periods))
        curve = np.concatenate((np.zeros((rows, 1)), np.cumsum(increments, axis=1)), axis=1)
        out[done : done + rows] = (np.maximum.accumulate(curve, axis=1) - curve).max(axis=1)
        done += rows
    return out


def test_the_simulated_statistic_agrees_with_the_path_statistic() -> None:
    """The simulation helper and :func:`maximum_drawdown` must be the same thing."""
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0004, 0.01, 400)
    curve = np.concatenate(([0.0], np.cumsum(returns)))
    by_hand = float((np.maximum.accumulate(curve) - curve).max())
    assert maximum_drawdown(returns).depth == pytest.approx(by_hand)


def test_discrete_monitoring_understates_the_drawdown_and_the_gap_dies_like_one_over_root_n() -> (
    None
):
    """Daily marks see less of the path than the path contains.

    The expected maximum drawdown of a walk observed ``n`` times approaches the
    continuous value from below, and the shortfall halves as ``n`` quadruples.
    Extrapolating in ``1 / sqrt(n)`` lands on the continuous law, which is the
    check: a bias with the wrong exponent would not.
    """
    horizon, drift, vol = 1.0, 0.0, 0.2
    exact = expected_maximum_drawdown(horizon, drift, vol)
    draws = 60_000
    shortfalls = []
    for index, periods in enumerate((64, 256, 1024)):
        sample = discrete_maximum_drawdowns(horizon, drift, vol, periods, draws, 100 + index)
        shortfalls.append(exact - float(sample.mean()))
    assert all(value > 0.0 for value in shortfalls)
    ratios = np.array(shortfalls[:-1]) / np.array(shortfalls[1:])
    assert np.all(ratios > 1.7)
    assert np.all(ratios < 2.3)


def test_the_size_of_a_daily_record_tested_against_the_continuous_null() -> None:
    """The continuous law is conservative on discretely observed data, measured.

    A test at the 5% level, applied to a drawdown read off 252 daily marks but
    compared against the law of the path underneath, rejects less often than it
    claims. That is the right direction -- it will not call a drawdown
    remarkable when it is not -- and the distortion is worth a number.
    """
    horizon, drift, vol = 1.0, 0.0, 0.2
    sample = discrete_maximum_drawdowns(horizon, drift, vol, 252, 120_000, 7)
    critical = drawdown_quantile(0.95, horizon, drift, vol)
    size = float((sample >= critical).mean())
    assert 0.02 < size < 0.045, size
    # Monitoring 16 times more often recovers most of it.
    denser = discrete_maximum_drawdowns(horizon, drift, vol, 4032, 60_000, 8)
    assert float((denser >= critical).mean()) > size


def test_derived_quantities_refuse_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        expected_maximum_drawdown(0.0, 0.0, 0.15)
    with pytest.raises(ValidationError):
        expected_maximum_drawdown(1.0, float("inf"), 0.15)
    with pytest.raises(ValidationError):
        drawdown_quantile(1.5, 1.0, 0.0, 0.15)
    with pytest.raises(ValidationError):
        mean_time_to_drawdown(-1.0, 0.0, 0.15)
    with pytest.raises(ValidationError):
        final_drawdown_exceedance(0.1, 1.0, 0.0, 0.0)


def test_the_conditioning_budget_is_looser_for_an_integral_than_for_a_probability() -> None:
    """And the slack is what the domain on the losing side is bought with.

    The boundary is a property of the dimensionless groups, so it is the same
    total Sharpe ratio at every horizon and volatility.
    """
    for horizon, vol in [(1.0, 0.15), (16.0, 0.3), (0.5, 0.05)]:
        low, high = 0.1, 10.0
        for _ in range(30):
            mid = 0.5 * (low + high)
            try:
                expected_maximum_drawdown(horizon, -mid * vol / math.sqrt(horizon), vol)
            except ValidationError:
                high = mid
            else:
                low = mid
        assert low == pytest.approx(3.23, abs=0.02)
    # A probability a caller reads directly is held to the strict budget, which
    # is five orders tighter, and that is deliberate.
    with pytest.raises(ValidationError, match="cancellation"):
        drawdown_survival(2.0, 4.0, -0.3, 0.15)
    assert 0.0 < drawdown_survival(2.0, 4.0, -0.3, 0.15, max_conditioning=1e12) <= 1.0
