"""Every entry point refuses a series that does not move.

A constant return series has a standard deviation of zero in real arithmetic
and almost never in floating point: subtracting the mean leaves rounding of
order ``eps * level``, so a guard written as ``sd == 0`` does not fire and the
Sharpe ratio comes back around 1e15. That number then wins every comparison it
is entered into, which is the worst shape a bug can take in a library whose
job is to decide which of a family of strategies is best.

The diagnosis existed in :mod:`holdout.pairwise`, in a comment naming both the
4.3e-19 and the 2.3e15. It had not reached the other five places that make the
same test, and this module is the sweep that says it has now.

The sweep is over lengths *and* magnitudes together, because the residue
depends on both: whether the mean of a constant array is exactly
representable, and so whether anything is left over at all, turns on the
value's binary expansion. That is why a guard at one multiple of epsilon fires
for some pairs and not others, and why no single hand-picked example would
have caught this.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from holdout import (
    InsufficientDataError,
    ValidationError,
    autocorrelation_adjusted_sharpe,
    estimate_sharpe,
    excess_returns,
    financing_impact,
    kurtosis,
    moments,
    sharpe_difference,
    sharpe_ratio,
    skewness,
    trial_correlation,
)
from holdout.series import CONSTANT_TOLERANCE, FloatArray, constant_floor

#: Lengths and magnitudes whose product covers the pairs where the residue is
#: nonzero. 0.001 at a thousand rows is the pair that was escaping.
LENGTHS = (10, 100, 1000, 2000, 10000)
LEVELS = (1e-9, 1e-6, 0.0007, 0.001, 1.0, 1234.5, -0.002)

SERIES_FUNCTIONS: tuple[Callable[[FloatArray], object], ...] = (
    sharpe_ratio,
    estimate_sharpe,
    skewness,
    kurtosis,
    moments,
)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("function", SERIES_FUNCTIONS, ids=lambda f: f.__name__)
def test_a_series_that_does_not_move_is_refused(
    function: Callable[[FloatArray], object], length: int, level: float
) -> None:
    with pytest.raises(ValidationError):
        function(np.full(length, level))


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("level", LEVELS)
def test_a_flat_trial_column_is_named(length: int, level: float) -> None:
    """And named by its index, since the caller has to find it among the others."""
    matrix = np.column_stack([np.linspace(0.0, 1.0, length), np.full(length, level)])
    with pytest.raises(ValidationError, match="trial column 1 has zero variance"):
        trial_correlation(matrix)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_financing_path_refuses_one_too(length: int, level: float) -> None:
    flat = np.full(length, level)
    with pytest.raises(ValidationError):
        financing_impact(flat, 0.0001)
    with pytest.raises(ValidationError):
        financing_impact(flat, np.zeros(length))


@pytest.mark.parametrize("length", [n for n in LENGTHS if n >= 20])
@pytest.mark.parametrize("level", LEVELS)
def test_a_pair_of_flat_series_is_refused_by_name(length: int, level: float) -> None:
    """``InsufficientDataError``, not ``ValidationError``: a flat series is not malformed.

    The two are siblings under ``HoldoutError`` rather than one inside the
    other, and this is the module that has to get the distinction right --
    everything else here raises the validation error, and the pairwise test is
    the one place that treats a motionless series as a shortage of information.
    """
    moving = np.linspace(-0.01, 0.01, length)
    with pytest.raises(InsufficientDataError, match="does not move"):
        sharpe_difference(np.full(length, level), moving)
    with pytest.raises(InsufficientDataError, match="does not move"):
        sharpe_difference(moving, np.full(length, level))


def test_the_autocorrelation_adjusted_ratio_refuses_one_as_well() -> None:
    with pytest.raises(ValidationError):
        autocorrelation_adjusted_sharpe(np.full(1000, 0.001), 12)


def test_a_rate_series_that_cancels_the_returns_is_refused() -> None:
    """The conversion can flatten a series that arrived with variance of its own."""
    x = np.asarray(np.random.default_rng(3).normal(0.0004, 0.01, 500))
    with pytest.raises(ValidationError, match="excess returns have zero variance"):
        financing_impact(x, x - 0.001)


# --- the tolerance itself ---------------------------------------------------


def test_the_tolerance_is_above_what_a_constant_series_leaves_behind() -> None:
    """One half of the choice: it has to catch the rounding, measured.

    Over the lengths and magnitudes above the one-dimensional reduction leaves
    at most about twice ``eps * level``. Eight has margin over that without
    being chosen to accommodate anything.
    """
    eps = float(np.finfo(np.float64).eps)
    worst = 0.0
    for length in LENGTHS:
        for level in LEVELS:
            flat = np.full(length, level)
            worst = max(worst, float(np.std(flat, ddof=1)) / (eps * abs(level)))
    assert worst < 4.0
    assert 2.0 * worst < CONSTANT_TOLERANCE


def test_the_tolerance_is_below_variation_anybody_could_mean() -> None:
    """The other half: it must not refuse a series that genuinely moves.

    A relative standard deviation of 1e-13 is a Sharpe ratio of order 1e13 and
    nothing anyone measures, and it is still accepted. Asserting only the first
    half would leave a guard that could be widened without limit.
    """
    series = 1.0 + np.resize(np.array([1e-13, -1e-13]), 1000)
    assert float(np.std(series, ddof=1)) > constant_floor(series)
    assert sharpe_ratio(series) > 1e12
    tighter = 1.0 + np.resize(np.array([1e-15, -1e-15]), 1000)
    assert float(np.std(tighter, ddof=1)) < constant_floor(tighter)


def test_the_floor_is_relative_to_the_series_own_magnitude() -> None:
    """So that the threshold means the same thing in fractions and in basis points."""
    fractions = np.asarray(np.random.default_rng(1).normal(0.0004, 0.01, 500))
    points = fractions * 10000.0
    assert constant_floor(points) == pytest.approx(constant_floor(fractions) * 10000.0)
    assert sharpe_ratio(points) == pytest.approx(sharpe_ratio(fractions), rel=1e-12)


def test_an_all_zero_series_is_refused_rather_than_compared_against_zero() -> None:
    """A floor of zero would admit a standard deviation of zero, which is the whole point."""
    zeros = np.zeros(100)
    assert constant_floor(zeros) > 0.0
    with pytest.raises(ValidationError):
        sharpe_ratio(zeros)


def test_the_axis_reduction_is_the_reason_the_check_runs_per_column() -> None:
    """The measurement behind the loop in trial_correlation, as a regression.

    ``std(axis=0)`` reduces along a strided axis and leaves far more rounding
    on a constant column than the one-dimensional reduction does — hundreds of
    times as much over the grid below. A guard built on the axis form would
    need a tolerance set by numpy's reduction order rather than by the data, so
    the loop is not a stylistic choice.
    """
    eps = float(np.finfo(np.float64).eps)
    worst_axis = 0.0
    worst_one_dimensional = 0.0
    for length in (500, 1000, 2000, 10000):
        for columns in (2, 3, 5, 30):
            for level in LEVELS:
                matrix = np.empty((length, columns))
                matrix[:] = np.linspace(0.0, 1.0, length)[:, None]
                matrix[:, 0] = level
                scale = eps * abs(level)
                worst_axis = max(worst_axis, float(matrix.std(axis=0, ddof=1)[0]) / scale)
                worst_one_dimensional = max(
                    worst_one_dimensional, float(np.std(matrix[:, 0], ddof=1)) / scale
                )
    # Swept rather than taken at one shape, because which (length, columns,
    # level) triple is worst is not something to guess: the single case this
    # test was first written around came to 133, two orders short of the worst
    # the grid holds.
    assert worst_one_dimensional < 4.0
    assert worst_axis > 100.0
    assert worst_axis > 50.0 * worst_one_dimensional
    assert worst_axis > CONSTANT_TOLERANCE


def test_a_series_that_moves_is_still_accepted_everywhere() -> None:
    """The guards are a floor, not a new refusal: ordinary data goes through untouched."""
    x = np.asarray(np.random.default_rng(11).normal(0.0004, 0.01, 1200))
    assert abs(sharpe_ratio(x)) < 1.0
    assert abs(skewness(x)) < 1.0
    assert estimate_sharpe(x, periods_per_year=252).n == 1200
    assert trial_correlation(np.column_stack([x, x * 2.0 + 0.001])).shape == (2, 2)
    assert np.isfinite(excess_returns(x, 0.0002)).all()
    assert financing_impact(x, 0.0002).shift > 0.0
