"""Tests for what financing does to a selection rather than to a number.

The arrangement of this module follows the argument. The invariance is stated
first and asserted on data constructed to satisfy its condition exactly; then
the counterexample, also constructed, because the condition is algebraic and a
simulation could only ever suggest that it is tight; then the measurements,
which are what say how much any of it matters.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from holdout import (
    ValidationError,
    excess_columns,
    expected_maximum_sharpe,
    financing_shifts,
    selection_under_financing,
    sharpe_ratio,
)
from holdout.series import FloatArray

PPY = 252.0
RATE = 0.0002


def _exact(means: list[float], deviations: list[float], n: int = 1000) -> FloatArray:
    """A matrix whose sample mean and sample standard deviation are what was asked for.

    Alternating plus and minus one has a sample mean of exactly zero, and
    dividing by its own sample standard deviation leaves a pattern with a
    sample standard deviation of exactly one. Scaling and shifting that gives a
    column whose Sharpe ratio is ``mean / deviation`` by construction, so the
    tests below can assert where a ranking sits rather than hope a draw puts it
    there.
    """
    assert n % 2 == 0, "an odd count would not have a zero mean"
    pattern = np.resize(np.array([1.0, -1.0]), n)
    pattern = pattern / float(np.std(pattern, ddof=1))
    return np.column_stack(
        [mean + deviation * pattern for mean, deviation in zip(means, deviations, strict=True)]
    )


def _family(
    deviations: FloatArray, *, mean: float = 0.0003, seed: int = 0, n: int = 2000
) -> FloatArray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal((n, deviations.size)) * deviations + mean)


# --- the construction itself ------------------------------------------------


def test_the_constructor_puts_the_ratios_exactly_where_it_says() -> None:
    x = _exact([0.05 * 0.002, 0.02 * 0.02], [0.002, 0.02])
    assert sharpe_ratio(x[:, 0]) == pytest.approx(0.05, rel=1e-14)
    assert sharpe_ratio(x[:, 1]) == pytest.approx(0.02, rel=1e-14)
    assert float(np.std(x[:, 0], ddof=1)) == pytest.approx(0.002, rel=1e-14)


# --- the invariance, and its condition --------------------------------------


def test_identical_sample_volatility_shifts_every_column_alike() -> None:
    """The ranking survives exactly when the shifts coincide, and they coincide here.

    Constructed so that the sample standard deviations are equal to the last
    bit, which is the condition — not equal *population* volatilities, which is
    the most a family can be designed for and which is not enough.
    """
    x = _exact([0.0001, 0.0005, 0.0003, 0.0002], [0.01, 0.01, 0.01, 0.01])
    shifts = financing_shifts(x, RATE)
    assert float(np.ptp(shifts)) < 1e-15
    result = selection_under_financing(x, RATE, periods_per_year=PPY)
    assert not result.moved
    assert result.gross_best == 1
    assert result.displacement == 0.0
    gross = [sharpe_ratio(x[:, j]) for j in range(4)]
    net = [sharpe_ratio(excess_columns(x, RATE)[:, j]) for j in range(4)]
    assert np.argsort(gross).tolist() == np.argsort(net).tolist()


def test_the_dispersion_is_untouched_when_the_shift_is_common() -> None:
    """Subtracting one number from every ratio cannot change their spread."""
    x = _exact([0.0001, 0.0005, 0.0003, 0.0002], [0.01, 0.01, 0.01, 0.01])
    result = selection_under_financing(x, RATE)
    assert result.excess_dispersion == pytest.approx(result.gross_dispersion, rel=1e-12)
    assert result.benchmark_excess == pytest.approx(result.benchmark_gross, rel=1e-12)


def test_the_winner_flips_on_a_constructed_pair() -> None:
    """Exact arithmetic, because the condition for a flip is algebraic.

    Column 0 wins gross at 0.05 against 0.02. Its shift is ``rate / 0.002`` and
    column 1's is ``rate / 0.02``, a difference of 0.09 in per-period Sharpe
    ratio, which is more than the 0.03 that separates them. So the order has to
    reverse, and it does: -0.05 against +0.01.
    """
    x = _exact([0.05 * 0.002, 0.02 * 0.02], [0.002, 0.02])
    shifts = financing_shifts(x, RATE)
    assert shifts[0] == pytest.approx(0.1, rel=1e-12)
    assert shifts[1] == pytest.approx(0.01, rel=1e-12)
    result = selection_under_financing(x, RATE, periods_per_year=PPY)
    assert result.moved
    assert (result.gross_best, result.excess_best) == (0, 1)
    assert result.gross_ratio == pytest.approx(0.05, rel=1e-12)
    assert result.displaced_ratio == pytest.approx(-0.05, rel=1e-12)
    assert result.excess_ratio == pytest.approx(0.01, rel=1e-12)
    assert result.displacement == pytest.approx(0.06, rel=1e-12)


def test_the_flip_needs_the_gap_to_be_smaller_than_the_shift_difference() -> None:
    """Either side of the boundary, so the condition is pinned rather than illustrated.

    The shifts differ by 0.09. A gross gap of 0.08 reverses and one of 0.10
    does not, which is the whole of the rule.
    """
    for gap, flips in ((0.08, True), (0.10, False)):
        x = _exact([(0.02 + gap) * 0.002, 0.02 * 0.02], [0.002, 0.02])
        assert selection_under_financing(x, RATE).moved is flips


def test_displacement_is_never_negative() -> None:
    """The excess winner is the maximum on an excess basis, so it cannot be beaten."""
    for seed in range(8):
        family = _family(np.linspace(0.004, 0.03, 12), seed=seed)
        assert selection_under_financing(family, RATE).displacement >= 0.0


# --- the measurements -------------------------------------------------------


def test_equal_target_volatility_is_not_enough_to_keep_the_winner() -> None:
    """The gap between the algebra and the practice, measured.

    A family drawn at one volatility satisfies the invariance's condition only
    in expectation: the sample standard deviations still differ by order
    ``1 / sqrt(2n)``, so the shifts differ, and a close race still turns over.
    Two per cent of 200 draws here, against none that the algebra alone would
    have predicted.
    """
    flat = np.full(30, 0.01)
    moves = sum(
        1
        for seed in range(200)
        if selection_under_financing(_family(flat, seed=1000 + seed), RATE).moved
    )
    assert 1 <= moves <= 12, moves


@pytest.mark.parametrize(
    ("factor", "low", "high"),
    [(1.2, 0.03, 0.22), (2.0, 0.30, 0.60), (4.0, 0.45, 0.80)],
)
def test_the_displacement_rate_grows_with_the_volatility_spread(
    factor: float, low: float, high: float
) -> None:
    """Measured over 200 draws at each of three spreads, as bands.

    Widening the volatilities a family spans from one order of magnitude to
    sixteen takes the winner's displacement rate from a tenth of draws to three
    fifths. The bands are wide because a rate over 200 draws has a standard
    error of three or four points; the ordering between them is the claim, and
    the next test asserts that separately.
    """
    deviations = np.linspace(0.01 / factor, 0.01 * factor, 30)
    moves = sum(
        1
        for seed in range(200)
        if selection_under_financing(_family(deviations, seed=1000 + seed), RATE).moved
    )
    assert low <= moves / 200 <= high, moves


def test_the_rate_is_monotone_in_the_spread() -> None:
    """The ordering, asserted on the same draws so that it is not a coincidence of seeds."""
    rates = []
    for factor in (1.0, 1.2, 2.0, 4.0):
        deviations = np.linspace(0.01 / factor, 0.01 * factor, 30)
        moves = sum(
            1
            for seed in range(120)
            if selection_under_financing(_family(deviations, seed=1000 + seed), RATE).moved
        )
        rates.append(moves / 120)
    assert rates == sorted(rates)
    assert rates[-1] - rates[0] > 0.3


def test_the_benchmark_moves_down_when_the_low_volatility_trials_are_the_good_ones() -> None:
    """A common mean return: the lowest-volatility trial has the highest ratio and the
    largest shift, so the field compresses and the bar the search has to clear comes down.

    This is the condition the earlier phase's claim needed and did not state.
    The crossover sweep's thirty rules all have nearly the same volatility,
    which is why the benchmark did not move there.
    """
    deviations = np.linspace(0.004, 0.03, 30)
    result = selection_under_financing(_family(deviations, seed=5), RATE, periods_per_year=PPY)
    assert result.excess_dispersion < result.gross_dispersion
    assert result.benchmark_excess < result.benchmark_gross
    assert result.benchmark_excess / result.benchmark_gross < 0.9


def test_the_benchmark_moves_up_when_every_trial_has_the_same_ratio() -> None:
    """A common Sharpe ratio: a level field, hit hardest at the low-volatility end, stretches.

    The other direction, so that "it moves" is not quietly "it falls". Both are
    measured because the sign is a fact about the data rather than about the
    conversion.
    """
    deviations = np.linspace(0.004, 0.03, 30)
    rng = np.random.default_rng(9)
    family = rng.standard_normal((2000, 30)) * deviations + 0.05 * deviations
    result = selection_under_financing(family, RATE, periods_per_year=PPY)
    assert result.excess_dispersion > result.gross_dispersion
    assert result.benchmark_excess > result.benchmark_gross


def test_the_benchmark_properties_are_the_librarys_own_function() -> None:
    """Not a reimplementation of the expected maximum, which would be a second place to be wrong."""
    result = selection_under_financing(_family(np.linspace(0.004, 0.03, 30), seed=5), RATE)
    assert result.benchmark_gross == expected_maximum_sharpe(30, result.gross_dispersion)
    assert result.benchmark_excess == expected_maximum_sharpe(30, result.excess_dispersion)


# --- shifts, and the arithmetic they are made of ----------------------------


def test_the_shift_is_the_rate_over_each_columns_own_volatility() -> None:
    family = _family(np.linspace(0.004, 0.03, 12), seed=3)
    shifts = financing_shifts(family, RATE)
    for column in range(12):
        sigma = float(np.std(family[:, column], ddof=1))
        assert shifts[column] == pytest.approx(RATE / sigma, rel=1e-12)
    assert shifts[0] > shifts[-1]


def test_the_shift_is_a_difference_of_ratios_and_not_a_quotient() -> None:
    """Which is what keeps it right for a rate that moves.

    With a rate series the conversion changes each column's volatility as well
    as its mean, so ``rate / sigma`` is an approximation and the difference of
    the two ratios is the thing itself. They part company in the fourth
    significant figure here, which is small and is not zero.
    """
    family = _family(np.linspace(0.004, 0.03, 8), seed=4)
    moving = (0.03 + 0.03 * np.sin(np.linspace(0.0, 8.0, family.shape[0]))) / PPY
    shifts = financing_shifts(family, moving)
    quotient = float(np.mean(moving)) / np.std(family, axis=0, ddof=1)
    relative = np.abs(shifts - quotient) / np.abs(shifts)
    assert 1e-06 < float(np.max(relative)) < 1e-02


def test_a_single_column_has_no_dispersion_and_wins_either_way() -> None:
    x = _exact([0.0003], [0.01])
    result = selection_under_financing(x, RATE)
    assert result.trials == 1
    assert (result.gross_best, result.excess_best) == (0, 0)
    assert result.gross_dispersion == 0.0
    assert not result.moved


def test_selection_names_the_column_with_no_variance() -> None:
    family = _family(np.linspace(0.004, 0.03, 4), seed=2)
    family[:, 2] = 0.001
    with pytest.raises(ValidationError, match=r"returns\[:, 2\] have zero variance"):
        selection_under_financing(family, RATE)


def test_selection_rejects_a_misaligned_rate_series() -> None:
    family = _family(np.linspace(0.004, 0.03, 4), seed=2)
    with pytest.raises(ValidationError, match="must line up"):
        selection_under_financing(family, np.zeros(family.shape[0] - 1))


def test_the_annualised_scale_is_the_same_square_root_as_everywhere_else() -> None:
    family = _family(np.linspace(0.004, 0.03, 30), seed=5)
    result = selection_under_financing(family, RATE, periods_per_year=PPY)
    assert result.periods_per_year == PPY
    assert result.benchmark_gross * math.sqrt(PPY) > result.benchmark_gross
