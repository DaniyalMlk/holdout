"""The two-strategy comparison, and the size of both tests measured.

Three kinds of check here, and the third is the one that decides whether the
module is worth having.

*Exact invariances.* A Sharpe ratio is invariant to rescaling its series, so the
difference of two is, so the statistic is — to the last bits, not approximately.
Swapping the two series flips the sign and changes nothing else. Those two pin the
whole computation against a transposed gradient or a variance that quietly depends
on the units.

*The closed form against its own algebra.* Memmel's variance reduces to known
expressions in cases where it can be written down independently: at zero
correlation, at equal Sharpe ratios, and against the two separate Lo standard
errors that the naive comparison would have used. Those tests are what catch the
uncorrected Jobson-Korkie form, which differs from the corrected one only in one
term and produces a plausible number.

*The size of both tests under a true null.* Which is the only thing that says the
robust variance is worth its bandwidth. Measured here on 400 replications for
speed and in ``examples/sharpe_difference.py`` on 2,000, where the figures the
documentation quotes come from.
"""

from __future__ import annotations

import math
import statistics

import numpy as np
import pytest

from holdout.exceptions import InsufficientDataError, ValidationError
from holdout.pairwise import (
    MINIMUM_OBSERVATIONS,
    SharpeDifference,
    VarianceMethod,
    newey_west_bandwidth,
    newey_west_covariance,
    sharpe_difference,
)
from holdout.sharpe import sharpe_standard_error

CRITICAL = 1.959963984540054


def paired(
    *,
    count: int = 500,
    correlation: float = 0.7,
    persistence: float = 0.0,
    first_mean: float = 0.0005,
    second_mean: float = 0.0005,
    volatility: float = 0.01,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Two correlated series, optionally serially dependent, with equal Sharpe ratios.

    The innovations are rescaled so an AR(1) series keeps the unconditional
    volatility it was asked for. Without that, turning the persistence up changes
    the Sharpe ratios as well as their dependence, and a size measurement would be
    reading two changes at once.
    """
    rng = np.random.default_rng(seed)
    independent = rng.normal(0.0, 1.0, count)
    other = correlation * independent + math.sqrt(1.0 - correlation * correlation) * rng.normal(
        0.0, 1.0, count
    )
    scale = math.sqrt(1.0 - persistence * persistence) if persistence else 1.0
    for series in (independent, other):
        if persistence:
            for index in range(1, count):
                series[index] += persistence * series[index - 1]
    return (
        first_mean + volatility * scale * independent,
        second_mean + volatility * scale * other,
    )


# -- exact invariances --------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-4, 0.25, 40.0, 1000.0])
@pytest.mark.parametrize("method", list(VarianceMethod))
def test_the_statistic_does_not_care_what_units_the_returns_are_in(
    scale: float, method: VarianceMethod
) -> None:
    """A Sharpe ratio is scale invariant, so everything built on one has to be.

    Asserted to a relative 1e-12 rather than approximately, because the invariance
    is exact: every term in both variances is a ratio of quantities of the same
    degree in the series. A variance with a stray absolute term in it passes an
    approximate check on one scale and fails this on four.
    """
    first, second = paired(second_mean=0.0002)
    plain = sharpe_difference(first, second, method=method)
    scaled = sharpe_difference(first * scale, second / scale, method=method)
    assert scaled.difference == pytest.approx(plain.difference, rel=1e-12)
    assert scaled.statistic == pytest.approx(plain.statistic, rel=1e-12)
    assert scaled.closed_form_error == pytest.approx(plain.closed_form_error, rel=1e-12)
    assert scaled.robust_error == pytest.approx(plain.robust_error, rel=1e-12)
    assert scaled.correlation == pytest.approx(plain.correlation, rel=1e-12)


@pytest.mark.parametrize("method", list(VarianceMethod))
def test_swapping_the_two_series_flips_the_sign_and_nothing_else(
    method: VarianceMethod,
) -> None:
    """Both variances are symmetric in the two series, so only the sign moves.

    The Ledoit-Wolf gradient has a minus sign on the second series' two components
    and the covariance is symmetric, so the quadratic form is unchanged — which is
    only true if the gradient's ordering matches the moment matrix's columns. That
    is the pairing this test exists to check.
    """
    first, second = paired(second_mean=0.0001)
    forward = sharpe_difference(first, second, method=method)
    backward = sharpe_difference(second, first, method=method)
    assert backward.difference == pytest.approx(-forward.difference, rel=1e-12)
    assert backward.statistic == pytest.approx(-forward.statistic, rel=1e-12)
    assert backward.p_value == pytest.approx(forward.p_value, rel=1e-12)
    assert backward.closed_form_error == pytest.approx(forward.closed_form_error, rel=1e-12)
    assert backward.robust_error == pytest.approx(forward.robust_error, rel=1e-12)
    assert backward.first == pytest.approx(forward.second)
    assert backward.second == pytest.approx(forward.first)


def test_adding_a_constant_return_moves_the_ratios_and_not_the_correlation() -> None:
    """A sanity check on which quantity each argument feeds.

    Shifting a series' mean changes its Sharpe ratio and leaves the correlation
    alone, so the difference moves and the correlation does not. A variance that
    had picked up the shift would fail here while every scale test still passed.
    """
    first, second = paired()
    base = sharpe_difference(first, second)
    shifted = sharpe_difference(first + 0.001, second)
    assert shifted.correlation == pytest.approx(base.correlation, rel=1e-12)
    assert shifted.difference > base.difference
    assert shifted.first > base.first
    assert shifted.second == pytest.approx(base.second, rel=1e-12)


# -- the closed form against its own algebra ----------------------------------


def test_at_zero_correlation_the_closed_form_is_the_two_lo_errors_added() -> None:
    """Which is the one case where the naive comparison is right.

    Memmel's variance is ``(2 - 2r + (Sa^2 + Sb^2 - 2 Sa Sb r^2) / 2) / n``. At
    ``r = 0`` that is ``(2 + (Sa^2 + Sb^2) / 2) / n``, which is exactly
    ``(1 + Sa^2/2)/n + (1 + Sb^2/2)/n`` — the sum of the two Lo variances. So
    independent series are the case where differencing two separate standard errors
    happens to be correct, and the test pins that seam.
    """
    first, second = paired(correlation=0.0, second_mean=0.0002, count=4000, seed=7)
    result = sharpe_difference(first, second)
    assert abs(result.correlation) < 0.05
    independent = math.sqrt(
        sharpe_standard_error(result.first, result.observations) ** 2
        + sharpe_standard_error(result.second, result.observations) ** 2
    )
    # Not exact, because the sample correlation is only near zero. Close enough that
    # the identity is visible and far from the correlated case below.
    assert result.closed_form_error == pytest.approx(independent, rel=0.02)


def test_correlation_shrinks_the_closed_form_error_a_lot() -> None:
    """The half of this test that two separate standard errors cannot see.

    Two strategies on the same market move together, and the difference of two
    things that move together varies less than either. Treating them as independent
    gives an interval far too wide, so the naive comparison is conservative rather
    than wrong — which is worse in one way, because a real difference goes
    unreported and nothing looks broken.
    """
    errors = {}
    for correlation in (0.0, 0.5, 0.9, 0.99):
        first, second = paired(correlation=correlation, second_mean=0.0002, count=4000)
        errors[correlation] = sharpe_difference(first, second).closed_form_error
    values = [errors[key] for key in sorted(errors)]
    assert values == sorted(values, reverse=True)
    assert errors[0.99] < 0.2 * errors[0.0]


def test_the_closed_form_matches_a_hand_computation() -> None:
    """Memmel's expression, written out again from the reported quantities.

    The uncorrected Jobson-Korkie variance differs from this in one term, and the
    difference is invisible in any test that only checks the statistic is finite
    and the p-value is between zero and one.
    """
    first, second = paired(second_mean=0.0002)
    result = sharpe_difference(first, second)
    expected = math.sqrt(
        (
            2.0
            - 2.0 * result.correlation
            + 0.5
            * (
                result.first**2
                + result.second**2
                - 2.0 * result.first * result.second * result.correlation**2
            )
        )
        / result.observations
    )
    assert result.closed_form_error == pytest.approx(expected, rel=1e-14)


# -- the HAC covariance -------------------------------------------------------


def test_a_zero_bandwidth_is_the_plain_sample_covariance() -> None:
    """No autocovariance terms, which is the White heteroskedasticity-consistent case.

    Still free of the normality assumption and not of the independence one, and the
    test is against the sample covariance computed a different way.
    """
    rng = np.random.default_rng(3)
    series = rng.normal(0.0, 1.0, (400, 3))
    estimated = newey_west_covariance(series, 0)
    reference = np.cov(series, rowvar=False, ddof=0)
    assert estimated == pytest.approx(reference, abs=1e-12)


def test_the_hac_covariance_picks_up_positive_autocorrelation() -> None:
    """A persistent series has a long-run variance above its sample variance.

    ``(1 + rho) / (1 - rho)`` times it in the limit, which at ``rho = 0.5`` is
    three. The Bartlett estimate at a rule-of-thumb bandwidth reaches part of the
    way there and not all of it — truncation biases it down, and that is why the
    robust test still over-rejects on dependent data rather than fixing it.
    """
    rng = np.random.default_rng(4)
    innovations = rng.normal(0.0, 1.0, 4000)
    series = np.empty(4000)
    series[0] = innovations[0]
    for index in range(1, 4000):
        series[index] = 0.5 * series[index - 1] + innovations[index]
    column = series.reshape(-1, 1)
    plain = float(newey_west_covariance(column, 0)[0, 0])
    corrected = float(newey_west_covariance(column, newey_west_bandwidth(series.size))[0, 0])
    assert corrected > 1.5 * plain
    assert corrected < 3.0 * plain


def test_the_bandwidth_rule_is_the_conventional_one() -> None:
    assert newey_west_bandwidth(100) == 4
    assert newey_west_bandwidth(1000) == 6
    assert newey_west_bandwidth(1) == 1
    with pytest.raises(ValidationError, match="must be positive"):
        newey_west_bandwidth(0)


def test_a_bandwidth_reaching_past_the_sample_is_refused() -> None:
    rng = np.random.default_rng(5)
    series = rng.normal(0.0, 1.0, (30, 2))
    with pytest.raises(ValidationError, match="reaches past the end"):
        newey_west_covariance(series, 30)
    with pytest.raises(ValidationError, match="must not be negative"):
        newey_west_covariance(series, -1)
    with pytest.raises(ValidationError, match="one row per observation"):
        newey_west_covariance(series[:, 0], 2)


# -- size, which is the reason the second estimator exists --------------------


def rejection_rates(
    *, persistence: float, trials: int = 400, count: int = 500
) -> tuple[float, float, float]:
    """How often each variance rejects a true null, and the mean error ratio."""
    closed = 0
    robust = 0
    ratios = []
    for trial in range(trials):
        first, second = paired(count=count, persistence=persistence, seed=1000 + trial)
        result = sharpe_difference(first, second)
        ratios.append(result.error_ratio)
        if abs(result.difference / result.closed_form_error) > CRITICAL:
            closed += 1
        if abs(result.difference / result.robust_error) > CRITICAL:
            robust += 1
    return closed / trials, robust / trials, statistics.mean(ratios)


def test_both_tests_hold_their_size_on_independent_returns() -> None:
    """Where the closed form's assumptions hold, so neither should have an edge.

    Both come out near the nominal 5% and the two standard errors agree to within a
    per cent, which is the result that says the robust version costs nothing when it
    is not needed.
    """
    closed, robust, ratio = rejection_rates(persistence=0.0)
    assert 0.02 < closed < 0.09
    assert 0.02 < robust < 0.09
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_serial_dependence_breaks_the_closed_form_and_only_bends_the_robust_one() -> None:
    """The measurement the module exists for, stated as it came out.

    At a persistence of 0.6 the closed form rejects a true null about a third of
    the time against a nominal 5%, because it assumes independence and the
    difference of two persistent series has far more sampling variability than its
    formula allows for. The robust variance cuts that to about one in nine — much
    better and not right, because a truncated Bartlett kernel recovers only part of
    the long-run variance. Both halves of that go in the documentation.
    """
    closed, robust, ratio = rejection_rates(persistence=0.6)
    assert closed > 0.20
    assert robust < closed / 2.0
    assert robust > 0.05
    assert ratio > 1.3


# -- refusals -----------------------------------------------------------------


def test_the_same_series_twice_is_refused_rather_than_called_insignificant() -> None:
    """0/0 is not 1, and the two answers mean different things.

    'These are the same strategy' and 'the difference between these two strategies
    is not significant' are not the same statement, and a p-value of one would read
    as the second.
    """
    first, _ = paired()
    with pytest.raises(ValidationError, match="identical"):
        sharpe_difference(first, first)
    with pytest.raises(ValidationError, match="identical"):
        sharpe_difference(first, first.copy())


def test_a_series_that_does_not_move_has_no_sharpe_ratio() -> None:
    first, _ = paired()
    flat = np.full(first.size, 0.001)
    with pytest.raises(InsufficientDataError, match="does not move"):
        sharpe_difference(first, flat)


def test_perfect_correlation_and_equal_ratios_leaves_no_sampling_error() -> None:
    """The closed form's degenerate case, reached by scaling a series.

    A series and a positive multiple of it have the same Sharpe ratio and a
    correlation of one, so Memmel's variance is exactly zero. Refusing names the
    correlation, because that is the quantity that explains it.
    """
    first, _ = paired()
    with pytest.raises(InsufficientDataError, match="perfect correlation"):
        sharpe_difference(first, first * 3.0, method=VarianceMethod.JOBSON_KORKIE_MEMMEL)


def test_misaligned_series_are_refused_rather_than_trimmed() -> None:
    """Trimming would be a guess about which end, and the wrong guess is invisible.

    A one-period misalignment destroys the correlation the test is built on and
    leaves every number looking reasonable.
    """
    first, second = paired()
    with pytest.raises(ValidationError, match="the same length"):
        sharpe_difference(first, second[:-1])


def test_too_few_observations_is_refused() -> None:
    first, second = paired(count=MINIMUM_OBSERVATIONS - 1)
    with pytest.raises(InsufficientDataError, match="at least 20"):
        sharpe_difference(first, second)


def test_the_boundary_conditions_on_the_arguments() -> None:
    first, second = paired()
    result = sharpe_difference(first, second)
    with pytest.raises(ValidationError, match="level must be in"):
        result.confidence_interval(0.0)
    with pytest.raises(ValidationError, match="level must be in"):
        result.confidence_interval(1.0)
    with pytest.raises(ValidationError, match="periods_per_year was not given"):
        _ = result.annualised_difference
    with pytest.raises(ValueError, match="periods_per_year"):
        sharpe_difference(first, second, periods_per_year=0.0)


# -- the shape of the result --------------------------------------------------


def test_both_variances_are_always_reported_whichever_is_asked_for() -> None:
    """Because the disagreement between them is the finding when there is one.

    Choosing one and discarding the other would hide exactly the case the module
    was written to surface.
    """
    first, second = paired(persistence=0.5, second_mean=0.0002)
    closed = sharpe_difference(first, second, method=VarianceMethod.JOBSON_KORKIE_MEMMEL)
    robust = sharpe_difference(first, second, method=VarianceMethod.LEDOIT_WOLF)
    assert closed.difference == pytest.approx(robust.difference, rel=1e-14)
    assert closed.closed_form_error == pytest.approx(robust.closed_form_error, rel=1e-14)
    assert closed.robust_error == pytest.approx(robust.robust_error, rel=1e-14)
    assert closed.standard_error == closed.closed_form_error
    assert robust.standard_error == robust.robust_error
    assert robust.error_ratio == pytest.approx(robust.robust_error / robust.closed_form_error)
    assert abs(closed.statistic) > abs(robust.statistic)


def test_the_interval_and_the_p_value_agree_about_the_null() -> None:
    """A two-sided 95% interval excludes zero exactly when the p-value is under 5%.

    Both are built from the same statistic, so this is a consistency check on the
    two ways of presenting it rather than on the statistic itself — and it is the
    check that catches a one-sided p-value dressed as a two-sided one.
    """
    for mean in (0.0005, 0.0008, 0.0012, 0.0020):
        first, second = paired(count=2000, first_mean=mean, seed=42)
        result = sharpe_difference(first, second)
        low, high = result.confidence_interval(0.95)
        assert (low > 0.0 or high < 0.0) == (result.p_value < 0.05)


def test_annualisation_is_the_root_of_time_applied_at_the_edge() -> None:
    first, second = paired(second_mean=0.0002)
    result = sharpe_difference(first, second, periods_per_year=252.0)
    assert result.annualised_difference == pytest.approx(result.difference * math.sqrt(252.0))
    assert isinstance(result, SharpeDifference)


def test_a_larger_bandwidth_is_allowed_and_changes_the_robust_error_only() -> None:
    first, second = paired(persistence=0.5)
    errors = [
        sharpe_difference(first, second, bandwidth=lag).robust_error for lag in (0, 2, 10, 40)
    ]
    assert errors == sorted(errors)
    closed = {
        sharpe_difference(first, second, bandwidth=lag).closed_form_error for lag in (0, 2, 10, 40)
    }
    assert len(closed) == 1
    default = sharpe_difference(first, second)
    assert default.bandwidth == newey_west_bandwidth(first.size)
