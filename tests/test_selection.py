"""Inference on the strategy that was selected.

The references here are independent of the module. The log upper tail is
checked against ``scipy.stats.norm.logsf``; the conditional law against a
direct ratio of tails where that ratio still works, and against a Monte Carlo
of the truncation where it does not; the intervals against their own coverage
under a known truth. The selection bias is checked against
:func:`holdout.expected_maximum_normal`, which this module never calls.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import norm

from holdout import expected_maximum_normal
from holdout.exceptions import InsufficientDataError, ValidationError
from holdout.selection import (
    _MILLS,
    LOG_TAIL_ASYMPTOTIC_FROM,
    conditional_interval,
    log_upper_tail,
    median_unbiased_mean,
    naive_interval,
    select_best,
    selection_coverage,
    selection_inference,
    truncated_tail,
)

# --------------------------------------------------------------------------
# The log upper tail, against scipy
# --------------------------------------------------------------------------


def test_log_upper_tail_matches_an_independent_implementation_everywhere() -> None:
    worst = 0.0
    where = 0.0
    for step in range(1, 4001):
        z = step / 10.0
        mine = log_upper_tail(z)
        reference = float(norm.logsf(z))
        relative = abs(mine - reference) / abs(reference)
        if relative > worst:
            worst, where = relative, z
    assert worst < 5e-15, (worst, where)
    # Deliberately no assertion about *where* the worst disagreement sits. An
    # earlier version of this test asserted it was out in the tail, which was
    # true against one version of the reference and false against an older one
    # -- it is a fact about the reference's error profile, not about this code.
    # The claim that belongs here is made in the next test instead, against a
    # reference that cannot change.
    assert 0.1 <= where <= 400.0


def test_the_series_branch_is_already_exact_where_it_takes_over() -> None:
    # The version-independent form of the claim: compare the series to erfc
    # below the crossover, where erfc is still accurate and the two overlap.
    # The error falls like a high power of z, so by the time the series is
    # actually used it is below machine precision and anything left over is
    # the reference's error rather than the series'.
    def series(z: float) -> float:
        inverse = 1.0 / (z * z)
        total = 0.0
        for coefficient in reversed(_MILLS):
            total = coefficient + inverse * total
        return -0.5 * z * z - math.log(z) - 0.5 * math.log(2.0 * math.pi) + math.log(total)

    errors = []
    for z in (8.0, 12.0, 16.0, LOG_TAIL_ASYMPTOTIC_FROM):
        exact = math.log(0.5 * math.erfc(z / math.sqrt(2.0)))
        errors.append(abs(series(z) - exact) / abs(exact))
    assert all(later < earlier for earlier, later in itertools.pairwise(errors))
    assert errors[0] < 1e-9
    assert errors[-1] < 1e-15
    # And at the crossover the two branches are the same number.
    assert log_upper_tail(LOG_TAIL_ASYMPTOTIC_FROM) == pytest.approx(
        math.log(0.5 * math.erfc(LOG_TAIL_ASYMPTOTIC_FROM / math.sqrt(2.0))), rel=1e-14
    )


def test_log_upper_tail_is_continuous_across_the_crossover() -> None:
    # Both branches at the *same* argument. Evaluating them either side of the
    # crossover would measure the slope of the tail, which is -z and swamps
    # the thing being checked by eight orders of magnitude.
    z = LOG_TAIL_ASYMPTOTIC_FROM
    erfc_branch = math.log(0.5 * math.erfc(z / math.sqrt(2.0)))
    series_branch = log_upper_tail(z)
    assert abs(series_branch - erfc_branch) / abs(erfc_branch) < 1e-14
    assert series_branch == pytest.approx(float(norm.logsf(z)), rel=1e-15)


def test_log_upper_tail_handles_the_negative_half_and_zero() -> None:
    assert log_upper_tail(0.0) == pytest.approx(math.log(0.5))
    assert log_upper_tail(-5.0) == pytest.approx(float(norm.logsf(-5.0)))
    with pytest.raises(ValidationError, match="must be finite"):
        log_upper_tail(math.inf)


def test_the_direct_ratio_this_replaces_has_already_failed() -> None:
    # The reason the log scale is not fastidiousness: erfc reaches exactly
    # zero near 38 standard deviations, so a ratio of two tails past that is
    # 0/0 rather than something small.
    assert 0.5 * math.erfc(39.0 / math.sqrt(2.0)) == 0.0
    with pytest.raises(ZeroDivisionError):
        _ = (0.5 * math.erfc(50.0 / math.sqrt(2.0))) / (0.5 * math.erfc(45.0 / math.sqrt(2.0)))
    # And the module's answer there is an ordinary number.
    deep = truncated_tail(50.0, 0.0, 1.0, 45.0)
    assert 0.0 < deep < 1e-100
    assert deep == pytest.approx(float(np.exp(norm.logsf(50.0) - norm.logsf(45.0))), rel=1e-10)


# --------------------------------------------------------------------------
# The conditional law
# --------------------------------------------------------------------------


def test_truncated_tail_matches_the_direct_ratio_where_that_works() -> None:
    for mean in (-2.0, 0.0, 1.5):
        for lower in (-1.0, 0.0, 2.0):
            for x in (lower, lower + 0.5, lower + 3.0):
                direct = float(np.exp(norm.logsf(x, mean) - norm.logsf(lower, mean)))
                assert truncated_tail(x, mean, 1.0, lower) == pytest.approx(
                    min(direct, 1.0), rel=1e-12
                )


def test_truncated_tail_matches_a_simulation_of_the_truncation() -> None:
    # Draw from the untruncated normal, keep what clears the barrier, and
    # count. No formula of the module's in it at all.
    generator = np.random.default_rng(7)
    mean, deviation, lower = 0.4, 1.3, 1.0
    draws = generator.normal(mean, deviation, size=400_000)
    kept = draws[draws >= lower]
    for x in (1.2, 2.0, 3.0):
        empirical = float((kept >= x).mean())
        exact = truncated_tail(x, mean, deviation, lower)
        error = abs(empirical - exact)
        assert error < 4.0 * math.sqrt(exact * (1 - exact) / len(kept)) + 1e-6, (x, error)


def test_a_point_below_the_truncation_is_certain_by_definition() -> None:
    assert truncated_tail(0.0, 1.0, 1.0, 2.0) == 1.0
    assert truncated_tail(2.0, 1.0, 1.0, 2.0) == 1.0


def test_the_conditional_tail_increases_in_the_mean() -> None:
    # The monotonicity the inversion rests on, and the direction is the one
    # that is easy to get backwards: a higher mean makes the observed value
    # less surprising from above.
    previous = -1.0
    for mean in (-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0):
        tail = truncated_tail(1.5, mean, 1.0, 1.0)
        assert tail > previous
        previous = tail
    assert truncated_tail(1.5, -500.0, 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert truncated_tail(1.5, 500.0, 1.0, 1.0) == pytest.approx(1.0, abs=1e-12)


def test_truncated_tail_validation() -> None:
    with pytest.raises(ValidationError, match="deviation must be positive"):
        truncated_tail(1.0, 0.0, 0.0, 0.5)
    with pytest.raises(ValidationError, match="must be finite"):
        truncated_tail(math.nan, 0.0, 1.0, 0.5)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_select_best_finds_the_winner_and_the_runner_up() -> None:
    selected = select_best([0.4, 1.2, 0.9, -0.3], [0.3, 0.25, 0.3, 0.4])
    assert selected.index == 1
    assert selected.estimate == 1.2
    assert selected.deviation == 0.25
    assert selected.threshold == 0.9
    assert selected.margin == pytest.approx((1.2 - 0.9) / 0.25)
    assert selected.candidates == 4


def test_select_best_breaks_ties_towards_the_lower_index() -> None:
    # So the selection is a function of the data rather than of the order the
    # candidates happened to arrive in.
    assert select_best([1.0, 1.0, 0.5], [0.2, 0.2, 0.2]).index == 0
    assert select_best([0.5, 1.0, 1.0], [0.2, 0.2, 0.2]).index == 1


def test_select_best_validation() -> None:
    with pytest.raises(InsufficientDataError, match="at least two candidates"):
        select_best([1.0], [0.2])
    with pytest.raises(ValidationError, match="against 1 standard errors"):
        select_best([1.0, 2.0], [0.2])
    with pytest.raises(ValidationError, match=r"deviations\[1\] must be positive"):
        select_best([1.0, 2.0], [0.2, -0.1])
    with pytest.raises(ValidationError, match=r"estimates\[0\] must be finite"):
        select_best([math.nan, 2.0], [0.2, 0.3])


# --------------------------------------------------------------------------
# The interval, against its own coverage and against the naive one
# --------------------------------------------------------------------------


def test_the_conditional_interval_brackets_the_estimate_it_inverts() -> None:
    interval = conditional_interval(1.5, 0.4, 0.3, level=0.95)
    assert interval.lower < interval.median < interval.upper
    assert interval.median < 1.5  # the selection bias, always downwards
    assert interval.bounded


def test_the_inversion_reproduces_the_quantiles_it_was_asked_for() -> None:
    # The definition of the interval: at the lower bound the observed estimate
    # sits at conditional tail alpha/2, at the upper bound at 1 - alpha/2.
    estimate, deviation, threshold, level = 1.5, 0.4, 1.1, 0.9
    interval = conditional_interval(estimate, deviation, threshold, level)
    alpha = 1.0 - level
    assert truncated_tail(estimate, interval.lower, deviation, threshold) == pytest.approx(
        alpha / 2.0, abs=1e-9
    )
    assert truncated_tail(estimate, interval.upper, deviation, threshold) == pytest.approx(
        1.0 - alpha / 2.0, abs=1e-9
    )
    assert truncated_tail(estimate, interval.median, deviation, threshold) == pytest.approx(
        0.5, abs=1e-9
    )


def test_a_clear_winner_gets_back_the_naive_interval() -> None:
    # The correction is not a tax on every selection. Six standard errors
    # clear of the runner-up, conditioning changes nothing a reader would see.
    interval = conditional_interval(2.0, 0.3, 2.0 - 6.0 * 0.3, level=0.95)
    assert interval.lower == pytest.approx(interval.naive_lower, abs=1e-4)
    assert interval.upper == pytest.approx(interval.naive_upper, abs=1e-4)
    assert interval.inflation == pytest.approx(1.0, abs=1e-4)
    assert median_unbiased_mean(2.0, 0.3, 2.0 - 6.0 * 0.3) == pytest.approx(2.0, abs=1e-4)


def test_a_close_race_puts_the_bound_a_long_way_out_at_a_known_rate() -> None:
    # Finite, which is not the natural guess, and at the rate
    # log(1/alpha) * deviation / margin.
    deviation, estimate = 0.3, 1.0
    for margin in (0.03, 0.1, 0.3):
        interval = conditional_interval(estimate, deviation, estimate - margin * deviation)
        predicted = estimate - deviation * math.log(1.0 / 0.025) / margin
        assert interval.bounded
        assert interval.lower == pytest.approx(predicted, rel=0.01)
    # Three hundredths of a standard error puts it 123 standard errors out,
    # and a thirtieth puts it 111: log(1/alpha) / margin, nothing else.
    tight = conditional_interval(1.0, 0.3, 1.0 - 0.03 * 0.3)
    assert (tight.lower - 1.0) / 0.3 == pytest.approx(-122.96, rel=0.01)
    looser = conditional_interval(1.0, 0.3, 1.0 - (1.0 / 30.0) * 0.3)
    assert (looser.lower - 1.0) / 0.3 == pytest.approx(-110.67, rel=0.01)


def test_shrinkage_falls_away_as_the_margin_grows() -> None:
    estimate, deviation = 1.5, 1.0
    shrinkages = [
        estimate - median_unbiased_mean(estimate, deviation, estimate - margin * deviation)
        for margin in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(shrinkages))
    assert shrinkages[0] == pytest.approx(6.84, abs=0.05)
    assert shrinkages[3] == pytest.approx(0.314, abs=0.01)
    assert shrinkages[-1] == pytest.approx(0.0, abs=1e-4)


def test_an_exact_tie_raises_rather_than_returning_an_infinity() -> None:
    with pytest.raises(ValidationError, match="does not exceed the runner-up"):
        conditional_interval(1.0, 0.3, 1.0)
    with pytest.raises(ValidationError, match="does not exceed the runner-up"):
        median_unbiased_mean(1.0, 0.3, 1.5)


def test_interval_validation() -> None:
    with pytest.raises(ValidationError, match=r"level must lie in \(0, 1\)"):
        conditional_interval(1.0, 0.3, 0.5, level=1.0)
    with pytest.raises(ValidationError, match="deviation must be positive"):
        conditional_interval(1.0, 0.0, 0.5)
    with pytest.raises(ValidationError, match=r"level must lie in \(0, 1\)"):
        naive_interval(1.0, 0.3, level=0.0)


def test_the_naive_interval_is_the_textbook_one() -> None:
    low, high = naive_interval(1.0, 0.5, level=0.95)
    assert low == pytest.approx(1.0 - 1.959963984540054 * 0.5, abs=1e-9)
    assert high == pytest.approx(1.0 + 1.959963984540054 * 0.5, abs=1e-9)
    low, high = naive_interval(0.0, 1.0, level=0.99)
    assert high == pytest.approx(2.5758293035489004, abs=1e-9)


def test_selection_inference_gathers_the_whole_answer() -> None:
    inference = selection_inference([1.0, 0.99, 0.5, 0.0], [0.3, 0.3, 0.3, 0.3])
    assert inference.selected.index == 0
    assert inference.shrinkage > 0.0
    assert inference.shrinkage == pytest.approx(
        inference.selected.estimate - inference.interval.median
    )
    assert inference.interval.inflation > 5.0


# --------------------------------------------------------------------------
# Coverage: the finding the module exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidates", "naive_floor", "naive_ceiling"),
    [(2, 0.93, 0.97), (5, 0.86, 0.91), (10, 0.75, 0.81), (50, 0.26, 0.35)],
)
def test_the_naive_interval_under_covers_more_as_the_table_grows(
    candidates: int, naive_floor: float, naive_ceiling: float
) -> None:
    result = selection_coverage([0.0] * candidates, [1.0] * candidates, paths=2000, seed=1)
    assert naive_floor < result["naive_coverage"] < naive_ceiling
    assert 0.93 < result["conditional_coverage"] < 0.97


def test_the_selection_bias_is_the_expected_maximum() -> None:
    # Checked against a function this module never calls, so the agreement is
    # between a simulation and a closed form rather than within one of them.
    for candidates in (2, 5, 10, 50):
        result = selection_coverage([0.0] * candidates, [1.0] * candidates, paths=4000, seed=1)
        assert result["mean_selection_bias"] == pytest.approx(
            expected_maximum_normal(candidates), rel=0.03
        )


def test_the_mean_width_is_the_wrong_summary_and_the_median_is_the_right_one() -> None:
    result = selection_coverage([0.0] * 10, [1.0] * 10, paths=2000, seed=1)
    naive = result["mean_naive_width"]
    assert naive == pytest.approx(2 * 1.959963984540054, abs=1e-6)
    # The mean is many times the naive width; the median is a small multiple.
    assert result["mean_bounded_conditional_width"] / naive > 8.0
    assert 1.5 < result["median_bounded_conditional_width"] / naive < 4.0
    assert result["median_bounded_conditional_width"] < result["mean_bounded_conditional_width"]


def test_a_clear_favourite_costs_almost_nothing_and_rescues_the_naive_interval() -> None:
    result = selection_coverage([4.0] + [0.0] * 9, [1.0] * 10, paths=2000, seed=2)
    naive = result["mean_naive_width"]
    assert result["naive_coverage"] > 0.93
    assert result["median_bounded_conditional_width"] / naive < 1.15
    assert result["mean_selection_bias"] < 0.15


def test_coverage_holds_with_unequal_standard_errors() -> None:
    # Nothing in the construction assumed the candidates were comparable, and
    # the naive interval's failure is milder here because the widest-error
    # candidate wins most selections and is allowed the most room.
    result = selection_coverage([0.3] * 4, [0.5, 1.0, 1.5, 2.0], paths=2000, seed=3)
    assert 0.93 < result["conditional_coverage"] < 0.97
    assert result["naive_coverage"] < 0.94


def test_coverage_validation() -> None:
    with pytest.raises(ValidationError, match="against 1 standard errors"):
        selection_coverage([0.0, 0.0], [1.0])
    with pytest.raises(InsufficientDataError, match="at least two candidates"):
        selection_coverage([0.0], [1.0])
    with pytest.raises(ValidationError, match="path count must be positive"):
        selection_coverage([0.0, 0.0], [1.0, 1.0], paths=0)
    with pytest.raises(ValidationError, match=r"level must lie in \(0, 1\)"):
        selection_coverage([0.0, 0.0], [1.0, 1.0], level=2.0)
    with pytest.raises(ValidationError, match=r"deviations\[0\] must be positive"):
        selection_coverage([0.0, 0.0], [0.0, 1.0])


def test_coverage_is_reproducible_from_its_seed() -> None:
    first = selection_coverage([0.0] * 5, [1.0] * 5, paths=500, seed=11)
    second = selection_coverage([0.0] * 5, [1.0] * 5, paths=500, seed=11)
    assert first == second
    third = selection_coverage([0.0] * 5, [1.0] * 5, paths=500, seed=12)
    assert third["naive_coverage"] != first["naive_coverage"]


# --------------------------------------------------------------------------
# The assumption, tested rather than restated
# --------------------------------------------------------------------------


def test_correlated_candidates_make_the_two_intervals_change_places() -> None:
    # The module's one assumption is that the winner's estimate is independent
    # of the losers', which a parameter sweep over neighbouring settings
    # violates. The failure is not a small loss of validity: the conditional
    # interval becomes both wider and less valid, while the naive one recovers.
    naive = []
    conditional = []
    widths = []
    biases = []
    for rho in (0.0, 0.5, 0.9):
        result = selection_coverage([0.0] * 10, [1.0] * 10, paths=2000, seed=4, correlation=rho)
        naive.append(result["naive_coverage"])
        conditional.append(result["conditional_coverage"])
        widths.append(result["median_bounded_conditional_width"])
        biases.append(result["mean_selection_bias"])
    # The naive interval improves with correlation and the conditional one decays.
    assert naive[0] < naive[1] < naive[2]
    assert conditional[0] > conditional[2]
    assert conditional[0] > 0.93
    assert conditional[2] < 0.90
    assert naive[2] > conditional[2]
    # Wider and less valid at once, which is the part worth asserting.
    assert widths[0] < widths[1] < widths[2]
    # Because correlation shrinks the winner's curse itself.
    assert biases[0] > 2.5 * biases[2]


def test_the_crossover_sits_near_three_quarters() -> None:
    below = selection_coverage([0.0] * 10, [1.0] * 10, paths=3000, seed=4, correlation=0.5)
    above = selection_coverage([0.0] * 10, [1.0] * 10, paths=3000, seed=4, correlation=0.9)
    assert below["conditional_coverage"] > below["naive_coverage"]
    assert above["conditional_coverage"] < above["naive_coverage"]


def test_more_candidates_push_the_crossover_further_out() -> None:
    # With fifty candidates the naive interval has so much further to fall
    # that the conditional one is still the better of the two at a
    # correlation of a half.
    result = selection_coverage([0.0] * 50, [1.0] * 50, paths=1500, seed=5, correlation=0.5)
    assert result["naive_coverage"] < 0.75
    assert result["conditional_coverage"] > 0.92


def test_correlation_validation() -> None:
    with pytest.raises(ValidationError, match=r"correlation must lie in \[0, 1\)"):
        selection_coverage([0.0, 0.0], [1.0, 1.0], correlation=1.0)
    with pytest.raises(ValidationError, match=r"correlation must lie in \[0, 1\)"):
        selection_coverage([0.0, 0.0], [1.0, 1.0], correlation=-0.1)
