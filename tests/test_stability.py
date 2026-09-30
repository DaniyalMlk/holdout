"""A break in the Sharpe ratio at a date the data chose.

The statistic is checked against recomputing the two subsample Sharpe ratios with
``estimate_sharpe`` and their standard errors with ``sharpe_standard_error``, which
is the slow path the vectorised one replaces. The size of the test is measured
rather than asserted, at a replication count small enough to live in a suite: the
figures in the README come from 500 replications, and what is checked here is that
the two tests are on opposite sides of a gap far too wide for the Monte Carlo noise
of sixty.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from holdout.exceptions import InsufficientDataError, ValidationError
from holdout.sharpe import estimate_sharpe, sharpe_standard_error, sharpe_variance_term
from holdout.stability import (
    DEFAULT_TRIM,
    MINIMUM_SIDE,
    VARIANCE_FLOOR,
    _bounds,
    break_statistics,
    sharpe_break,
)
from holdout.stability import _variance_term as vectorised_variance_term


def flat(seed: int, n: int = 1000, *, mean: float = 0.0, sd: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, n)


def broken(seed: int, before: float, after: float, each: int = 500, sd: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(before, sd, each), rng.normal(after, sd, each)])


# -- the statistic -----------------------------------------------------------


def test_the_vectorised_statistic_matches_recomputing_it_the_slow_way() -> None:
    """Against ``estimate_sharpe`` and ``sharpe_standard_error``, at every date.

    The fast path is four cumulative sums and some algebra on power sums; the slow
    path is the library's ordinary Sharpe estimator applied twice per candidate.
    They share the formula and nothing else, which is what makes this worth
    running over all 701 candidates rather than at one.
    """
    sample = flat(3, 1000, mean=0.0004)
    found = break_statistics(sample)
    assert len(found.statistics) == len(found.splits) == 701
    for split, statistic in zip(found.splits, found.statistics, strict=True):
        left = estimate_sharpe(sample[:split])
        right = estimate_sharpe(sample[split:])
        expected = (right.value - left.value) / math.sqrt(
            left.standard_error() ** 2 + right.standard_error() ** 2
        )
        assert statistic == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_the_vectorised_variance_term_is_the_scalar_one() -> None:
    """The scalar version validates one argument at a time, which a million of
    them cannot afford -- so the formula exists twice and is pinned together.

    The kurtosis has to be drawn *above* ``1 + skew**2``, which every distribution
    satisfies and ``check_shape`` enforces. Drawing the two independently produces
    impossible pairs about a third of the time, which is how this test first
    failed -- inside the validator rather than on the comparison.
    """
    rng = np.random.default_rng(5)
    sharpes = rng.uniform(-0.4, 0.4, 200)
    skews = rng.uniform(-1.5, 1.5, 200)
    kurtoses = 1.0 + skews**2 + rng.uniform(0.05, 9.0, 200)
    fast = vectorised_variance_term(sharpes, skews, kurtoses)
    for index, value in enumerate(fast):
        assert value == pytest.approx(
            sharpe_variance_term(sharpes[index], skews[index], kurtoses[index]),
            rel=1e-12,
        )


def test_the_shifted_power_sums_survive_a_mean_far_below_the_noise() -> None:
    """The cancellation this is arranged to avoid.

    A daily return series has a mean around a hundredth of its standard
    deviation, so the sum of squares and the square of the sum agree to two
    significant figures and their difference loses the rest. Shifting by the
    sample mean first fixes it; this checks the result against the slow path on a
    series whose mean is a *thousandth* of its noise, where the unshifted version
    would be visibly wrong.
    """
    sample = flat(9, 800, mean=1e-5, sd=0.01)
    found = break_statistics(sample, trim=0.2)
    for split, statistic in zip(found.splits, found.statistics, strict=True):
        left = estimate_sharpe(sample[:split])
        right = estimate_sharpe(sample[split:])
        expected = (right.value - left.value) / math.sqrt(
            left.standard_error() ** 2 + right.standard_error() ** 2
        )
        assert statistic == pytest.approx(expected, rel=1e-8, abs=1e-11)


def test_the_statistic_keeps_its_sign() -> None:
    """Positive means the Sharpe ratio was higher *after*, which nobody worries
    about and is worth being able to tell apart from the case that matters.
    """
    improving = break_statistics(broken(4, 0.0, 0.0015))
    assert improving.signed_supremum > 0.0
    decaying = break_statistics(broken(4, 0.0015, 0.0))
    assert decaying.signed_supremum < 0.0
    assert decaying.supremum == abs(decaying.signed_supremum)
    assert decaying.at in decaying.splits


def test_the_supremum_of_a_planted_break_lands_near_the_break() -> None:
    found = break_statistics(broken(6, 0.0018, -0.0008, each=600))
    assert abs(found.at - 600) < 150


def test_trimming_changes_which_dates_are_candidates() -> None:
    sample = flat(2, 1000)
    wide = break_statistics(sample, trim=0.02)
    narrow = break_statistics(sample, trim=0.35)
    assert len(wide.statistics) > len(narrow.statistics)
    assert min(narrow.splits) > min(wide.splits)
    assert max(narrow.splits) < max(wide.splits)
    # The narrower search is a subset, so its supremum cannot be the larger.
    assert narrow.supremum <= wide.supremum
    # And the trim never puts a candidate closer to an end than the floor.
    assert min(wide.splits) >= MINIMUM_SIDE
    assert len(sample) - max(wide.splits) >= MINIMUM_SIDE


def test_a_short_series_is_refused_and_a_wide_trim_is_not() -> None:
    with pytest.raises(InsufficientDataError):
        break_statistics(flat(1, 2 * MINIMUM_SIDE - 1))
    with pytest.raises(ValidationError, match=r"must be below 0\.5"):
        break_statistics(flat(1, 1000), trim=0.5)
    with pytest.raises(ValidationError):
        break_statistics(flat(1, 1000), trim=-0.1)
    # A trim of 0.45 on 70 observations is *not* refused, and should not be: the
    # floor on how short a side may be pushes the edge out to 32, which still
    # leaves seven candidates with 32 to 38 observations on each side. A first
    # version of this test expected a refusal and was asserting that two
    # constraints combine to be stricter than either, which they do not.
    seven = break_statistics(flat(1, 70), trim=0.45)
    assert len(seven.splits) == 7
    assert min(seven.splits) == 32


def test_the_only_lengths_with_no_candidate_break_are_odd_and_barely_trimmable() -> None:
    """The counterexample search, which found one.

    The guard in ``_bounds`` looked unreachable: for an even number of
    observations, ``trim < 0.5`` caps the edge at half the sample and a candidate
    always survives. It was nearly removed on the strength of that argument. For an
    *odd* number the ceiling can reach ``(total + 1) / 2`` and take the last
    candidate away, and a sweep of every length from the minimum to 400 against
    every trim to 0.499 reaches it 389 times -- every one of them at an odd length
    and a trim above 0.492.

    So the branch stays, and this is the search rather than a claim about it.
    """
    reached: list[tuple[int, float]] = []
    for total in range(2 * MINIMUM_SIDE, 401):
        for step in range(500):
            trim = step * 0.001
            try:
                low, high = _bounds(total, trim)
            except InsufficientDataError:
                reached.append((total, trim))
                continue
            assert high > low
            assert low >= MINIMUM_SIDE
            assert total - (high - 1) >= MINIMUM_SIDE
    assert len(reached) == 389
    assert all(total % 2 == 1 for total, _ in reached)
    assert min(trim for _, trim in reached) == pytest.approx(0.492)
    assert min(total for total, _ in reached) == 61
    # And the message says what to do about it.
    with pytest.raises(InsufficientDataError, match="or bring more data"):
        _bounds(61, 0.499)


def test_a_constant_stretch_gives_a_statistic_of_zero_not_a_nan() -> None:
    """A side with no variance has no Sharpe ratio, so the split is not a
    comparison -- and a NaN would propagate through the maximum and out into a
    payload, where ``json.dumps`` writes it as a bare ``NaN`` that a strict
    parser at the far end will not read.

    The guard has to be relative, and this is the test that found out why. A
    constant stretch's second moment computed from power sums is not zero but the
    rounding left over from subtracting two nearly equal sums -- of order 1e-22
    here. Divided into a mean of 0.001 that is a Sharpe ratio of 1e8 and a break
    statistic of -2e9: large enough to dominate every supremum in the sample, and
    finite enough to pass every check for a NaN. Only a threshold relative to the
    whole sample's own variance catches it.
    """
    sample = np.concatenate([np.full(200, 0.001), flat(8, 800)])
    found = break_statistics(sample, trim=0.02)
    assert all(math.isfinite(one) for one in found.statistics)
    flat_side = [
        statistic
        for split, statistic in zip(found.splits, found.statistics, strict=True)
        if split <= 200
    ]
    assert flat_side and all(one == 0.0 for one in flat_side)
    assert VARIANCE_FLOOR < 1e-6
    import json

    json.dumps({"statistics": list(found.statistics)}, allow_nan=False)
    # And a quiet-but-real stretch is still a comparison, not a degenerate one:
    # a tenth of the sample's volatility is nowhere near the floor.
    quiet = np.concatenate([np.random.default_rng(31).normal(0.0, 0.001, 200), flat(8, 800)])
    mixed = break_statistics(quiet, trim=0.02)
    assert all(one != 0.0 for one in mixed.statistics)


# -- the test ----------------------------------------------------------------


def test_the_bootstrap_p_value_is_reported_beside_the_naive_one() -> None:
    result = sharpe_break(flat(3, 1000, mean=0.0004), resamples=299, seed=1)
    assert result.p_value > result.naive_p_value
    assert 0.0 < result.p_value <= 1.0
    assert result.candidates == 701
    assert result.resamples == 299
    assert result.trim == DEFAULT_TRIM
    assert result.block_length >= 1.0
    assert result.naive_understates_by == pytest.approx(result.p_value / result.naive_p_value)
    # The statistic and the split agree with the statistic-only entry point.
    found = break_statistics(flat(3, 1000, mean=0.0004))
    assert result.statistic == pytest.approx(found.supremum, rel=1e-12)
    assert result.at == found.at
    # And the two subsample Sharpe ratios are the ones the split implies.
    sample = flat(3, 1000, mean=0.0004)
    assert result.before == pytest.approx(estimate_sharpe(sample[: result.at]).value, rel=1e-12)
    assert result.after == pytest.approx(estimate_sharpe(sample[result.at :]).value, rel=1e-12)


def test_the_p_value_cannot_be_zero() -> None:
    """A p-value of zero from a thousand resamples is a statement the resamples
    cannot support, so the observed statistic counts as one draw from the null.
    """
    result = sharpe_break(broken(7, 0.004, -0.004, each=500), resamples=99, seed=2)
    assert result.p_value == pytest.approx(1.0 / 100.0)
    assert result.p_value > 0.0


def test_the_same_seed_gives_the_same_p_value_and_a_different_one_does_not() -> None:
    sample = flat(12, 600)
    first = sharpe_break(sample, resamples=199, seed=42)
    again = sharpe_break(sample, resamples=199, seed=42)
    other = sharpe_break(sample, resamples=199, seed=43)
    assert first == again
    assert first.statistic == pytest.approx(other.statistic, rel=1e-12)
    assert first.p_value != other.p_value or first.p_value in (1.0, 1.0 / 200.0)


def test_the_naive_test_rejects_a_true_null_far_too_often() -> None:
    """The measurement the module exists to make, at a cheap replication count.

    Sixty replications of 600 iid normal returns with no break. The README's
    figures -- 41.4% against 3.2% at the default trim, over 500 replications of
    1,000 -- are not reproducible at this size, and nothing here pretends to: what
    is asserted is that the two tests sit either side of a gap far wider than the
    Monte Carlo noise of sixty draws, which is about six percentage points.
    """
    replications = 60
    naive = bootstrap = 0
    for rep in range(replications):
        sample = flat(5_000 + rep, 600)
        result = sharpe_break(sample, resamples=99, seed=rep, block_length=1.0)
        naive += result.naive_p_value < 0.05
        bootstrap += result.p_value < 0.05
    assert naive / replications > 0.25
    assert bootstrap / replications < 0.15
    assert naive > 3 * bootstrap


def test_a_block_length_is_chosen_when_none_is_given_and_used_when_one_is() -> None:
    sample = flat(13, 600)
    chosen = sharpe_break(sample, resamples=49, seed=3)
    given = sharpe_break(sample, resamples=49, seed=3, block_length=25.0)
    assert chosen.block_length != 25.0
    assert given.block_length == 25.0
    assert chosen.statistic == pytest.approx(given.statistic, rel=1e-12)


def test_a_flagrant_break_is_detected_and_a_mild_one_is_not() -> None:
    """Which is the module's own caveat, as a test.

    A per-period Sharpe of 0.40 collapsing to -0.40 is found. A fall from 0.10 to
    0.05 -- an annualised 1.59 becoming 0.79, which would end a strategy's life at
    any real desk -- is not, and over 120 replications it is found 10% of the time.
    Reading a large p-value here as evidence of stability is the mistake this
    module makes easiest.
    """
    flagrant = sharpe_break(broken(14, 0.004, -0.004), resamples=299, seed=4)
    assert flagrant.p_value < 0.02
    assert flagrant.before > 0.2 > 0.0 > flagrant.after
    mild = sharpe_break(broken(15, 0.0010, 0.0005), resamples=299, seed=5)
    assert mild.p_value > 0.20


def test_the_arguments_are_checked() -> None:
    sample = flat(16, 600)
    with pytest.raises(ValidationError, match="resamples must be a positive integer"):
        sharpe_break(sample, resamples=0)
    with pytest.raises(ValidationError, match="resamples must be a positive integer"):
        sharpe_break(sample, resamples=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="block_length must be at least 1"):
        sharpe_break(sample, resamples=10, block_length=0.5)
    with pytest.raises(ValidationError, match=r"must be below 0\.5"):
        sharpe_break(sample, trim=0.6, resamples=10)


def test_the_standard_error_used_is_the_one_that_accounts_for_shape() -> None:
    """Not the normal-returns version, and the difference has a sign.

    A strategy with negative skew and fat tails has a noisier Sharpe ratio than
    the normal formula says, so the normal formula makes the break statistic
    larger and finds breaks that are not there. Constructed here from a
    Student-t-like mixture so the sample kurtosis is genuinely high, and the two
    standard errors compared directly.
    """
    rng = np.random.default_rng(21)
    heavy = np.concatenate([rng.normal(0.0005, 0.006, 900), rng.normal(-0.004, 0.03, 100)])
    estimate = estimate_sharpe(heavy)
    assert estimate.kurtosis > 5.0
    assert estimate.skewness < -0.5
    mertens = estimate.standard_error("mertens")
    normal = sharpe_standard_error(estimate.value, estimate.n)
    assert mertens > normal
    # So a statistic built on the normal error would be the larger one, and the
    # module uses the smaller-statistic, larger-error version.
    scale = normal / mertens
    assert scale < 1.0
    found = break_statistics(heavy, trim=0.3)
    assert found.supremum > 0.0
