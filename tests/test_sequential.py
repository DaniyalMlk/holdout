"""A confidence sequence, and the two things that make it worth having.

**The validity is checked as an identity, not cited.** Everything in this module
rests on one claim: that the normal mixture is a martingale. So that claim is
integrated. One step of the mixture against eighty Gauss-Hermite nodes
reproduces its own current value to 4e-16, at several partial sums and several
variance levels — and the boundary is checked to be exactly the level set where
that martingale reaches ``1 / alpha``, in both directions, because a boundary
that has drifted from its martingale has no theorem behind it.

**The tuning is checked against the condition it was derived from.** The mixing
variance is the solution of ``r - log(1 + r) = 2 log(1 / alpha)``; the fixed
point is verified to satisfy that equation, and separately verified to be where
the boundary is narrowest, by sweeping the mixing variance around it. The sweep
also establishes that the optimum sits in a flat region — three parts in ten
thousand over a ten per cent move in either direction — which is the useful
thing to know about a tuning choice.

**Then the two numbers the phase exists to produce.** Monitoring an ordinary 95%
interval and stopping at the first crossing rejects 33.7% of the time at sixty
observations and 61.9% at five thousand, against returns whose mean is exactly
zero; and the rate *grows with the horizon*, which is what tells you it is not a
multiplicity to be corrected for but a rule with no size at all. The sequence
costs a factor of 1.548560 on the width at its tuning point and a factor of 3.40
on the number of observations — and that factor is almost invariant to the
strategy, 3.4047 at an annual Sharpe ratio of 0.3 against 3.3977 at 2.0.

**And the slack is explained rather than tuned away.** The sequence crosses
1.35% of the time by its tuning point, not 5%. Extending the same paths to four,
twenty and a hundred times that point takes it to 2.46%, 3.48% and 3.64%: the
budget is being spent, in the part of the future the simulation was not looking
at. A test that measured only the first number would have read as a bound three
times too conservative.

**The estimated variance is measured, not waved at.** Substituting the sample
standard deviation for the true one raises the crossing rate from 1.47% to
1.98% — a 35% increase in false alarms, still inside the nominal 5%. Stated as a
number because "approximately valid" is not one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from holdout import (
    NORMAL_95,
    AnytimeBound,
    ConfidenceSequence,
    InsufficientDataError,
    ValidationError,
    confidence_sequence,
    minimum_track_record_length,
    mixing_variance,
    mixture_boundary,
    mixture_martingale,
    peeking_rejection_rate,
    sequential_track_record_length,
)
from holdout.sequential import _inverse_normal, _tightest_ratio

ONE_SIDED_95 = 1.6448536269514722


# -- the martingale, which is where the validity comes from --------------------


def gauss_hermite(nodes: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Probabilist's Hermite nodes and weights, so the measure is the standard
    normal and the weights sum to ``sqrt(2 pi)``."""
    abscissae, weights = np.polynomial.hermite_e.hermegauss(nodes)
    return abscissae, weights / math.sqrt(2.0 * math.pi)


@pytest.mark.parametrize(
    ("total", "count"),
    [(0.0, 1), (0.05, 10), (-0.2, 50), (0.4, 200), (-0.01, 3)],
)
def test_the_mixture_is_a_martingale(total: float, count: int) -> None:
    """``E[M(n+1) | F(n)] = M(n)``, integrated rather than asserted. This is the
    whole of the validity: Ville's inequality applies to a non-negative
    martingale starting at one, and nothing else in this module needs proving."""
    sigma = 0.017
    mixing = mixing_variance(250.0 * sigma * sigma)
    variance = count * sigma * sigma
    abscissae, weights = gauss_hermite()
    expected = sum(
        weight * mixture_martingale(total + sigma * node, variance + sigma * sigma, mixing)
        for node, weight in zip(abscissae, weights, strict=True)
    )
    assert expected == pytest.approx(mixture_martingale(total, variance, mixing), rel=1e-14)


def test_the_mixture_starts_at_one() -> None:
    mixing = mixing_variance(100.0)
    assert mixture_martingale(0.0, 0.0, mixing) == pytest.approx(1.0, rel=1e-15)


@pytest.mark.parametrize("variance", [0.0, 1.0, 37.0, 1000.0])
@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
def test_the_boundary_is_the_level_set_of_the_martingale(variance: float, alpha: float) -> None:
    """Both directions: the martingale at the boundary is exactly ``1 / alpha``,
    and just inside the boundary it is below it."""
    mixing = mixing_variance(100.0, alpha=alpha)
    edge = mixture_boundary(variance, mixing, alpha=alpha)
    assert mixture_martingale(edge, variance, mixing) == pytest.approx(1.0 / alpha, rel=1e-12)
    assert mixture_martingale(edge * 0.999, variance, mixing) < 1.0 / alpha
    assert mixture_martingale(edge * 1.001, variance, mixing) > 1.0 / alpha


def test_the_boundary_grows_with_the_variance_and_shrinks_with_alpha() -> None:
    mixing = mixing_variance(100.0)
    levels = [mixture_boundary(v, mixing) for v in (1.0, 10.0, 100.0, 1000.0)]
    assert levels == sorted(levels)
    tight = [
        mixture_boundary(100.0, mixing_variance(100.0, alpha=a), alpha=a)
        for a in (0.2, 0.1, 0.05, 0.01)
    ]
    assert tight == sorted(tight)


# -- the tuning ----------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.2, 0.1, 0.05, 0.01, 1e-4])
def test_the_fixed_point_solves_the_equation_it_was_derived_from(
    alpha: float,
) -> None:
    ratio = _tightest_ratio(alpha)
    assert ratio - math.log1p(ratio) == pytest.approx(2.0 * math.log(1.0 / alpha), rel=1e-12)
    assert mixing_variance(100.0, alpha=alpha) == pytest.approx(100.0 / ratio, rel=1e-15)


def test_the_fixed_point_is_where_the_boundary_is_narrowest() -> None:
    """Derived by differentiation, so it is verified by sweeping. The sweep also
    says the optimum is flat: a ten per cent move in the mixing variance costs
    three parts in ten thousand, so the choice is not delicate."""
    target = 100.0
    tuned = mixing_variance(target)
    best = mixture_boundary(target, tuned)
    for factor in (0.25, 0.5, 0.9, 1.1, 2.0, 4.0):
        assert mixture_boundary(target, tuned * factor) > best
    nearby = max(mixture_boundary(target, tuned * factor) for factor in (0.9, 1.1))
    assert nearby / best - 1.0 < 1e-3
    assert nearby / best - 1.0 > 1e-5


def test_the_tuning_is_in_variance_units_not_observation_counts() -> None:
    """Which matters: a caller passing a horizon when they mean a variance
    process is out by the square of their volatility, and the two agree exactly
    when the volatility is one."""
    assert mixing_variance(250.0 * 0.02**2) == pytest.approx(
        mixing_variance(250.0) * 0.02**2, rel=1e-15
    )


# -- what monitoring a fixed-sample interval costs -----------------------------


def test_peeking_at_a_fixed_sample_interval_rejects_far_too_often() -> None:
    """The practice this module replaces, simulated. A 5% test that rejects a
    third of the time at sixty observations."""
    assert peeking_rejection_rate(60, paths=4000) == pytest.approx(0.337, abs=0.02)
    assert peeking_rejection_rate(250, paths=4000) == pytest.approx(0.441, abs=0.02)


def test_the_peeking_rate_grows_with_the_horizon() -> None:
    """The reason it is not a multiplicity correction. There is no horizon to
    correct for: the longer you look the more certain the rejection, because the
    partial sum's fluctuations grow like ``sqrt(2 n log log n)`` and the interval
    only like ``sqrt(n)``."""
    rates = [peeking_rejection_rate(h, paths=3000) for h in (60, 250, 1000, 4000)]
    assert rates == sorted(rates)
    assert rates[0] > 0.25
    assert rates[-1] > 0.55


def test_the_sequence_holds_its_level_where_the_fixed_interval_does_not() -> None:
    sequence = peeking_rejection_rate(250, paths=4000, sequence=True)
    fixed = peeking_rejection_rate(250, paths=4000)
    assert sequence < 0.05
    assert fixed > 0.4
    assert fixed / sequence > 10.0


def test_the_slack_is_spent_further_out_than_the_tuning_point() -> None:
    """1.35% by the tuning point, 3.64% by a hundred times it. The bound is not
    three times too conservative; the other two thirds of its budget are being
    spent where this simulation was not looking."""
    alpha = 0.05
    tune = 60
    mixing = mixing_variance(float(tune), alpha=alpha)
    generator = np.random.default_rng(3)
    paths = 4000
    rates = []
    crossed = np.zeros(paths, dtype=bool)
    running = np.zeros(paths)
    seen = 0
    for horizon in (tune, tune * 4, tune * 20, tune * 100):
        while seen < horizon:
            size = min(4000, horizon - seen)
            draws = generator.standard_normal((paths, size))
            totals = running[:, None] + np.cumsum(draws, axis=1)
            level = np.asarray(
                [
                    mixture_boundary(float(n), mixing, alpha=alpha)
                    for n in range(seen + 1, seen + size + 1)
                ]
            )
            crossed |= (np.abs(totals) >= level).any(axis=1)
            running = totals[:, -1]
            seen += size
        rates.append(float(crossed.mean()))
    assert rates == sorted(rates)
    assert rates[0] < 0.025
    assert rates[-1] < alpha
    assert rates[-1] > 2.0 * rates[0]


# -- the price, in width and in observations -----------------------------------


def test_the_width_ratio_depends_only_on_the_fraction_of_the_horizon() -> None:
    """1.548560 at the tuning point, 1.950657 a tenth of the way, 4.563248 a
    hundredth — and the same figures to six decimal places at tuning points of
    100, 1,000, 10,000 and 100,000 observations. The ratio is a function of
    ``n / target`` alone, which is what makes the horizon the only thing a
    caller has to choose."""
    tables = []
    for tune in (100, 1000, 10_000, 100_000):
        mixing = mixing_variance(float(tune))
        tables.append(
            tuple(
                mixture_boundary(float(int(tune * fraction)), mixing)
                / (NORMAL_95 * math.sqrt(int(tune * fraction)))
                for fraction in (1.0, 0.1, 0.01)
            )
        )
    for table in tables[1:]:
        assert table == pytest.approx(tables[0], rel=1e-12)
    assert tables[0][0] == pytest.approx(1.548560, abs=1e-6)
    assert tables[0][1] == pytest.approx(1.950657, abs=1e-6)
    assert tables[0][2] == pytest.approx(4.563248, abs=1e-6)


@pytest.mark.parametrize(
    ("annual", "expected"),
    [(0.3, 3.4047), (0.5, 3.4040), (1.0, 3.4047), (2.0, 3.3977)],
)
def test_the_sequential_record_length_is_a_fixed_multiple_of_the_fixed_one(
    annual: float, expected: float
) -> None:
    """The multiple is a property of the mixture and the confidence level, not
    of the strategy: it moves by two parts in a thousand over a sevenfold range
    of Sharpe ratios."""
    fixed = math.ceil((ONE_SIDED_95 / (annual / math.sqrt(252.0))) ** 2)
    sequential = sequential_track_record_length(annual)
    assert sequential / fixed == pytest.approx(expected, abs=1e-3)


def test_self_tuning_the_record_length_is_worth_doing() -> None:
    """Tuning at the fixed-sample answer tunes for the wrong horizon, since the
    sequential answer is three times longer. Iterating to the fixed point saves
    five per cent of the data."""
    loose = sequential_track_record_length(0.5, self_tuned=False)
    tight = sequential_track_record_length(0.5, self_tuned=True)
    assert tight < loose
    assert loose / tight == pytest.approx(1.056, abs=0.01)
    # And it is a fixed point: tuning at the answer returns the answer.
    mixing = mixing_variance(float(tight))
    per_period = 0.5 / math.sqrt(252.0)
    assert tight * per_period >= mixture_boundary(float(tight), mixing)
    assert (tight - 1) * per_period < mixture_boundary(float(tight - 1), mixing)


def test_the_sequential_length_exceeds_the_gaussian_fixed_sample_one() -> None:
    """Against the library's own fixed-sample answer, on the Gaussian shape
    where the two are comparable."""
    annual = 1.0
    per_period = annual / math.sqrt(252.0)
    fixed = minimum_track_record_length(per_period, confidence=0.95)
    sequential = sequential_track_record_length(annual)
    assert sequential > fixed
    assert 3.0 < sequential / fixed < 4.0


def test_a_higher_confidence_demands_a_longer_record() -> None:
    lengths = [sequential_track_record_length(0.5, alpha=a) for a in (0.1, 0.05, 0.01)]
    assert lengths == sorted(lengths)


# -- the sequence over a series ------------------------------------------------


def returns_with_drift(
    count: int, sharpe: float, *, seed: int = 11, sigma: float = 0.01
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    drift = sharpe / math.sqrt(252.0) * sigma
    return drift + sigma * generator.standard_normal(count)


def test_the_sequence_reports_one_row_per_observation_from_the_start() -> None:
    values = returns_with_drift(400, 1.0)
    sequence = confidence_sequence(values, volatility=0.01)
    assert len(sequence.bounds) == 399
    assert sequence.bounds[0].observations == 2
    assert sequence.last.observations == 400
    assert isinstance(sequence.last, AnytimeBound)
    assert isinstance(sequence, ConfidenceSequence)
    assert not sequence.estimated_volatility


def test_every_row_is_centred_on_the_running_mean() -> None:
    values = returns_with_drift(200, 0.8)
    sequence = confidence_sequence(values, volatility=0.01)
    for bound in sequence.bounds:
        assert bound.mean == pytest.approx(float(np.mean(values[: bound.observations])), rel=1e-12)
        assert bound.lower < bound.mean < bound.upper
        assert bound.half_width == pytest.approx(0.5 * (bound.upper - bound.lower), rel=1e-15)


def test_the_band_narrows_as_the_record_lengthens() -> None:
    sequence = confidence_sequence(returns_with_drift(1000, 1.0), volatility=0.01)
    widths = [bound.half_width for bound in sequence.bounds]
    assert widths == sorted(widths, reverse=True)


def test_a_strategy_with_an_edge_crosses_and_a_strategy_without_one_does_not() -> None:
    strong = confidence_sequence(
        returns_with_drift(6000, 2.5, seed=4), volatility=0.01, target=6000
    )
    flat = confidence_sequence(returns_with_drift(6000, 0.0, seed=4), volatility=0.01, target=6000)
    assert strong.crossing is not None
    assert flat.crossing is None
    assert strong.last.lower > 0.0
    assert strong.annualised_lower == pytest.approx(strong.last.lower * 252.0, rel=1e-15)
    assert strong.sharpe_lower() > 0.0
    assert strong.sharpe_lower() < 2.5


def test_tuning_for_a_longer_horizon_widens_the_band_over_the_data() -> None:
    """The trade the parameter exists to expose: a band narrowest at ten
    thousand observations is wider at five hundred than one narrowest there."""
    values = returns_with_drift(500, 1.0)
    near = confidence_sequence(values, volatility=0.01, target=500)
    far = confidence_sequence(values, volatility=0.01, target=10_000)
    assert far.last.half_width > near.last.half_width
    assert far.mixing > near.mixing


def test_estimating_the_volatility_costs_coverage_and_the_cost_is_measured() -> None:
    """A 35% increase in the false-alarm rate, from 1.47% to 1.98%, at 250
    observations. Inside the nominal 5% because the mixture has slack there,
    which is the reason the substitution is allowed at all and not a reason to
    call it exact."""
    alpha = 0.05
    count = 250
    paths = 3000
    generator = np.random.default_rng(7)
    draws = generator.standard_normal((paths, count))
    totals = np.cumsum(draws, axis=1)
    squares = np.cumsum(draws**2, axis=1)
    counts = np.arange(1, count + 1, dtype=float)
    sample = (squares - totals**2 / counts) / np.maximum(counts - 1.0, 1.0)
    mixing = mixing_variance(float(count), alpha=alpha)
    known = np.asarray([mixture_boundary(float(n), mixing, alpha=alpha) for n in counts])
    spread = counts * sample + mixing
    estimated = np.sqrt(spread * np.log(spread / (mixing * alpha * alpha)))
    crossed_known = float((np.abs(totals[:, 1:]) >= known[1:]).any(axis=1).mean())
    crossed_estimated = float((np.abs(totals[:, 1:]) >= estimated[:, 1:]).any(axis=1).mean())
    assert crossed_known < alpha
    assert crossed_estimated < alpha
    assert crossed_estimated > crossed_known
    assert 1.1 < crossed_estimated / crossed_known < 1.8


def test_the_volatility_is_estimated_when_it_is_not_given() -> None:
    values = returns_with_drift(300, 1.0)
    sequence = confidence_sequence(values)
    assert sequence.estimated_volatility
    assert sequence.volatility == pytest.approx(float(np.std(values, ddof=1)), rel=1e-15)


# -- refusals ------------------------------------------------------------------


def test_a_constant_series_has_no_interval() -> None:
    with pytest.raises(InsufficientDataError, match="no volatility"):
        confidence_sequence([0.01] * 50)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"volatility": 0.0}, "must be positive"),
        ({"volatility": -0.01}, "must be positive"),
        ({"volatility": float("inf")}, "must be positive"),
        ({"target": 0}, "positive count"),
        ({"start": 0}, "start must be between"),
        ({"start": 999}, "start must be between"),
    ],
)
def test_the_sequence_refuses_what_it_cannot_measure(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        confidence_sequence(returns_with_drift(100, 1.0), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [0.0, -1.0, float("inf"), float("nan")])
def test_the_mixing_variance_refuses_a_target_that_is_not_a_variance(
    target: float,
) -> None:
    with pytest.raises(ValidationError, match="positive and finite"):
        mixing_variance(target)


@pytest.mark.parametrize("mixing", [0.0, -1.0])
def test_the_boundary_and_martingale_refuse_a_bad_mixture(mixing: float) -> None:
    with pytest.raises(ValidationError, match="mixing variance must be positive"):
        mixture_boundary(10.0, mixing)
    with pytest.raises(ValidationError, match="mixing variance must be positive"):
        mixture_martingale(1.0, 10.0, mixing)


def test_a_negative_variance_process_is_refused() -> None:
    with pytest.raises(ValidationError, match="cannot be negative"):
        mixture_boundary(-1.0, 1.0)
    with pytest.raises(ValidationError, match="cannot be negative"):
        mixture_martingale(0.0, -1.0, 1.0)


@pytest.mark.parametrize("sharpe", [0.0, -0.5])
def test_a_non_positive_sharpe_has_no_sequential_record_length(sharpe: float) -> None:
    with pytest.raises(ValidationError, match="not positive"):
        sequential_track_record_length(sharpe)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"horizon": 0}, "positive count"), ({"horizon": 10, "paths": 0}, "path count")],
)
def test_the_simulation_refuses_a_degenerate_request(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        peeking_rejection_rate(**kwargs)  # type: ignore[arg-type]


def test_the_normal_quantile_helper_agrees_with_the_cached_values() -> None:
    assert _inverse_normal(0.95) == pytest.approx(ONE_SIDED_95, rel=1e-12)
    assert _inverse_normal(0.975) == pytest.approx(NORMAL_95, rel=1e-12)
