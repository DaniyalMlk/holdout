"""Tests for the financing conversion.

Several of these are equalities rather than tolerances, which is deliberate:
the relations between the two conventions are exact in real arithmetic, so a
tolerance here would be hiding whichever of the two implementations drifted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from holdout import (
    ValidationError,
    annual_rate,
    excess_columns,
    excess_returns,
    expected_maximum_sharpe,
    financing_impact,
    levered_excess,
    levered_returns,
    period_rate,
    sharpe_ratio,
)
from holdout.series import FloatArray

PPY = 252.0


def _series(n: int = 2520, *, seed: int = 7) -> FloatArray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(0.0004, 0.01, n), dtype=np.float64)


def _ramp(n: int, low: float, high: float) -> FloatArray:
    """A rate path sweeping ``low`` to ``high`` and back, annualised."""
    wave = 0.5 + 0.5 * np.sin(np.linspace(0.0, 8.0, n))
    return np.asarray(low + (high - low) * wave, dtype=np.float64)


# --- the rate, as quoted and as needed -------------------------------------


def test_simple_period_rate_is_division() -> None:
    assert period_rate(0.05, PPY, compounding="simple") == 0.05 / PPY


def test_compound_period_rate_compounds_back_to_the_annual_rate() -> None:
    """Exactly, through the inverse; to within 1e-12 through the growth factor.

    The gap between those two statements is the whole reason the conversion is
    written with ``log1p`` and ``expm1``. ``annual_rate`` takes the logarithm
    of ``1 + r`` without ever forming ``1 + r``, and comes back on the nose.
    Writing ``(1 + r) ** 252`` forms it, which costs the small rate its
    leading digits before the exponentiation has even started.
    """
    r = period_rate(0.05, PPY, compounding="compound")
    assert annual_rate(r, PPY, compounding="compound") == pytest.approx(0.05, rel=1e-15, abs=0.0)
    through_the_factor = (1.0 + r) ** PPY - 1.0
    assert 1e-14 < abs(through_the_factor - 0.05) / 0.05 < 1e-12


def test_the_direct_root_loses_the_leading_digits_of_a_small_rate() -> None:
    """The measurement behind the implementation choice, as a regression.

    ``(1 + a) ** (1 / n) - 1`` builds a number a hair above one and subtracts
    one from it, so what survives is whatever the growth factor's last bits
    happened to hold. At 0.25% over 252 periods the per-period rate is around
    1e-05, the growth factor is good to 2e-16 absolute, and the quotient of
    those is the 1e-11 of relative error below. ``expm1`` of a scaled
    ``log1p`` never forms the factor and so never pays that.
    """
    annual = 0.0025
    exact = period_rate(annual, PPY, compounding="compound")
    direct = math.pow(1.0 + annual, 1.0 / PPY) - 1.0
    assert abs(direct - exact) / exact > 1e-12
    assert annual_rate(exact, PPY, compounding="compound") == pytest.approx(
        annual, rel=1e-15, abs=0.0
    )
    naive_round_trip = math.pow(1.0 + direct, PPY) - 1.0
    assert abs(naive_round_trip - annual) / annual > 1e-12


@pytest.mark.parametrize("compounding", ["simple", "compound"])
@pytest.mark.parametrize("annual", [-0.9, -0.01, 0.0, 0.0025, 0.05, 0.4])
@pytest.mark.parametrize("ppy", [1.0, 4.0, 12.0, 252.0, 365.0])
def test_annual_rate_inverts_period_rate(annual: float, ppy: float, compounding: str) -> None:
    assert compounding in ("simple", "compound")
    kind: object = compounding
    per = period_rate(annual, ppy, compounding=kind)  # type: ignore[arg-type]
    back = annual_rate(per, ppy, compounding=kind)  # type: ignore[arg-type]
    assert back == pytest.approx(annual, rel=1e-14, abs=1e-18)


def test_the_two_compoundings_agree_at_one_period_a_year() -> None:
    """With a single period there is nothing to compound, so they must coincide."""
    assert period_rate(0.05, 1.0, compounding="simple") == pytest.approx(
        period_rate(0.05, 1.0, compounding="compound"), rel=1e-15
    )


def test_simple_exceeds_compound_for_a_positive_rate() -> None:
    """Jensen on a concave root: dividing overstates the per-period rate."""
    for annual in (0.001, 0.02, 0.05, 0.2):
        assert period_rate(annual, PPY, compounding="simple") > period_rate(
            annual, PPY, compounding="compound"
        )


def test_the_compounding_difference_is_small_beside_the_rate() -> None:
    """The measured size of the convention's effect, which is the reason to call it minor.

    At 5% over 252 periods the two per-period rates differ by under 5e-06,
    about 2.4% of the rate. The Sharpe shift the rate itself produces is
    ``rate / sigma``, so the convention is worth 2.4% of a haircut that is
    itself around a third of a Sharpe ratio: real, and not the thing to worry
    about first.
    """
    simple = period_rate(0.05, PPY, compounding="simple")
    compound = period_rate(0.05, PPY, compounding="compound")
    gap = simple - compound
    assert 4.0e-06 < gap < 5.0e-06
    assert 0.02 < gap / simple < 0.03
    x = _series()
    shift = gap / float(np.std(x, ddof=1)) * math.sqrt(PPY)
    assert 0.005 < shift < 0.01


@pytest.mark.parametrize("bad", [-1.0, -1.5, math.nan, math.inf])
def test_period_rate_rejects_impossible_annual_rates(bad: float) -> None:
    with pytest.raises(ValidationError, match="annual rate must be"):
        period_rate(bad, PPY, compounding="simple")


@pytest.mark.parametrize("ppy", [0.0, -1.0, math.nan])
def test_period_rate_rejects_an_impossible_frequency(ppy: float) -> None:
    with pytest.raises(ValidationError, match="periods_per_year must be positive"):
        period_rate(0.05, ppy, compounding="simple")


def test_period_rate_rejects_an_unknown_compounding() -> None:
    with pytest.raises(ValidationError, match="compounding must be"):
        period_rate(0.05, PPY, compounding="continuous")  # type: ignore[arg-type]


def test_annual_rate_rejects_an_unknown_compounding() -> None:
    with pytest.raises(ValidationError, match="compounding must be"):
        annual_rate(0.0002, PPY, compounding="continuous")  # type: ignore[arg-type]


def test_annual_rate_refuses_to_compound_a_rate_below_minus_one() -> None:
    with pytest.raises(ValidationError, match="cannot be compounded"):
        annual_rate(-1.5, PPY, compounding="compound")
    assert annual_rate(-1.5, 4.0, compounding="simple") == -6.0


def test_annual_rate_rejects_a_non_finite_rate() -> None:
    with pytest.raises(ValidationError, match="period rate must be finite"):
        annual_rate(math.nan, PPY, compounding="simple")


# --- the excess return, and what the convention costs ----------------------


def test_arithmetic_excess_is_subtraction() -> None:
    x = _series(64)
    rf = period_rate(0.05, PPY, compounding="simple")
    assert np.array_equal(excess_returns(x, rf), x - rf)


def test_geometric_excess_is_the_arithmetic_one_rescaled() -> None:
    """``(1+r)/(1+rf) - 1 == (r - rf)/(1 + rf)``, in real arithmetic exactly.

    Not asserted bit for bit: the two sides are different floating-point
    expressions of the same number and round differently. The bound is tight
    enough to catch any algebraic error and loose enough to survive a
    different libm.
    """
    x = _series()
    rf = period_rate(0.05, PPY, compounding="simple")
    geometric = excess_returns(x, rf, convention="geometric")
    rescaled = (x - rf) / (1.0 + rf)
    assert np.max(np.abs(geometric - rescaled)) < 1e-15


def test_a_constant_rate_makes_the_convention_free() -> None:
    """The headline fact: for a constant rate the convention cannot move a Sharpe ratio.

    The geometric excess is the arithmetic one times ``1/(1+rf)``, a positive
    constant, and a Sharpe ratio is invariant to positive scaling.
    """
    x = _series()
    rf = period_rate(0.05, PPY, compounding="simple")
    arithmetic = sharpe_ratio(excess_returns(x, rf))
    geometric = sharpe_ratio(excess_returns(x, rf, convention="geometric"))
    assert abs(arithmetic - geometric) < 1e-15
    assert arithmetic != pytest.approx(sharpe_ratio(x), abs=1e-6)


def test_a_constant_rate_does_move_the_mean_and_the_volatility() -> None:
    """The other half of the same fact, so that "free" is not read as "identical"."""
    x = _series()
    rf = period_rate(0.2, PPY, compounding="simple")
    arithmetic = excess_returns(x, rf)
    geometric = excess_returns(x, rf, convention="geometric")
    factor = 1.0 / (1.0 + rf)
    assert float(np.mean(geometric)) == pytest.approx(float(np.mean(arithmetic)) * factor)
    assert float(np.std(geometric, ddof=1)) == pytest.approx(
        float(np.std(arithmetic, ddof=1)) * factor
    )
    assert float(np.std(geometric, ddof=1)) < float(np.std(arithmetic, ddof=1))


def test_a_moving_rate_breaks_the_invariance_by_a_measured_amount() -> None:
    """With the rate moving, the scaling moves with it and the ratios part company.

    A decade of daily returns against a cash rate sweeping 0% to 6% puts the
    two Sharpe ratios about 4e-04 apart in relative terms. Quoted as a band:
    the digits are not portable and the order of magnitude is the claim.
    """
    x = _series()
    rf = _ramp(x.size, 0.0, 0.06) / PPY
    arithmetic = sharpe_ratio(excess_returns(x, rf))
    geometric = sharpe_ratio(excess_returns(x, rf, convention="geometric"))
    relative = abs(arithmetic - geometric) / abs(arithmetic)
    assert 1e-05 < relative < 1e-02


def test_excess_returns_accepts_a_rate_series() -> None:
    x = _series(128)
    rf = np.linspace(0.0, 0.0002, x.size)
    assert np.array_equal(excess_returns(x, rf), x - rf)


def test_excess_returns_rejects_a_misaligned_rate_series() -> None:
    with pytest.raises(ValidationError, match="rate has 3 observation"):
        excess_returns(_series(10), [0.0, 0.0, 0.0])


def test_excess_returns_rejects_a_non_finite_scalar_rate() -> None:
    with pytest.raises(ValidationError, match="rate must be finite"):
        excess_returns(_series(10), math.nan)


def test_excess_returns_rejects_an_unknown_convention() -> None:
    with pytest.raises(ValidationError, match="convention must be"):
        excess_returns(_series(10), 0.0, convention="logarithmic")  # type: ignore[arg-type]


def test_the_geometric_convention_refuses_a_rate_that_wipes_out_the_divisor() -> None:
    """``(1 + rf)`` at or below zero is a division this convention cannot survive."""
    with pytest.raises(ValidationError, match="not positive here"):
        excess_returns(_series(10), -1.0, convention="geometric")
    with pytest.raises(ValidationError, match=r"rate\[3\]"):
        rates = np.zeros(10)
        rates[3] = -1.2
        excess_returns(_series(10), rates, convention="geometric")
    # The arithmetic convention never divides, so the same rate is fine there.
    assert excess_returns(_series(10), -1.0)[0] == pytest.approx(_series(10)[0] + 1.0)


# --- the matrix path --------------------------------------------------------


def test_excess_columns_shifts_every_column_by_the_same_rate() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0.0003, 0.01, (500, 4))
    rf = 0.0002
    out = excess_columns(x, rf)
    assert out.shape == x.shape
    assert np.array_equal(out, x - rf)
    for j in range(x.shape[1]):
        assert np.array_equal(out[:, j], excess_returns(x[:, j], rf))


def test_excess_columns_promotes_a_single_series_to_one_column() -> None:
    x = _series(64)
    out = excess_columns(x, 0.0001)
    assert out.shape == (64, 1)
    assert np.array_equal(out[:, 0], excess_returns(x, 0.0001))


def test_excess_columns_hits_a_low_volatility_column_hardest() -> None:
    """One cash rate, different volatilities, so different shifts: ``rate / sigma_i``.

    This is the mechanism by which financing can change which strategy looks
    best, and it is worth pinning as arithmetic before it is demonstrated as an
    outcome.
    """
    rng = np.random.default_rng(11)
    vols = np.array([0.002, 0.01, 0.04])
    x = rng.normal(0.0, 1.0, (4000, 3)) * vols
    x = x + 0.0004  # the same mean return on every column
    rf = 0.0002
    net = excess_columns(x, rf)
    shifts = [sharpe_ratio(x[:, j]) - sharpe_ratio(net[:, j]) for j in range(3)]
    assert shifts[0] > shifts[1] > shifts[2] > 0.0
    for j in range(3):
        assert shifts[j] == pytest.approx(rf / float(np.std(net[:, j], ddof=1)), rel=1e-12)


def test_excess_columns_rejects_a_misaligned_rate_series() -> None:
    with pytest.raises(ValidationError, match="must line up"):
        excess_columns(np.zeros((10, 2)) + np.linspace(0, 1, 10)[:, None], np.zeros(9))


def test_excess_columns_rejects_an_unknown_convention() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValidationError, match="convention must be"):
        excess_columns(rng.normal(0, 0.01, (20, 2)), 0.0, convention="log")  # type: ignore[arg-type]


def test_excess_columns_geometric_matches_the_series_path_column_by_column() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(0.0003, 0.01, (300, 3))
    rf = _ramp(300, 0.0, 0.05) / PPY
    out = excess_columns(x, rf, convention="geometric")
    for j in range(3):
        assert np.array_equal(out[:, j], excess_returns(x[:, j], rf, convention="geometric"))


# --- leverage ---------------------------------------------------------------


@pytest.mark.parametrize("leverage", [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 20.0])
def test_leverage_at_no_spread_is_a_pure_scaling(leverage: float) -> None:
    """``L * (r - rf)``, so every scale-free statistic is untouched.

    This is the sense in which a Sharpe ratio "does not care" about leverage,
    and the whole of that claim: it holds when the borrowing is at the
    financing rate exactly and nowhere else.
    """
    x = _series()
    rf = period_rate(0.04, PPY, compounding="compound")
    base = excess_returns(x, rf)
    levered = levered_excess(x, leverage=leverage, rate=rf, spread=0.0)
    assert np.max(np.abs(levered - leverage * base)) < 1e-16 * leverage
    assert sharpe_ratio(levered) == pytest.approx(sharpe_ratio(base), rel=1e-14, abs=1e-16)


def test_leverage_of_one_changes_nothing_at_all() -> None:
    """Nothing is borrowed, so the spread cannot reach the return. Bit for bit."""
    x = _series(256)
    rf = period_rate(0.04, PPY, compounding="compound")
    for spread in (0.0, 0.001, 0.05):
        assert np.array_equal(
            levered_excess(x, leverage=1.0, rate=rf, spread=spread),
            excess_returns(x, rf),
        )


def test_a_spread_costs_exactly_what_the_closed_form_says() -> None:
    """``SR`` falls by ``(1 - 1/L) * spread / sigma``, which is an identity, not a tolerance.

    At a 1% annualised spread, 16% annualised volatility and three turns of
    leverage, that is 0.04 of annualised Sharpe ratio — a quarter of the
    multiplicity haircut this library computes for fifty trials, from a line
    item nobody reports.
    """
    x = _series()
    rf = period_rate(0.04, PPY, compounding="compound")
    spread = period_rate(0.01, PPY, compounding="compound")
    base = excess_returns(x, rf)
    sigma = float(np.std(base, ddof=1))
    for leverage in (1.0, 2.0, 3.0, 5.0):
        levered = levered_excess(x, leverage=leverage, rate=rf, spread=spread)
        predicted = sharpe_ratio(base) - (1.0 - 1.0 / leverage) * spread / sigma
        assert sharpe_ratio(levered) == pytest.approx(predicted, rel=0.0, abs=1e-15)
    three_turns = levered_excess(x, leverage=3.0, rate=rf, spread=spread)
    drop = (sharpe_ratio(base) - sharpe_ratio(three_turns)) * math.sqrt(PPY)
    assert 0.04 < drop < 0.045


def test_a_lender_does_not_pay_the_borrowers_spread() -> None:
    """Below one turn there is nothing borrowed; the remainder is lent at the rate itself."""
    x = _series()
    rf = period_rate(0.04, PPY, compounding="compound")
    base = excess_returns(x, rf)
    for spread in (0.0, 0.0005, 0.01):
        half = levered_excess(x, leverage=0.5, rate=rf, spread=spread)
        assert np.max(np.abs(half - 0.5 * base)) < 1e-17


def test_levered_returns_is_the_total_return_not_the_excess_one() -> None:
    """The gross series, which is what a drawdown is measured on."""
    x = _series(512)
    rf = 0.0002
    spread = 0.0001
    total = levered_returns(x, leverage=2.0, rate=rf, spread=spread)
    assert np.max(np.abs(total - (2.0 * x - (rf + spread)))) < 1e-18
    assert np.array_equal(levered_excess(x, leverage=2.0, rate=rf, spread=spread), total - rf)
    assert np.max(np.abs(total)) > np.max(np.abs(x))


def test_levered_excess_composes_under_the_geometric_convention() -> None:
    x = _series(512)
    rf = _ramp(512, 0.0, 0.05) / PPY
    total = levered_returns(x, leverage=2.5, rate=rf, spread=0.0001)
    assert np.array_equal(
        levered_excess(x, leverage=2.5, rate=rf, spread=0.0001, convention="geometric"),
        excess_returns(total, rf, convention="geometric"),
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_leverage_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValidationError, match="leverage must be positive"):
        levered_returns(_series(16), leverage=bad, rate=0.0)


def test_spread_must_be_finite() -> None:
    with pytest.raises(ValidationError, match="spread must be finite"):
        levered_returns(_series(16), leverage=2.0, rate=0.0, spread=math.nan)


# --- what the conversion cost ----------------------------------------------


def test_a_constant_rate_shift_is_exactly_the_closed_form() -> None:
    """``rate / sigma``, with nothing left over.

    Subtracting a constant moves the mean and leaves the standard deviation
    alone, so the Sharpe shift *is* the ratio, and ``residual`` is zero rather
    than small. It is asserted at 1e-15 and measures 1e-17.
    """
    x = _series()
    rf = period_rate(0.05, PPY, compounding="compound")
    impact = financing_impact(x, rf, periods_per_year=PPY)
    assert impact.shift == pytest.approx(impact.predicted_shift, rel=0.0, abs=1e-15)
    assert abs(impact.residual) < 1e-15
    assert impact.mean_rate == rf
    assert impact.observations == x.size
    assert impact.gross > impact.excess


def test_the_financing_haircut_is_about_a_third_of_a_sharpe_ratio() -> None:
    """The number this module exists for, on a decade of 16%-volatility returns.

    0.31 of annualised Sharpe at a 5% cash rate. The comparison worth making
    is against the haircut this library already computes, so the test computes
    that one too rather than quoting it: the deflated Sharpe ratio's benchmark
    for the best of fifty trials whose Sharpe ratios are dispersed by sampling
    error alone. Financing comes to between a third and two thirds of it.
    Both are the same kind of adjustment to the same number, and only one of
    them is usually reported.
    """
    x = _series()
    impact = financing_impact(
        x, period_rate(0.05, PPY, compounding="compound"), periods_per_year=PPY
    )
    assert 0.30 < impact.annualised_shift < 0.32
    multiplicity = expected_maximum_sharpe(50, 1.0 / x.size) * math.sqrt(PPY)
    assert 0.6 < multiplicity < 0.8
    assert 1.0 / 3.0 < impact.annualised_shift / multiplicity < 2.0 / 3.0
    annual_volatility = float(np.std(x, ddof=1)) * math.sqrt(PPY)
    assert 0.15 < annual_volatility < 0.16
    # The simple convention would put the same haircut 2.4% higher, which is
    # the compounding difference showing up where it is finally visible.
    simple = financing_impact(x, period_rate(0.05, PPY, compounding="simple"), periods_per_year=PPY)
    assert 1.02 < simple.annualised_shift / impact.annualised_shift < 1.03


def test_a_moving_rate_leaves_a_residual_the_closed_form_misses() -> None:
    """Because it moves the standard deviation too, which the closed form assumes it does not.

    Over a cash rate sweeping 0% to 6%, the residual is about 1e-04 of the
    shift: the closed form is a good approximation and not an identity, and
    the point of reporting both is that the gap says which it is.
    """
    x = _series()
    rf = _ramp(x.size, 0.0, 0.06) / PPY
    impact = financing_impact(x, rf, periods_per_year=PPY)
    assert impact.residual != 0.0
    assert 1e-05 < abs(impact.residual / impact.shift) < 1e-03
    assert impact.mean_rate == pytest.approx(float(np.mean(rf)))


def test_the_two_conventions_disagree_about_the_residual_and_not_the_shift() -> None:
    x = _series()
    rf = _ramp(x.size, 0.0, 0.06) / PPY
    arithmetic = financing_impact(x, rf, periods_per_year=PPY)
    geometric = financing_impact(x, rf, convention="geometric", periods_per_year=PPY)
    assert geometric.shift == pytest.approx(arithmetic.shift, rel=1e-03)
    assert abs(geometric.residual) > 1e-07
    assert geometric.residual * arithmetic.residual < 0.0


def test_a_zero_rate_takes_nothing() -> None:
    x = _series(512)
    impact = financing_impact(x, 0.0, periods_per_year=PPY)
    assert impact.shift == 0.0
    assert impact.predicted_shift == 0.0
    assert impact.gross == impact.excess


def test_annualised_shift_needs_a_frequency() -> None:
    impact = financing_impact(_series(64), 0.0001)
    assert impact.periods_per_year is None
    with pytest.raises(ValidationError, match="no annual scale"):
        _ = impact.annualised_shift


def test_financing_impact_refuses_a_series_with_no_variance() -> None:
    with pytest.raises(ValidationError, match="returns have zero variance"):
        financing_impact(np.full(32, 0.001), 0.0001)


def test_financing_impact_refuses_a_rate_that_flattens_the_excess_series() -> None:
    """A rate series equal to the returns leaves nothing with any variance to divide by."""
    x = _series(64)
    with pytest.raises(ValidationError, match="excess returns have zero variance"):
        financing_impact(x, x)
