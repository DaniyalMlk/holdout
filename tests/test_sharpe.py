from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from holdout import (
    InsufficientDataError,
    ValidationError,
    autocorrelation_adjusted_sharpe,
    estimate_sharpe,
    serial_correlation_factor,
    sharpe_ratio,
    sharpe_standard_error,
)
from holdout.sharpe import check_shape, sharpe_variance_term


def test_sharpe_ratio_by_hand() -> None:
    x = [0.02, -0.01, 0.03, 0.00, 0.01]
    mean = 0.01
    sd = math.sqrt(sum((v - mean) ** 2 for v in x) / 4)
    assert sharpe_ratio(x) == pytest.approx(mean / sd)
    assert sharpe_ratio(x, periods_per_year=12) == pytest.approx(mean / sd * math.sqrt(12))


def test_risk_free_scalar_and_series() -> None:
    x = np.array([0.02, -0.01, 0.03, 0.00, 0.01])
    rf = np.array([0.001, 0.001, 0.002, 0.002, 0.001])
    assert sharpe_ratio(x, risk_free=0.001) == pytest.approx(sharpe_ratio(x - 0.001))
    assert sharpe_ratio(x, risk_free=rf) == pytest.approx(sharpe_ratio(x - rf))
    with pytest.raises(ValidationError, match="risk_free has 2"):
        sharpe_ratio(x, risk_free=[0.0, 0.0])
    with pytest.raises(ValidationError, match="finite"):
        sharpe_ratio(x, risk_free=math.nan)


def test_zero_variance_is_an_error() -> None:
    with pytest.raises(ValidationError, match="zero variance"):
        sharpe_ratio([0.01] * 20)


@given(
    st.floats(min_value=-1.0, max_value=1.0),
    st.floats(min_value=-3.0, max_value=3.0),
    st.floats(min_value=0.0, max_value=20.0),
)
def test_mertens_forms_are_identical(sr: float, skew: float, extra: float) -> None:
    kurt = 1.0 + skew**2 + extra
    first = 1.0 + sr**2 / 2.0 - skew * sr + (kurt - 3.0) / 4.0 * sr**2
    assert sharpe_variance_term(sr, skew, kurt) == pytest.approx(first, abs=1e-12)
    assert sharpe_variance_term(sr, skew, kurt) >= 0.0


def test_normal_case_is_lo() -> None:
    assert sharpe_standard_error(0.1, 250) == pytest.approx(math.sqrt((1 + 0.005) / 250))


def test_impossible_moments_are_rejected() -> None:
    # Excess kurtosis 0 passed for a skewed series: no distribution has it.
    with pytest.raises(ValidationError, match="excess kurtosis may have been passed"):
        sharpe_standard_error(0.1, 250, skewness=-1.5, kurtosis=0.0)
    with pytest.raises(ValidationError, match="finite"):
        check_shape(math.nan, 3.0)
    with pytest.raises(InsufficientDataError):
        sharpe_standard_error(0.1, 1)


def test_standard_error_matches_the_sampling_distribution_for_normal_returns() -> None:
    rng = np.random.default_rng(11)
    reps, n, sr = 4000, 500, 0.1
    x = rng.normal(size=(reps, n)) + sr
    estimates = x.mean(axis=1) / x.std(axis=1, ddof=1)
    # Monte Carlo error on a standard deviation from 4000 draws is about 1.1%.
    assert estimates.std(ddof=1) == pytest.approx(sharpe_standard_error(sr, n), rel=0.04)


def test_mertens_corrects_what_the_normal_formula_misses() -> None:
    # Negatively skewed returns (skew -2, kurtosis 9): the normal formula
    # understates the sampling error by about 10%, Mertens' is within MC error.
    rng = np.random.default_rng(11)
    reps, n, sr = 4000, 500, 0.1
    rng.normal(size=(reps, n))  # keep the stream aligned with the normal case
    x = -(rng.exponential(size=(reps, n)) - 1.0) + sr
    estimates = x.mean(axis=1) / x.std(axis=1, ddof=1)
    observed = estimates.std(ddof=1)
    mertens = sharpe_standard_error(sr, n, skewness=-2.0, kurtosis=9.0)
    normal = sharpe_standard_error(sr, n)
    assert observed == pytest.approx(mertens, rel=0.04)
    assert normal < 0.93 * observed


def test_estimate_carries_what_inference_needs() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0.0005, 0.01, 1000)
    est = estimate_sharpe(x, periods_per_year=252)
    assert est.value == pytest.approx(sharpe_ratio(x))
    assert est.annualised == pytest.approx(sharpe_ratio(x, periods_per_year=252))
    assert est.n == 1000
    assert est.standard_error("normal") == pytest.approx(sharpe_standard_error(est.value, 1000))
    assert est.annualised_standard_error() == pytest.approx(est.standard_error() * math.sqrt(252))
    lo, hi = est.confidence_interval(0.95)
    assert lo < est.value < hi
    assert (hi - lo) / 2 == pytest.approx(1.959964 * est.standard_error(), rel=1e-6)
    alo, ahi = est.confidence_interval(0.95, annualise=True)
    assert alo == pytest.approx(lo * math.sqrt(252))
    assert ahi == pytest.approx(hi * math.sqrt(252))


def test_estimate_without_frequency_refuses_to_annualise() -> None:
    est = estimate_sharpe(np.random.default_rng(1).normal(size=50))
    with pytest.raises(ValidationError, match="periods_per_year"):
        _ = est.annualised
    with pytest.raises(ValidationError, match="method"):
        est.standard_error("bogus")  # type: ignore[arg-type]


def test_confidence_interval_covers_at_the_nominal_rate() -> None:
    rng = np.random.default_rng(21)
    true_sr, hits, reps = 0.08, 0, 2000
    for _ in range(reps):
        x = rng.standard_t(6, size=400) / math.sqrt(1.5) + true_sr
        lo, hi = estimate_sharpe(x).confidence_interval(0.9)
        hits += lo <= true_sr <= hi
    # Binomial standard error at 2000 draws is about 0.7 percentage points.
    assert hits / reps == pytest.approx(0.9, abs=0.025)


def test_no_serial_correlation_gives_square_root_of_time() -> None:
    assert serial_correlation_factor(12, []) == pytest.approx(math.sqrt(12))
    assert serial_correlation_factor(12, np.zeros(30)) == pytest.approx(math.sqrt(12))
    assert serial_correlation_factor(1, [0.9]) == pytest.approx(1.0)


@pytest.mark.parametrize("phi", [-0.4, -0.1, 0.1, 0.3, 0.6])
@pytest.mark.parametrize("q", [2, 12, 52])
def test_eta_matches_the_ar1_closed_form(phi: float, q: int) -> None:
    rho = phi ** np.arange(1, q)
    inner = phi / (1 - phi) * (q - (1 - phi**q) / (1 - phi))
    assert serial_correlation_factor(q, rho) == pytest.approx(q / math.sqrt(q + 2 * inner))


def test_positive_autocorrelation_shrinks_the_annual_ratio() -> None:
    q = 12
    assert serial_correlation_factor(q, 0.3 ** np.arange(1, q)) < math.sqrt(q)
    assert serial_correlation_factor(q, (-0.3) ** np.arange(1, q)) > math.sqrt(q)


def test_inconsistent_autocorrelations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="non-positive variance"):
        serial_correlation_factor(3, [-0.9, -0.9])
    with pytest.raises(ValidationError, match="within"):
        serial_correlation_factor(3, [1.5])
    with pytest.raises(ValidationError, match="positive integer"):
        serial_correlation_factor(0, [])


def test_adjusted_sharpe_matches_the_sharpe_of_aggregated_returns() -> None:
    # The adjusted annual ratio should estimate the Sharpe ratio of
    # non-overlapping 12-period sums; the naive sqrt(12) scaling should not.
    rng = np.random.default_rng(3)
    q, phi, blocks = 12, 0.4, 40_000
    e = rng.normal(size=q * blocks)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, x.size):
        x[t] = phi * x[t - 1] + e[t]
    x = x * 0.01 + 0.002
    aggregated = sharpe_ratio(x.reshape(blocks, q).sum(axis=1))
    adjusted = autocorrelation_adjusted_sharpe(x, q)
    naive = sharpe_ratio(x, periods_per_year=q)
    assert adjusted == pytest.approx(aggregated, rel=0.03)
    assert naive > 1.25 * aggregated


def test_adjusted_sharpe_arguments() -> None:
    x = np.random.default_rng(0).normal(0.01, 0.02, 200)
    assert autocorrelation_adjusted_sharpe(x, 12, max_lag=0) == pytest.approx(
        sharpe_ratio(x, periods_per_year=12)
    )
    with pytest.raises(ValidationError, match="positive integer"):
        autocorrelation_adjusted_sharpe(x, 12.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="max_lag"):
        autocorrelation_adjusted_sharpe(x, 12, max_lag=-1)
