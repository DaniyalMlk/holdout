from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from holdout import ValidationError, autocorrelation, kurtosis, moments, skewness
from holdout.exceptions import InsufficientDataError

SAMPLES = {
    "normal": np.random.default_rng(1).normal(0.001, 0.01, 500),
    "lognormal": np.random.default_rng(2).lognormal(0.0, 0.6, 300) - 1.0,
    "student": np.random.default_rng(3).standard_t(5, 1000) * 0.01,
    "short": np.array([0.01, -0.03, 0.02, 0.05, -0.01]),
}


@pytest.mark.parametrize("name", sorted(SAMPLES))
@pytest.mark.parametrize("bias", [True, False])
def test_skewness_matches_scipy(name: str, bias: bool) -> None:
    x = SAMPLES[name]
    assert skewness(x, bias=bias) == pytest.approx(stats.skew(x, bias=bias), rel=1e-12)


@pytest.mark.parametrize("name", sorted(SAMPLES))
@pytest.mark.parametrize("bias", [True, False])
def test_kurtosis_matches_scipy(name: str, bias: bool) -> None:
    x = SAMPLES[name]
    excess = stats.kurtosis(x, fisher=True, bias=bias)
    assert kurtosis(x, excess=True, bias=bias) == pytest.approx(excess, rel=1e-12)
    assert kurtosis(x, bias=bias) == pytest.approx(excess + 3.0, rel=1e-12)


def test_kurtosis_is_raw_by_default() -> None:
    x = np.random.default_rng(4).normal(size=200_000)
    assert kurtosis(x) == pytest.approx(3.0, abs=0.05)
    assert skewness(x) == pytest.approx(0.0, abs=0.02)


def test_moments_bundle_agrees_with_the_single_functions() -> None:
    x = SAMPLES["lognormal"]
    m = moments(x)
    assert m.n == x.size
    assert m.mean == pytest.approx(float(np.mean(x)))
    assert m.std == pytest.approx(float(np.std(x, ddof=1)))
    assert m.skewness == pytest.approx(skewness(x))
    assert m.kurtosis == pytest.approx(kurtosis(x))
    assert m.excess_kurtosis == pytest.approx(kurtosis(x, excess=True))
    assert moments(x, ddof=0).std == pytest.approx(float(np.std(x)))
    with pytest.raises(ValidationError, match="ddof"):
        moments(x, ddof=x.size)


@pytest.mark.parametrize("value", [0.0, 0.01, 1e6])
def test_constant_series_is_rejected_not_nan(value: float) -> None:
    with pytest.raises(ValidationError, match="zero variance"):
        skewness(np.full(50, value))
    with pytest.raises(ValidationError, match="zero variance"):
        kurtosis(np.full(50, value))


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_subnormal=False),
        min_size=3,
        max_size=60,
    )
)
def test_sample_moments_satisfy_pearsons_inequality(values: list[float]) -> None:
    # Any sample is itself a distribution, so kurt >= 1 + skew**2 must hold;
    # the Sharpe ratio inference relies on this to keep variances non-negative.
    x = np.array(values)
    if np.ptp(x) < 1e-6:
        return
    m = moments(x)
    assert m.kurtosis >= 1.0 + m.skewness**2 - 1e-9


def test_autocorrelation_matches_the_textbook_estimator() -> None:
    rng = np.random.default_rng(5)
    e = rng.normal(size=5000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, e.size):
        x[t] = 0.6 * x[t - 1] + e[t]
    rho = autocorrelation(x, 3)
    d = x - x.mean()
    expected = [float(np.sum(d[k:] * d[:-k]) / np.sum(d * d)) for k in (1, 2, 3)]
    np.testing.assert_allclose(rho, expected, rtol=1e-12)
    np.testing.assert_allclose(rho, [0.6, 0.36, 0.216], atol=0.04)


def test_autocorrelation_edge_cases() -> None:
    assert autocorrelation([1.0, 2.0, 3.0], 0).size == 0
    with pytest.raises(InsufficientDataError):
        autocorrelation([1.0, 2.0, 3.0], 3)
    with pytest.raises(ValidationError):
        autocorrelation([1.0, 2.0, 3.0], -1)
    with pytest.raises(ValidationError, match="zero variance"):
        autocorrelation([1.0, 1.0, 1.0], 1)
