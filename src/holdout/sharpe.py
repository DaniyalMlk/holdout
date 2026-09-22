"""The Sharpe ratio and its sampling error.

Conventions, fixed once here and used everywhere else:

* A Sharpe ratio is **per period** unless it says otherwise. Annualisation is a
  presentation step applied at the edge, because every inference formula in
  the literature (Lo 2002, Mertens 2002, Bailey and López de Prado 2012) is
  stated for the per-period ratio and the sample size in periods. Mixing an
  annualised ratio with a daily sample size is the most common way these
  formulas get misapplied, and the error is a factor of ``sqrt(252)``.
* Kurtosis is raw (3 for a normal distribution). See :mod:`holdout.moments`.

The standard error under non-normal returns is Mertens' (2002)::

    Var[SR] = (1 + SR**2 / 2 - skew * SR + (kurt - 3) / 4 * SR**2) / n
            = (1 - skew * SR + (kurt - 1) / 4 * SR**2) / n

The second form is the one that appears inside the probabilistic Sharpe ratio;
the two are algebraically identical and a test holds them to each other. With
``skew = 0`` and ``kurt = 3`` it reduces to Lo's (2002) iid-normal result
``(1 + SR**2 / 2) / n``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import InsufficientDataError, ValidationError
from .moments import autocorrelation, moments
from .series import FloatArray, as_returns, check_periods_per_year, check_probability

__all__ = [
    "SharpeRatio",
    "autocorrelation_adjusted_sharpe",
    "check_shape",
    "estimate_sharpe",
    "serial_correlation_factor",
    "sharpe_ratio",
    "sharpe_standard_error",
    "sharpe_variance_term",
]

Method = Literal["normal", "mertens"]
_NORMAL = NormalDist()


def check_shape(skewness: float, kurtosis: float) -> tuple[float, float]:
    """Validate a (skewness, raw kurtosis) pair as moments some distribution could have.

    Every distribution satisfies Pearson's inequality ``kurt >= 1 + skew**2``.
    A pair that violates it — most often excess kurtosis passed where raw was
    expected, for a series with any skew at all — cannot come from data, and
    the variance formulas built on it can go negative. Rejecting it here is
    what stops a confident-looking number being computed from impossible
    inputs.
    """
    s = float(skewness)
    k = float(kurtosis)
    if not (math.isfinite(s) and math.isfinite(k)):
        raise ValidationError(f"skewness and kurtosis must be finite, got {s!r} and {k!r}")
    if k < 1.0 + s * s - 1e-12:
        raise ValidationError(
            f"kurtosis {k:g} with skewness {s:g} violates kurtosis >= 1 + skewness**2, "
            "which every distribution satisfies; kurtosis is raw here (3 for a normal), "
            "so an excess kurtosis may have been passed"
        )
    return s, k


def sharpe_variance_term(sharpe: float, skewness: float = 0.0, kurtosis: float = 3.0) -> float:
    """``1 - skew * SR + (kurt - 1) / 4 * SR**2``: ``n`` times the variance of the estimator."""
    s, k = check_shape(skewness, kurtosis)
    sr = float(sharpe)
    if not math.isfinite(sr):
        raise ValidationError(f"sharpe must be finite, got {sr!r}")
    # Non-negative whenever check_shape passes: it is at least (skew * SR / 2 - 1)**2.
    return max(1.0 - s * sr + (k - 1.0) / 4.0 * sr * sr, 0.0)


def sharpe_standard_error(
    sharpe: float,
    n: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Asymptotic standard error of a per-period Sharpe ratio estimated from ``n`` periods."""
    if n < 2:
        raise InsufficientDataError(f"n must be at least 2, got {n}")
    return math.sqrt(sharpe_variance_term(sharpe, skewness, kurtosis) / n)


def _excess(returns: ArrayLike, risk_free: float | ArrayLike) -> FloatArray:
    x = as_returns(returns)
    rf = np.asarray(risk_free, dtype=np.float64)
    if rf.ndim == 0:
        if not math.isfinite(float(rf)):
            raise ValidationError(f"risk_free must be finite, got {float(rf)!r}")
        return x - float(rf)
    rf = as_returns(rf, name="risk_free", min_length=1)
    if rf.shape != x.shape:
        raise ValidationError(f"risk_free has {rf.size} observations but returns has {x.size}")
    return x - rf


def sharpe_ratio(
    returns: ArrayLike,
    *,
    risk_free: float | ArrayLike = 0.0,
    periods_per_year: float | None = None,
) -> float:
    """Sample Sharpe ratio of ``returns`` in excess of ``risk_free`` (both per period).

    Uses the ``ddof=1`` standard deviation. Returns the per-period ratio, or
    the ratio scaled by ``sqrt(periods_per_year)`` when that is given.
    """
    ppy = check_periods_per_year(periods_per_year)
    excess = _excess(returns, risk_free)
    sd = float(np.std(excess, ddof=1))
    if sd == 0.0 or sd <= np.finfo(np.float64).eps * float(np.max(np.abs(excess))):
        raise ValidationError("excess returns have zero variance, so the Sharpe ratio is undefined")
    sr = float(np.mean(excess)) / sd
    return sr * math.sqrt(ppy) if ppy is not None else sr


@dataclass(frozen=True)
class SharpeRatio:
    """A per-period Sharpe ratio together with what is needed to judge it.

    ``value`` is per period. ``annualised`` and the annualised standard error
    use the square-root-of-time rule, which is only right for serially
    uncorrelated returns; see :func:`autocorrelation_adjusted_sharpe` when that
    assumption is doubtful.
    """

    value: float
    n: int
    skewness: float
    kurtosis: float
    periods_per_year: float | None = None

    def standard_error(self, method: Method = "mertens") -> float:
        """Per-period standard error; ``"normal"`` ignores the observed skew and kurtosis."""
        if method == "normal":
            return sharpe_standard_error(self.value, self.n)
        if method == "mertens":
            return sharpe_standard_error(
                self.value, self.n, skewness=self.skewness, kurtosis=self.kurtosis
            )
        raise ValidationError(f"method must be 'normal' or 'mertens', got {method!r}")

    @property
    def annualised(self) -> float:
        return self.value * self._scale()

    def annualised_standard_error(self, method: Method = "mertens") -> float:
        return self.standard_error(method) * self._scale()

    def confidence_interval(
        self, level: float = 0.95, method: Method = "mertens", *, annualise: bool = False
    ) -> tuple[float, float]:
        """Two-sided normal-approximation interval for the true Sharpe ratio."""
        check_probability(level, "level")
        z = _NORMAL.inv_cdf(0.5 + level / 2.0)
        scale = self._scale() if annualise else 1.0
        half = z * self.standard_error(method) * scale
        centre = self.value * scale
        return centre - half, centre + half

    def _scale(self) -> float:
        if self.periods_per_year is None:
            raise ValidationError("periods_per_year was not given, so there is no annual scale")
        return math.sqrt(self.periods_per_year)


def estimate_sharpe(
    returns: ArrayLike,
    *,
    risk_free: float | ArrayLike = 0.0,
    periods_per_year: float | None = None,
) -> SharpeRatio:
    """Estimate the per-period Sharpe ratio with the moments its inference needs."""
    ppy = check_periods_per_year(periods_per_year)
    excess = _excess(returns, risk_free)
    sr = sharpe_ratio(excess)
    m = moments(excess)
    return SharpeRatio(
        value=sr, n=m.n, skewness=m.skewness, kurtosis=m.kurtosis, periods_per_year=ppy
    )


def serial_correlation_factor(q: int, autocorrelations: ArrayLike) -> float:
    """Lo's (2002) factor ``eta(q)`` mapping a per-period Sharpe ratio to ``q`` periods.

    ``eta(q) = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k)``. With no
    serial correlation it is ``sqrt(q)``. ``autocorrelations`` holds
    ``rho_1, rho_2, ...``; lags beyond its length are taken as zero, and
    entries beyond ``q - 1`` are ignored.
    """
    if q < 1:
        raise ValidationError(f"q must be a positive integer, got {q}")
    rho = np.asarray(autocorrelations, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(rho)) or np.any(np.abs(rho) > 1.0):
        raise ValidationError("autocorrelations must be finite and within [-1, 1]")
    lags = min(q - 1, rho.size)
    k = np.arange(1, lags + 1, dtype=np.float64)
    variance_ratio = q + 2.0 * float(np.sum((q - k) * rho[:lags]))
    if variance_ratio <= 0.0:
        raise ValidationError(
            "these autocorrelations imply a non-positive variance for the q-period return; "
            "they cannot all hold at once"
        )
    return q / math.sqrt(variance_ratio)


def autocorrelation_adjusted_sharpe(
    returns: ArrayLike,
    periods_per_year: int,
    *,
    risk_free: float | ArrayLike = 0.0,
    max_lag: int | None = None,
) -> float:
    """Annualised Sharpe ratio using Lo's serial-correlation adjustment.

    Estimates ``rho_1 .. rho_max_lag`` from the sample (default: all ``q - 1``
    lags) and scales the per-period ratio by ``eta(q)`` rather than
    ``sqrt(q)``. Positive autocorrelation — typical of smoothed or illiquid
    marks — makes the naive annualisation overstate the ratio.

    Each extra lag adds estimation noise; for daily data annualised to a year,
    restricting ``max_lag`` to the lags that are plausibly non-zero is usually
    wiser than estimating 251 of them.
    """
    q = int(periods_per_year)
    if q != periods_per_year or q < 1:
        raise ValidationError(
            f"periods_per_year must be a positive integer, got {periods_per_year}"
        )
    excess = _excess(returns, risk_free)
    lags = q - 1 if max_lag is None else int(max_lag)
    if lags < 0:
        raise ValidationError(f"max_lag must be non-negative, got {max_lag}")
    rho = autocorrelation(excess, lags) if lags > 0 else np.empty(0)
    return sharpe_ratio(excess) * serial_correlation_factor(q, rho)
