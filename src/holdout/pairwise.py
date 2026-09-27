"""Comparing exactly two strategies, which is the comparison people actually make.

The rest of this library is built for families. :func:`~holdout.spa.reality_check`
and :func:`~holdout.spa.superior_predictive_ability` test many candidates against
one benchmark with the selection accounted for;
:func:`~holdout.mcs.model_confidence_set` finds the set that cannot be told apart
from the best. All of that machinery exists to control an error rate over a family,
and pointing it at a family of one is using a multiple-testing correction on a
single test.

The pairwise question has its own answer and the library did not have it. Nor is
differencing two :func:`~holdout.sharpe.estimate_sharpe` standard errors a
substitute, because it is wrong twice. It ignores the correlation between the two
series, which for two strategies trading the same market is usually large and
always *reduces* the variance of the difference — so the naive interval is too
wide and the test too conservative. And it treats a Sharpe ratio as if it were a
mean, when it is a ratio of two estimated moments, so the delta method has terms
the difference of two independent standard errors does not.

Two estimators of the same difference are returned from one call.

**Jobson-Korkie with Memmel's correction.** The closed form under independent
normal returns::

    Var[Sa - Sb] = (2 - 2r + (Sa^2 + Sb^2 - 2 Sa Sb r^2) / 2) / n

with ``r`` the correlation of the two series. Jobson and Korkie (1981) derived it
with an error in the variance; Memmel (2003) corrected it, and the corrected form
is what every practitioner reference quotes. Its assumptions are exactly the ones
financial returns do not satisfy.

**Ledoit and Wolf (2008).** The same difference, with its variance from the delta
method over the four sample moments ``(mean_a, mean_b, E[a^2], E[b^2])`` and a
heteroskedasticity-and-autocorrelation-consistent estimate of their covariance. It
makes no distributional assumption and allows serial dependence, at the cost of a
kernel bandwidth that has to come from somewhere.

The interesting part is what happens when the two disagree, so both are always
returned rather than one being chosen. What measuring them showed is in
``examples/sharpe_difference.py`` and it is not quite the story the literature
implies: on independent normal data the two agree closely and both hold their
size, and the gap only opens where the returns are serially dependent — where the
closed form rejects a true null far more often than it should and the robust one
is closer without being right.

Sign and scale conventions. The difference is ``first - second``, so a positive
one favours the first series. Both estimators are invariant to rescaling either
series, because a Sharpe ratio is, and the tests assert that rather than assuming
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray, as_returns, check_periods_per_year

__all__ = [
    "SharpeDifference",
    "VarianceMethod",
    "newey_west_bandwidth",
    "newey_west_covariance",
    "sharpe_difference",
]

_NORMAL = NormalDist()

#: Fewest paired observations a difference will be computed from. Two Sharpe
#: ratios from ten returns each are two numbers, and the asymptotic variance of
#: their difference is a statement about a limit neither is anywhere near.
MINIMUM_OBSERVATIONS = 20


class VarianceMethod(str, Enum):
    """Which variance of the difference to use for the headline statistic."""

    #: Jobson-Korkie with Memmel's correction. Closed form, assumes independent
    #: normal returns.
    JOBSON_KORKIE_MEMMEL = "jobson_korkie_memmel"
    #: Ledoit-Wolf: the delta method over four moments with a HAC covariance.
    #: Robust to non-normality and serial dependence, at the cost of a bandwidth.
    LEDOIT_WOLF = "ledoit_wolf"


def newey_west_bandwidth(observations: int) -> int:
    """``floor(4 (n / 100) ** (2/9))``, the rule of thumb, at least one.

    Chosen because it is the convention rather than because it is optimal:
    Andrews' (1991) automatic bandwidth is data-dependent and better, and
    reporting a bandwidth that came from a formula in the documentation is
    honest in a way an automatic one silently applied is not. The bandwidth is
    an argument on :func:`sharpe_difference`, and the result says what was used.
    """
    if observations < 1:
        raise ValidationError(f"observations must be positive, got {observations!r}")
    return max(1, int(4.0 * (observations / 100.0) ** (2.0 / 9.0)))


def newey_west_covariance(series: NDArray[np.float64], bandwidth: int) -> NDArray[np.float64]:
    """HAC covariance of the mean of a multivariate series, times ``n``.

    ``series`` has one row per observation and one column per moment. Returns the
    long-run covariance ``Gamma_0 + sum_j w_j (Gamma_j + Gamma_j')`` under a
    Bartlett kernel ``w_j = 1 - j / (bandwidth + 1)``, which is the estimate whose
    ``/ n`` is the variance of the sample mean.

    The Bartlett kernel is not a detail. Summing the autocovariances with equal
    weights gives an estimator that is not guaranteed positive semi-definite, so a
    variance computed from it can come out negative — which does not look like an
    error, it looks like a nan appearing in a standard error two steps later.
    """
    if series.ndim != 2:
        raise ValidationError(
            f"a HAC covariance needs one row per observation, got shape {series.shape}"
        )
    count = series.shape[0]
    if bandwidth < 0:
        raise ValidationError(f"bandwidth must not be negative, got {bandwidth!r}")
    if bandwidth >= count:
        raise ValidationError(
            f"a bandwidth of {bandwidth} needs more than {count} observations; at that "
            "lag the kernel reaches past the end of the sample"
        )
    centred = series - series.mean(axis=0)
    total = centred.T @ centred / count
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        cross = centred[lag:].T @ centred[:-lag] / count
        total = total + weight * (cross + cross.T)
    return np.asarray(total, dtype=np.float64)


@dataclass(frozen=True)
class SharpeDifference:
    """The difference of two Sharpe ratios, by two variances at once."""

    #: Per-period Sharpe ratio of the first series.
    first: float
    #: Per-period Sharpe ratio of the second series.
    second: float
    #: ``first - second``. Positive favours the first series.
    difference: float
    observations: int
    correlation: float
    #: Standard error under Jobson-Korkie with Memmel's correction.
    closed_form_error: float
    #: Standard error from Ledoit-Wolf's delta method with a HAC covariance.
    robust_error: float
    #: Bandwidth the HAC estimate used.
    bandwidth: int
    #: Which of the two the headline :attr:`statistic` and :attr:`p_value` use.
    method: VarianceMethod
    periods_per_year: float | None = None

    @property
    def standard_error(self) -> float:
        """The error belonging to :attr:`method`."""
        return (
            self.closed_form_error
            if self.method is VarianceMethod.JOBSON_KORKIE_MEMMEL
            else self.robust_error
        )

    @property
    def statistic(self) -> float:
        """The difference in standard errors. Asymptotically standard normal."""
        return self.difference / self.standard_error

    @property
    def p_value(self) -> float:
        """Two-sided, against the null that the two Sharpe ratios are equal."""
        return 2.0 * (1.0 - _NORMAL.cdf(abs(self.statistic)))

    @property
    def error_ratio(self) -> float:
        """Robust error over closed-form error.

        Above one means the closed form is understating the uncertainty, which is
        the direction that matters: it is the direction in which a difference
        looks significant and is not.
        """
        return self.robust_error / self.closed_form_error

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Two-sided normal-approximation interval for the true difference."""
        if not 0.0 < level < 1.0:
            raise ValidationError(f"level must be in (0, 1), got {level!r}")
        half = _NORMAL.inv_cdf(0.5 + level / 2.0) * self.standard_error
        return self.difference - half, self.difference + half

    @property
    def annualised_difference(self) -> float:
        """The difference scaled by the root of the periods in a year.

        The same square-root-of-time rule the rest of this library applies at the
        edge, and with the same caveat: it is right for serially uncorrelated
        returns, which is the assumption the robust variance exists because of.
        """
        if self.periods_per_year is None:
            raise ValidationError("periods_per_year was not given, so there is no annual scale")
        return self.difference * math.sqrt(self.periods_per_year)


#: Multiple of machine epsilon below which a series is treated as constant. The
#: test has to be *relative*, because the standard deviation of a constant array
#: is not exactly zero: subtracting the mean leaves rounding of order
#: ``eps * level``, and on a series of 0.001 repeated 500 times numpy returns
#: 4.3e-19 rather than 0. An absolute guard at zero therefore does not fire, the
#: Sharpe ratio comes out at 2.3e15, and every number downstream is arithmetic on
#: rounding error.
CONSTANT_TOLERANCE = 8.0


def _deviation(series: FloatArray, name: str) -> float:
    """Sample standard deviation, refusing a series that does not move.

    Relative to the series' own magnitude, so the threshold means the same thing
    whether returns are quoted as fractions or in basis points.
    """
    deviation = float(series.std(ddof=1))
    scale = float(np.abs(series).max()) or 1.0
    floor = CONSTANT_TOLERANCE * float(np.finfo(np.float64).eps) * scale
    if deviation <= floor:
        raise InsufficientDataError(
            f"{name} does not move: its standard deviation is {deviation:.4g}, which is "
            f"within rounding of zero for a series of this magnitude ({scale:.4g}). A "
            "Sharpe ratio needs a positive one — computed from this it would come back "
            "as a very large number rather than as an error."
        )
    return deviation


def _memmel_error(first: float, second: float, correlation: float, count: int) -> float:
    variance = (
        2.0
        - 2.0 * correlation
        + 0.5 * (first * first + second * second - 2.0 * first * second * correlation**2)
    ) / count
    if variance <= 0.0:
        raise InsufficientDataError(
            f"the closed-form variance of the difference came out {variance:.4g}. Under "
            "Memmel's correction that happens as the two series approach perfect "
            f"correlation — here it is {correlation:.6f} — where the difference of the "
            "two ratios has no sampling error left to speak of and the statistic is not "
            "defined."
        )
    return math.sqrt(variance)


def _ledoit_wolf_error(
    first_series: FloatArray,
    second_series: FloatArray,
    bandwidth: int,
) -> float:
    """Delta method over ``(mean_a, mean_b, E[a^2], E[b^2])`` with a HAC covariance.

    The gradient is where this is easy to get wrong, so it is written out. With
    ``s = sqrt(g - m ** 2)`` the Sharpe ratio is ``m / s`` and

        d(m/s)/dm = g / s ** 3        d(m/s)/dg = -m / (2 s ** 3)

    both of which follow from differentiating ``m (g - m^2) ** -0.5`` and
    collecting terms. The ``g / s ** 3`` is the part that looks wrong and is not:
    the derivative in the mean picks up a second term because the denominator
    depends on the mean too.
    """
    count = first_series.size
    moments = np.column_stack(
        (
            first_series,
            second_series,
            first_series * first_series,
            second_series * second_series,
        )
    )
    mean_a, mean_b, second_a, second_b = (float(value) for value in moments.mean(axis=0))
    variance_a = second_a - mean_a * mean_a
    variance_b = second_b - mean_b * mean_b
    cube_a = variance_a**1.5
    cube_b = variance_b**1.5
    gradient = np.array(
        [
            second_a / cube_a,
            -second_b / cube_b,
            -mean_a / (2.0 * cube_a),
            mean_b / (2.0 * cube_b),
        ]
    )
    covariance = newey_west_covariance(moments, bandwidth)
    variance = float(gradient @ covariance @ gradient) / count
    if variance <= 0.0:
        raise InsufficientDataError(
            f"the robust variance of the difference came out {variance:.4g}. A Bartlett "
            "kernel cannot produce a negative one from a full-rank sample, so this is "
            "two series whose moments are collinear — most often the same series twice, "
            "or one a fixed multiple of the other."
        )
    return math.sqrt(variance)


def sharpe_difference(
    first: ArrayLike,
    second: ArrayLike,
    *,
    method: VarianceMethod = VarianceMethod.LEDOIT_WOLF,
    bandwidth: int | None = None,
    periods_per_year: float | None = None,
) -> SharpeDifference:
    """Test whether two paired return series have different Sharpe ratios.

    The series must be aligned and the same length: the correlation between them
    is half of what makes this test different from two separate ones, and a
    misalignment destroys it while leaving every number looking plausible.

    ``method`` decides only which variance the headline statistic and p-value use.
    Both are computed and both are on the result, because the disagreement between
    them is the finding when there is one. The default is Ledoit-Wolf, because the
    assumptions the closed form needs are exactly the ones return series do not
    satisfy.

    ``bandwidth`` is the Bartlett-kernel truncation for the robust variance, and
    defaults to :func:`newey_west_bandwidth`. Zero uses no autocovariances at all,
    which is the White heteroskedasticity-consistent case: still free of the
    normality assumption and not of the independence one.
    """
    left = as_returns(first, name="first", min_length=MINIMUM_OBSERVATIONS)
    right = as_returns(second, name="second", min_length=MINIMUM_OBSERVATIONS)
    if left.size != right.size:
        raise ValidationError(
            f"the two series are paired, so they must be the same length: got "
            f"{left.size} and {right.size}. Aligning them is the caller's job because "
            "only the caller knows which end to trim."
        )
    if np.array_equal(left, right):
        raise ValidationError(
            "the two series are identical, so the difference is exactly zero and so is "
            "its variance. The statistic is 0/0 and there is nothing to test — which is "
            "a different answer from 'the difference is not significant', and returning "
            "a p-value of one here would blur the two."
        )
    annual = check_periods_per_year(periods_per_year)
    count = left.size
    lag = newey_west_bandwidth(count) if bandwidth is None else bandwidth

    deviation_a = _deviation(left, "first")
    deviation_b = _deviation(right, "second")
    sharpe_a = float(left.mean()) / deviation_a
    sharpe_b = float(right.mean()) / deviation_b
    correlation = float(np.corrcoef(left, right)[0, 1])

    return SharpeDifference(
        first=sharpe_a,
        second=sharpe_b,
        difference=sharpe_a - sharpe_b,
        observations=count,
        correlation=correlation,
        closed_form_error=_memmel_error(sharpe_a, sharpe_b, correlation, count),
        robust_error=_ledoit_wolf_error(left, right, lag),
        bandwidth=lag,
        method=VarianceMethod(method),
        periods_per_year=annual,
    )
