"""Probabilistic and deflated Sharpe ratios.

The probabilistic Sharpe ratio (Bailey and López de Prado 2012) is the
probability that the true Sharpe ratio exceeds a benchmark, given the observed
ratio, the sample length, and the shape of the returns::

    PSR(SR*) = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR**2) )

The deflated Sharpe ratio (Bailey and López de Prado 2014) is the same
probability with the benchmark raised to the Sharpe ratio that the *best* of
``N`` unskilled trials would be expected to show. That is the whole idea: the
right null for a selected strategy is not "zero skill" but "the best of N
attempts at zero skill".

Everything here is per period. The worked example in the 2014 paper states an
annualised variance of trial Sharpe ratios and converts at the end; the test
suite reproduces it that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import InsufficientDataError, ValidationError
from .series import as_matrix, check_probability
from .sharpe import SharpeRatio, estimate_sharpe, sharpe_variance_term

__all__ = [
    "EULER_MASCHERONI",
    "DeflatedSharpe",
    "deflate_trials",
    "deflated_sharpe_ratio",
    "expected_maximum_normal",
    "expected_maximum_sharpe",
    "minimum_track_record_length",
    "probabilistic_sharpe_ratio",
]

EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()

MaximumMethod = Literal["approximation", "exact"]


def _check_n(n: int) -> int:
    if int(n) != n or n < 2:
        raise InsufficientDataError(f"n must be an integer of at least 2, got {n}")
    return int(n)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n: int,
    *,
    benchmark: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true per-period Sharpe ratio exceeds ``benchmark``.

    ``sharpe`` and ``benchmark`` are per period, ``n`` is the number of
    periods, ``kurtosis`` is raw. Uses ``n - 1`` as in the original paper.
    """
    n = _check_n(n)
    if not math.isfinite(benchmark):
        raise ValidationError(f"benchmark must be finite, got {benchmark!r}")
    term = sharpe_variance_term(sharpe, skewness, kurtosis)
    if term == 0.0:
        # Degenerate two-point distributions only: the estimate has no error.
        return 1.0 if sharpe > benchmark else 0.0 if sharpe < benchmark else 0.5
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(term)
    return _NORMAL.cdf(z)


def minimum_track_record_length(
    sharpe: float,
    *,
    benchmark: float = 0.0,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Number of periods needed for ``PSR(benchmark)`` to reach ``confidence``.

    ``MinTRL = 1 + (1 - skew * SR + (kurt - 1) / 4 * SR**2) * (z / (SR - SR*))**2``.
    It is the inverse of :func:`probabilistic_sharpe_ratio` in ``n``, and a test
    holds the two to each other. Undefined — and an error — when the observed
    ratio does not exceed the benchmark, since no amount of data at that ratio
    would establish that it does.
    """
    check_probability(confidence, "confidence")
    if not sharpe > benchmark:
        raise ValidationError(
            f"sharpe {sharpe:g} does not exceed benchmark {benchmark:g}; "
            "no track record length makes it significant"
        )
    z = _NORMAL.inv_cdf(confidence)
    term = sharpe_variance_term(sharpe, skewness, kurtosis)
    return 1.0 + term * (z / (sharpe - benchmark)) ** 2


def expected_maximum_normal(n_trials: int) -> float:
    """Exact ``E[max(Z_1..Z_N)]`` for ``N`` independent standard normals.

    Integrates ``x * N * phi(x) * Phi(x)**(N - 1)`` numerically. Accurate to
    about 1e-10 for ``N`` up to 10**7, which :func:`expected_maximum_sharpe`
    uses as the reference against which its closed-form approximation is
    measured.
    """
    if int(n_trials) != n_trials or n_trials < 1:
        raise ValidationError(f"n_trials must be a positive integer, got {n_trials}")
    n = int(n_trials)
    if n == 1:
        return 0.0
    x = np.linspace(-9.0, 9.0 + math.sqrt(2.0 * math.log(n)), 40_001)
    cdf = np.array([_NORMAL.cdf(float(v)) for v in x])
    pdf = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    with np.errstate(divide="ignore"):
        log_cdf = np.where(cdf > 0.0, np.log(np.maximum(cdf, 1e-300)), -np.inf)
    density = n * pdf * np.exp((n - 1) * log_cdf)
    y = x * density
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def expected_maximum_sharpe(
    n_trials: int,
    variance: float,
    *,
    mean: float = 0.0,
    method: MaximumMethod = "approximation",
) -> float:
    """Expected maximum of ``N`` independent Sharpe ratios with the given mean and variance.

    ``variance`` is the cross-sectional variance of the trials' Sharpe ratios,
    on the same (per-period) scale as the result. The default is the
    approximation used in the deflated Sharpe ratio paper::

        E[max] ~= mean + sqrt(variance) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))

    with ``g`` the Euler–Mascheroni constant. ``method="exact"`` integrates the
    normal order statistic instead. The approximation is asymptotic. Measured
    against the exact value it overstates by 2.6% at five trials, 0.9% at a
    hundred and 0.1% at a million — so for three or more trials it errs towards
    a higher benchmark and a more conservative deflated ratio — but it
    understates by 8% at two trials.
    """
    if int(n_trials) != n_trials or n_trials < 1:
        raise ValidationError(f"n_trials must be a positive integer, got {n_trials}")
    if not (math.isfinite(variance) and variance >= 0.0):
        raise ValidationError(f"variance must be non-negative and finite, got {variance!r}")
    if not math.isfinite(mean):
        raise ValidationError(f"mean must be finite, got {mean!r}")
    n = int(n_trials)
    if n == 1:
        return mean
    if method == "exact":
        scale = expected_maximum_normal(n)
    elif method == "approximation":
        g = EULER_MASCHERONI
        scale = (1.0 - g) * _NORMAL.inv_cdf(1.0 - 1.0 / n) + g * _NORMAL.inv_cdf(
            1.0 - 1.0 / (n * math.e)
        )
    else:
        raise ValidationError(f"method must be 'approximation' or 'exact', got {method!r}")
    return mean + math.sqrt(variance) * scale


def deflated_sharpe_ratio(
    sharpe: float,
    n: int,
    *,
    n_trials: int,
    trials_variance: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trials_mean: float = 0.0,
    method: MaximumMethod = "approximation",
) -> float:
    """Probability that the selected strategy's true Sharpe ratio exceeds zero, after selection.

    Equal to :func:`probabilistic_sharpe_ratio` with the benchmark set to
    :func:`expected_maximum_sharpe` of the trials. ``sharpe`` is the selected
    strategy's per-period ratio over ``n`` periods; ``trials_variance`` is the
    per-period variance of all ``n_trials`` trials' ratios.
    """
    benchmark = expected_maximum_sharpe(n_trials, trials_variance, mean=trials_mean, method=method)
    return probabilistic_sharpe_ratio(
        sharpe, n, benchmark=benchmark, skewness=skewness, kurtosis=kurtosis
    )


@dataclass(frozen=True)
class DeflatedSharpe:
    """The deflated Sharpe ratio of the best of a set of trials, with its inputs."""

    selected: int
    estimate: SharpeRatio
    n_trials: float
    trials_mean: float
    trials_variance: float
    expected_maximum: float
    probabilistic: float
    deflated: float


def deflate_trials(
    returns: ArrayLike,
    *,
    selected: int | None = None,
    n_trials: float | None = None,
    method: MaximumMethod = "approximation",
) -> DeflatedSharpe:
    """Deflate the best column of a ``(periods, trials)`` matrix of trial returns.

    ``selected`` defaults to the column with the highest Sharpe ratio.
    ``n_trials`` defaults to the number of columns; pass an effective number
    (see :mod:`holdout.trials`) when the trials are correlated, since counting
    near-duplicates as independent over-deflates.

    The mean of the trial ratios is set to zero rather than estimated, as in
    the paper's null of no skill; the variance is the sample variance of the
    trial ratios.
    """
    matrix = as_matrix(returns, min_columns=2)
    ratios = np.array([estimate_sharpe(matrix[:, j]).value for j in range(matrix.shape[1])])
    column = int(np.argmax(ratios)) if selected is None else int(selected)
    if not 0 <= column < matrix.shape[1]:
        raise ValidationError(f"selected must index one of {matrix.shape[1]} columns, got {column}")
    trials = float(matrix.shape[1]) if n_trials is None else float(n_trials)
    if not trials >= 1.0:
        raise ValidationError(f"n_trials must be at least 1, got {trials}")
    estimate = estimate_sharpe(matrix[:, column])
    variance = float(np.var(ratios, ddof=1))
    # Non-integer effective counts are rounded for the order-statistic formula.
    expected = expected_maximum_sharpe(max(1, round(trials)), variance, method=method)
    psr = probabilistic_sharpe_ratio(
        estimate.value, estimate.n, skewness=estimate.skewness, kurtosis=estimate.kurtosis
    )
    dsr = probabilistic_sharpe_ratio(
        estimate.value,
        estimate.n,
        benchmark=expected,
        skewness=estimate.skewness,
        kurtosis=estimate.kurtosis,
    )
    return DeflatedSharpe(
        selected=column,
        estimate=estimate,
        n_trials=trials,
        trials_mean=0.0,
        trials_variance=variance,
        expected_maximum=expected,
        probabilistic=psr,
        deflated=dsr,
    )
