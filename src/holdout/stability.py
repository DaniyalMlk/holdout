"""Was the edge there throughout? The maximum problem, over dates.

Every other test here guards a maximum taken over *strategies*.
:func:`~holdout.spa.superior_predictive_ability` asks whether the best of K beats
a benchmark once K is accounted for, :func:`~holdout.spa.romano_wolf` asks which
ones do, :func:`~holdout.deflated.deflated_sharpe_ratio` raises the benchmark to
what the best of N unskilled trials would show. None of them asks whether the
strategy earned its Sharpe ratio evenly or earned all of it in one stretch.

The obvious check is to split the sample and compare the two Sharpe ratios. It has
the flaw this library exists to point out, twice.

**The split point is chosen after looking at the equity curve.** Nobody picks the
midpoint of a track record that has an obvious cliff in it two thirds of the way
along. The comparison gets made at the most damaging date available and read
against a critical value for one date fixed in advance.

**Searching honestly does not fix it either.** Take the largest statistic over
every candidate date and a normal critical value is the wrong distribution, for
exactly the reason White's Reality Check exists: the maximum of several hundred
correlated statistics is not distributed like one of them. Measured over 500
replications of 1,000 iid normal returns with no break at all, reading the
supremum against a two-sided normal 5% critical value rejects **41.4%** of the
time. The bootstrap here rejects 3.2%, against a nominal 5% and a Monte Carlo
standard error of about one point.

So the statistic is a supremum by construction -- there is no single-date version
of it to misuse -- and its distribution comes from the stationary bootstrap, which
handles the maximum and any serial correlation in the same pass. The naive p-value
is reported beside it, because the gap between the two is the entire point and a
reader who has been quoting the naive one should see what it was worth.

Three decisions worth stating.

**The statistic uses this library's own Sharpe standard error.** Two disjoint
subsamples, each with its Mertens (2002) standard error from its own skew and
kurtosis, and their difference standardised by the root of the sum of squares. The
normal-returns standard error would be simpler and wrong in the direction that
matters: a strategy with negative skew and fat tails has a noisier Sharpe ratio
than the normal formula says, and a break test that understates the noise finds
breaks that are not there.

**Trimming is an argument, and it changes the answer.** Near either end the
subsample is short, its skew and kurtosis are barely estimated, and the statistic
is dominated by that rather than by anything about the strategy. The default is to
ignore the first and last 15%, which is the conventional choice; at 2% the naive
test's size rises to 58.8% and the bootstrap's to 7.0%, so the bootstrap absorbs
most of the extra and not all of it. The bootstrap is conservative at the default
trim and liberal at an aggressive one, which is worth knowing before choosing one
to suit an answer.

**The test has very little power, and that is the most useful thing it says.**
Over 120 replications at 500 observations either side -- two four-year halves of
daily returns, a longer track record than most things get -- a strategy whose
per-period Sharpe ratio falls from 0.10 to 0.05 is detected 10% of the time, one
that falls from 0.10 to 0.00 is detected 25%, and one that goes from 0.12 to
-0.04, which is an annualised 1.90 turning into -0.63, is detected 52%. So a large
p-value here is close to no evidence at all, and reading one as confirmation that
a strategy is stable is the mistake this module makes easiest. It is worth having
anyway: when it *does* reject, it rejects against the right distribution.

**The subsample moments come from cumulative power sums, on data shifted by the
sample mean.** A thousand resamples over a thousand days is 1.4 million subsample
Sharpe ratios with their skew and kurtosis, which cannot be a Python loop. Power
sums make it four cumulative sums per resample. They also cancel catastrophically
on returns whose mean is a hundredth of their standard deviation, which is every
return series there is, so the sums are taken on ``x - mean(x)`` and the mean added
back; subtracting a constant leaves every subsample's variance, skew and kurtosis
unchanged and its mean shifted by exactly that constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .bootstrap import Seed, optimal_block_length, stationary_bootstrap_indices
from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray, as_returns, check_probability

__all__ = [
    "DEFAULT_TRIM",
    "BreakStatistics",
    "SharpeBreak",
    "break_statistics",
    "sharpe_break",
]

#: Fraction of the sample ignored at each end. Fifteen per cent is the
#: conventional choice and it is a choice: the statistic has no useful meaning
#: where the shorter side is too short to estimate a fourth moment on.
DEFAULT_TRIM = 0.15

#: Smallest subsample either side of a candidate break. A Sharpe ratio needs a
#: variance, its standard error needs a skew and a kurtosis, and a fourth moment
#: from fewer than thirty observations is not an estimate of anything.
MINIMUM_SIDE = 30


def _standard_normal_two_sided(statistic: float) -> float:
    """``2 (1 - Phi(|z|))``, by the error function rather than a table."""
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def _subsample_moments(
    shifted: FloatArray, offset: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Sharpe ratio, skewness, kurtosis and count, for every prefix.

    ``shifted`` is ``(..., n)`` with each row's own mean already subtracted and
    ``offset`` is that mean, broadcastable against the leading axes. Prefix ``k``
    then has mean ``S1/k + offset`` while its central moments are those of
    ``shifted`` unchanged, because subtracting a constant moves a mean and leaves
    every moment about that mean alone.

    Returns arrays over ``k = 1 .. n``. The first entries are meaningless -- a
    prefix of one observation has no variance -- and the caller trims them.
    """
    powers = [np.cumsum(shifted**power, axis=-1) for power in (1, 2, 3, 4)]
    counts = np.arange(1, shifted.shape[-1] + 1, dtype=np.float64)
    first, second, third, fourth = (one / counts for one in powers)
    mean = first
    # Rounding can push a central second moment a hair below zero on a nearly
    # constant stretch. Clipping at zero rather than at a small positive number
    # keeps the zero-variance case detectable downstream instead of hiding it.
    m2 = np.maximum(second - mean**2, 0.0)
    m3 = third - 3.0 * mean * second + 2.0 * mean**3
    m4 = fourth - 4.0 * mean * third + 6.0 * mean**2 * second - 3.0 * mean**4
    with np.errstate(divide="ignore", invalid="ignore"):
        unbiased = m2 * counts / np.maximum(counts - 1.0, 1.0)
        sharpe = (mean + offset) / np.sqrt(unbiased)
        skewness = m3 / m2**1.5
        kurtosis = m4 / m2**2
    return sharpe, skewness, kurtosis, counts


def _statistics(sample: FloatArray, low: int, high: int) -> FloatArray:
    """Break statistics for splits after row ``k``, ``k`` in ``[low, high)``.

    ``sample`` is ``(..., n)``. Vectorised over the leading axes so a whole
    bootstrap runs in one pass.
    """
    total = sample.shape[-1]
    offsets = np.mean(sample, axis=-1, keepdims=True)
    shifted = sample - offsets
    forward = _subsample_moments(shifted, offsets)
    backward = _subsample_moments(shifted[..., ::-1], offsets)
    left_sharpe, left_skew, left_kurt, counts = forward
    right_sharpe, right_skew, right_kurt, _ = backward

    # Split after row k: the left side is the prefix of length k, the right side
    # the suffix of length n - k, which is the reversed array's prefix of that
    # length. Index k - 1 and n - k - 1 respectively.
    left = slice(low - 1, high - 1)
    right_index = total - np.arange(low, high) - 1
    left_variance = (
        _variance_term(left_sharpe[..., left], left_skew[..., left], left_kurt[..., left])
        / counts[left]
    )
    right_variance = (
        _variance_term(
            right_sharpe[..., right_index],
            right_skew[..., right_index],
            right_kurt[..., right_index],
        )
        / counts[right_index]
    )
    difference = right_sharpe[..., right_index] - left_sharpe[..., left]
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = difference / np.sqrt(left_variance + right_variance)
    # A side with no variance has no Sharpe ratio, so the split is not a
    # comparison. Zero rather than a NaN, because a NaN would propagate through
    # the maximum and out into a payload.
    cleaned: FloatArray = np.nan_to_num(statistic, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned


def _variance_term(sharpe: FloatArray, skewness: FloatArray, kurtosis: FloatArray) -> FloatArray:
    """Mertens' ``n`` times the variance of a Sharpe estimator, vectorised.

    The scalar version lives in :func:`~holdout.sharpe.sharpe_variance_term` and
    validates its arguments one at a time, which a million of them cannot afford.
    The formula is the same and a test pins them together.
    """
    term = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe
    return np.maximum(term, 0.0)


def _bounds(total: int, trim: float) -> tuple[int, int]:
    check_probability(trim, "trim", open_interval=False)
    if trim >= 0.5:
        raise ValidationError(
            f"trim must be below 0.5, got {trim!r}; trimming half from each end "
            "leaves no candidate break"
        )
    edge = max(math.ceil(trim * total), MINIMUM_SIDE)
    low = edge
    high = min(total - edge, total) + 1
    if high <= low:
        raise InsufficientDataError(
            f"{total} observations with trim {trim} leaves no candidate break with at "
            f"least {MINIMUM_SIDE} on each side; a Sharpe ratio's standard error needs "
            "a skew and a kurtosis, and a fourth moment from fewer than that is not "
            "an estimate"
        )
    return low, high


@dataclass(frozen=True)
class BreakStatistics:
    """The statistic at every candidate break date.

    Attributes:
        statistics: Signed, one per candidate. Positive means the Sharpe ratio
            was *higher* after the break than before, which is the direction
            nobody worries about and is worth keeping the sign for.
        splits: Number of observations before each candidate break.
        trim: The fraction ignored at each end.
    """

    statistics: tuple[float, ...]
    splits: tuple[int, ...]
    trim: float

    @property
    def supremum(self) -> float:
        """The largest absolute statistic."""
        return max(abs(one) for one in self.statistics)

    @property
    def at(self) -> int:
        """Observations before the break where the supremum falls."""
        worst = max(range(len(self.statistics)), key=lambda i: abs(self.statistics[i]))
        return self.splits[worst]

    @property
    def signed_supremum(self) -> float:
        """The supremum with its sign, so the direction is not lost."""
        worst = max(range(len(self.statistics)), key=lambda i: abs(self.statistics[i]))
        return self.statistics[worst]


def break_statistics(returns: ArrayLike, *, trim: float = DEFAULT_TRIM) -> BreakStatistics:
    """Standardised Sharpe-ratio difference at every candidate break date.

    At each split, the Sharpe ratio after less the Sharpe ratio before, divided by
    the root of the sum of the two Mertens standard errors squared. The two sides
    are disjoint, so treating their estimates as independent is right up to the
    serial correlation across the boundary -- which the bootstrap in
    :func:`sharpe_break` accounts for and this function does not claim to.
    """
    sample = as_returns(returns, min_length=2 * MINIMUM_SIDE)
    low, high = _bounds(sample.size, trim)
    values = _statistics(sample, low, high)
    return BreakStatistics(
        statistics=tuple(float(one) for one in values),
        splits=tuple(range(low, high)),
        trim=float(trim),
    )


@dataclass(frozen=True)
class SharpeBreak:
    """A test for a break in the Sharpe ratio at an unknown date.

    Attributes:
        statistic: The supremum of the absolute standardised difference.
        at: Observations before the break where it falls.
        before: Sharpe ratio of the sample up to that point, per period.
        after: Sharpe ratio from that point on, per period.
        p_value: From the bootstrap. The one to read.
        naive_p_value: What a two-sided normal critical value would have said
            about the same statistic, as if the date had been fixed in advance.
            Reported so that the gap is visible, never as the answer.
        candidates: How many dates the supremum was taken over.
        resamples: Bootstrap resamples used.
        block_length: Mean block length of the stationary bootstrap.
        trim: The fraction ignored at each end.
    """

    statistic: float
    at: int
    before: float
    after: float
    p_value: float
    naive_p_value: float
    candidates: int
    resamples: int
    block_length: float
    trim: float

    @property
    def naive_understates_by(self) -> float:
        """Ratio of the two p-values. How much the naive one overstates the case."""
        if self.naive_p_value <= 0.0:
            return math.inf
        return self.p_value / self.naive_p_value


def sharpe_break(
    returns: ArrayLike,
    *,
    trim: float = DEFAULT_TRIM,
    resamples: int = 1000,
    block_length: float | None = None,
    seed: Seed = None,
) -> SharpeBreak:
    """Test for a break in the Sharpe ratio at a date chosen by the data.

    The statistic is the supremum over candidate dates, and its null distribution
    comes from the stationary bootstrap: each resample is a series with no break
    in it, so the supremum computed on each resample is a draw from the null. That
    handles the maximum and any serial correlation together, which is why the
    bootstrap is not optional here.

    ``block_length`` defaults to Politis and White's choice for this series. The
    bootstrap destroys a real break by construction -- blocks from before and
    after get mixed -- which is exactly what makes the resample a null draw rather
    than a problem.
    """
    sample = as_returns(returns, min_length=2 * MINIMUM_SIDE)
    low, high = _bounds(sample.size, trim)
    observed = _statistics(sample, low, high)
    worst = int(np.argmax(np.abs(observed)))
    statistic = float(abs(observed[worst]))
    split = low + worst

    if int(resamples) != resamples or resamples < 1:
        raise ValidationError(f"resamples must be a positive integer, got {resamples!r}")
    length = optimal_block_length(sample) if block_length is None else float(block_length)
    indices = stationary_bootstrap_indices(sample.size, length, int(resamples), seed=seed)
    draws = np.max(np.abs(_statistics(sample[indices], low, high)), axis=-1)
    # The observed statistic counts as one draw from the null, which keeps the
    # p-value away from exactly zero: a p-value of zero from a thousand resamples
    # is a statement the resamples cannot support.
    p_value = float((1.0 + np.count_nonzero(draws >= statistic)) / (1.0 + resamples))

    left, right = sample[:split], sample[split:]
    return SharpeBreak(
        statistic=statistic,
        at=split,
        before=_sharpe(left),
        after=_sharpe(right),
        p_value=p_value,
        naive_p_value=_standard_normal_two_sided(statistic),
        candidates=high - low,
        resamples=int(resamples),
        block_length=length,
        trim=float(trim),
    )


def _sharpe(sample: FloatArray) -> float:
    """Per-period Sharpe ratio, or zero for a stretch with no variance.

    Zero rather than a raise: this is reported beside a statistic that has
    already been computed, and a degenerate side has already been given a
    statistic of zero, so raising here would refuse to report a result the test
    has produced.
    """
    deviation = float(np.std(sample, ddof=1))
    scale = float(np.max(np.abs(sample))) if sample.size else 0.0
    if deviation <= np.finfo(np.float64).eps * max(scale, 1.0):
        return 0.0
    return float(np.mean(sample)) / deviation
