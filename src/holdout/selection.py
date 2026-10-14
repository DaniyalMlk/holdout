"""The strategy that was chosen, and what its record is worth once that is allowed for.

:mod:`holdout.deflated` asks whether the best of a set of records is
significant and returns a probability. :mod:`holdout.multiple` haircuts it.
Neither answers the question that comes next, which is the one a committee
actually asks: *what is the selected strategy's Sharpe ratio?* The number
carried forward is almost always the winning estimate itself, and that
estimate was chosen for being large.

The device here is conditioning. Write ``x_i`` for the estimated Sharpe of
each candidate and ``s_i`` for its standard error, and suppose the estimates
are independent and approximately normal about their true values. The winner
is whichever ``x_i`` is largest. Hold the losers' estimates fixed and ask what
the winner's estimate was drawn from: it was drawn from ``N(mu_k, s_k)``
*restricted to the event that it exceeded the best of the others*. So the
conditional law is a truncated normal with a known truncation point, and every
quantity below -- the interval, the point estimate -- is an exact statement
about that law rather than an approximation to it.

Nothing here needs the losers' standard errors, and nothing here is a
correction factor applied to the naive answer.

**The ordinary interval's coverage collapses in the number of candidates.**
Under a null where every candidate has the same true Sharpe and a unit
standard error, the naive 95% interval on the *selected* strategy's estimate
covers that strategy's truth 95.1% of the time out of two candidates, 88.3%
out of five, 77.8% out of ten and **30.5% out of fifty**. The conditional
interval holds 94.8% to 95.2% across all four. The naive interval is not
slightly optimistic; past a few dozen candidates it is wrong twice as often
as it is right.

**The bias it is missing is the expected maximum, and that is checkable
against something already here.** The mean gap between the winning estimate
and the winner's own truth comes out 0.557, 1.156, 1.532 and 2.240 at two,
five, ten and fifty candidates, against
:func:`holdout.deflated.expected_maximum_normal` at 0.564, 1.163, 1.539 and
2.249 -- agreement to within 1.4%, from a simulation that knows nothing about
that function.

**The price is the width, and the mean width is the wrong way to quote it.**
The conditional intervals average 25.4 to 67.2 wide against the naive 3.92,
which sounds unusable. Their *medians* are 6.0, 9.0, 10.6 and 14.7 -- between
1.5 and 3.7 times the naive width. The gap between the two summaries is a
heavy right tail made entirely of near-ties at the top of the table, and
quoting the mean would describe the typical case by the worst one.

**The interval is finite however close the race was, which is not the natural
guess.** A winner that barely cleared the runner-up feels as though it should
be consistent with any true value at all. It is not: the conditional tail runs
from zero to one as the mean runs over the line, so every level is attained
at some finite mean. What actually happens is that the bound runs a long way
out, at a rate that is a formula rather than a surprise --
``log(1 / alpha) * deviation / margin``. At a margin of three hundredths of a
standard error the lower bound is 123 standard errors below the estimate
(-35.891 against the small-margin prediction's -35.889, on an estimate of 1.0
with a standard error of 0.3). An *exact* tie is the one degenerate case, and
it raises rather than returning an infinity.

**The one assumption is independence, and where it fails the two intervals
change places.** The conditional law holds the losers' estimates fixed and
treats the winner's as independent of them, which a parameter sweep over
neighbouring settings badly violates. Imposing an equicorrelation on ten
otherwise identical candidates (``selection_coverage(..., correlation=rho)``):

====  ================  ======================  ============
rho   naive coverage    conditional coverage    median width
====  ================  ======================  ============
0.00  78.5%             94.9%                   10.8
0.50  86.1%             94.6%                   14.1
0.75  90.5%             91.4%                   19.0
0.80  91.4%             90.0%                   21.0
0.90  93.2%             84.1%                   28.9
0.99  94.3%             55.3%                   90.0
====  ================  ======================  ============

They cross at about ``rho = 0.78``. Correlation shrinks the winner's curse --
the mean selection bias falls from 1.53 to 0.15 across that range -- so the
naive interval recovers, while the conditional one becomes *both wider and
less valid*, which is not a trade a reader would guess. Above a correlation
of roughly three quarters this module is the wrong tool and the ordinary
interval is the better one. With fifty candidates the crossing is much
further out, because the naive interval has so much further to fall: at
``rho = 0.5`` it covers 68.2% against the conditional interval's 94.6%.

**And the correction costs nothing when the winner is clear, so it is not a
tax on every selection.** With one candidate four standard errors ahead of
nine others, the median conditional width is 4.10 against the naive 3.92 -- a
5% premium -- and the naive coverage has itself recovered to 94.3%. The
median-unbiased estimate behaves the same way: at a margin of four standard
errors over the runner-up it is the raw estimate to four decimal places, at
two standard errors it is 0.031 below, at one 0.314 below, and at a tenth of
a standard error it is 6.84 below. Everything in this module is a statement
about how close the race was, and nothing in it is a statement about how many
candidates there were.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .exceptions import InsufficientDataError, ValidationError

FloatArray = NDArray[np.float64]

__all__ = [
    "LOG_TAIL_ASYMPTOTIC_FROM",
    "ConditionalInterval",
    "Selected",
    "SelectionInference",
    "conditional_interval",
    "log_upper_tail",
    "median_unbiased_mean",
    "naive_interval",
    "select_best",
    "selection_coverage",
    "selection_inference",
    "truncated_tail",
]

# Above this many standard deviations the upper tail is taken from its
# asymptotic series rather than from ``erfc``. Chosen by a sweep over both the
# crossover and the number of terms: see :func:`log_upper_tail`.
LOG_TAIL_ASYMPTOTIC_FROM = 20.0

# ``(-1)**n (2n-1)!!``: the Mills-ratio asymptotic series for the upper tail.
_MILLS = (1.0, -1.0, 3.0, -15.0, 105.0, -945.0, 10395.0)

_SQRT_TWO = math.sqrt(2.0)
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return value


def _positive(name: str, value: float) -> float:
    _finite(name, value)
    if value <= 0.0:
        raise ValidationError(f"{name} must be positive, got {value!r}")
    return value


def log_upper_tail(z: float) -> float:
    """``log P(Z >= z)`` for a standard normal, accurate at every ``z``.

    Every quantity in this module is a ratio of two upper tails, and both can
    be far below the smallest positive double. ``math.erfc`` underflows to
    exactly zero around ``z = 38``, so the direct ratio becomes ``0.0 / 0.0``
    and returns a nan -- not a small number, a missing one. Below
    :data:`LOG_TAIL_ASYMPTOTIC_FROM` this is the log of ``erfc``, which is
    accurate there; above it, the Mills-ratio asymptotic series

        ``log P(Z >= z) = -z**2/2 - log(z sqrt(2 pi))
                          + log(1 - 1/z**2 + 3/z**4 - 15/z**6 + ...)``

    with coefficients ``(-1)**n (2n-1)!!``.

    **The crossover and the term count were swept rather than chosen, and the
    answer sits in a flat region rather than on a cliff.** Worst relative
    error against an independent implementation over ``z`` from 0.1 to 400:

    ========  ========  ========  ========
    crossover  5 terms   6 terms   7 terms
    ========  ========  ========  ========
    8          2.2e-08   3.7e-09   7.3e-10
    12         1.9e-10   1.4e-11   1.3e-12
    15         1.4e-11   6.6e-13   3.8e-14
    20         4.4e-13   1.2e-14   6.5e-16
    25         3.1e-14   7.1e-16   6.5e-16
    ========  ========  ========  ========

    Twenty and seven is the first cell at machine precision, and its
    neighbours in both directions are there too, so the choice is not
    balanced on a boundary. The worst error in that cell is attained at
    ``z = 368``, not at the crossover, which is the sign that the piecewise
    definition has stopped being the limiting thing.
    """
    _finite("z", z)
    if z < LOG_TAIL_ASYMPTOTIC_FROM:
        tail = 0.5 * math.erfc(z / _SQRT_TWO)
        if tail <= 0.0:  # pragma: no cover - unreachable below the crossover
            raise ValidationError(f"the tail underflowed below the crossover at z={z!r}")
        return math.log(tail)
    inverse = 1.0 / (z * z)
    series = 0.0
    for coefficient in reversed(_MILLS):
        series = coefficient + inverse * series
    return -0.5 * z * z - math.log(z) - _LOG_SQRT_TWO_PI + math.log(series)


def truncated_tail(x: float, mean: float, deviation: float, lower: float) -> float:
    """``P(X >= x | X >= lower)`` for ``X ~ N(mean, deviation**2)``.

    The conditional survival function of the truncated normal, computed as the
    exponential of a difference of two log tails so that it survives where
    both tails have underflowed. Asking about a point below the truncation is
    certain by definition rather than an error: the conditioning event already
    guarantees it.
    """
    _finite("x", x)
    _finite("mean", mean)
    _positive("deviation", deviation)
    _finite("lower", lower)
    if x <= lower:
        return 1.0
    numerator = log_upper_tail((x - mean) / deviation)
    denominator = log_upper_tail((lower - mean) / deviation)
    return math.exp(numerator - denominator)


@dataclass(frozen=True)
class Selected:
    """Which candidate won, and by how much.

    ``threshold`` is the best of the *other* estimates, which is the
    truncation point of the winner's conditional law. ``margin`` is how far
    the winner cleared it, in its own standard errors: the single number that
    decides whether conditioning changes the answer at all.
    """

    index: int
    estimate: float
    deviation: float
    threshold: float
    candidates: int

    @property
    def margin(self) -> float:
        """How far the winner cleared the runner-up, in its standard errors."""
        return (self.estimate - self.threshold) / self.deviation


def select_best(estimates: Sequence[float], deviations: Sequence[float]) -> Selected:
    """The largest estimate, and the runner-up that it had to beat.

    Ties are broken towards the lower index, which makes the selection a
    function of the data rather than of the iteration order. A tie leaves a
    margin of exactly zero, and every conditional quantity below is then
    degenerate in the way it should be: the data say nothing about which of
    two indistinguishable candidates is better.
    """
    if len(estimates) != len(deviations):
        raise ValidationError(
            f"{len(estimates)} estimates against {len(deviations)} standard errors"
        )
    if len(estimates) < 2:
        raise InsufficientDataError(
            "selection inference needs at least two candidates; with one there was no selection"
        )
    for index, (estimate, deviation) in enumerate(zip(estimates, deviations, strict=True)):
        _finite(f"estimates[{index}]", estimate)
        _positive(f"deviations[{index}]", deviation)
    best = 0
    for index in range(1, len(estimates)):
        if estimates[index] > estimates[best]:
            best = index
    threshold = max(estimate for index, estimate in enumerate(estimates) if index != best)
    return Selected(
        index=best,
        estimate=estimates[best],
        deviation=deviations[best],
        threshold=threshold,
        candidates=len(estimates),
    )


@dataclass(frozen=True)
class ConditionalInterval:
    """An interval for the selected strategy's true Sharpe, and its price.

    ``lower`` is ``-inf`` when the data place no lower bound at all, which is
    not a failure: a winner that barely cleared the runner-up is consistent
    with any true value, because a low enough one is still occasionally the
    largest draw. The naive interval never admits that and is wrong to.
    """

    lower: float
    upper: float
    median: float
    level: float
    naive_lower: float
    naive_upper: float

    @property
    def width(self) -> float:
        """``upper - lower``, infinite when the lower bound is."""
        return self.upper - self.lower

    @property
    def naive_width(self) -> float:
        return self.naive_upper - self.naive_lower

    @property
    def inflation(self) -> float:
        """How many times wider than the naive interval. Infinite when unbounded."""
        return self.width / self.naive_width

    @property
    def bounded(self) -> bool:
        return math.isfinite(self.lower)


def _inverse_normal(probability: float) -> float:
    """The standard normal quantile, by bisection on ``erfc``.

    A bisection rather than a rational approximation because this is called
    once per interval and the accuracy is then the machine's rather than the
    approximation's.
    """
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if 0.5 * math.erfc(-middle / _SQRT_TWO) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def naive_interval(estimate: float, deviation: float, level: float = 0.95) -> tuple[float, float]:
    """The interval that ignores the selection: ``estimate +- z * deviation``.

    Here to be reported beside the conditional one rather than to be used.
    """
    _finite("estimate", estimate)
    _positive("deviation", deviation)
    if not 0.0 < level < 1.0:
        raise ValidationError(f"level must lie in (0, 1), got {level!r}")
    half = _inverse_normal(0.5 * (1.0 + level)) * deviation
    return estimate - half, estimate + half


def _require_strict_winner(estimate: float, threshold: float) -> None:
    """An exact tie carries no information, and is an error rather than an infinity.

    If the winning estimate did not strictly exceed the runner-up's, the
    conditional law is degenerate: the conditioning event is "the estimate was
    at least the threshold", which the observation satisfies with certainty
    under every true value, so the confidence set is the whole line. Returning
    an interval of infinite width would be arithmetically right and useless;
    saying so is more use to a caller who can go and look at why two records
    tied.
    """
    if estimate <= threshold:
        raise ValidationError(
            f"the selected estimate {estimate!r} does not exceed the runner-up "
            f"{threshold!r}, so conditioning on the selection says nothing: every "
            "true value is consistent with it"
        )


def _invert(estimate: float, deviation: float, threshold: float, target: float) -> float:
    """The mean under which the observed estimate sits at conditional tail ``target``.

    ``truncated_tail(estimate, mean, deviation, threshold)`` is strictly
    *increasing* in ``mean`` -- raise the mean and the observed value stops
    looking like a large draw -- running from zero as the mean falls without
    bound to one as it rises. Every target in ``(0, 1)`` therefore has exactly
    one solution, and the interval is finite however close the race was. That
    is worth stating because the opposite is the natural guess: a winner that
    barely cleared the runner-up feels as though it should admit any true
    value at all, and what actually happens is that the bound runs to a
    hundred standard errors rather than to minus infinity.

    The bracket is grown outwards rather than guessed, because the distance to
    the solution is not a few standard errors: it goes like
    ``log(1 / target) * deviation / margin``, so a margin of a thirtieth of a
    standard error puts the lower bound a hundred and ten standard errors
    below the estimate.
    """
    low = min(threshold, estimate) - deviation
    scale = deviation
    for _ in range(400):
        if truncated_tail(estimate, low, deviation, threshold) < target:
            break
        low -= scale
        scale *= 1.5
    else:  # pragma: no cover - 400 growing steps overshoot any float
        return -math.inf
    high = estimate + deviation
    scale = deviation
    for _ in range(400):
        if truncated_tail(estimate, high, deviation, threshold) > target:
            break
        high += scale
        scale *= 1.5
    else:  # pragma: no cover - as above
        return math.inf
    for _ in range(200):
        middle = 0.5 * (low + high)
        if truncated_tail(estimate, middle, deviation, threshold) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def median_unbiased_mean(estimate: float, deviation: float, threshold: float) -> float:
    """The true Sharpe under which the winning estimate was the conditional median.

    The point estimate with no selection bias in it, in the sense that it is
    as likely to sit above the truth as below once the selection is
    conditioned on. It is always below the raw estimate, and :func:`Selected`
    says by how much it should be: the shortfall is a function of the margin
    over the runner-up.
    """
    _finite("estimate", estimate)
    _positive("deviation", deviation)
    _finite("threshold", threshold)
    _require_strict_winner(estimate, threshold)
    return _invert(estimate, deviation, threshold, 0.5)


def conditional_interval(
    estimate: float, deviation: float, threshold: float, level: float = 0.95
) -> ConditionalInterval:
    """A confidence interval for the selected strategy's true Sharpe ratio.

    Exact under the model, not asymptotic: the winning estimate's conditional
    law is a normal truncated below at ``threshold``, and the bounds are the
    means at which the observed estimate sits at the conditional quantiles
    ``alpha / 2`` and ``1 - alpha / 2``. Inverting a tail probability in the
    parameter is the standard construction and it inherits the conditional
    law's exactness.
    """
    _finite("estimate", estimate)
    _positive("deviation", deviation)
    _finite("threshold", threshold)
    if not 0.0 < level < 1.0:
        raise ValidationError(f"level must lie in (0, 1), got {level!r}")
    _require_strict_winner(estimate, threshold)
    alpha = 1.0 - level
    naive_low, naive_high = naive_interval(estimate, deviation, level)
    return ConditionalInterval(
        lower=_invert(estimate, deviation, threshold, alpha / 2.0),
        upper=_invert(estimate, deviation, threshold, 1.0 - alpha / 2.0),
        median=_invert(estimate, deviation, threshold, 0.5),
        level=level,
        naive_lower=naive_low,
        naive_upper=naive_high,
    )


@dataclass(frozen=True)
class SelectionInference:
    """Everything the selection says about the winner, in one object."""

    selected: Selected
    interval: ConditionalInterval

    @property
    def shrinkage(self) -> float:
        """How much the median-unbiased estimate falls short of the raw one."""
        return self.selected.estimate - self.interval.median


def selection_inference(
    estimates: Sequence[float], deviations: Sequence[float], level: float = 0.95
) -> SelectionInference:
    """Select the best record and say what its Sharpe ratio is worth.

    The whole of the module's answer: which candidate won, how far it cleared
    the runner-up in its own standard errors, the median-unbiased estimate of
    its true Sharpe, the conditional interval, and the naive interval beside
    it for comparison.
    """
    selected = select_best(estimates, deviations)
    return SelectionInference(
        selected=selected,
        interval=conditional_interval(
            selected.estimate, selected.deviation, selected.threshold, level
        ),
    )


def selection_coverage(
    truths: Sequence[float],
    deviations: Sequence[float],
    *,
    level: float = 0.95,
    paths: int = 4_000,
    seed: int = 0,
    correlation: float = 0.0,
) -> dict[str, float]:
    """Simulate how often each interval covers the *selected* strategy's truth.

    Draws each candidate's estimate as ``N(truth, deviation**2)``, selects the
    largest, and asks whether each interval contains the true Sharpe *of the
    one that was selected* -- which changes from path to path, and is the
    whole difficulty. Returns the two coverages, the mean selection bias, the
    widths and the fraction of paths on which the conditional bound ran off.

    ``correlation`` is an equicorrelation between the candidates' estimates,
    imposed through a single common factor. It exists to test this module's
    one real assumption rather than to restate it: the conditional law holds
    the losers' estimates fixed and treats the winner's as independent of
    them, which a parameter sweep over neighbouring settings badly violates.
    Whether that makes the interval conservative or anti-conservative is a
    question with an answer, and it is not the same answer at both ends.

    Here so the numbers in this module's docstring can be reproduced from the
    installed package rather than taken on trust.
    """
    if len(truths) != len(deviations):
        raise ValidationError(f"{len(truths)} truths against {len(deviations)} standard errors")
    if len(truths) < 2:
        raise InsufficientDataError("coverage under selection needs at least two candidates")
    if paths < 1:
        raise ValidationError(f"the path count must be positive, got {paths}")
    if not 0.0 < level < 1.0:
        raise ValidationError(f"level must lie in (0, 1), got {level!r}")
    for index, (truth, deviation) in enumerate(zip(truths, deviations, strict=True)):
        _finite(f"truths[{index}]", truth)
        _positive(f"deviations[{index}]", deviation)

    if not 0.0 <= correlation < 1.0:
        raise ValidationError(f"correlation must lie in [0, 1), got {correlation!r}")

    generator = np.random.default_rng(seed)
    centres: FloatArray = np.asarray(truths, dtype=np.float64)
    scales: FloatArray = np.asarray(deviations, dtype=np.float64)
    shocks: FloatArray = generator.standard_normal(size=(paths, len(truths)))
    if correlation > 0.0:
        common: FloatArray = generator.standard_normal(size=(paths, 1))
        shocks = math.sqrt(correlation) * common + math.sqrt(1.0 - correlation) * shocks
    draws: FloatArray = centres + scales * shocks

    naive_hits = 0
    conditional_hits = 0
    bias = 0.0
    unbounded = 0
    conditional_width = 0.0
    naive_width = 0.0
    widths: list[float] = []
    for path in range(paths):
        row = [float(value) for value in draws[path]]
        inference = selection_inference(row, deviations, level)
        truth = float(centres[inference.selected.index])
        bias += inference.selected.estimate - truth
        if inference.interval.naive_lower <= truth <= inference.interval.naive_upper:
            naive_hits += 1
        if inference.interval.lower <= truth <= inference.interval.upper:
            conditional_hits += 1
        naive_width += inference.interval.naive_width
        if inference.interval.bounded:
            conditional_width += inference.interval.width
            widths.append(inference.interval.width)
        else:
            unbounded += 1
    bounded = paths - unbounded
    return {
        "naive_coverage": naive_hits / paths,
        "conditional_coverage": conditional_hits / paths,
        "mean_selection_bias": bias / paths,
        "unbounded_fraction": unbounded / paths,
        "mean_naive_width": naive_width / paths,
        "mean_bounded_conditional_width": (conditional_width / bounded if bounded else math.inf),
        # The widths have a heavy right tail -- a near-tie at the top puts a
        # bound a hundred standard errors out -- so the mean overstates what a
        # typical conditional interval looks like by a factor of several. Both
        # are reported because neither alone is the honest summary.
        "median_bounded_conditional_width": (
            sorted(widths)[len(widths) // 2] if widths else math.inf
        ),
    }
