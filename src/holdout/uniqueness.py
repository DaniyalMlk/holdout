"""How much information each observation actually carries, when labels overlap.

:mod:`holdout.splits` handles overlapping labels in the *splitter*. Purging
drops every training observation whose label window touches a test window, and
:func:`~holdout.splits.leakage_audit` proves it worked. That is the right fix
for cross-validation and it leaves the other half of the problem untouched.

Overlapping labels are not only a leakage hazard, they are a counting error.
Twenty observations whose five-day windows all cover the same week carry about
one week of information between them, and every estimator that averages over
observations — a Sharpe ratio, a bootstrap, a fitted model — treats them as
twenty. The symptom is a t statistic too large for the data behind it, and
purging does nothing about it because nothing has leaked. The observations are
simply redundant.

**Concurrency** is the count of label windows spanning each bar.
**Uniqueness** is the reciprocal: an observation's average uniqueness is the
mean of ``1 / concurrency`` over its own window, so an observation alone on
every bar it covers scores one and twenty observations sharing every bar score
a twentieth each. Summing uniqueness over observations gives an **effective
sample size**, which is the number this module exists to produce: the count to
divide by, instead of ``n``.

One exact identity holds whatever the overlap structure, and it is the check
that the arithmetic is right::

    sum over i of (uniqueness_i * length_i) == number of covered bars

because each side counts ``sum over covered bars of concurrency / concurrency``.
It holds with gaps, with nesting, with duplicated windows, and it is asserted
on randomised structures rather than on an example.

**The sequential bootstrap is the part worth being sceptical about.** It draws
with a probability proportional to each candidate's uniqueness given what is
already drawn, which sounds like it should fix the redundancy at its source.
Measured, it fixes almost none of it in the regime it is sold for: at full size
over heavily overlapping labels the uniform bootstrap already achieves 99.5% of
a ceiling that no sampling scheme can exceed, and the sequential one gets
99.8%, a gain of ``+0.44%`` against a standard error of ``0.09%``. The ceiling
is the point — total concurrency over the draw is ``size * length`` however it
is drawn, so uniqueness is capped at ``span / (size * length)``. Where that cap
is not binding, which means a draw small against the span, the sequential
scheme is genuinely better, by up to about ten per cent.
:func:`sequential_bootstrap` carries the table.

**This module needs integer bar indices where** :mod:`holdout.splits` **accepts
any comparable numbers**, and the difference is not an oversight. Purging only
asks whether two closed intervals intersect, which is an ordering question that
timestamps-as-floats answer perfectly well. Concurrency asks *how much* of a
window is shared, which is a measure question, and on a continuum the natural
measure makes a point label — the common case where an observation is one bar,
so that start equals end — a set of measure zero with no uniqueness at all. The
counting measure on bars is the one that gives the intended answer, so the
labels have to be bars.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bootstrap import Seed, as_generator
from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray

__all__ = [
    "Concurrency",
    "IndexArray",
    "SequentialDraw",
    "average_uniqueness",
    "concurrency",
    "drawn_uniqueness",
    "effective_sample_size",
    "sequential_bootstrap",
    "time_decay_weights",
    "uniqueness_weights",
]

IndexArray = NDArray[np.int64]


def _bars(start: ArrayLike, end: ArrayLike) -> tuple[IndexArray, IndexArray]:
    """Validate label windows as inclusive integer bar ranges."""
    first = np.asarray(start).reshape(-1)
    last = np.asarray(end).reshape(-1)
    if first.size != last.size:
        raise ValidationError(f"start has {first.size} entries but end has {last.size}")
    if first.size == 0:
        raise InsufficientDataError("at least one label is needed")
    for name, values in (("start", first), ("end", last)):
        if not np.issubdtype(values.dtype, np.integer):
            as_float = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(as_float)):
                raise ValidationError(f"label {name} values must be finite")
            if not np.all(as_float == np.floor(as_float)):
                raise ValidationError(
                    f"label {name} values must be whole bar indices; this module "
                    "measures how much of a window is shared, which is a count of "
                    "bars rather than an ordering. holdout.splits takes any "
                    "comparable numbers because purging only needs the ordering."
                )
    low = np.asarray(first, dtype=np.int64)
    high = np.asarray(last, dtype=np.int64)
    bad = np.flatnonzero(high < low)
    if bad.size:
        index = int(bad[0])
        raise ValidationError(f"label {index} ends ({high[index]}) before it starts ({low[index]})")
    # No sortedness requirement, deliberately. `holdout.splits` insists the
    # observations arrive sorted by label start, because its splits are
    # contiguous ranges of positions and that only means anything in time
    # order. Nothing here is positional: the difference array does not care
    # what order the windows arrive in, and the uniquenesses come back in input
    # order either way. Insisting anyway would make a bootstrap draw -- which
    # is a multiset in draw order -- have to be sorted before it could be
    # scored, for no gain.
    return low, high


class Concurrency:
    """How many label windows span each bar, over the span the labels cover.

    Attributes:
        counts: One count per bar from :attr:`first` to :attr:`last` inclusive.
            Zero on a bar no window covers, which is a gap in the labels rather
            than an error.
        first: The bar ``counts[0]`` describes.
        last: The bar ``counts[-1]`` describes.
    """

    __slots__ = ("_counts", "_first", "_reciprocal_prefix")

    def __init__(self, counts: IndexArray, first: int) -> None:
        self._counts = counts
        self._first = int(first)
        reciprocal = np.zeros(counts.size, dtype=np.float64)
        covered = counts > 0
        reciprocal[covered] = 1.0 / counts[covered]
        # Prefix sums of 1/c, so an average over any window is two lookups.
        self._reciprocal_prefix = np.concatenate(([0.0], np.cumsum(reciprocal, dtype=np.float64)))

    @property
    def counts(self) -> IndexArray:
        return self._counts

    @property
    def first(self) -> int:
        return self._first

    @property
    def last(self) -> int:
        return self._first + self._counts.size - 1

    @property
    def covered_bars(self) -> int:
        """Bars at least one window spans. The right-hand side of the identity."""
        return int(np.count_nonzero(self._counts))

    @property
    def peak(self) -> int:
        """The largest number of windows over any one bar."""
        return int(self._counts.max())

    def window_uniqueness(self, start: IndexArray, end: IndexArray) -> FloatArray:
        """Mean of ``1 / concurrency`` over each inclusive window, by prefix sum."""
        low = start - self._first
        high = end - self._first + 1
        total = self._reciprocal_prefix[high] - self._reciprocal_prefix[low]
        lengths = (end - start + 1).astype(np.float64)
        result: FloatArray = total / lengths
        return result

    def __repr__(self) -> str:
        return (
            f"Concurrency(bars {self._first}..{self.last}, "
            f"covered {self.covered_bars}, peak {self.peak})"
        )


def concurrency(start: ArrayLike, end: ArrayLike) -> Concurrency:
    """Count the label windows spanning each bar.

    Built from a difference array rather than by marking each window's bars, so
    the cost is one pass over the observations plus one over the bars instead of
    the product of the two. That matters here: the whole point of this module is
    the case where windows are long and numerous.

    Args:
        start: First bar of each label window. Integers, sorted ascending.
        end: Last bar, inclusive. Integers, each at least its own start.

    Returns:
        A :class:`Concurrency`.

    Raises:
        ValidationError: If the arrays disagree in length, hold non-integer or
            non-finite values, are not sorted by start, or any window ends
            before it begins.
        InsufficientDataError: If there are no labels.
    """
    low, high = _bars(start, end)
    first = int(low.min())
    size = int(high.max()) - first + 1
    deltas = np.zeros(size + 1, dtype=np.int64)
    np.add.at(deltas, low - first, 1)
    np.add.at(deltas, high - first + 1, -1)
    counts: IndexArray = np.cumsum(deltas[:-1], dtype=np.int64)
    return Concurrency(counts, first)


def average_uniqueness(start: ArrayLike, end: ArrayLike) -> FloatArray:
    """Each observation's mean ``1 / concurrency`` over its own label window.

    One for an observation that shares no bar with any other; ``1 / n`` for each
    of ``n`` observations sharing every bar.

    Args:
        start: First bar of each window.
        end: Last bar, inclusive.

    Returns:
        One uniqueness per observation, in input order, each in ``(0, 1]``.

    Raises:
        ValidationError: As :func:`concurrency`.
    """
    low, high = _bars(start, end)
    return concurrency(low, high).window_uniqueness(low, high)


def effective_sample_size(start: ArrayLike, end: ArrayLike) -> float:
    """Total uniqueness: the number of observations the sample is worth.

    ``n`` when nothing overlaps and one when everything does, and the number to
    divide by in place of ``n`` when it is in between. It is not a bound on
    anything — it is an average of local redundancies, and a sample can be
    worth fewer independent observations than this if the dependence runs
    through something other than the label windows.

    Args:
        start: First bar of each window.
        end: Last bar, inclusive.

    Returns:
        The sum of the average uniquenesses.
    """
    return float(average_uniqueness(start, end).sum())


def uniqueness_weights(start: ArrayLike, end: ArrayLike) -> FloatArray:
    """Sample weights proportional to average uniqueness, summing to one.

    The simplest correct thing: an observation that shares its window with
    nineteen others gets a twentieth of the weight of one that shares with
    nobody. Uniform, exactly, when nothing overlaps.

    Args:
        start: First bar of each window.
        end: Last bar, inclusive.

    Returns:
        Weights summing to one.
    """
    unique = average_uniqueness(start, end)
    weights: FloatArray = unique / unique.sum()
    return weights


def time_decay_weights(uniqueness: ArrayLike, *, decay: float = 1.0) -> FloatArray:
    """Discount older observations on an information scale, not a calendar one.

    The schedule is linear in *cumulative uniqueness* rather than in time, and
    that is the whole idea. A stretch in which fifty overlapping observations
    say the same thing should age like the one week of information it is, not
    like fifty observations; measuring the decay against accumulated uniqueness
    does that automatically, and measuring it against the index does not.

    ``decay`` is the weight the oldest observation would receive, read on the
    same scale:

    * ``1.0`` — no decay at all, every weight one.
    * ``0.5`` — the oldest observation is worth half the newest.
    * ``0.0`` — linear from zero, so the oldest observation is worth nothing.
    * negative — the oldest observations are worth nothing *and* a share of the
      sample is discarded outright: at ``-1.0`` the older half of the
      cumulative uniqueness is zeroed, and the schedule rises from zero over
      the newer half.

    Weights are returned on the scale the schedule defines, with the newest
    observation at one, rather than normalised to sum to one. Normalising here
    would make the decay invisible, because the only thing a caller can read
    off these numbers is their ratios.

    Args:
        uniqueness: Average uniqueness per observation, in time order — the
            output of :func:`average_uniqueness`.
        decay: Between -1 and 1 inclusive.

    Returns:
        One weight per observation, non-negative, the last equal to one.

    Raises:
        ValidationError: If ``decay`` is outside ``[-1, 1]``, or the uniqueness
            array is empty, non-finite or non-positive.
    """
    values = np.asarray(uniqueness, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise InsufficientDataError("at least one observation is needed")
    if not np.all(np.isfinite(values)):
        raise ValidationError("uniqueness values must be finite")
    if np.any(values <= 0.0):
        raise ValidationError("uniqueness values must be positive")
    rate = float(decay)
    if not -1.0 <= rate <= 1.0:
        raise ValidationError(f"decay must lie in [-1, 1], got {decay!r}")

    cumulative = np.cumsum(values, dtype=np.float64)
    total = float(cumulative[-1])
    if rate >= 0.0:
        slope = (1.0 - rate) / total
    else:
        # The share `1 + rate` of the cumulative uniqueness that survives. At
        # rate = -1 that share is zero, the slope is infinite in the limit, and
        # only the final observation keeps any weight -- which is the documented
        # end of the range rather than a division to guard.
        surviving = 1.0 + rate
        if surviving == 0.0:
            weights: FloatArray = np.where(cumulative >= total, 1.0, 0.0).astype(np.float64)
            return weights
        slope = 1.0 / (surviving * total)
    intercept = 1.0 - slope * total
    schedule: FloatArray = np.clip(intercept + slope * cumulative, 0.0, None)
    return schedule


class SequentialDraw:
    """A sequential bootstrap draw, with the uniqueness it achieved.

    Attributes:
        indices: Positions drawn, in the order drawn, with repeats.
        achieved: Average uniqueness of the drawn multiset, scored under its own
            concurrency. One would mean a sample with no redundancy at all.
        uniform: Average uniqueness of a uniform draw of the same size from the
            same labels, for comparison. The pair is the point: the advantage
            of drawing sequentially is a measurement, not a property.
    """

    __slots__ = ("achieved", "indices", "uniform")

    def __init__(self, indices: IndexArray, achieved: float, uniform: float) -> None:
        self.indices = indices
        self.achieved = achieved
        self.uniform = uniform

    @property
    def advantage(self) -> float:
        """``achieved / uniform - 1``, the fractional gain over drawing uniformly."""
        return self.achieved / self.uniform - 1.0

    def __repr__(self) -> str:
        return (
            f"SequentialDraw({self.indices.size} draws, achieved "
            f"{self.achieved:.4f} against uniform {self.uniform:.4f})"
        )


def drawn_uniqueness(start: ArrayLike, end: ArrayLike, indices: ArrayLike) -> float:
    """Average uniqueness of a drawn multiset, under the concurrency it creates.

    This is how any bootstrap draw gets scored, sequential or not, which is
    what makes the comparison between them a measurement rather than a claim.
    Repeats count: drawing the same observation twice puts its window over
    every one of its bars twice, and both copies are scored against that.

    Args:
        start: First bar of each label window, over the whole sample.
        end: Last bar, inclusive.
        indices: Positions drawn, with repeats.

    Returns:
        Mean average uniqueness over the drawn multiset.

    Raises:
        ValidationError: If any index is out of range.
        InsufficientDataError: If nothing was drawn.
    """
    low, high = _bars(start, end)
    drawn = np.asarray(indices, dtype=np.int64).reshape(-1)
    if drawn.size == 0:
        raise InsufficientDataError("nothing was drawn")
    if drawn.min() < 0 or drawn.max() >= low.size:
        raise ValidationError(
            f"indices must lie in [0, {low.size - 1}], got [{int(drawn.min())}, {int(drawn.max())}]"
        )
    return float(average_uniqueness(low[drawn], high[drawn]).mean())


def sequential_bootstrap(
    start: ArrayLike,
    end: ArrayLike,
    *,
    size: int | None = None,
    seed: Seed = None,
) -> SequentialDraw:
    """Draw with a probability proportional to uniqueness given what is drawn.

    A uniform bootstrap over overlapping labels reproduces the overlap: if half
    the sample describes one week, half of every resample describes that week
    too. Drawing sequentially fixes the sampling rather than reweighting it
    afterwards. Before each draw, every candidate is scored by what its average
    uniqueness *would* be if it were added to the draws already made, and the
    scores are normalised into probabilities.

    **How much this is worth is a ratio, and most of the time it is worth very
    little.** Total concurrency over the drawn multiset is fixed at
    ``size * length`` whatever is drawn, so the achievable average uniqueness
    is capped at ``span / (size * length)``, or one if that exceeds one — and
    the cap is attained by flattening the concurrency, which is all any
    sampling scheme can try to do. Over sixty seeds, two hundred observations,
    twenty-bar rolling windows, as a fraction of that cap:

    ======  ======  ==========  =======  ================
    draws   cap     sequential  uniform  gain
    ======  ======  ==========  =======  ================
    200     0.0548  0.998       0.995    +0.44% +- 0.09%
    100     0.1095  0.995       0.984    +1.03% +- 0.19%
    40      0.2737  0.978       0.946    +3.47% +- 0.55%
    20      0.5475  0.897       0.830    +8.78% +- 1.37%
    10      1.0     0.729       0.673    +10.43% +- 2.31%
    5       1.0     0.865       0.809    +9.37% +- 2.80%
    ======  ======  ==========  =======  ================

    The gain rises as the draw shrinks and then plateaus around ten per cent;
    at full size it is under half a per cent, because the uniform bootstrap
    already reaches 99.5% of a cap no scheme can beat. That is the opposite of
    how this is usually described. The heavy-overlap, full-size resample is the
    case the sequential bootstrap is motivated by and the case in which it
    cannot help, because there the binding constraint is arithmetic rather than
    algorithmic: two hundred twenty-bar windows laid over a two-hundred-and-
    nineteen-bar span force a mean concurrency near eighteen whichever windows
    are chosen.

    The remedy for heavy overlap is therefore to draw *fewer* observations, and
    :func:`effective_sample_size` is the number saying how many. Drawing all
    ``n`` of them cleverly does not recover information that is not there.

    The cost is one pass over the bars and one over the observations per draw,
    not the product: the running concurrency is kept as a count per bar and the
    reciprocals are prefix-summed once a step, after which each candidate's
    score is two lookups.

    Args:
        start: First bar of each label window.
        end: Last bar, inclusive.
        size: Number of draws. Defaults to the number of observations.
        seed: An integer, a NumPy generator, or ``None``.

    Returns:
        A :class:`SequentialDraw`, carrying the uniform draw's score beside its
        own so that the comparison does not have to be taken on trust.

    Raises:
        ValidationError: As :func:`concurrency`, or if ``size`` is not positive.
    """
    low, high = _bars(start, end)
    count = low.size
    draws = count if size is None else int(size)
    if draws <= 0:
        raise ValidationError(f"size must be a positive number of draws, got {size!r}")
    generator = as_generator(seed)

    first = int(low.min())
    bars = int(high.max()) - first + 1
    offsets_low = low - first
    offsets_high = high - first + 1
    lengths = (high - low + 1).astype(np.float64)
    # One extra slot so a window ending on the last bar has somewhere to
    # decrement; the difference array is the same trick `concurrency` uses.
    running = np.zeros(bars, dtype=np.int64)
    picked = np.empty(draws, dtype=np.int64)

    for step in range(draws):
        reciprocal = 1.0 / (running + 1.0)
        prefix = np.concatenate(([0.0], np.cumsum(reciprocal, dtype=np.float64)))
        scores = (prefix[offsets_high] - prefix[offsets_low]) / lengths
        total = float(scores.sum())
        # Every score is strictly positive -- a reciprocal of a positive count,
        # averaged -- so the total cannot vanish and there is no degenerate
        # distribution to fall back from.
        chosen = int(generator.choice(count, p=scores / total))
        picked[step] = chosen
        running[offsets_low[chosen] : offsets_high[chosen]] += 1

    achieved = drawn_uniqueness(low, high, picked)
    uniform_indices = generator.integers(0, count, size=draws)
    uniform = drawn_uniqueness(low, high, uniform_indices)
    return SequentialDraw(indices=picked, achieved=achieved, uniform=uniform)
