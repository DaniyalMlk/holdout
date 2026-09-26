"""The model confidence set: which models cannot be told apart from the best.

Every other test here needs a benchmark. :func:`~holdout.spa.superior_predictive_ability`
asks whether any strategy beats a designated one, :func:`~holdout.spa.romano_wolf`
asks which ones do, and neither has anything to say about a set of twenty
candidates with no incumbent among them.

The tempting move — nominate the sample-best as the benchmark and test the rest
against it — is not a repair. The benchmark is then chosen by the same data the
test is run on, so under the null the "benchmark" is systematically the luckiest
series present and every comparison is biased towards finding nothing. That is
the data-snooping problem in a new place rather than a solution to it.

Hansen, Lunde and Nason (2011) turn the question around. Start with all the
models. Test the hypothesis that they are all equally good. If it is rejected,
drop the one the test most implicates and try again on what is left. Stop when
the hypothesis is not rejected: what remains is the model confidence set, and it
contains the best model with probability at least ``1 - alpha`` asymptotically.

Two things about it are worth understanding before reading a result.

**It is a confidence set, so its size is information.** A set holding eighteen
of twenty models is not a failure of the procedure; it is the procedure saying
that two thousand observations cannot separate eighteen strategies. Reporting the
single best of those eighteen as the winner is precisely the claim the set exists
to deny. Conversely a set of one is a strong statement, and worth checking for
the mundane explanations — a strategy whose returns are on a different scale, or
one that is an affine transform of another.

**The elimination rule and the test statistic are separate choices**, and the
paper gives two of each. :attr:`Statistic.MAX` studentises each model's average
performance against the current set and takes the largest deviation;
:attr:`Statistic.RANGE` takes the largest studentised difference over all pairs.
They do not agree on borderline cases: the range statistic is driven by the two
extremes of the set and so is more affected by one wild model, while the max
statistic compares everything to the set average and is more affected by the
composition of the set. Both are here because neither dominates.

**Convention: higher is better.** The input is a matrix of *performance* —
returns, or negated losses — one column per model, matching the rest of this
library. A loss matrix must be negated before it is passed, and the docstring of
:func:`model_confidence_set` says what happens if it is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bootstrap import Seed, bootstrap_column_means
from .exceptions import ValidationError
from .series import FloatArray, check_labels, check_probability

__all__ = [
    "Elimination",
    "ModelConfidenceSet",
    "Statistic",
    "model_confidence_set",
]


class Statistic(str, Enum):
    """Which equivalence statistic to test the current set with."""

    #: ``max_i t_i``, where ``t_i`` studentises model ``i``'s average performance
    #: against the average of the set. Scales to many models: the statistic is a
    #: maximum over ``m`` quantities rather than over ``m(m-1)/2``.
    MAX = "max"
    #: ``max_{i,j} |t_ij|``, the largest studentised pairwise difference. Closer
    #: to the question as usually asked — "is any pair distinguishable" — and
    #: more sensitive to a single outlying model, because one extreme column
    #: enters every pair it belongs to.
    RANGE = "range"


@dataclass(frozen=True)
class Elimination:
    """One step of the sequential test, recorded so the path can be read."""

    #: Index of the model dropped at this step, in the original column order.
    model: int
    #: The equivalence statistic on the set as it stood before the drop.
    statistic: float
    #: Bootstrap p-value of that statistic. Not the model's confidence-set
    #: p-value: that one is the running maximum, since a model cannot be more
    #: confidently retained than the step that removed it.
    step_pvalue: float
    #: How many models were in the set when this step ran.
    size: int


@dataclass(frozen=True)
class ModelConfidenceSet:
    """The set, the p-value behind every model in it, and how it was reached."""

    #: ``(k,)`` confidence-set p-value per model, in the original column order.
    #: A model is in the set at level ``alpha`` when its p-value exceeds
    #: ``alpha``. These are the running maxima of the step p-values, which makes
    #: them monotone in the elimination order and the sets nested in ``alpha``.
    pvalues: FloatArray
    #: The eliminations in the order they happened, worst first.
    eliminations: tuple[Elimination, ...]
    #: Index of the model with the highest average performance. In the set by
    #: construction — the elimination rule never drops it while anything worse
    #: is present.
    best: int
    alpha: float
    statistic: Statistic
    block_length: float
    n_bootstrap: int
    names: tuple[str, ...]
    #: ``(k,)`` average performance per model. Reported rather than left for the
    #: caller to recompute, because the orderings below are by this and a reader
    #: comparing a set against a differently computed mean will find they differ
    #: in the last digits and wonder which is wrong.
    performance: FloatArray

    @property
    def included(self) -> NDArray[np.int64]:
        """Indices in the set at :attr:`alpha`, best average performance first."""
        hits = np.flatnonzero(self.pvalues > self.alpha)
        order = np.argsort(-self.performance[hits], kind="stable")
        result: NDArray[np.int64] = hits[order].astype(np.int64)
        return result

    @property
    def excluded(self) -> NDArray[np.int64]:
        """Indices not in the set, in the order they were eliminated."""
        dropped = [
            step.model
            for step in self.eliminations
            if self.pvalues[step.model] <= self.alpha
        ]
        return np.asarray(dropped, dtype=np.int64)

    @property
    def included_names(self) -> tuple[str, ...]:
        return tuple(self.names[index] for index in self.included)

    def at(self, alpha: float) -> NDArray[np.int64]:
        """The set at a different level, without rerunning the bootstrap.

        The p-values are all the procedure produces, so every level is available
        from one run. The sets are nested: a larger ``alpha`` can only remove
        models, never add them, because the p-values do not depend on it.
        """
        level = check_probability(alpha, "alpha")
        hits = np.flatnonzero(self.pvalues > level)
        order = np.argsort(-self.performance[hits], kind="stable")
        result: NDArray[np.int64] = hits[order].astype(np.int64)
        return result


def _studentised(
    means: FloatArray, resampled: FloatArray, active: NDArray[np.int64], statistic: Statistic
) -> tuple[float, FloatArray, FloatArray]:
    """The statistic, its bootstrap null, and the per-model scores behind it.

    Everything is a linear function of column means, which is why the bootstrap
    is over means and not over the matrix. Writing ``m`` for a resampled vector
    of means restricted to the active set:

    * ``MAX`` compares each model to the set average. The centred quantity is
      ``size / (size - 1) * (m_i - mean(m))``, which is the average of
      ``m_i - m_j`` over the other members, up to that factor. The scale factor
      is not cosmetic — dropping it makes the statistic shrink as the set does,
      and the sequence of tests would then get easier to pass for a reason that
      has nothing to do with the models.
    * ``RANGE`` works on the pairwise differences directly.

    The standard errors come from the bootstrap deviations rather than from a
    long-run variance estimator, which is the point of resampling in blocks: the
    serial dependence is in the resamples, so it is in the standard errors
    without anyone having to model it.
    """
    size = active.size
    centre = means[active]
    draws = resampled[:, active]
    if statistic is Statistic.MAX:
        factor = size / (size - 1.0)
        scores = factor * (centre - centre.mean())
        deviations = factor * (draws - draws.mean(axis=1, keepdims=True)) - scores
        scale = np.sqrt(np.mean(deviations * deviations, axis=0))
        _refuse_degenerate(scale, active)
        # Performance, so a *low* score is a candidate for elimination: the
        # statistic is the largest shortfall against the set, not the largest
        # lead over it.
        t = -scores / scale
        null = -deviations / scale
        return float(np.max(t)), t, np.max(null, axis=1)
    differences = centre[:, None] - centre[None, :]
    draw_differences = draws[:, :, None] - draws[:, None, :]
    deviations = draw_differences - differences
    scale = np.sqrt(np.mean(deviations * deviations, axis=0))
    upper = np.triu_indices(size, k=1)
    _refuse_degenerate(scale[upper], active)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(scale > 0.0, differences / scale, 0.0)
        null_pairs = np.where(scale > 0.0, deviations / scale, 0.0)
    # Model i's worst studentised shortfall against any single other model, so
    # the elimination rule and the statistic come from the same array.
    per_model = np.max(-t, axis=1)
    statistic_value = float(np.max(np.abs(t[upper])))
    null = np.max(np.abs(null_pairs[:, upper[0], upper[1]]), axis=1)
    return statistic_value, per_model, null


def _refuse_degenerate(scale: FloatArray, active: NDArray[np.int64]) -> None:
    """Refuse a standard error that is zero to the precision of the arithmetic.

    The obvious test — ``scale > 0`` — does not catch the case it is for. Two
    identical columns have means that differ in the last bits of the mantissa
    rather than not at all, so their bootstrap deviations come out around 1e-17
    instead of exactly zero and the check passes. What follows is worse than a
    crash: the studentised difference is a real number divided by 1e-17, the
    statistic is 3e16, and the model is eliminated with a p-value of zero as
    though the evidence against it were overwhelming.

    So the threshold is relative to the largest standard error in the same set,
    which is the only scale available here and the right one — the question is
    whether *this* difference has a standard error worth dividing by compared to
    the others being studentised alongside it.
    """
    largest = float(np.max(scale)) if scale.size else 0.0
    flat = np.flatnonzero(~(scale > largest * 1e-10))
    if flat.size:
        raise ValidationError(
            "two models in the set have the same performance in every bootstrap "
            "resample to within floating-point error, so their difference has no "
            f"standard error to studentise by (among columns {active.tolist()}). "
            "Duplicate columns cause this; so does a column that is another plus "
            "a constant, which is not equal to it but differs from it by the same "
            "amount in every period."
        )


def model_confidence_set(
    performance: ArrayLike,
    *,
    alpha: float = 0.10,
    statistic: Statistic | str = Statistic.MAX,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: Seed = None,
    names: Sequence[str] | None = None,
) -> ModelConfidenceSet:
    """The model confidence set of Hansen, Lunde and Nason (2011).

    ``performance`` is ``(periods, k)`` with **higher better** — returns, or
    losses negated — matching the convention everywhere else here. Passing a loss
    matrix unnegated does not fail: it returns the set of models that cannot be
    distinguished from the *worst* one, which looks entirely plausible. The
    ordering of :attr:`ModelConfidenceSet.included` is the thing to sanity-check
    against what the columns mean.

    ``alpha`` defaults to 0.10 rather than 0.05, following the paper. The set is
    a confidence region and the usual asymmetry of hypothesis testing does not
    apply: a smaller ``alpha`` gives a *larger* set, so the conventional choice
    here errs towards a bigger region rather than a smaller one.

    The bootstrap runs once. Every level is then available from
    :meth:`ModelConfidenceSet.at`, because the p-values do not depend on
    ``alpha`` — only the reading of them does.

    Raises :class:`~holdout.exceptions.ValidationError` when two columns cannot
    be told apart in any resample, which is a duplicated or affinely transformed
    column rather than a statistical finding.
    """
    level = check_probability(alpha, "alpha")
    kind = Statistic(statistic)
    sampled = bootstrap_column_means(
        performance,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        seed=seed,
        name="performance",
    )
    means = sampled.means
    count = means.size
    if count < 2:
        raise ValidationError(
            f"a confidence set over {count} model says nothing; pass at least two columns"
        )
    labels = check_labels(names, count, "names")

    active = np.arange(count, dtype=np.int64)
    pvalues = np.ones(count)
    eliminations: list[Elimination] = []
    running = 0.0
    while active.size > 1:
        value, scores, null = _studentised(means, sampled.resampled, active, kind)
        # The bootstrap null is centred at the sample, so its own realisations
        # are the reference distribution for the statistic.
        step = float(np.mean(null >= value))
        running = max(running, step)
        worst = int(active[int(np.argmax(scores))])
        pvalues[worst] = running
        eliminations.append(
            Elimination(model=worst, statistic=value, step_pvalue=step, size=int(active.size))
        )
        active = active[active != worst]
    # The survivor keeps the initial value of one, and that is not a placeholder.
    # A model is in the set at level alpha when the procedure stopped before the
    # step that would have removed it; the last model standing was in the active
    # set at every step, so it is in the set at every level. Assigning it the
    # running maximum instead would drop it from the set whenever the data
    # rejected equality all the way down — leaving an empty set, which is not a
    # possible answer to "which models survive".

    return ModelConfidenceSet(
        pvalues=pvalues,
        eliminations=tuple(eliminations),
        best=int(np.argmax(means)),
        alpha=level,
        statistic=kind,
        block_length=sampled.block_length,
        n_bootstrap=int(n_bootstrap),
        names=tuple(labels),
        performance=means,
    )
