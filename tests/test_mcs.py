"""The model confidence set, checked against what it claims rather than itself.

Four things are asserted here that a wrong implementation would fail: a set of
equally good models keeps nearly all of them, a set with a real winner drops the
rest, the true best model is in the set at least ``1 - alpha`` of the time under
simulation, and the set is nested in ``alpha`` so a level can be read off without
rerunning anything.

The simulated coverage is the one that matters. Everything else can be satisfied
by a procedure that returns plausible-looking sets for the wrong reasons.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from holdout import ValidationError
from holdout.mcs import Statistic, model_confidence_set

T, K, B = 500, 8, 300
BOTH = [Statistic.MAX, Statistic.RANGE]


def equally_good(generator: np.random.Generator, models: int = K) -> NDArray[np.float64]:
    return generator.normal(size=(T, models))


def one_clear_winner(generator: np.random.Generator) -> NDArray[np.float64]:
    """One model a long way ahead, and a pair of near-copies of it.

    The near-copies are the interesting part. They are not independent tries, and
    a procedure that treated them as such would find three winners where there is
    one strategy implemented three ways — which is what the cross-sectional
    resampling exists to prevent.
    """
    values = generator.normal(size=(T, K))
    values[:, 0] += 0.5
    values[:, 1] = values[:, 0] + generator.normal(scale=0.05, size=T)
    values[:, 2] = values[:, 0] + generator.normal(scale=0.05, size=T)
    return values


# -- the two things it has to get right --------------------------------------


@pytest.mark.parametrize("statistic", BOTH)
def test_equally_good_models_are_nearly_all_retained(statistic: Statistic) -> None:
    """Nothing separates them, so the set should say so.

    The assertion is on the average size over twenty draws rather than on one,
    because a single set of eight can lose one model to an unlucky sample without
    anything being wrong. Measured over those twenty draws: 7.85 of 8 under the
    max statistic and 7.80 under the range, with the smallest set seven in both
    cases. Nothing is separated because nothing differs, which is the answer.
    """
    generator = np.random.default_rng(11)
    sizes = []
    for _ in range(20):
        result = model_confidence_set(
            equally_good(generator),
            statistic=statistic,
            n_bootstrap=B,
            block_length=4,
            seed=generator,
        )
        sizes.append(result.included.size)
    assert sum(sizes) / len(sizes) >= 7.0
    assert min(sizes) >= 4


@pytest.mark.parametrize("statistic", BOTH)
def test_a_clear_winner_and_its_copies_survive_and_the_rest_do_not(
    statistic: Statistic,
) -> None:
    result = model_confidence_set(
        one_clear_winner(np.random.default_rng(12)),
        statistic=statistic,
        n_bootstrap=B,
        block_length=4,
        seed=7,
    )
    kept = set(result.included.tolist())
    assert kept <= {0, 1, 2}, "no model without the edge should survive"
    assert 0 in kept
    assert result.best in kept
    # And the losers were eliminated before the winners were considered.
    order = [step.model for step in result.eliminations]
    assert set(order[:5]).isdisjoint({0, 1, 2})


# -- coverage, by simulation -------------------------------------------------


@pytest.mark.parametrize("statistic", BOTH)
def test_the_best_model_is_in_the_set_at_least_one_minus_alpha_of_the_time(
    statistic: Statistic,
) -> None:
    """The claim the procedure is for, checked rather than quoted.

    Five models with means spread finely enough that the sample best is often not
    the true best — the case where a procedure that quietly conditioned on the
    sample ordering would look fine on the two tests above and fail here.

    Measured over 120 replications at ``alpha = 0.10``: the true best model was in
    the set 120 times out of 120 under both statistics. The bound is asymptotic
    and one-sided, so over-coverage is expected and is not a defect: the set is
    conservative because the elimination sequence stops at the first failure to
    reject rather than testing every subset.
    """
    generator = np.random.default_rng(13)
    replications = 120
    contained = 0
    for _ in range(replications):
        values = generator.normal(size=(200, 5)) + np.array([0.0, 0.04, 0.08, 0.12, 0.16])
        result = model_confidence_set(
            values,
            alpha=0.10,
            statistic=statistic,
            n_bootstrap=200,
            block_length=3,
            seed=generator,
        )
        contained += 4 in set(result.included.tolist())
    assert contained / replications >= 0.90


def test_the_set_is_never_empty_even_when_everything_is_rejected() -> None:
    """Two models a mile apart: the loser goes, the winner cannot.

    The last model standing keeps a p-value of one, and that is a statement about
    the procedure rather than a placeholder. It was in the active set at every
    step, so it is in the set at every level; giving it the running maximum
    instead would return an empty set whenever equality was rejected all the way
    down, and an empty set is not a possible answer to which models survive.
    """
    generator = np.random.default_rng(14)
    values = generator.normal(size=(300, 2))
    values[:, 1] += 2.0
    result = model_confidence_set(values, n_bootstrap=B, block_length=3, seed=1)
    assert result.included.tolist() == [1]
    assert result.pvalues[1] == 1.0
    assert result.pvalues[0] < 0.05
    assert result.at(0.99).tolist() == [1]


# -- reading the result ------------------------------------------------------


def test_the_sets_are_nested_in_alpha() -> None:
    """One bootstrap, every level. A larger alpha can only remove models."""
    generator = np.random.default_rng(15)
    values = generator.normal(size=(400, 6)) + np.linspace(0.0, 0.25, 6)
    result = model_confidence_set(values, n_bootstrap=500, block_length=3, seed=2)
    previous = set(range(6))
    sizes = []
    for alpha in (0.01, 0.05, 0.10, 0.25, 0.5):
        current = set(result.at(alpha).tolist())
        assert current <= previous
        previous = current
        sizes.append(len(current))
    assert sizes[0] >= sizes[-1]
    assert result.included.tolist() == result.at(result.alpha).tolist()


def test_the_step_pvalues_run_up_to_the_confidence_set_pvalues() -> None:
    """The confidence-set p-value is the running maximum of the step p-values.

    Which is what makes the sets nested: without the maximum a later step could
    return a smaller p-value than an earlier one, and a model eliminated late
    would drop out of the set at a level where a model eliminated before it
    stayed in.
    """
    generator = np.random.default_rng(16)
    values = generator.normal(size=(400, 7)) + np.linspace(0.0, 0.3, 7)
    result = model_confidence_set(values, n_bootstrap=400, block_length=3, seed=3)
    running = 0.0
    for step in result.eliminations:
        running = max(running, step.step_pvalue)
        assert result.pvalues[step.model] == pytest.approx(running)
        assert 0.0 <= step.step_pvalue <= 1.0
    sizes = [step.size for step in result.eliminations]
    assert sizes == list(range(7, 1, -1))


def test_the_ordering_is_by_average_performance() -> None:
    generator = np.random.default_rng(17)
    values = generator.normal(size=(400, 5)) + np.linspace(0.0, 0.4, 5)
    result = model_confidence_set(
        values, alpha=0.01, n_bootstrap=300, block_length=3, seed=4, names=list("abcde")
    )
    included = result.included
    assert list(included) == sorted(included, key=lambda i: -result.performance[i])
    assert result.included_names == tuple("abcde"[i] for i in included)
    assert result.performance == pytest.approx(values.mean(axis=0))
    assert result.best == 4


def test_excluded_is_the_elimination_order_of_the_models_that_went() -> None:
    generator = np.random.default_rng(18)
    values = generator.normal(size=(300, 4))
    values[:, 3] -= 1.0
    result = model_confidence_set(values, n_bootstrap=B, block_length=3, seed=5)
    excluded = result.excluded.tolist()
    assert 3 in excluded
    assert set(excluded).isdisjoint(set(result.included.tolist()))
    assert len(excluded) + result.included.size == 4
    order = [step.model for step in result.eliminations]
    assert excluded == [model for model in order if model in set(excluded)]


# -- the convention, and what breaking it does -------------------------------


def test_losses_passed_without_negating_select_the_worst_models() -> None:
    """The documented trap, pinned so the documentation stays true.

    Higher is better here. A loss matrix passed straight in returns the models
    that cannot be distinguished from the *worst* one, and it looks entirely
    reasonable doing it — same shape of output, same p-values in range. The only
    thing that gives it away is which models are in the set, which is why the
    ordering is reported alongside.
    """
    generator = np.random.default_rng(19)
    losses = generator.normal(size=(400, 4)) ** 2
    losses[:, 0] *= 0.4  # model 0 has the smallest loss, so it is the best
    right = model_confidence_set(-losses, n_bootstrap=B, block_length=3, seed=6)
    wrong = model_confidence_set(losses, n_bootstrap=B, block_length=3, seed=6)
    assert right.best == 0
    assert 0 in set(right.included.tolist())
    assert wrong.best != 0
    assert 0 not in set(wrong.included.tolist())
    # Not a mirror image of each other, and it is worth being precise about why:
    # negating reverses the elimination order, so the two runs are sequences of
    # *different* tests on different subsets, and only the first step is the same
    # hypothesis. The best model here has a p-value of 1 in one run and 0.0 in the
    # other — the two outputs are not related by a transformation a reader could
    # apply to recover one from the other.
    assert right.pvalues[0] == 1.0
    assert wrong.pvalues[0] < 0.05
    assert set(right.included.tolist()).isdisjoint(set(wrong.included.tolist()))


def test_a_duplicated_column_is_refused_by_the_range_statistic() -> None:
    """Their difference has zero variance in every resample, so nothing to divide by."""
    generator = np.random.default_rng(20)
    values = generator.normal(size=(300, 3))
    values[:, 2] = values[:, 1]
    with pytest.raises(ValidationError, match="no standard error"):
        model_confidence_set(
            values, statistic=Statistic.RANGE, n_bootstrap=B, block_length=3, seed=7
        )


def test_a_column_that_is_another_plus_a_constant_is_refused_by_the_range() -> None:
    """Not equal to it, and still without a standard error between them.

    The difference is 0.3 in every period, so it is 0.3 in every resample and the
    pairwise standard error is zero — the second model is better with certainty
    rather than with a measurable margin. Studentising that gives infinity, so the
    range statistic refuses rather than eliminating a model on a p-value of zero.

    The max statistic does not refuse, and should not: it never forms that pairwise
    standard error, and a column 0.3 above another really is the better of the two.
    It returns the set containing only that model, which is correct. The pair is
    degenerate to one statistic and informative to the other.
    """
    generator = np.random.default_rng(21)
    values = generator.normal(size=(300, 3))
    values[:, 2] = values[:, 1] + 0.3
    with pytest.raises(ValidationError, match="no standard error"):
        model_confidence_set(
            values, statistic=Statistic.RANGE, n_bootstrap=B, block_length=3, seed=8
        )
    under_max = model_confidence_set(
        values, statistic=Statistic.MAX, n_bootstrap=B, block_length=3, seed=8
    )
    assert under_max.included.tolist() == [2]


def test_a_standard_error_of_zero_is_detected_relative_to_the_others() -> None:
    """The reason the guard is not ``scale > 0``, which is the obvious version.

    Two identical columns have means that differ in the last bits of the mantissa
    rather than not at all, so their bootstrap deviations come out around 1e-17
    instead of zero and an absolute test passes them. The statistic is then a real
    number over 1e-17 — of order 1e16 — and the model is eliminated with a p-value
    of zero, as though the evidence against it were overwhelming rather than
    absent. Asserted here by constructing the case that produces it: a column
    built by adding zero, which is bit-identical in value and not in provenance.
    """
    generator = np.random.default_rng(28)
    values = generator.normal(size=(300, 3))
    values[:, 2] = values[:, 1] + 0.0
    with pytest.raises(ValidationError, match="floating-point error"):
        model_confidence_set(
            values, statistic=Statistic.RANGE, n_bootstrap=B, block_length=3, seed=8
        )


def test_the_max_statistic_tolerates_a_duplicated_column() -> None:
    """And that is a real difference between the two, not an oversight.

    The max statistic never forms a pairwise standard error: each model is
    compared to the average of the set. Two identical columns therefore sit at the
    same place with the same standard error and neither breaks anything — they get
    the same p-value and both stay or both go.

    What it costs is the reading. A set of ten "models" three of which are copies
    of one is not ten strategies that could not be separated, and the procedure has
    no way to tell the caller that. The range statistic refuses instead, which is
    the more useful behaviour and the reason the refusal exists at all.
    """
    generator = np.random.default_rng(20)
    values = generator.normal(size=(300, 3))
    values[:, 2] = values[:, 1]
    result = model_confidence_set(
        values, statistic=Statistic.MAX, n_bootstrap=B, block_length=3, seed=7
    )
    assert result.included.size == 3
    assert result.pvalues[1] == pytest.approx(result.pvalues[2])


def test_every_column_identical_is_refused_by_both() -> None:
    """The one degeneracy the max statistic does see: no spread at all."""
    generator = np.random.default_rng(27)
    values = np.tile(generator.normal(size=(300, 1)), (1, 3))
    for statistic in BOTH:
        with pytest.raises(ValidationError, match="no standard error"):
            model_confidence_set(values, statistic=statistic, n_bootstrap=B, block_length=3, seed=9)


def test_one_model_is_refused() -> None:
    generator = np.random.default_rng(22)
    with pytest.raises(ValidationError, match="says nothing"):
        model_confidence_set(generator.normal(size=(300, 1)), n_bootstrap=B, seed=9)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_an_alpha_outside_the_open_unit_interval_is_refused(alpha: float) -> None:
    generator = np.random.default_rng(23)
    with pytest.raises(ValidationError):
        model_confidence_set(generator.normal(size=(300, 3)), alpha=alpha, seed=10)


def test_an_unknown_statistic_is_refused() -> None:
    generator = np.random.default_rng(24)
    with pytest.raises(ValueError, match="quadratic"):
        model_confidence_set(
            generator.normal(size=(300, 3)), statistic="quadratic", n_bootstrap=B, seed=11
        )


def test_the_same_seed_gives_the_same_set() -> None:
    generator = np.random.default_rng(25)
    values = generator.normal(size=(300, 5)) + np.linspace(0.0, 0.2, 5)
    first = model_confidence_set(values, n_bootstrap=B, block_length=3, seed=12)
    second = model_confidence_set(values, n_bootstrap=B, block_length=3, seed=12)
    assert first.pvalues == pytest.approx(second.pvalues)
    assert first.included.tolist() == second.included.tolist()


def test_the_block_length_is_estimated_when_it_is_not_given() -> None:
    """And it is longer on a series with serial dependence, which is the point."""
    generator = np.random.default_rng(26)
    independent = generator.normal(size=(500, 4))
    dependent = np.empty_like(independent)
    dependent[0] = independent[0]
    for t in range(1, 500):
        dependent[t] = 0.7 * dependent[t - 1] + independent[t]
    plain = model_confidence_set(independent, n_bootstrap=B, seed=13)
    serial = model_confidence_set(dependent, n_bootstrap=B, seed=13)
    assert serial.block_length > plain.block_length
    assert plain.block_length > 0.0
    assert plain.n_bootstrap == B


def test_the_range_statistic_gives_the_smaller_set_on_borderline_data() -> None:
    """Measured, because "they do not always agree" is worth more as a number.

    Ten models on 800 observations, three of them sharing a small genuine edge —
    weak enough that both statistics are working near their limit. Over fifteen
    independent samples the max statistic retained 8.13 models on average and the
    range statistic 7.07, with the range set the smaller of the two on fourteen of
    the fifteen and larger on one. So the ordering is a tendency and not a
    guarantee, which is worth stating as such.

    Neither is wrong. The range statistic is a maximum over 45 studentised pairs
    and the max statistic a maximum over 10 deviations from the set average, so
    the range one has more chances to find a gap and finds them slightly more
    often. The default here is the max statistic: a larger set is the weaker
    claim, and on borderline data the weaker claim is the honest one.
    """
    generator = np.random.default_rng(101)
    samples = []
    for _ in range(15):
        base = generator.normal(0.0008, 0.01, size=(800, 1))
        samples.append(
            np.hstack(
                [
                    base + generator.normal(0.0, 0.001, size=(800, 3)),
                    generator.normal(-0.0005, 0.01, size=(800, 7)),
                ]
            )
        )
    sizes = {}
    for statistic in BOTH:
        sizes[statistic] = [
            model_confidence_set(
                values, statistic=statistic, n_bootstrap=B, block_length=4, seed=31
            ).included.size
            for values in samples
        ]
    wide = sizes[Statistic.MAX]
    narrow = sizes[Statistic.RANGE]
    assert sum(narrow) / len(narrow) < sum(wide) / len(wide) - 0.5
    assert sum(a <= b for a, b in zip(narrow, wide, strict=True)) >= 13
