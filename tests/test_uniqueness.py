"""Uniqueness, sample weights, and the bootstrap that is supposed to need them.

Three things carry this suite.

:class:`TestIdentity` asserts ``sum(uniqueness * length) == covered bars`` on
randomised overlap structures. Both sides count ``sum over covered bars of
c / c``, so it holds whatever the labels look like — gaps, nesting, duplicates,
a single bar — and it is the one check that the prefix-sum arithmetic is right
rather than merely self-consistent.

:class:`TestHandComputed` works the small cases out on paper. A chain of
windows overlapping by half, two identical windows, a window nested inside
another: each has an answer that can be written down, and writing it down is
the only way to catch an implementation that is consistently wrong.

:class:`TestSequentialBootstrapIsMostlyPointless` is the measurement. The
achievable uniqueness of a draw is capped at ``span / (size * length)``, since
total concurrency over the draw does not depend on what is drawn. The tests
assert that the cap holds, that the uniform bootstrap already reaches 99% of it
at full size, and that the sequential scheme's real gain lives entirely in the
small-draw regime.
"""

from __future__ import annotations

import numpy as np
import pytest

from holdout.exceptions import InsufficientDataError, ValidationError
from holdout.splits import purged_kfold
from holdout.uniqueness import (
    average_uniqueness,
    concurrency,
    drawn_uniqueness,
    effective_sample_size,
    sequential_bootstrap,
    time_decay_weights,
    uniqueness_weights,
)


def _rolling(count: int, length: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    start = np.arange(0, count * step, step, dtype=np.int64)
    return start, start + length - 1


def _random_labels(generator: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    start = np.sort(generator.integers(0, 200, size=count)).astype(np.int64)
    length = generator.integers(1, 30, size=count).astype(np.int64)
    return start, start + length - 1


class TestIdentity:
    """``sum(uniqueness * length) == covered bars``, whatever the structure."""

    @pytest.mark.parametrize("seed", range(12))
    def test_on_random_overlap_structures(self, seed: int) -> None:
        generator = np.random.default_rng(seed)
        start, end = _random_labels(generator, int(generator.integers(1, 60)))
        unique = average_uniqueness(start, end)
        lengths = (end - start + 1).astype(np.float64)
        assert float((unique * lengths).sum()) == pytest.approx(
            concurrency(start, end).covered_bars, rel=1e-12
        )

    def test_with_a_gap_in_the_labels(self) -> None:
        start = np.array([0, 1, 10, 11])
        end = np.array([2, 3, 12, 13])
        counts = concurrency(start, end)
        assert list(counts.counts) == [1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1]
        assert counts.covered_bars == 8
        unique = average_uniqueness(start, end)
        lengths = (end - start + 1).astype(np.float64)
        assert float((unique * lengths).sum()) == pytest.approx(8.0)

    def test_with_duplicated_windows(self) -> None:
        start = np.array([3, 3, 3, 3])
        end = np.array([7, 7, 7, 7])
        assert list(average_uniqueness(start, end)) == [0.25] * 4
        assert effective_sample_size(start, end) == pytest.approx(1.0)
        assert concurrency(start, end).covered_bars == 5

    def test_with_one_window_nested_inside_another(self) -> None:
        start = np.array([0, 2])
        end = np.array([9, 3])
        counts = concurrency(start, end)
        assert list(counts.counts) == [1, 1, 2, 2, 1, 1, 1, 1, 1, 1]
        # The outer window: eight bars alone and two shared, over ten bars.
        assert average_uniqueness(start, end)[0] == pytest.approx((8 + 2 * 0.5) / 10)
        # The inner one shares both of its bars.
        assert average_uniqueness(start, end)[1] == pytest.approx(0.5)

    def test_uniqueness_always_lies_in_the_unit_interval(self) -> None:
        for seed in range(8):
            generator = np.random.default_rng(100 + seed)
            start, end = _random_labels(generator, 40)
            unique = average_uniqueness(start, end)
            assert np.all(unique > 0.0)
            assert np.all(unique <= 1.0 + 1e-15)


class TestHandComputed:
    """The cases with an answer that can be written down."""

    def test_non_overlapping_point_labels_are_all_unique(self) -> None:
        start = np.arange(10)
        unique = average_uniqueness(start, start)
        assert list(unique) == [1.0] * 10
        assert effective_sample_size(start, start) == 10.0
        assert list(uniqueness_weights(start, start)) == [0.1] * 10

    def test_non_overlapping_multi_bar_labels_are_all_unique(self) -> None:
        start, end = _rolling(5, length=4, step=4)
        assert list(average_uniqueness(start, end)) == [1.0] * 5
        assert effective_sample_size(start, end) == 5.0

    def test_identical_windows_share_everything(self) -> None:
        start = np.zeros(5, dtype=np.int64)
        end = np.full(5, 4, dtype=np.int64)
        assert list(average_uniqueness(start, end)) == [0.2] * 5
        assert effective_sample_size(start, end) == pytest.approx(1.0)

    def test_a_chain_overlapping_by_half(self) -> None:
        """Windows ``[0,3], [2,5], [4,7], ...`` — worked out by hand.

        The interior bars are each covered twice and the two bars at each end
        once, so an interior window is ``4 * (1/2) / 4 = 1/2`` and an end
        window is ``(2 * 1 + 2 * (1/2)) / 4 = 3/4``.
        """
        start = np.arange(0, 12, 2, dtype=np.int64)
        end = start + 3
        unique = average_uniqueness(start, end)
        assert unique[0] == pytest.approx(0.75)
        assert unique[-1] == pytest.approx(0.75)
        assert list(np.round(unique[1:-1], 12)) == [0.5] * 4
        assert effective_sample_size(start, end) == pytest.approx(3.5)

    def test_concurrency_of_a_rolling_window_peaks_at_the_window_length(self) -> None:
        start, end = _rolling(50, length=7)
        counts = concurrency(start, end)
        assert counts.peak == 7
        assert counts.first == 0
        assert counts.last == 55
        assert counts.covered_bars == 56
        assert "peak 7" in repr(counts)

    def test_the_effective_sample_falls_as_windows_lengthen(self) -> None:
        sizes = [effective_sample_size(*_rolling(100, length=L)) for L in (1, 2, 5, 10, 50)]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] == pytest.approx(100.0)
        # A hundred observations of fifty-bar rolling windows are worth about
        # three: the span is 149 bars and each covers fifty of them.
        assert 2.5 < sizes[-1] < 3.5


class TestWeights:
    """Both schedules, against hand computations."""

    def test_uniqueness_weights_sum_to_one(self) -> None:
        for seed in range(6):
            generator = np.random.default_rng(7 * seed)
            start, end = _random_labels(generator, 30)
            weights = uniqueness_weights(start, end)
            assert float(weights.sum()) == pytest.approx(1.0)
            assert np.all(weights > 0.0)

    def test_uniqueness_weights_are_proportional_to_uniqueness(self) -> None:
        start = np.array([0, 0, 0, 10])
        end = np.array([4, 4, 4, 14])
        weights = uniqueness_weights(start, end)
        # Three observations at a third each and one alone: 1/3, 1/3, 1/3, 1.
        assert weights[3] == pytest.approx(3.0 * weights[0])

    def test_no_decay_leaves_every_weight_at_one(self) -> None:
        unique = np.full(8, 0.5)
        assert list(time_decay_weights(unique, decay=1.0)) == [1.0] * 8

    def test_linear_decay_from_zero(self) -> None:
        """``decay=0`` with eight equal uniquenesses: ``k/8`` for ``k = 1..8``."""
        unique = np.full(8, 0.5)
        weights = time_decay_weights(unique, decay=0.0)
        assert list(np.round(weights, 12)) == [k / 8 for k in range(1, 9)]

    def test_half_decay_starts_the_oldest_at_half_the_newest(self) -> None:
        unique = np.full(8, 0.5)
        weights = time_decay_weights(unique, decay=0.5)
        assert weights[-1] == pytest.approx(1.0)
        # The schedule is linear in cumulative uniqueness and reaches 0.5 at
        # zero, so the first observation sits half a step above that.
        assert weights[0] == pytest.approx(0.5 + 0.5 / 8)

    def test_a_negative_decay_discards_a_share_of_the_sample(self) -> None:
        unique = np.full(8, 0.5)
        weights = time_decay_weights(unique, decay=-0.5)
        assert list(weights[:4]) == [0.0] * 4
        assert np.all(weights[4:] > 0.0)
        assert weights[-1] == pytest.approx(1.0)

    def test_the_extreme_negative_decay_keeps_only_the_newest(self) -> None:
        unique = np.full(8, 0.5)
        weights = time_decay_weights(unique, decay=-1.0)
        assert list(weights) == [0.0] * 7 + [1.0]

    def test_decay_runs_on_information_and_not_on_the_index(self) -> None:
        """The reason the schedule is cumulative in uniqueness.

        Two samples of the same length: in the first every observation is
        unique, in the second the older half is a pile of redundant copies
        carrying one observation's worth of information between them. Linear
        decay should treat the redundant pile as the single old observation it
        is, so the newer half keeps far more of its weight than it would under
        a schedule linear in the index.
        """
        plain = np.full(20, 1.0)
        redundant = np.concatenate([np.full(10, 0.1), np.full(10, 1.0)])
        # With every uniqueness equal, cumulative uniqueness *is* the index, so
        # `plain` is exactly the index-linear schedule and is the comparison.
        under_plain = time_decay_weights(plain, decay=0.0)
        under_redundant = time_decay_weights(redundant, decay=0.0)
        old_plain = float(under_plain[:10].sum()) / float(under_plain.sum())
        old_redundant = float(under_redundant[:10].sum()) / float(under_redundant.sum())
        # The index-linear schedule hands the older half 26% of the weight. On
        # an information scale the same ten observations get 7.8%, because they
        # are one observation's worth out of eleven -- which is 9%, so the
        # schedule lands within about a point of the share they deserve.
        assert old_plain == pytest.approx(0.262, abs=0.005)
        assert old_redundant == pytest.approx(0.078, abs=0.005)
        assert old_redundant < old_plain / 3.0

    def test_the_weights_are_monotone_in_time_for_any_decay(self) -> None:
        generator = np.random.default_rng(3)
        unique = generator.uniform(0.05, 1.0, size=40)
        for decay in (1.0, 0.7, 0.3, 0.0, -0.3, -0.9):
            weights = time_decay_weights(unique, decay=decay)
            assert np.all(np.diff(weights) >= -1e-15)
            assert weights[-1] == pytest.approx(1.0)


class TestSequentialBootstrapIsMostlyPointless:
    """The measurement, and the arithmetic that explains it."""

    @staticmethod
    def _cap(count: int, length: int, draws: int) -> float:
        start, end = _rolling(count, length)
        span = int(end.max() - start.min() + 1)
        return min(1.0, span / (draws * length))

    @pytest.mark.parametrize(("length", "draws"), [(20, 200), (20, 100), (20, 40), (50, 500)])
    def test_no_scheme_can_beat_the_cap(self, length: int, draws: int) -> None:
        """Total concurrency over the draw is ``draws * length`` however it is drawn.

        So the mean of ``1 / c`` over the drawn windows is bounded by flattening
        ``c``, which gives ``span / (draws * length)``. Both schemes are
        checked against it, because a scheme that beat it would be a bug in the
        scoring rather than a discovery.
        """
        count = 200 if length == 20 else 500
        start, end = _rolling(count, length)
        cap = self._cap(count, length, draws)
        for seed in range(4):
            draw = sequential_bootstrap(start, end, size=draws, seed=seed)
            assert draw.achieved <= cap + 1e-12
            assert draw.uniform <= cap + 1e-12

    def test_at_full_size_the_uniform_bootstrap_is_already_at_the_cap(self) -> None:
        """Which is why drawing cleverly buys nothing in the regime it is sold for."""
        start, end = _rolling(200, length=20)
        cap = self._cap(200, 20, 200)
        achieved = []
        uniform = []
        for seed in range(12):
            draw = sequential_bootstrap(start, end, seed=100 + seed)
            achieved.append(draw.achieved)
            uniform.append(draw.uniform)
        assert float(np.mean(uniform)) / cap > 0.99
        assert float(np.mean(achieved)) / cap > 0.99
        # The whole advantage is under half a per cent, and the spread across
        # seeds is comparable to it.
        advantage = float(np.mean(achieved)) / float(np.mean(uniform)) - 1.0
        assert 0.0 < advantage < 0.01

    def test_the_gain_rises_as_the_draw_shrinks(self) -> None:
        """Monotone over the sizes where it is resolved, and it plateaus below them.

        Thirty replications, which is enough to order these four and not enough
        to separate ten draws from five: over sixty seeds those two come out at
        +10.43% and +9.37% with standard errors of 2.31% and 2.80%, so the
        plateau is asserted as a floor rather than as an ordering.
        """
        start, end = _rolling(200, length=20)
        gains = {}
        for draws in (200, 100, 40, 20):
            achieved = []
            uniform = []
            for seed in range(30):
                draw = sequential_bootstrap(start, end, size=draws, seed=seed)
                achieved.append(draw.achieved)
                uniform.append(draw.uniform)
            gains[draws] = float(np.mean(achieved)) / float(np.mean(uniform)) - 1.0
        assert gains[20] > gains[40] > gains[100] > gains[200]
        assert gains[200] < 0.01
        assert gains[20] > 0.05

    def test_it_helps_most_when_nothing_overlaps_at_all(self) -> None:
        """Because then the only redundancy left is drawing the same point twice.

        Point labels overlap with nothing, so a uniform resample's redundancy
        is entirely duplicate draws, and avoiding those is the one thing the
        sequential scheme does well. That the gain is largest here, where the
        overlap problem does not exist, is the clearest sign that the scheme is
        not solving the overlap problem.
        """
        start = np.arange(200, dtype=np.int64)
        draw = sequential_bootstrap(start, start, seed=7)
        assert draw.advantage > 0.10
        assert draw.achieved < 1.0

    def test_the_draw_is_the_size_asked_for_and_in_range(self) -> None:
        start, end = _rolling(50, length=5)
        for size in (1, 7, 50, 120):
            draw = sequential_bootstrap(start, end, size=size, seed=1)
            assert draw.indices.size == size
            assert draw.indices.min() >= 0
            assert draw.indices.max() < 50

    def test_it_is_reproducible_from_a_seed(self) -> None:
        start, end = _rolling(60, length=6)
        first = sequential_bootstrap(start, end, seed=12345)
        second = sequential_bootstrap(start, end, seed=12345)
        assert list(first.indices) == list(second.indices)
        assert first.achieved == second.achieved
        third = sequential_bootstrap(start, end, seed=54321)
        assert list(third.indices) != list(first.indices)

    def test_a_generator_is_accepted_as_well_as_a_seed(self) -> None:
        start, end = _rolling(30, length=4)
        draw = sequential_bootstrap(start, end, seed=np.random.default_rng(9))
        assert draw.indices.size == 30

    def test_the_repr_names_both_numbers(self) -> None:
        start, end = _rolling(30, length=4)
        draw = sequential_bootstrap(start, end, seed=2)
        text = repr(draw)
        assert "30 draws" in text
        assert "uniform" in text

    def test_drawn_uniqueness_counts_repeats(self) -> None:
        start = np.arange(5, dtype=np.int64)
        # Five distinct point labels score one each.
        assert drawn_uniqueness(start, start, [0, 1, 2, 3, 4]) == pytest.approx(1.0)
        # The same one five times scores a fifth each.
        assert drawn_uniqueness(start, start, [2, 2, 2, 2, 2]) == pytest.approx(0.2)
        # And a draw out of order is scored the same as one in order, which is
        # what lets a bootstrap draw be scored without sorting it first.
        assert drawn_uniqueness(start, start, [3, 0, 4, 1, 2]) == pytest.approx(
            drawn_uniqueness(start, start, [0, 1, 2, 3, 4])
        )


class TestAgainstTheSplitter:
    """The two halves of the overlapping-label problem, in the same place."""

    def test_purging_removes_leakage_and_leaves_the_redundancy(self) -> None:
        """Which is the sentence this module exists to make true of the library.

        A purged split is free of leakage — the splitter's own audit says so —
        and its training set is still full of observations saying the same
        thing. The effective sample size of that training set is a fraction of
        its length, and nothing in the splitter notices.
        """
        count, length = 120, 10
        start, end = _rolling(count, length)
        splits = purged_kfold(4, n=count, start=start, end=end)
        for split in splits:
            train = split.train
            assert train.size > 0
            effective = effective_sample_size(start[train], end[train])
            assert effective < 0.25 * train.size

    def test_non_overlapping_labels_leave_nothing_for_this_module_to_do(self) -> None:
        count = 60
        start, end = _rolling(count, length=1)
        splits = purged_kfold(5, n=count, start=start, end=end)
        for split in splits:
            train = split.train
            assert effective_sample_size(start[train], end[train]) == pytest.approx(
                float(train.size)
            )


class TestValidation:
    """What gets refused, and what the message says."""

    def test_mismatched_arrays(self) -> None:
        with pytest.raises(ValidationError, match="entries"):
            average_uniqueness([0, 1, 2], [0, 1])

    def test_no_labels(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least one label"):
            average_uniqueness([], [])

    def test_a_window_that_ends_before_it_starts(self) -> None:
        with pytest.raises(ValidationError, match=r"ends .* before it starts"):
            average_uniqueness([0, 5], [3, 2])

    def test_non_integer_labels_say_why_integers_are_needed(self) -> None:
        with pytest.raises(ValidationError, match="whole bar indices"):
            average_uniqueness([0.0, 1.5], [1.0, 2.5])

    def test_whole_valued_floats_are_accepted(self) -> None:
        """Because a timestamp-free label array often arrives as floats."""
        unique = average_uniqueness([0.0, 4.0], [3.0, 7.0])
        assert list(unique) == [1.0, 1.0]

    def test_non_finite_labels(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            average_uniqueness([0.0, np.nan], [1.0, 2.0])
        with pytest.raises(ValidationError, match="finite"):
            average_uniqueness([0.0, 1.0], [1.0, np.inf])

    def test_unsorted_labels_are_fine_here(self) -> None:
        """Unlike in the splitter, where the splits are positional."""
        unique = average_uniqueness([4, 0], [7, 3])
        assert list(unique) == [1.0, 1.0]

    @pytest.mark.parametrize("decay", [1.01, -1.01, 2.0, -5.0])
    def test_a_decay_outside_the_range(self, decay: float) -> None:
        with pytest.raises(ValidationError, match=r"decay must lie in \[-1, 1\]"):
            time_decay_weights(np.full(5, 0.5), decay=decay)

    def test_bad_uniqueness_input_to_the_decay(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least one observation"):
            time_decay_weights([])
        with pytest.raises(ValidationError, match="finite"):
            time_decay_weights([0.5, np.nan])
        with pytest.raises(ValidationError, match="positive"):
            time_decay_weights([0.5, 0.0])

    def test_a_non_positive_draw_size(self) -> None:
        start, end = _rolling(10, length=2)
        with pytest.raises(ValidationError, match="positive number of draws"):
            sequential_bootstrap(start, end, size=0)
        with pytest.raises(ValidationError, match="positive number of draws"):
            sequential_bootstrap(start, end, size=-3)

    def test_an_out_of_range_index(self) -> None:
        start = np.arange(5, dtype=np.int64)
        with pytest.raises(ValidationError, match=r"indices must lie in \[0, 4\]"):
            drawn_uniqueness(start, start, [0, 5])
        with pytest.raises(ValidationError, match="indices must lie in"):
            drawn_uniqueness(start, start, [-1, 2])

    def test_an_empty_draw(self) -> None:
        start = np.arange(5, dtype=np.int64)
        with pytest.raises(InsufficientDataError, match="nothing was drawn"):
            drawn_uniqueness(start, start, [])

    def test_a_bad_seed(self) -> None:
        start, end = _rolling(10, length=2)
        with pytest.raises(ValidationError, match="seed must be"):
            sequential_bootstrap(start, end, seed="nope")  # type: ignore[arg-type]
