from __future__ import annotations

import time

import numpy as np
import pytest

from holdout import HoldoutError, ValidationError, kurtosis
from holdout.synthetic import crossover_sweep, garch_returns
from holdout.trials import effective_number_of_trials, trial_correlation


def test_garch_returns_have_fat_tails_and_volatility_clustering() -> None:
    x = garch_returns(20_000, seed=1)
    assert kurtosis(x) > 5.0
    sq = x**2 - np.mean(x**2)
    lag1 = float(sq[1:] @ sq[:-1] / (sq @ sq))
    assert lag1 > 0.1
    # Unconditional variance omega / (1 - alpha - beta) = 5e-5.
    assert np.var(x) == pytest.approx(5e-5, rel=0.25)


def test_garch_arguments() -> None:
    with pytest.raises(ValidationError):
        garch_returns(10, alpha=0.5, beta=0.6)
    with pytest.raises(ValidationError):
        garch_returns(10, dof=2.0)


def test_sweep_shape_and_names() -> None:
    sweep = crossover_sweep(1000, fast=(5, 10), slow=(20, 50), seed=3)
    assert sweep.table.names == ["ma5_20", "ma5_50", "ma10_20", "ma10_50"]
    assert sweep.table.values.shape == (1000, 4)
    assert sweep.market.shape == (1000,)
    assert np.all(np.isfinite(sweep.table.values))
    # Each strategy's return is plus or minus the market's (or zero on a tie).
    ratio = np.abs(sweep.table.values) - np.abs(sweep.market)[:, None]
    assert np.all((np.abs(ratio) < 1e-15) | (sweep.table.values == 0))


def test_sweep_members_are_highly_correlated() -> None:
    sweep = crossover_sweep(seed=7)
    c = trial_correlation(sweep.table.values)
    assert effective_number_of_trials(c, method="eigenvalue") < len(sweep.table.names) / 2


def test_sweep_is_reproducible_and_validates_length() -> None:
    a = crossover_sweep(600, fast=(5,), slow=(40,), seed=1)
    b = crossover_sweep(600, fast=(5,), slow=(40,), seed=1)
    np.testing.assert_array_equal(a.table.values, b.table.values)
    with pytest.raises(ValidationError, match="n_days"):
        crossover_sweep(100)


class TestDegenerateSweepGrids:
    """A grid that yields no backtestable pair is refused from the parameters.

    Every case here used to fail inside numpy or inside a builtin, after the
    GARCH simulation had already run, with an error that was not a
    ``HoldoutError`` and did not name the argument at fault.
    """

    def test_an_empty_fast_or_slow_is_refused_by_name(self) -> None:
        # The empty slow raised out of max() in the n_days guard, before that
        # guard could express anything; the empty fast was not checked at all
        # and fell through to column_stack.
        with pytest.raises(ValidationError, match="fast must contain at least one window"):
            crossover_sweep(600, fast=(), slow=(40,))
        with pytest.raises(ValidationError, match="slow must contain at least one window"):
            crossover_sweep(600, fast=(5,), slow=())

    def test_a_grid_whose_every_pair_is_skipped_is_refused(self) -> None:
        # A crossover needs the fast window strictly shorter than the slow one.
        # fast=(40,), slow=(40,) offers one pair and keeps none.
        with pytest.raises(ValidationError, match="the grid is empty"):
            crossover_sweep(600, fast=(40,), slow=(40,))
        with pytest.raises(ValidationError, match="the grid is empty"):
            crossover_sweep(600, fast=(50, 60), slow=(40, 50))

    def test_the_refusal_is_catchable_as_a_library_error(self) -> None:
        # The point of the hierarchy: a caller separates "these inputs cannot
        # describe a sweep" from a bug in their own code with one except.
        for fast, slow in (((), (40,)), ((5,), ()), ((40,), (40,))):
            with pytest.raises(HoldoutError):
                crossover_sweep(600, fast=fast, slow=slow)

    def test_an_empty_grid_costs_no_simulation(self) -> None:
        # Deciding validity from the data meant simulating n_days + max(slow)
        # days of GARCH before discovering there was nothing to backtest. The
        # bound is loose on purpose -- it is here to catch the simulation
        # running at all, not to time it.
        started = time.perf_counter()
        with pytest.raises(ValidationError, match="the grid is empty"):
            crossover_sweep(2_000_000, fast=(40,), slow=(40,))
        assert time.perf_counter() - started < 0.5

    def test_an_empty_grid_is_reported_before_a_short_n_days(self) -> None:
        # Both are wrong here: no pair survives AND 10 days is too few. The
        # grid is reported, because an empty grid is structural -- no n_days
        # makes it backtestable -- whereas a short sample is fixable by the
        # caller. Pinned because the two guards are adjacent and reordering
        # them is a silent change in what a doubly-invalid call is told.
        with pytest.raises(ValidationError, match="the grid is empty"):
            crossover_sweep(10, fast=(40,), slow=(40,))

    def test_the_n_days_guard_still_fires_for_a_grid_that_is_valid(self) -> None:
        # The new checks run first, so this confirms they did not shadow the
        # length guard for any grid that has pairs to backtest.
        with pytest.raises(ValidationError, match="n_days must be at least 80"):
            crossover_sweep(10, fast=(5,), slow=(40,))


class TestGridSizing:
    """Pairs offered against pairs kept, asserted rather than assumed."""

    @pytest.mark.parametrize(
        ("fast", "slow", "offered", "kept"),
        [
            ((5, 10, 15, 20, 30), (40, 60, 90, 120, 180, 250), 30, 30),  # the default
            ((5, 10), (20, 50), 4, 4),
            ((10, 50), (20, 40), 4, 2),  # 50 is not shorter than either slow window
            ((3, 5, 7), (5, 7, 9), 9, 6),  # the interleaved case, where it matters
        ],
    )
    def test_pairs_kept_against_pairs_offered(
        self, fast: tuple[int, ...], slow: tuple[int, ...], offered: int, kept: int
    ) -> None:
        sweep = crossover_sweep(600, fast=fast, slow=slow, seed=1)
        assert len(fast) * len(slow) == offered
        assert len(sweep.parameters) == kept
        assert len(sweep.table.names) == kept
        assert sweep.table.values.shape == (600, kept)
        # Every kept pair is a genuine crossover, and nothing valid was dropped.
        assert sweep.parameters == [(f, s) for f in fast for s in slow if f < s]

    def test_the_grid_order_is_the_nesting_order_of_the_two_tuples(self) -> None:
        # The winner of a sweep is chosen by index, so column order is part of
        # the result. fast varies slowest.
        sweep = crossover_sweep(600, fast=(3, 5), slow=(7, 9), seed=1)
        assert sweep.parameters == [(3, 7), (3, 9), (5, 7), (5, 9)]
        assert sweep.table.names == ["ma3_7", "ma3_9", "ma5_7", "ma5_9"]
