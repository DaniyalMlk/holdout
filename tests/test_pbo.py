from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from holdout import InsufficientDataError, ValidationError
from holdout.pbo import cscv_partitions, probability_of_backtest_overfitting


def sharpe_columns(x: NDArray[np.float64]) -> NDArray[np.float64]:
    result: NDArray[np.float64] = x.mean(axis=0) / x.std(axis=0, ddof=1)
    return result


def direct_logits(x: NDArray[np.float64], n_blocks: int) -> list[float]:
    """CSCV written as a plain loop, straight from the paper."""
    blocks = np.array_split(np.arange(x.shape[0]), n_blocks)
    out = []
    for chosen in cscv_partitions(n_blocks):
        train = np.concatenate([blocks[b] for b in chosen])
        test = np.concatenate([blocks[b] for b in range(n_blocks) if b not in chosen])
        r_in, r_out = sharpe_columns(x[train]), sharpe_columns(x[test])
        best = int(np.argmax(r_in))
        rank = 1 + int(np.sum(r_out < r_out[best]))
        w = rank / (x.shape[1] + 1)
        out.append(math.log(w / (1 - w)))
    return out


def test_partition_count() -> None:
    assert len(cscv_partitions(16)) == 12_870
    assert len(cscv_partitions(2)) == 2
    parts = cscv_partitions(6)
    assert len(parts) == 20
    # Symmetric: every block is in-sample in exactly half the partitions.
    counts = np.bincount(np.concatenate(parts), minlength=6)
    assert np.all(counts == 10)
    for bad in (0, 3, 24, 4.5):
        with pytest.raises(ValidationError):
            cscv_partitions(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("rows", [160, 163, 999])
def test_fast_path_matches_the_direct_loop(rows: int) -> None:
    rng = np.random.default_rng(rows)
    x = rng.normal(0.0, 0.01, size=(rows, 12)) + rng.normal(0, 0.0005, size=12)
    result = probability_of_backtest_overfitting(x, n_blocks=8)
    np.testing.assert_allclose(result.logits, direct_logits(x, 8), rtol=1e-10)


def test_general_metric_path_matches_the_fast_path() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0.0002, 0.01, size=(400, 9))
    fast = probability_of_backtest_overfitting(x, n_blocks=10)
    slow = probability_of_backtest_overfitting(x, n_blocks=10, metric=sharpe_columns)
    np.testing.assert_allclose(fast.logits, slow.logits, rtol=1e-10)
    np.testing.assert_array_equal(fast.selected, slow.selected)
    np.testing.assert_allclose(fast.in_sample, slow.in_sample, rtol=1e-10)
    np.testing.assert_allclose(fast.out_of_sample, slow.out_of_sample, rtol=1e-10)
    mean = probability_of_backtest_overfitting(x, n_blocks=10, metric="mean")
    by_callable = probability_of_backtest_overfitting(
        x, n_blocks=10, metric=lambda d: d.mean(axis=0)
    )
    np.testing.assert_allclose(mean.logits, by_callable.logits, rtol=1e-10)


def test_pure_noise_sits_near_one_half() -> None:
    rng = np.random.default_rng(0)
    values = [
        probability_of_backtest_overfitting(rng.normal(size=(800, 20)), n_blocks=10).pbo
        for _ in range(30)
    ]
    assert np.mean(values) == pytest.approx(0.5, abs=0.06)


def test_a_genuine_edge_sits_near_zero() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000, 20))
    x[:, 3] += 0.25
    result = probability_of_backtest_overfitting(x)
    assert result.pbo < 0.01
    assert result.selection_frequency()[3] > 0.99
    assert result.probability_of_loss < 0.01


def test_winners_that_must_lose_sit_near_one() -> None:
    # Every strategy has exactly zero mean over the full sample, so whatever
    # does best on one half does correspondingly worse on the other.
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 20))
    x -= x.mean(axis=0)
    result = probability_of_backtest_overfitting(x, metric="mean")
    assert result.pbo == 1.0
    assert result.probability_of_loss == 1.0
    slope, _, _ = result.degradation
    assert slope < 0


def test_degradation_regression() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(600, 15)) * 0.01
    result = probability_of_backtest_overfitting(x, n_blocks=8)
    slope, intercept, r2 = result.degradation
    fitted = np.polyfit(result.in_sample, result.out_of_sample, 1)
    assert slope == pytest.approx(fitted[0])
    assert intercept == pytest.approx(fitted[1])
    assert 0.0 <= r2 <= 1.0
    assert result.n_partitions == 70
    assert result.n_strategies == 15
    assert result.selection_frequency().sum() == pytest.approx(1.0)


def test_logits_are_bounded_by_the_number_of_strategies() -> None:
    rng = np.random.default_rng(4)
    n = 10
    result = probability_of_backtest_overfitting(rng.normal(size=(300, n)), n_blocks=6)
    edge = math.log(n)  # rank n of n: w = n / (n + 1)
    assert np.all(np.abs(result.logits) <= edge + 1e-12)


def test_ties_share_an_average_rank() -> None:
    # Two identical columns and a third: the winner's rank is averaged with its twin.
    rng = np.random.default_rng(6)
    a = rng.normal(0.1, 1.0, size=200)
    x = np.column_stack([a, a, rng.normal(-0.5, 1.0, size=200)])
    result = probability_of_backtest_overfitting(x, n_blocks=4)
    w = 2.5 / 4  # ranks 2 and 3 averaged, over N + 1 = 4
    np.testing.assert_allclose(result.logits, math.log(w / (1 - w)))


def test_arguments() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 5))
    with pytest.raises(InsufficientDataError, match="blocks"):
        probability_of_backtest_overfitting(x[:20], n_blocks=16)
    with pytest.raises(InsufficientDataError):
        probability_of_backtest_overfitting(x[:, :1])
    with pytest.raises(ValidationError, match="metric"):
        probability_of_backtest_overfitting(x, metric="sortino")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="3 scores"):
        probability_of_backtest_overfitting(x, n_blocks=4, metric=lambda d: d.mean(axis=0)[:3])
    with pytest.raises(ValidationError, match="non-finite"):
        probability_of_backtest_overfitting(
            x, n_blocks=4, metric=lambda d: np.full(d.shape[1], np.nan)
        )
    flat = x.copy()
    flat[:, 2] = 0.01
    with pytest.raises(ValidationError, match="strategy 2 has zero variance"):
        probability_of_backtest_overfitting(flat, n_blocks=4)


def test_fast_path_survives_a_large_mean_relative_to_the_spread() -> None:
    # Mean 1 and spread 1e-4: raw sums of squares would lose about eight digits.
    rng = np.random.default_rng(8)
    x = 1.0 + 1e-4 * rng.normal(size=(320, 6))
    result = probability_of_backtest_overfitting(x, n_blocks=8)
    np.testing.assert_allclose(result.logits, direct_logits(x, 8), rtol=1e-12)
    slow = probability_of_backtest_overfitting(x, n_blocks=8, metric=sharpe_columns)
    np.testing.assert_allclose(result.in_sample, slow.in_sample, rtol=1e-9)
