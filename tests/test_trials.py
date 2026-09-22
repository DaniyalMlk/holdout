from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from holdout import ValidationError
from holdout.deflated import deflate_trials
from holdout.trials import check_correlation, effective_number_of_trials, trial_correlation


def blocks(sizes: list[int], within: float = 1.0) -> NDArray[np.float64]:
    n = sum(sizes)
    c = np.zeros((n, n))
    start = 0
    for size in sizes:
        c[start : start + size, start : start + size] = within
        start += size
    np.fill_diagonal(c, 1.0)
    return c


@pytest.mark.parametrize("method", ["average", "eigenvalue", "participation"])
def test_extremes(method: str) -> None:
    assert effective_number_of_trials(np.eye(20), method=method) == pytest.approx(20)  # type: ignore[arg-type]
    assert effective_number_of_trials(np.ones((20, 20)), method=method) == pytest.approx(1)  # type: ignore[arg-type]
    assert effective_number_of_trials(np.eye(1), method=method) == 1.0  # type: ignore[arg-type]


@pytest.mark.parametrize("sizes", [[5, 5], [10, 3, 1], [4, 4, 4, 4], [1, 1, 18]])
def test_eigenvalue_method_counts_perfectly_correlated_blocks(sizes: list[int]) -> None:
    assert effective_number_of_trials(blocks(sizes), method="eigenvalue") == pytest.approx(
        len(sizes)
    )


@pytest.mark.parametrize("k", [2, 3, 5])
def test_participation_counts_equal_blocks_but_underweights_small_ones(k: int) -> None:
    assert effective_number_of_trials(blocks([4] * k), method="participation") == pytest.approx(k)
    # Blocks of 10, 3 and 1: 196 / 110.
    got = effective_number_of_trials(blocks([10, 3, 1]), method="participation")
    assert got == pytest.approx(196 / 110)


def test_participation_equicorrelated_closed_form() -> None:
    for rho in (0.1, 0.5, 0.9):
        c = np.full((12, 12), rho)
        np.fill_diagonal(c, 1.0)
        expected = 12 / (1 + 11 * rho**2)
        assert effective_number_of_trials(c, method="participation") == pytest.approx(expected)


def test_eigenvalue_method_jumps_where_an_eigenvalue_crosses_an_integer() -> None:
    # Twelve equicorrelated trials: the top eigenvalue 1 + 11 rho reaches 6 at
    # rho = 5/11 and the Li-Ji count drops from 8 to 7 there.
    def count(rho: float) -> float:
        c = np.full((12, 12), rho)
        np.fill_diagonal(c, 1.0)
        return effective_number_of_trials(c, method="eigenvalue")

    assert count(5 / 11 - 1e-6) == pytest.approx(8.0, abs=1e-4)
    assert count(5 / 11 + 1e-6) == pytest.approx(7.0, abs=1e-4)


def test_average_method_does_not_count_blocks() -> None:
    # Two blocks of five identical trials: truly two, the average method says 6.0.
    got = effective_number_of_trials(blocks([5, 5]), method="average")
    rho = 40 / 90
    assert got == pytest.approx(rho + (1 - rho) * 10)
    assert got == pytest.approx(6.0)


def test_partial_correlation_lies_between_the_extremes() -> None:
    c = np.full((10, 10), 0.5)
    np.fill_diagonal(c, 1.0)
    for method in ("average", "eigenvalue", "participation"):
        got = effective_number_of_trials(c, method=method)
        assert 1.0 < got < 10.0
    assert effective_number_of_trials(c, method="average") == pytest.approx(5.5)


def test_negative_average_correlation_is_floored() -> None:
    c = np.array([[1.0, -0.4, -0.4], [-0.4, 1.0, -0.2], [-0.4, -0.2, 1.0]])
    assert effective_number_of_trials(c, method="average") == pytest.approx(3.0)
    assert effective_number_of_trials(c, method="eigenvalue") <= 3.0


def test_monotone_in_correlation() -> None:
    values = []
    for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
        c = np.full((12, 12), rho)
        np.fill_diagonal(c, 1.0)
        values.append(
            [effective_number_of_trials(c, method=m) for m in ("average", "participation")]
        )
    for column in zip(*values, strict=True):
        assert list(column) == sorted(column, reverse=True)


def test_correlation_validation() -> None:
    with pytest.raises(ValidationError, match="square"):
        check_correlation(np.ones((2, 3)))
    with pytest.raises(ValidationError, match="symmetric"):
        check_correlation(np.array([[1.0, 0.2], [0.3, 1.0]]))
    with pytest.raises(ValidationError, match="diagonal"):
        check_correlation(np.array([[2.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValidationError, match="non-finite"):
        check_correlation(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValidationError, match=r"\[-1, 1\]"):
        check_correlation(np.array([[1.0, 1.5], [1.5, 1.0]]))
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    with pytest.raises(ValidationError, match="positive semi-definite"):
        check_correlation(bad)
    with pytest.raises(ValidationError, match="method"):
        effective_number_of_trials(np.eye(3), method="median")  # type: ignore[arg-type]


def test_trial_correlation_from_returns() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=(2000, 1))
    x = np.hstack([base + 0.1 * rng.normal(size=(2000, 1)) for _ in range(6)])
    x = np.hstack([x, rng.normal(size=(2000, 4))])
    c = trial_correlation(x)
    np.testing.assert_allclose(c, np.corrcoef(x, rowvar=False))
    # Six near-copies (correlation ~0.99) and four independent trials. The
    # estimators disagree, which is why the method is a required argument:
    # Li-Ji gives 6 (the block's eigenvalue 5.94 contributes 1.94), the
    # average method 7.0, the participation ratio 2.5.
    assert effective_number_of_trials(c, method="eigenvalue") == pytest.approx(6.0, abs=0.1)
    assert effective_number_of_trials(c, method="average") == pytest.approx(7.0, abs=0.1)
    assert effective_number_of_trials(c, method="participation") == pytest.approx(2.5, abs=0.1)
    with pytest.raises(ValidationError, match="column 1"):
        trial_correlation(np.column_stack([rng.normal(size=10), np.full(10, 0.01)]))


def test_effective_trials_feed_the_deflated_ratio() -> None:
    # Forty variants that are near-copies of one strategy should not be
    # deflated as forty independent attempts.
    rng = np.random.default_rng(9)
    base = rng.normal(0.0008, 0.01, size=(1000, 1))
    x = base + 0.001 * rng.normal(size=(1000, 40))
    n_eff = effective_number_of_trials(trial_correlation(x), method="eigenvalue")
    assert n_eff < 3
    naive = deflate_trials(x)
    adjusted = deflate_trials(x, n_trials=n_eff)
    assert adjusted.deflated > naive.deflated
