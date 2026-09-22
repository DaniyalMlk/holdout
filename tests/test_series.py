from __future__ import annotations

import math

import numpy as np
import pytest

from holdout import InsufficientDataError, ValidationError, as_matrix, as_returns
from holdout.series import check_labels, check_periods_per_year, check_probability


def test_as_returns_accepts_lists_and_arrays() -> None:
    out = as_returns([0.01, -0.02, 0.03])
    assert out.dtype == np.float64
    assert out.shape == (3,)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_as_returns_names_the_first_non_finite_index(bad: float) -> None:
    with pytest.raises(ValidationError, match=r"returns\[2\]"):
        as_returns([0.1, 0.2, bad, 0.3, bad])


def test_as_returns_rejects_scalars_matrices_and_text() -> None:
    with pytest.raises(ValidationError, match="scalar"):
        as_returns(0.5)
    with pytest.raises(ValidationError, match="one-dimensional"):
        as_returns([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValidationError, match="numbers"):
        as_returns(["a", "b"])


def test_as_returns_enforces_minimum_length() -> None:
    with pytest.raises(InsufficientDataError, match="1 observation"):
        as_returns([0.1])
    assert as_returns([0.1], min_length=1).size == 1


def test_errors_are_value_errors() -> None:
    # Callers catching ValueError keep working.
    with pytest.raises(ValueError):
        as_returns([math.nan, 1.0])
    with pytest.raises(ValueError):
        as_returns([1.0])


def test_as_matrix_promotes_vectors_and_locates_bad_cells() -> None:
    assert as_matrix([0.1, 0.2, 0.3]).shape == (3, 1)
    grid = np.zeros((4, 3))
    grid[3, 1] = np.nan
    with pytest.raises(ValidationError, match=r"\[3, 1\]"):
        as_matrix(grid)


def test_as_matrix_minimum_shape() -> None:
    with pytest.raises(InsufficientDataError, match="strategy column"):
        as_matrix(np.zeros((5, 1)), min_columns=2)
    with pytest.raises(InsufficientDataError, match="observation"):
        as_matrix(np.zeros((1, 4)))
    with pytest.raises(ValidationError, match="two-dimensional"):
        as_matrix(np.zeros((2, 2, 2)))


@pytest.mark.parametrize("bad", [0, -252, math.nan, math.inf])
def test_periods_per_year_must_be_positive(bad: float) -> None:
    with pytest.raises(ValidationError):
        check_periods_per_year(bad)
    assert check_periods_per_year(None) is None
    assert check_periods_per_year(252) == 252.0


def test_probability_bounds() -> None:
    assert check_probability(0.5, "p") == 0.5
    with pytest.raises(ValidationError):
        check_probability(0.0, "p")
    assert check_probability(0.0, "p", open_interval=False) == 0.0
    with pytest.raises(ValidationError):
        check_probability(1.5, "p", open_interval=False)


def test_labels() -> None:
    assert check_labels(None, 2) == ["s0", "s1"]
    with pytest.raises(ValidationError, match="unique"):
        check_labels(["a", "a"], 2)
    with pytest.raises(ValidationError, match="entries"):
        check_labels(["a"], 2)
