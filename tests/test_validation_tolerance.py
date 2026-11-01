"""What ``check_correlation``'s tolerance actually decides.

The function names four properties it validates and takes a ``tolerance``
keyword. Two of the four -- symmetry and the unit diagonal -- are
``np.allclose`` calls, and ``np.allclose`` keeps ``rtol=1e-5`` unless it is
told otherwise. Its comparison is ``|a - b| <= atol + rtol * |b|``, so with
``atol`` wired to ``tolerance`` and ``rtol`` left alone, the relative term sets
the bound for every entry of a correlation matrix and the knob can loosen the
check but not tighten it.

The tests below measure the bound by bisection rather than reasoning about the
formula, and assert two things about it: that it tracks the tolerance, and that
it does not depend on the size of the entry being compared. Both fail on a
relative comparison -- the second one by a factor of nine across the entries
used here -- and both hold once the comparison is absolute.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from holdout import ValidationError
from holdout.trials import check_correlation


def asymmetric(rho: float, gap: float) -> NDArray[np.float64]:
    """A 2x2 correlation matrix whose two off-diagonal entries differ by ``gap``."""
    return np.array([[1.0, rho], [rho + gap, 1.0]])


def off_diagonal(diagonal: float) -> NDArray[np.float64]:
    """A symmetric matrix whose diagonal sits ``diagonal`` away from one, below it.

    Below one deliberately: a diagonal entry above one is caught by the
    ``[-1, 1]`` range check further down the function, which would hide
    whatever the unit-diagonal check does or does not do.
    """
    return np.array([[1.0 - diagonal, 0.3], [0.3, 1.0]])


def largest_accepted(
    build: Callable[[float], NDArray[np.float64]], tolerance: float, ceiling: float = 1e-3
) -> float:
    """The largest deviation ``check_correlation`` still accepts, by bisection.

    ``build`` maps a deviation to a matrix. The search needs the deviation to be
    rejected at the ceiling and accepted at zero, and both are asserted rather
    than assumed, because a bisection between two rejections or two acceptances
    silently returns one of its own bounds.
    """

    def accepted(deviation: float) -> bool:
        try:
            check_correlation(build(deviation), tolerance=tolerance)
        except ValidationError:
            return False
        return True

    assert accepted(0.0), "the undeviated matrix must be accepted"
    assert not accepted(ceiling), f"a deviation of {ceiling} must be rejected"
    low, high = 0.0, ceiling
    for _ in range(80):
        middle = (low + high) / 2.0
        if accepted(middle):
            low = middle
        else:
            high = middle
    return low


@pytest.mark.parametrize("tolerance", [1e-10, 1e-8, 1e-6])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_the_symmetry_bound_is_the_tolerance_it_was_given(tolerance: float, rho: float) -> None:
    bound = largest_accepted(lambda gap: asymmetric(rho, gap), tolerance)
    assert bound == pytest.approx(tolerance, rel=1e-6, abs=0.0)


@pytest.mark.parametrize("tolerance", [1e-10, 1e-8, 1e-6])
def test_the_symmetry_bound_does_not_depend_on_how_large_the_entry_is(tolerance: float) -> None:
    # The claim the parametrisation above cannot make on its own: a relative
    # comparison would give a bound nine times larger at rho=0.9 than at
    # rho=0.1, and an absolute one gives the same bound at both.
    small = largest_accepted(lambda gap: asymmetric(0.1, gap), tolerance)
    large = largest_accepted(lambda gap: asymmetric(0.9, gap), tolerance)
    assert small == pytest.approx(large, rel=1e-6, abs=0.0)


@pytest.mark.parametrize("tolerance", [1e-10, 1e-8, 1e-6])
def test_the_diagonal_bound_is_the_tolerance_it_was_given(tolerance: float) -> None:
    bound = largest_accepted(off_diagonal, tolerance)
    assert bound == pytest.approx(tolerance, rel=1e-6, abs=0.0)


def test_a_matrix_the_default_rejects_and_a_loose_tolerance_accepts() -> None:
    # Both halves, so that neither a function which rejects everything nor one
    # which accepts everything passes. 4e-6 is under numpy's default relative
    # bound for these entries, which is the whole point of the case.
    sloppy = asymmetric(0.5, 4e-6)
    with pytest.raises(ValidationError, match="symmetric"):
        check_correlation(sloppy)
    with pytest.raises(ValidationError, match="symmetric"):
        check_correlation(sloppy, tolerance=0.0)
    accepted = check_correlation(sloppy, tolerance=1e-5)
    assert accepted[0, 1] == accepted[1, 0]

    uneven = off_diagonal(5e-6)
    with pytest.raises(ValidationError, match="diagonal"):
        check_correlation(uneven)
    np.testing.assert_allclose(np.diag(check_correlation(uneven, tolerance=1e-5)), [1 - 5e-6, 1.0])


def test_what_the_caller_gets_back_when_the_matrix_is_accepted() -> None:
    # The averaging is deliberate and stays: within the tolerance the two
    # entries are the same number measured twice. What was wrong was getting it
    # back after asking for a tolerance the asymmetry exceeded -- a caller who
    # passes a tight tolerance to catch a carelessly estimated matrix was handed
    # a tidy one instead of the error.
    inside = check_correlation(asymmetric(0.9, 8e-9), tolerance=1e-8)
    assert inside[0, 1] == inside[1, 0] == pytest.approx(0.9 + 4e-9, rel=1e-12, abs=0.0)
    with pytest.raises(ValidationError, match="symmetric"):
        check_correlation(asymmetric(0.9, 2e-8), tolerance=1e-8)


def test_the_other_two_checks_are_unchanged() -> None:
    # The range check and the positive-semi-definite check read the tolerance
    # directly rather than through allclose, and neither moved.
    check_correlation(np.array([[1.0, 1.0 + 5e-9], [1.0 + 5e-9, 1.0]]), tolerance=1e-8)
    with pytest.raises(ValidationError, match=r"\[-1, 1\]"):
        check_correlation(np.array([[1.0, 1.05], [1.05, 1.0]]), tolerance=1e-8)
    indefinite = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    with pytest.raises(ValidationError, match="positive semi-definite"):
        check_correlation(indefinite)
    # The PSD floor is deliberately not allowed below 1e-10 per dimension: an
    # eigenvalue of a matrix assembled from rounded correlations carries
    # rounding of that order whatever the caller asks for.
    rounded = np.array([[1.0, 1.0 - 1e-12], [1.0 - 1e-12, 1.0]])
    check_correlation(rounded, tolerance=0.0)
    assert math.isfinite(float(np.linalg.eigvalsh(rounded)[0]))
