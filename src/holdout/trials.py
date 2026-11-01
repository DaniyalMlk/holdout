"""Effective number of independent trials.

The deflated Sharpe ratio's benchmark grows with the number of trials, and it
assumes they are independent. They rarely are: a parameter sweep over a
lookback window produces a family of strategies whose returns are nearly
identical. Counting fifty near-copies as fifty trials over-deflates; counting
them as one under-deflates if they are not quite copies. The estimators here
map a correlation matrix of trial returns to a number in ``[1, N]``.

None of the estimators is right in general — "how many independent trials is
this?" has no unique answer between the extremes — so the method is a required
argument and the behaviour of each is pinned down by tests:

* ``"average"`` — ``rho + (1 - rho) * N`` with ``rho`` the mean off-diagonal
  correlation (floored at zero), the interpolation used by Bailey and López de
  Prado. Exact at the two extremes (independent: ``N``; identical: 1),
  continuous, and blind to structure: two blocks of five identical trials
  count as 6.0.
* ``"eigenvalue"`` — Li and Ji (2005): ``sum f(|lambda_i|)`` over the
  eigenvalues of the correlation matrix with ``f(x) = [x >= 1] + (x - floor(x))``.
  Counts ``k`` blocks of perfectly correlated trials as exactly ``k``, but is
  discontinuous: when an eigenvalue crosses an integer the count drops by
  almost one. For twelve equicorrelated trials it falls from 8.0 to 7.0 as the
  correlation passes 5/11.
* ``"participation"`` — the participation ratio ``N**2 / sum(lambda_i**2)``.
  Continuous, ``k`` for ``k`` equal-sized perfect blocks, ``N / (1 + (N - 1) rho**2)``
  for equicorrelated trials, but dominated by the largest block, so small
  independent groups are under-counted.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ValidationError
from .series import FloatArray, as_matrix, constant_floor

__all__ = ["check_correlation", "effective_number_of_trials", "trial_correlation"]

EffectiveMethod = Literal["average", "eigenvalue", "participation"]


def check_correlation(matrix: ArrayLike, *, tolerance: float = 1e-8) -> FloatArray:
    """Validate a correlation matrix: square, symmetric, unit diagonal, PSD.

    ``tolerance`` is absolute and is the whole of the bound for the symmetry,
    unit-diagonal and range checks: an asymmetry or a departure from one of more
    than ``tolerance`` is an error whatever the size of the entry. The
    positive-semi-definite check scales it by the dimension and floors it at
    ``1e-10``, because an eigenvalue of a matrix assembled from correlations
    rounded to a few decimal places carries rounding of that order however
    strictly the caller wants the rest of the matrix read.

    Within the tolerance the two off-diagonal entries of a pair are one number
    measured twice, and their average is what comes back.
    """
    c = np.asarray(matrix, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 1:
        raise ValidationError(f"a correlation matrix must be square, got shape {c.shape}")
    if not np.all(np.isfinite(c)):
        raise ValidationError("the correlation matrix contains non-finite values")
    # `rtol=0.0` explicitly, or `tolerance` decides almost nothing. numpy's
    # comparison is `|a - b| <= atol + rtol * |b|` and it keeps `rtol=1e-5`
    # unless told otherwise, so with only `atol` wired up the relative term set
    # the bound for every entry of a correlation matrix: at the documented
    # default of 1e-8 an asymmetry of 9e-6 was accepted between two entries of
    # 0.9, and the tolerance supplied a thousandth of that. The knob could
    # loosen the check and not tighten it, and a caller passing a tight
    # tolerance to reject a carelessly estimated matrix was handed back the
    # symmetrised average instead of an error.
    if not np.allclose(c, c.T, rtol=0.0, atol=tolerance):
        raise ValidationError("the correlation matrix is not symmetric")
    if not np.allclose(np.diag(c), 1.0, rtol=0.0, atol=tolerance):
        raise ValidationError("the correlation matrix must have ones on its diagonal")
    if np.any(np.abs(c) > 1.0 + tolerance):
        raise ValidationError("correlations must lie in [-1, 1]")
    smallest = float(np.linalg.eigvalsh((c + c.T) / 2.0)[0])
    if smallest < -max(tolerance, 1e-10) * c.shape[0]:
        raise ValidationError(
            f"the correlation matrix is not positive semi-definite (smallest eigenvalue "
            f"{smallest:.3g}); correlations estimated pairwise on different samples can do this"
        )
    return (c + c.T) / 2.0


def trial_correlation(returns: ArrayLike) -> FloatArray:
    """Pearson correlation matrix of the columns of a ``(periods, trials)`` matrix."""
    x = as_matrix(returns, min_rows=3)
    # Column by column, not with an axis argument. The guard compares a
    # standard deviation against a multiple of epsilon, and ``std(axis=0)``
    # reduces along a strided axis: on a constant column it leaves rounding of
    # hundreds of times ``eps * level`` where the one-dimensional reduction
    # leaves under four, measured over a grid of lengths, column counts and
    # magnitudes in tests/test_constant_series.py. A
    # tolerance set to absorb that would be a tolerance chosen to accommodate
    # numpy's reduction order rather than the data's rounding.
    for column in range(x.shape[1]):
        series = x[:, column]
        if float(np.std(series, ddof=1)) <= constant_floor(series):
            raise ValidationError(f"trial column {column} has zero variance")
    c = np.corrcoef(x, rowvar=False)
    return np.atleast_2d(np.asarray(c, dtype=np.float64))


def effective_number_of_trials(correlation: ArrayLike, *, method: EffectiveMethod) -> float:
    """Effective number of independent trials implied by a correlation matrix.

    ``method`` is one of ``"average"``, ``"eigenvalue"`` or ``"participation"``;
    see the module documentation for how they differ.
    """
    c = check_correlation(correlation)
    n = c.shape[0]
    if n == 1:
        return 1.0
    if method == "average":
        off = c[~np.eye(n, dtype=bool)]
        rho = min(max(float(off.mean()), 0.0), 1.0)
        return float(rho + (1.0 - rho) * n)
    eigenvalues = np.abs(np.linalg.eigvalsh(c))
    if method == "eigenvalue":
        # Snap eigenvalues like 0.9999999999 that are integers in exact arithmetic,
        # or the floor below would move them by one.
        rounded = np.round(eigenvalues)
        snapped = np.where(np.abs(eigenvalues - rounded) < 1e-9, rounded, eigenvalues)
        total = float(np.sum((snapped >= 1.0) + (snapped - np.floor(snapped))))
        return min(max(total, 1.0), float(n))
    if method == "participation":
        return float(n * n / np.sum(eigenvalues**2))
    raise ValidationError(
        f"method must be 'average', 'eigenvalue' or 'participation', got {method!r}"
    )
