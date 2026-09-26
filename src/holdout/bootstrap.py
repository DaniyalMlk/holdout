"""The stationary bootstrap and its block length.

Returns are serially dependent — volatility clusters, and smoothed marks make
the returns themselves autocorrelated — so resampling individual days
destroys exactly the structure that determines how noisy an average is. The
stationary bootstrap (Politis and Romano 1994) resamples blocks instead, with
block lengths drawn from a geometric distribution with mean ``b``. Randomising
the length makes the resampled series stationary, which fixed blocks are not.

Resampling works on **row indices**, applied to every column of a matrix at
once, so the dependence *between* strategies is kept as well as the
dependence over time. That is what the tests of superior predictive ability
need.

The block length is chosen by Politis and White (2004) with the correction in
Patton, Politis and White (2009). For an AR(1) process with coefficient
``phi`` their optimal stationary-bootstrap block length has the closed form
``(2 phi / (1 - phi**2)) ** (2/3) * n ** (1/3)``, which the test suite checks
the estimator against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray, as_matrix, as_returns

__all__ = [
    "ColumnMeans",
    "as_generator",
    "bootstrap_column_means",
    "optimal_block_length",
    "stationary_bootstrap_indices",
]

Seed = int | np.random.Generator | None


def as_generator(seed: Seed) -> np.random.Generator:
    """A NumPy generator from an integer seed, an existing generator, or ``None``."""
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or (isinstance(seed, int) and not isinstance(seed, bool)):
        return np.random.default_rng(seed)
    raise ValidationError(f"seed must be an int, a numpy Generator or None, got {seed!r}")


def stationary_bootstrap_indices(
    n: int, block_length: float, n_samples: int, *, seed: Seed = None
) -> NDArray[np.int64]:
    """``(n_samples, n)`` row indices for the stationary bootstrap.

    Each resample starts at a uniformly random row; each following row either
    continues the current block (with probability ``1 - 1/block_length``,
    wrapping around the end of the sample) or jumps to a new uniformly random
    row. Block lengths are therefore geometric with mean ``block_length``.
    """
    if int(n) != n or n < 2:
        raise InsufficientDataError(f"n must be an integer of at least 2, got {n}")
    if not (math.isfinite(block_length) and block_length >= 1.0):
        raise ValidationError(f"block_length must be at least 1, got {block_length!r}")
    if int(n_samples) != n_samples or n_samples < 1:
        raise ValidationError(f"n_samples must be a positive integer, got {n_samples}")
    rng = as_generator(seed)
    n, n_samples = int(n), int(n_samples)
    jump = rng.random((n_samples, n)) < 1.0 / block_length
    jump[:, 0] = True
    fresh = rng.integers(0, n, size=(n_samples, n))
    # Position within the current block: 0 at every jump, counting up otherwise.
    positions = np.arange(n)
    last_jump = np.maximum.accumulate(np.where(jump, positions, 0), axis=1)
    starts = np.take_along_axis(fresh, last_jump, axis=1)
    indices: NDArray[np.int64] = ((starts + positions - last_jump) % n).astype(np.int64)
    return indices


def optimal_block_length(returns: ArrayLike) -> float:
    """Politis–White (2004, corrected 2009) block length for the stationary bootstrap.

    Steps, as in the paper:

    1. ``m_hat`` is the smallest lag after which ``K_N = max(5, sqrt(log10 n))``
       consecutive sample autocorrelations all lie inside
       ``+-2 sqrt(log10(n) / n)``; the bandwidth is ``M = 2 * m_hat``.
    2. With the flat-top window ``lambda(t) = 1`` for ``|t| <= 1/2`` and
       ``2 (1 - |t|)`` up to ``|t| = 1``, estimate
       ``G = sum lambda(k/M) |k| R(k)`` and ``g = sum lambda(k/M) R(k)``.
    3. ``b = (2 G**2 / D) ** (1/3) * n ** (1/3)`` with ``D = 2 g**2``.

    The result is capped at ``min(3 sqrt(n), n / 3)`` and floored at 1. A
    nearly white-noise series gives a value close to 1.
    """
    x = as_returns(returns, min_length=16)
    n = x.size
    d = x - x.mean()
    gamma0 = float(d @ d) / n
    if gamma0 <= 0.0:
        raise ValidationError("the series has zero variance")
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = min(math.ceil(math.sqrt(n)) + k_n, n - 1)
    acov = np.array([float(d[k:] @ d[: n - k]) / n for k in range(m_max + 1)])
    rho = np.abs(acov / gamma0)
    band = 2.0 * math.sqrt(math.log10(n) / n)
    m_hat = None
    for m in range(m_max - k_n + 1):
        if np.all(rho[m + 1 : m + 1 + k_n] < band):
            m_hat = m
            break
    bandwidth = m_max if m_hat is None else min(2 * max(m_hat, 1), m_max)
    k = np.arange(1, bandwidth + 1, dtype=np.float64)
    t = k / bandwidth
    window = np.where(t <= 0.5, 1.0, 2.0 * (1.0 - t))
    big_g = 2.0 * float(np.sum(window * k * acov[1 : bandwidth + 1]))
    small_g = gamma0 + 2.0 * float(np.sum(window * acov[1 : bandwidth + 1]))
    cap = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    if small_g <= 0.0:
        return float(cap)
    b = (2.0 * big_g**2 / (2.0 * small_g**2)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), cap))


@dataclass(frozen=True)
class ColumnMeans:
    """Column means of a matrix, and of each stationary-bootstrap resample of it."""

    #: ``(k,)`` sample means, one per column.
    means: FloatArray
    #: ``(n_bootstrap, k)`` means of the resamples. Every row of this used the
    #: *same* resampled time indices across all ``k`` columns, which is the
    #: whole reason this is one function rather than ``k`` calls.
    resampled: FloatArray
    #: Rows in the original matrix.
    n: int
    #: The block length used, estimated if it was not given.
    block_length: float


def bootstrap_column_means(
    matrix: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: Seed = None,
    name: str = "matrix",
    min_rows: int = 16,
) -> ColumnMeans:
    """Resample the rows of a matrix and take the column means of each resample.

    Everything any test in this library asks of the bootstrap is a function of
    these means. A pairwise difference of two columns has the mean
    ``mean_i - mean_j`` in every resample, so a test over all ``k(k-1)/2`` pairs
    needs this ``(B, k)`` array and not a ``(B, k, k)`` one — which is the
    difference between megabytes and gigabytes once ``k`` is in the dozens.

    One block length is used for every column, estimated as the average of the
    per-column estimates when it is not supplied. Resampling the columns on
    different indices would break the cross-sectional dependence, and that
    dependence is the reason a set of near-identical strategies is not twenty
    independent tries.

    Memory is bounded by resampling in chunks rather than materialising the full
    ``(B, n, k)`` gather, which at a thousand replications of a thousand rows and
    fifty columns would be 400 MB for a result that is 400 KB.
    """
    values = as_matrix(matrix, name=name, min_rows=min_rows)
    n, k = values.shape
    if int(n_bootstrap) != n_bootstrap or n_bootstrap < 100:
        raise ValidationError(f"n_bootstrap must be an integer of at least 100, got {n_bootstrap}")
    if block_length is None:
        block = float(np.mean([optimal_block_length(values[:, j]) for j in range(k)]))
    else:
        block = float(block_length)
    replications = int(n_bootstrap)
    indices = stationary_bootstrap_indices(n, block, replications, seed=as_generator(seed))
    resampled = np.empty((replications, k))
    chunk = max(1, int(4_000_000 // max(n * k, 1)))
    for lo in range(0, replications, chunk):
        hi = min(lo + chunk, replications)
        resampled[lo:hi] = values[indices[lo:hi]].mean(axis=1)
    return ColumnMeans(
        means=values.mean(axis=0), resampled=resampled, n=n, block_length=block
    )
