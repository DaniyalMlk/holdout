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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import InsufficientDataError, ValidationError
from .series import as_returns

__all__ = ["as_generator", "optimal_block_length", "stationary_bootstrap_indices"]

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
