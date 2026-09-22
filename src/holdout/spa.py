"""Tests of superior predictive ability.

The question: of ``K`` strategies, does *any* beat a benchmark once the fact
that ``K`` were tried is accounted for — and if so, which? The input is a
``(periods, K)`` matrix of **performance differentials**, strategy return
minus benchmark return (or benchmark loss minus strategy loss), so that
higher is better and the null hypothesis is ``max_k E[d_k] <= 0``.

* **White's Reality Check** (2000) bootstraps the distribution of
  ``max_k sqrt(n) (mean*_k - mean_k)``. It is valid, but it recentres every
  strategy to mean zero, so a hopeless strategy is treated as a contender
  for the maximum: adding poor strategies makes it more conservative, and it
  is not studentised, so the noisiest strategy dominates.
* **Hansen's SPA test** (2005) fixes both. Statistics are studentised, so a
  noisy strategy does not dominate the maximum; and a strategy clearly worse
  than the benchmark is not recentred, so in the bootstrap it keeps its
  negative mean and cannot reach the maximum. Three p-values bound the truth:
  *lower* (recentre at ``max(mean, 0)``), *consistent* (recentre only the
  strategies whose t-statistic is above ``-sqrt(2 log log n)``) and *upper*
  (recentre everything: the studentised Reality Check).
* **Romano–Wolf** (2005) stepdown answers "which": it rejects the strategies
  whose statistic beats the bootstrap maximum over the ones not yet rejected,
  removes them, and repeats, controlling family-wise error throughout.

Every test resamples rows with the stationary bootstrap, so all ``K``
strategies are resampled together and their cross-correlation is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bootstrap import Seed, as_generator, optimal_block_length, stationary_bootstrap_indices
from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray, as_matrix, check_probability

__all__ = [
    "RealityCheck",
    "RomanoWolf",
    "SPATest",
    "reality_check",
    "romano_wolf",
    "superior_predictive_ability",
]


@dataclass(frozen=True)
class _Resampled:
    means: FloatArray  # (K,)
    boot_means: FloatArray  # (B, K)
    omega: FloatArray  # (K,) standard deviation of sqrt(n) * mean
    n: int
    block_length: float


def _resample(
    differentials: ArrayLike,
    n_bootstrap: int,
    block_length: float | None,
    seed: Seed,
) -> _Resampled:
    d = as_matrix(differentials, name="differentials", min_rows=16)
    n, k = d.shape
    if int(n_bootstrap) != n_bootstrap or n_bootstrap < 100:
        raise ValidationError(f"n_bootstrap must be an integer of at least 100, got {n_bootstrap}")
    if block_length is None:
        # One length for all columns, so they are resampled with the same rows.
        block = float(np.mean([optimal_block_length(d[:, j]) for j in range(k)]))
    else:
        block = float(block_length)
    rng = as_generator(seed)
    indices = stationary_bootstrap_indices(n, block, int(n_bootstrap), seed=rng)
    means = d.mean(axis=0)
    boot = np.empty((int(n_bootstrap), k))
    # Resample in chunks to bound memory at roughly 32 MB.
    chunk = max(1, int(4_000_000 // max(n * k, 1)))
    for lo in range(0, int(n_bootstrap), chunk):
        hi = min(lo + chunk, int(n_bootstrap))
        boot[lo:hi] = d[indices[lo:hi]].mean(axis=1)
    omega = np.sqrt(n) * boot.std(axis=0, ddof=1)
    flat = np.flatnonzero(omega <= 0.0)
    if flat.size:
        raise ValidationError(f"differential column {int(flat[0])} has no bootstrap variation")
    return _Resampled(means=means, boot_means=boot, omega=omega, n=n, block_length=block)


@dataclass(frozen=True)
class RealityCheck:
    """White's Reality Check: ``statistic`` is ``max_k sqrt(n) mean_k``."""

    statistic: float
    pvalue: float
    best: int
    block_length: float
    n_bootstrap: int


def reality_check(
    differentials: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: Seed = None,
) -> RealityCheck:
    """White's (2000) Reality Check for data snooping."""
    r = _resample(differentials, n_bootstrap, block_length, seed)
    root_n = math.sqrt(r.n)
    statistic = float(np.max(root_n * r.means))
    null = np.max(root_n * (r.boot_means - r.means), axis=1)
    return RealityCheck(
        statistic=statistic,
        pvalue=float(np.mean(null >= statistic)),
        best=int(np.argmax(r.means)),
        block_length=r.block_length,
        n_bootstrap=int(n_bootstrap),
    )


@dataclass(frozen=True)
class SPATest:
    """Hansen's test for superior predictive ability.

    ``consistent`` is the p-value to report; ``lower`` and ``upper`` bound it.
    ``upper`` is the studentised Reality Check.
    """

    statistic: float
    lower: float
    consistent: float
    upper: float
    best: int
    block_length: float
    n_bootstrap: int


def superior_predictive_ability(
    differentials: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: Seed = None,
) -> SPATest:
    """Hansen's (2005) SPA test with lower, consistent and upper p-values."""
    r = _resample(differentials, n_bootstrap, block_length, seed)
    root_n = math.sqrt(r.n)
    t = root_n * r.means / r.omega
    statistic = float(max(np.max(t), 0.0))
    threshold = -math.sqrt(2.0 * math.log(math.log(r.n)))
    centres = {
        "lower": np.maximum(r.means, 0.0),
        "consistent": np.where(t >= threshold, r.means, 0.0),
        "upper": r.means,
    }
    pvalues = {}
    for name, mu in centres.items():
        z = root_n * (r.boot_means - mu) / r.omega
        null = np.maximum(np.max(z, axis=1), 0.0)
        pvalues[name] = float(np.mean(null >= statistic))
    return SPATest(
        statistic=statistic,
        lower=pvalues["lower"],
        consistent=pvalues["consistent"],
        upper=pvalues["upper"],
        best=int(np.argmax(t)),
        block_length=r.block_length,
        n_bootstrap=int(n_bootstrap),
    )


@dataclass(frozen=True)
class RomanoWolf:
    """Romano–Wolf stepdown: adjusted p-values and the strategies rejected at ``alpha``."""

    statistics: FloatArray
    adjusted_pvalues: FloatArray
    alpha: float
    block_length: float
    n_bootstrap: int

    @property
    def rejected(self) -> NDArray[np.int64]:
        """Indices of strategies found to beat the benchmark, strongest first."""
        hits = np.flatnonzero(self.adjusted_pvalues <= self.alpha)
        order = np.argsort(-self.statistics[hits], kind="stable")
        result: NDArray[np.int64] = hits[order].astype(np.int64)
        return result


def romano_wolf(
    differentials: ArrayLike,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: Seed = None,
) -> RomanoWolf:
    """Romano–Wolf (2005) studentised stepdown for which strategies beat the benchmark.

    Adjusted p-values follow the stepdown construction: sort the statistics
    in decreasing order; the ``j``-th strategy's p-value is the bootstrap
    probability that the maximum over it and every weaker strategy exceeds
    its statistic, made monotone. Rejecting where the adjusted p-value is at
    most ``alpha`` controls the family-wise error rate at ``alpha``.
    """
    check_probability(alpha, "alpha")
    r = _resample(differentials, n_bootstrap, block_length, seed)
    k = r.means.size
    if k < 1:
        raise InsufficientDataError("at least one strategy is needed")
    root_n = math.sqrt(r.n)
    t = root_n * r.means / r.omega
    z = root_n * (r.boot_means - r.means) / r.omega
    order = np.argsort(-t, kind="stable")
    # Maximum over the j-th strongest and everything weaker: a reversed running max.
    tail_max = np.maximum.accumulate(z[:, order][:, ::-1], axis=1)[:, ::-1]
    raw = np.mean(tail_max >= t[order][None, :], axis=0)
    adjusted_sorted = np.maximum.accumulate(raw)
    adjusted = np.empty(k)
    adjusted[order] = adjusted_sorted
    return RomanoWolf(
        statistics=t,
        adjusted_pvalues=adjusted,
        alpha=float(alpha),
        block_length=r.block_length,
        n_bootstrap=int(n_bootstrap),
    )
