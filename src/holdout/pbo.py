"""Probability of backtest overfitting (Bailey, Borwein, López de Prado and Zhu 2017).

Combinatorially symmetric cross-validation (CSCV) splits the ``T`` rows of a
``(T, N)`` matrix of strategy returns into ``S`` contiguous blocks and, for
every one of the ``C(S, S/2)`` ways of choosing half of them:

1. evaluates every strategy on the chosen half (in-sample) and on the rest
   (out-of-sample);
2. takes the in-sample winner and finds its relative rank ``w`` out-of-sample,
   ``rank / (N + 1)`` with rank 1 the worst;
3. records the logit ``log(w / (1 - w))``.

The probability of backtest overfitting is the share of partitions whose
logit is at most zero — the winner landing at or below the out-of-sample
median. Pure noise gives about one half; a search that finds something real
gives close to zero; a search whose winners systematically fail gives close
to one.

Every partition uses each block exactly once on each side across the whole
set of combinations, which is the "symmetric" part: no block is privileged as
the hold-out.

For the Sharpe ratio and the mean, partitions are evaluated from per-block
sums, counts and sums of squares, so the cost is one pass over the data plus
``C(S, S/2)`` small matrix products. Any other metric takes the general path,
which slices the data once per partition.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray, as_matrix

__all__ = ["PBOResult", "cscv_partitions", "probability_of_backtest_overfitting"]

Metric = Literal["sharpe", "mean"] | Callable[[FloatArray], FloatArray]
_MAX_PARTITIONS = 1_000_000


@dataclass(frozen=True)
class PBOResult:
    """Outcome of combinatorially symmetric cross-validation.

    One entry per partition in ``logits``, ``selected``, ``in_sample`` and
    ``out_of_sample``: the logit of the in-sample winner's out-of-sample
    relative rank, which strategy that winner was, and its metric on each
    side.
    """

    pbo: float
    logits: FloatArray
    selected: NDArray[np.int64]
    in_sample: FloatArray
    out_of_sample: FloatArray
    n_blocks: int
    n_strategies: int

    @property
    def n_partitions(self) -> int:
        return int(self.logits.size)

    @property
    def probability_of_loss(self) -> float:
        """Share of partitions in which the in-sample winner's out-of-sample metric is negative."""
        return float(np.mean(self.out_of_sample < 0.0))

    @property
    def degradation(self) -> tuple[float, float, float]:
        """``(slope, intercept, r_squared)`` of out-of-sample on in-sample metric.

        A slope near zero or negative means in-sample performance says nothing
        about — or the opposite of — what follows. ``nan`` values mean the
        in-sample metric did not vary across partitions.
        """
        x, y = self.in_sample, self.out_of_sample
        dx = x - x.mean()
        sxx = float(dx @ dx)
        if sxx <= 0.0:
            return math.nan, math.nan, math.nan
        slope = float(dx @ (y - y.mean())) / sxx
        intercept = float(y.mean()) - slope * float(x.mean())
        residual = y - (intercept + slope * x)
        syy = float((y - y.mean()) @ (y - y.mean()))
        r2 = 1.0 - float(residual @ residual) / syy if syy > 0.0 else math.nan
        return slope, intercept, r2

    def selection_frequency(self) -> FloatArray:
        """How often each strategy was the in-sample winner, as a share of partitions."""
        counts = np.bincount(self.selected, minlength=self.n_strategies)
        return counts / float(self.n_partitions)


def cscv_partitions(n_blocks: int) -> list[tuple[int, ...]]:
    """Every choice of ``n_blocks / 2`` training blocks, in lexicographic order."""
    if int(n_blocks) != n_blocks or n_blocks < 2 or n_blocks % 2:
        raise ValidationError(f"n_blocks must be an even integer of at least 2, got {n_blocks}")
    count = math.comb(int(n_blocks), int(n_blocks) // 2)
    if count > _MAX_PARTITIONS:
        raise ValidationError(
            f"n_blocks = {n_blocks} gives {count:,} partitions; use at most "
            f"{_MAX_PARTITIONS:,} (n_blocks <= 22)"
        )
    return list(combinations(range(int(n_blocks)), int(n_blocks) // 2))


def _block_bounds(rows: int, n_blocks: int) -> NDArray[np.int64]:
    # Contiguous, as equal as possible; the first rows % n_blocks blocks get one extra row.
    sizes = np.full(n_blocks, rows // n_blocks, dtype=np.int64)
    sizes[: rows % n_blocks] += 1
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def probability_of_backtest_overfitting(
    returns: ArrayLike,
    *,
    n_blocks: int = 16,
    metric: Metric = "sharpe",
) -> PBOResult:
    """Run CSCV over a ``(periods, strategies)`` return matrix.

    ``metric`` is ``"sharpe"`` (default), ``"mean"``, or a callable mapping a
    ``(rows, strategies)`` array to one score per strategy, higher being better.
    ``n_blocks`` must be even; the paper uses 16 (12,870 partitions).

    Blocks are contiguous so that serial dependence stays inside a block
    rather than straddling the in-sample/out-of-sample boundary everywhere.
    """
    x = as_matrix(returns, min_columns=2)
    rows, n = x.shape
    partitions = cscv_partitions(n_blocks)
    if rows < 2 * n_blocks:
        raise InsufficientDataError(
            f"{rows} periods cannot fill {n_blocks} blocks of at least two periods each"
        )
    bounds = _block_bounds(rows, n_blocks)
    all_blocks = frozenset(range(n_blocks))

    if isinstance(metric, str):
        if metric not in ("sharpe", "mean"):
            raise ValidationError(f"metric must be 'sharpe', 'mean' or a callable, got {metric!r}")
        # Centre each column first: the variance below is a difference of sums,
        # which cancels badly when the mean is large relative to the spread.
        centre = x.mean(axis=0)
        centred = x - centre
        sums = np.add.reduceat(centred, bounds[:-1], axis=0)
        squares = np.add.reduceat(centred * centred, bounds[:-1], axis=0)
        counts = np.diff(bounds).astype(np.float64)
        # One-hot membership of training blocks, one row per partition.
        membership = np.zeros((len(partitions), n_blocks))
        for i, chosen in enumerate(partitions):
            membership[i, list(chosen)] = 1.0
        outside = 1.0 - membership
        r_in = _from_sums(
            membership @ sums, membership @ squares, membership @ counts, centre, metric
        )
        r_out = _from_sums(outside @ sums, outside @ squares, outside @ counts, centre, metric)
    else:
        spans = [np.arange(bounds[b], bounds[b + 1]) for b in range(n_blocks)]
        r_in = np.empty((len(partitions), n))
        r_out = np.empty((len(partitions), n))
        for i, chosen in enumerate(partitions):
            train = np.concatenate([spans[b] for b in chosen])
            test = np.concatenate([spans[b] for b in sorted(all_blocks - set(chosen))])
            r_in[i] = _score(metric, x[train], n)
            r_out[i] = _score(metric, x[test], n)

    rows_index = np.arange(len(partitions))
    selected = np.argmax(r_in, axis=1).astype(np.int64)
    winner_out = r_out[rows_index, selected]
    below = np.sum(r_out < winner_out[:, None], axis=1)
    ties = np.sum(r_out == winner_out[:, None], axis=1)
    # Rank 1 is the worst; ties share the average rank.
    w = (below + (ties + 1.0) / 2.0) / (n + 1.0)
    logits = np.log(w / (1.0 - w))
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        logits=logits,
        selected=selected,
        in_sample=r_in[rows_index, selected],
        out_of_sample=winner_out,
        n_blocks=int(n_blocks),
        n_strategies=int(n),
    )


def _from_sums(
    total: FloatArray,
    squares: FloatArray,
    count: FloatArray,
    centre: FloatArray,
    metric: str,
) -> FloatArray:
    """Metric per (partition, strategy) from sums of centred returns."""
    c = count[:, None]
    shifted_mean = total / c
    mean: FloatArray = shifted_mean + centre
    if metric == "mean":
        return mean
    variance = (squares - c * shifted_mean * shifted_mean) / (c - 1.0)
    scale = np.maximum(squares / c, np.finfo(np.float64).tiny)
    flat = (variance <= 1e-14 * scale) | (squares <= 0.0)
    if np.any(flat):
        column = int(np.argwhere(flat)[0][1])
        raise ValidationError(
            f"strategy {column} has zero variance on part of the sample, "
            "so its Sharpe ratio is undefined there"
        )
    result: FloatArray = mean / np.sqrt(variance)
    return result


def _score(metric: Callable[[FloatArray], FloatArray], data: FloatArray, n: int) -> FloatArray:
    out = np.asarray(metric(data), dtype=np.float64).reshape(-1)
    if out.size != n:
        raise ValidationError(f"metric returned {out.size} scores for {n} strategies")
    if not np.all(np.isfinite(out)):
        raise ValidationError("metric returned a non-finite score")
    return out
