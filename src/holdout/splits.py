"""Cross-validation for series whose labels span several periods.

A label is the outcome a model is trained to predict — a five-day forward
return, the result of a trade held until a barrier is hit. Observation ``i``
is made at ``start[i]`` and its label is only known at ``end[i] >= start[i]``.
When two observations' label windows overlap, their labels share information,
and ordinary k-fold puts one in training and the other in test all the time.
The model is then scored on outcomes it has partly seen.

The splitters here follow López de Prado (2018, chapters 7 and 12):

* **Purging** drops every training observation whose label window overlaps
  the window spanned by a test group.
* **Embargo** additionally drops the training observations that immediately
  follow a test group, because serial correlation lets them carry
  information back across the boundary even without a label overlap.

:func:`leakage_audit` is the check: it raises if any training label overlaps
any test label in any split. Every splitter in this module is tested against
it, and plain k-fold on the same labels is tested to *fail* it.

Everything is in index space: splits hold positions ``0..n-1`` into arrays
sorted by ``start``. ``start`` and ``end`` can be any comparable numbers —
integer positions, or timestamps as floats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import HoldoutError, InsufficientDataError, ValidationError
from .series import FloatArray

__all__ = [
    "CombinatorialPurgedCV",
    "LeakageError",
    "Split",
    "combinatorial_purged_cv",
    "kfold",
    "leakage_audit",
    "number_of_paths",
    "purged_kfold",
    "walk_forward",
]

IndexArray = NDArray[np.int64]


class LeakageError(HoldoutError):
    """A training observation's label overlaps a test observation's label."""


@dataclass(frozen=True)
class Split:
    """One train/test split, as sorted positions into the observation arrays."""

    train: IndexArray
    test: IndexArray
    test_groups: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        if np.intersect1d(self.train, self.test).size:
            raise ValidationError("a split's train and test positions must be disjoint")


def _labels(
    n: int | None, start: ArrayLike | None, end: ArrayLike | None
) -> tuple[FloatArray, FloatArray]:
    if start is None and end is None:
        if n is None:
            raise ValidationError("give either n or the label start and end arrays")
        s = np.arange(n, dtype=np.float64)
        return s, s.copy()
    if start is None or end is None:
        raise ValidationError("start and end must be given together")
    s = np.asarray(start, dtype=np.float64).reshape(-1)
    e = np.asarray(end, dtype=np.float64).reshape(-1)
    if s.shape != e.shape:
        raise ValidationError(f"start has {s.size} entries but end has {e.size}")
    if n is not None and s.size != n:
        raise ValidationError(f"n is {n} but the label arrays have {s.size} entries")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(e))):
        raise ValidationError("label start and end times must be finite")
    if np.any(np.diff(s) < 0):
        raise ValidationError("observations must be sorted by label start")
    bad = np.flatnonzero(e < s)
    if bad.size:
        i = int(bad[0])
        raise ValidationError(f"label {i} ends ({e[i]:g}) before it starts ({s[i]:g})")
    return s, e


def _groups(n: int, n_groups: int) -> list[IndexArray]:
    if int(n_groups) != n_groups or n_groups < 2:
        raise ValidationError(
            f"the number of folds must be an integer of at least 2, got {n_groups}"
        )
    if n < n_groups:
        raise InsufficientDataError(f"{n} observations cannot form {n_groups} folds")
    return [g.astype(np.int64) for g in np.array_split(np.arange(n), int(n_groups))]


def _purge(
    candidates: IndexArray,
    test_groups: list[IndexArray],
    start: FloatArray,
    end: FloatArray,
    embargo: int,
) -> IndexArray:
    keep = np.ones(candidates.size, dtype=bool)
    s, e = start[candidates], end[candidates]
    n = start.size
    for group in test_groups:
        t0, t1 = start[group].min(), end[group].max()
        # Overlap of closed intervals [s, e] and [t0, t1].
        keep &= ~((s <= t1) & (e >= t0))
        if embargo:
            last = int(group.max())
            lo, hi = last + 1, min(last + embargo, n - 1)
            keep &= ~((candidates >= lo) & (candidates <= hi))
    kept: IndexArray = candidates[keep]
    return kept


def _check_embargo(embargo: int, n: int) -> int:
    if int(embargo) != embargo or embargo < 0:
        raise ValidationError(
            f"embargo must be a non-negative number of observations, got {embargo}"
        )
    if embargo >= n:
        raise ValidationError(f"an embargo of {embargo} would remove all {n} observations")
    return int(embargo)


def kfold(n: int, n_folds: int) -> list[Split]:
    """Plain contiguous k-fold with no purging — the baseline that leaks."""
    groups = _groups(n, n_folds)
    everything = np.arange(n, dtype=np.int64)
    return [
        Split(train=np.setdiff1d(everything, g), test=g, test_groups=(i,))
        for i, g in enumerate(groups)
    ]


def purged_kfold(
    n_folds: int,
    *,
    n: int | None = None,
    start: ArrayLike | None = None,
    end: ArrayLike | None = None,
    embargo: int = 0,
) -> list[Split]:
    """Contiguous k-fold with purging and an embargo of ``embargo`` observations.

    With no label arrays each label covers only its own period, and purging
    reduces to plain k-fold (plus the embargo).
    """
    s, e = _labels(n, start, end)
    size = s.size
    embargo = _check_embargo(embargo, size)
    groups = _groups(size, n_folds)
    everything = np.arange(size, dtype=np.int64)
    splits = []
    for i, g in enumerate(groups):
        train = _purge(np.setdiff1d(everything, g), [g], s, e, embargo)
        splits.append(Split(train=train, test=g, test_groups=(i,)))
    return splits


def walk_forward(
    n: int,
    *,
    train: int,
    test: int,
    step: int | None = None,
    expanding: bool = False,
    gap: int = 0,
    start: ArrayLike | None = None,
    end: ArrayLike | None = None,
) -> list[Split]:
    """Walk-forward splits: train on the past, test on the next ``test`` periods.

    ``train`` is the (initial, if ``expanding``) training length, ``step`` how
    far each window advances (default ``test``), and ``gap`` a number of
    periods skipped between the end of training and the start of testing.
    With label arrays, training observations whose labels reach into the test
    window are also purged, so a five-day forward return does not train a
    model that is then tested on those five days.
    """
    s, e = _labels(n, start, end)
    for name, value, low in (("train", train, 1), ("test", test, 1), ("gap", gap, 0)):
        if int(value) != value or value < low:
            raise ValidationError(f"{name} must be an integer of at least {low}, got {value}")
    step = test if step is None else step
    if int(step) != step or step < 1:
        raise ValidationError(f"step must be a positive integer, got {step}")
    if train + gap + test > n:
        raise InsufficientDataError(
            f"train {train} + gap {gap} + test {test} exceeds the {n} observations"
        )
    splits = []
    offset = 0
    while offset + train + gap + test <= n:
        first = 0 if expanding else offset
        window: IndexArray = np.arange(first, offset + train, dtype=np.int64)
        test_idx: IndexArray = np.arange(
            offset + train + gap, offset + train + gap + test, dtype=np.int64
        )
        train_idx = _purge(window, [test_idx], s, e, 0)
        if train_idx.size:
            splits.append(Split(train=train_idx, test=test_idx))
        offset += step
    return splits


def number_of_paths(n_groups: int, n_test_groups: int) -> int:
    """``k / N * C(N, k)``: backtest paths from combinatorial purged cross-validation."""
    if not 0 < n_test_groups < n_groups:
        raise ValidationError(
            f"n_test_groups must be between 1 and n_groups - 1, got {n_test_groups} of {n_groups}"
        )
    return math.comb(n_groups - 1, n_test_groups - 1)


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """All ``C(N, k)`` purged splits and the backtest paths they assemble into.

    Each group is tested in ``C(N - 1, k - 1)`` splits. Path ``j`` takes each
    group's ``j``-th appearance as a test group, so every path is a complete,
    non-overlapping out-of-sample history of the whole sample.
    """

    splits: tuple[Split, ...]
    groups: tuple[IndexArray, ...]
    n_test_groups: int

    @property
    def n_paths(self) -> int:
        return number_of_paths(len(self.groups), self.n_test_groups)

    def path_assignment(self) -> list[list[int]]:
        """``assignment[p][g]``: which split supplies group ``g`` on path ``p``."""
        seen = [0] * len(self.groups)
        assignment = [[-1] * len(self.groups) for _ in range(self.n_paths)]
        for index, split in enumerate(self.splits):
            for g in split.test_groups:
                assignment[seen[g]][g] = index
                seen[g] += 1
        return assignment

    def assemble_paths(self, predictions: list[ArrayLike]) -> FloatArray:
        """Stitch per-split out-of-sample values into ``(n_paths, n)`` backtest paths.

        ``predictions[i]`` holds one value per position in ``splits[i].test``,
        in the same order — typically the strategy return the model produced
        on that test observation.
        """
        if len(predictions) != len(self.splits):
            raise ValidationError(
                f"expected {len(self.splits)} prediction arrays, got {len(predictions)}"
            )
        values = []
        for i, (split, p) in enumerate(zip(self.splits, predictions, strict=True)):
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
            if arr.size != split.test.size:
                raise ValidationError(
                    f"predictions[{i}] has {arr.size} values for {split.test.size} test positions"
                )
            values.append(arr)
        n = int(sum(g.size for g in self.groups))
        paths = np.full((self.n_paths, n), np.nan)
        for p, row in enumerate(self.path_assignment()):
            for g, index in enumerate(row):
                split = self.splits[index]
                positions = self.groups[g]
                lookup = np.searchsorted(split.test, positions)
                paths[p, positions] = values[index][lookup]
        return paths


def combinatorial_purged_cv(
    n_groups: int,
    n_test_groups: int,
    *,
    n: int | None = None,
    start: ArrayLike | None = None,
    end: ArrayLike | None = None,
    embargo: int = 0,
) -> CombinatorialPurgedCV:
    """Combinatorial purged cross-validation over ``n_groups`` contiguous groups."""
    s, e = _labels(n, start, end)
    size = s.size
    embargo = _check_embargo(embargo, size)
    groups = _groups(size, n_groups)
    number_of_paths(n_groups, n_test_groups)
    everything = np.arange(size, dtype=np.int64)
    splits = []
    for chosen in combinations(range(n_groups), n_test_groups):
        test_groups = [groups[g] for g in chosen]
        test = np.concatenate(test_groups)
        train = _purge(np.setdiff1d(everything, test), test_groups, s, e, embargo)
        splits.append(Split(train=train, test=test, test_groups=tuple(chosen)))
    return CombinatorialPurgedCV(
        splits=tuple(splits), groups=tuple(groups), n_test_groups=int(n_test_groups)
    )


def leakage_audit(
    splits: list[Split] | tuple[Split, ...],
    start: ArrayLike,
    end: ArrayLike,
) -> None:
    """Raise :class:`LeakageError` if any training label overlaps any test label.

    Checks label windows pairwise, not just against the test span, so it is
    independent of how the splitter purged.
    """
    s, e = _labels(None, start, end)
    for index, split in enumerate(splits):
        if split.test.size == 0 or split.train.size == 0:
            continue
        ts, te = s[split.test], e[split.test]
        order = np.argsort(ts)
        ts, te = ts[order], te[order]
        running_end = np.maximum.accumulate(te)
        for i in split.train:
            # Any test label with start <= e_i and end >= s_i overlaps.
            k = int(np.searchsorted(ts, e[i], side="right"))
            if k and running_end[k - 1] >= s[i]:
                j = int(split.test[order][np.flatnonzero((ts[:k] <= e[i]) & (te[:k] >= s[i]))[0]])
                raise LeakageError(
                    f"split {index}: training observation {int(i)} (label {s[i]:g}-{e[i]:g}) "
                    f"overlaps test observation {j} (label {s[j]:g}-{e[j]:g})"
                )
