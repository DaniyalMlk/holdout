from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from holdout import InsufficientDataError, ValidationError
from holdout.splits import (
    LeakageError,
    Split,
    kfold,
    leakage_audit,
    purged_kfold,
    walk_forward,
)


def forward_labels(
    n: int, horizon: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    start = np.arange(n, dtype=np.float64)
    return start, np.minimum(start + horizon, n - 1)


def test_plain_kfold_leaks_when_labels_overlap() -> None:
    start, end = forward_labels(100, np.full(100, 5))
    with pytest.raises(LeakageError, match="overlaps test observation"):
        leakage_audit(kfold(100, 5), start, end)


def test_plain_kfold_is_clean_for_single_period_labels() -> None:
    start = np.arange(100.0)
    leakage_audit(kfold(100, 5), start, start)


@settings(max_examples=60, deadline=None)
@given(
    st.integers(min_value=30, max_value=300),
    st.integers(min_value=2, max_value=8),
    st.integers(min_value=0, max_value=20),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=2**31),
)
def test_purged_kfold_never_leaks(
    n: int, folds: int, max_horizon: int, embargo: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    start, end = forward_labels(n, rng.integers(0, max_horizon + 1, size=n))
    splits = purged_kfold(folds, start=start, end=end, embargo=min(embargo, n - 1))
    leakage_audit(splits, start, end)
    covered = np.sort(np.concatenate([s.test for s in splits]))
    np.testing.assert_array_equal(covered, np.arange(n))


def test_purging_removes_exactly_the_overlapping_labels() -> None:
    # Ten observations, labels three periods long; test fold is 4..5.
    start, end = forward_labels(10, np.full(10, 3))
    splits = purged_kfold(5, start=start, end=end)
    fold = splits[2]
    np.testing.assert_array_equal(fold.test, [4, 5])
    # Label [s, s+3] overlaps [4, 8] (test span: start 4, end 5 + 3) for s = 1..8.
    np.testing.assert_array_equal(fold.train, [0, 9])


def test_embargo_drops_the_observations_after_each_test_fold() -> None:
    splits = purged_kfold(4, n=40, embargo=3)
    np.testing.assert_array_equal(splits[0].train, np.arange(13, 40))
    np.testing.assert_array_equal(splits[1].train, np.r_[0:10, 23:40])
    np.testing.assert_array_equal(splits[3].train, np.arange(0, 30))


def test_label_validation() -> None:
    with pytest.raises(ValidationError, match="sorted"):
        purged_kfold(2, start=[2, 1, 3], end=[2, 1, 3])
    with pytest.raises(ValidationError, match="ends"):
        purged_kfold(2, start=[0, 1, 2], end=[0, 0, 2])
    with pytest.raises(ValidationError, match="together"):
        purged_kfold(2, start=[0, 1])
    with pytest.raises(ValidationError, match="either n"):
        purged_kfold(2)
    with pytest.raises(ValidationError, match="finite"):
        purged_kfold(2, start=[0, np.nan], end=[0, 1])
    with pytest.raises(ValidationError, match="entries"):
        purged_kfold(2, n=3, start=[0, 1], end=[0, 1])
    with pytest.raises(ValidationError, match="has 2 entries but end has 3"):
        purged_kfold(2, start=[0, 1], end=[0, 1, 2])
    with pytest.raises(InsufficientDataError):
        purged_kfold(5, n=3)
    with pytest.raises(ValidationError, match="folds"):
        purged_kfold(1, n=10)
    with pytest.raises(ValidationError, match="embargo"):
        purged_kfold(2, n=10, embargo=-1)
    with pytest.raises(ValidationError, match="embargo"):
        purged_kfold(2, n=10, embargo=10)


def test_split_rejects_overlap() -> None:
    with pytest.raises(ValidationError, match="disjoint"):
        Split(train=np.array([1, 2], dtype=np.int64), test=np.array([2, 3], dtype=np.int64))


def test_rolling_walk_forward() -> None:
    splits = walk_forward(20, train=8, test=4)
    assert [(s.train[0], s.train[-1], s.test[0], s.test[-1]) for s in splits] == [
        (0, 7, 8, 11),
        (4, 11, 12, 15),
        (8, 15, 16, 19),
    ]


def test_expanding_walk_forward_with_gap_and_step() -> None:
    splits = walk_forward(30, train=10, test=5, step=3, expanding=True, gap=2)
    for s in splits:
        assert s.train[0] == 0
        assert s.test[0] - s.train[-1] == 3
        assert s.test.size == 5
    assert len(splits) == 5
    sizes = [s.train.size for s in splits]
    assert sizes == [10, 13, 16, 19, 22]


def test_walk_forward_purges_labels_that_reach_into_the_test_window() -> None:
    start, end = forward_labels(40, np.full(40, 5))
    splits = walk_forward(40, train=20, test=5, start=start, end=end)
    leakage_audit(splits, start, end)
    # The last five training labels would end inside the test window.
    np.testing.assert_array_equal(splits[0].train, np.arange(0, 15))
    unpurged = walk_forward(40, train=20, test=5)
    with pytest.raises(LeakageError):
        leakage_audit(unpurged, start, end)


def test_walk_forward_arguments() -> None:
    with pytest.raises(InsufficientDataError):
        walk_forward(10, train=8, test=4)
    for kwargs in (
        {"train": 0, "test": 2},
        {"train": 2, "test": 0},
        {"train": 2, "test": 2, "gap": -1},
        {"train": 2, "test": 2, "step": 0},
    ):
        with pytest.raises(ValidationError):
            walk_forward(10, **kwargs)  # type: ignore[arg-type]
