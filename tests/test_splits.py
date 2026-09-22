from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from holdout import InsufficientDataError, ValidationError
from holdout.splits import (
    LeakageError,
    Split,
    combinatorial_purged_cv,
    kfold,
    leakage_audit,
    number_of_paths,
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


@pytest.mark.parametrize(
    ("groups", "k", "paths"), [(6, 2, 5), (10, 2, 9), (8, 3, 21), (5, 1, 1), (4, 3, 3)]
)
def test_number_of_paths_matches_the_closed_form(groups: int, k: int, paths: int) -> None:
    assert number_of_paths(groups, k) == paths
    assert number_of_paths(groups, k) == k * math.comb(groups, k) // groups
    cv = combinatorial_purged_cv(groups, k, n=groups * 7)
    assert len(cv.splits) == math.comb(groups, k)
    assert cv.n_paths == paths


def test_number_of_paths_arguments() -> None:
    with pytest.raises(ValidationError):
        number_of_paths(6, 0)
    with pytest.raises(ValidationError):
        number_of_paths(6, 6)


def test_every_path_covers_each_observation_exactly_once() -> None:
    n = 60
    start, end = forward_labels(n, np.full(n, 2))
    cv = combinatorial_purged_cv(6, 2, start=start, end=end, embargo=1)
    leakage_audit(cv.splits, start, end)
    assignment = cv.path_assignment()
    assert len(assignment) == 5
    for row in assignment:
        assert all(index >= 0 for index in row)
        for g, index in enumerate(row):
            assert g in cv.splits[index].test_groups
    # Each split contributes each of its test groups to exactly one path.
    used = sorted((index, g) for row in assignment for g, index in enumerate(row))
    expected = sorted((i, g) for i, s in enumerate(cv.splits) for g in s.test_groups)
    assert used == expected


def test_assembled_paths_put_each_prediction_in_its_place() -> None:
    n = 48
    cv = combinatorial_purged_cv(6, 2, n=n)
    # Encode (split, position) in the prediction so placement can be checked.
    predictions = [1000.0 * i + s.test for i, s in enumerate(cv.splits)]
    paths = cv.assemble_paths(predictions)  # type: ignore[arg-type]
    assert paths.shape == (5, n)
    assert not np.any(np.isnan(paths))
    np.testing.assert_array_equal(paths % 1000, np.tile(np.arange(n), (5, 1)))
    for p, row in enumerate(cv.path_assignment()):
        for g, index in enumerate(row):
            assert np.all(paths[p, cv.groups[g]] // 1000 == index)


def test_assemble_paths_arguments() -> None:
    cv = combinatorial_purged_cv(4, 2, n=20)
    with pytest.raises(ValidationError, match="prediction arrays"):
        cv.assemble_paths([np.zeros(10)])
    bad = [np.zeros(s.test.size) for s in cv.splits]
    bad[2] = np.zeros(3)
    with pytest.raises(ValidationError, match=r"predictions\[2\]"):
        cv.assemble_paths(bad)  # type: ignore[arg-type]


def test_cpcv_purges_around_every_test_group() -> None:
    start, end = forward_labels(60, np.full(60, 4))
    cv = combinatorial_purged_cv(6, 2, start=start, end=end, embargo=2)
    split = next(s for s in cv.splits if s.test_groups == (1, 4))
    # Groups are 10 wide, so the test groups are 10..19 and 40..49, and their
    # label spans are [10, 23] and [40, 53]. A training label [s, s + 4]
    # overlaps them for s in 6..23 and 36..53; the two-period embargoes (20-21,
    # 50-51) fall inside those ranges already.
    np.testing.assert_array_equal(split.train, np.r_[0:6, 24:36, 54:60])
    leakage_audit(cv.splits, start, end)


def test_embargo_extends_beyond_the_purge_when_labels_are_short() -> None:
    cv = combinatorial_purged_cv(6, 2, n=60, embargo=3)
    split = next(s for s in cv.splits if s.test_groups == (1, 4))
    np.testing.assert_array_equal(split.train, np.r_[0:10, 23:40, 53:60])
