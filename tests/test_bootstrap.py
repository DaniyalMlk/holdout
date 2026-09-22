from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import lfilter

from holdout import InsufficientDataError, ValidationError
from holdout.bootstrap import as_generator, optimal_block_length, stationary_bootstrap_indices


def ar1(phi: float, n: int, seed: int) -> NDArray[np.float64]:
    e = np.random.default_rng(seed).normal(size=n + 200)
    x: NDArray[np.float64] = lfilter([1.0], [1.0, -phi], e)[200:]
    return x


def test_indices_are_valid_and_reproducible() -> None:
    idx = stationary_bootstrap_indices(50, 4.0, 200, seed=3)
    assert idx.shape == (200, 50)
    assert idx.min() >= 0 and idx.max() < 50
    np.testing.assert_array_equal(idx, stationary_bootstrap_indices(50, 4.0, 200, seed=3))
    assert not np.array_equal(idx, stationary_bootstrap_indices(50, 4.0, 200, seed=4))


@pytest.mark.parametrize("block", [1.0, 3.0, 10.0, 40.0])
def test_mean_block_length(block: float) -> None:
    idx = stationary_bootstrap_indices(2000, block, 50, seed=1)
    # A block continues exactly when the next index is the previous one plus one (mod n).
    continues = (idx[:, 1:] - idx[:, :-1]) % 2000 == 1
    starts = 1 + np.sum(~continues, axis=1)
    mean_length = 2000 / starts.mean()
    # A jump can land on the next index by chance (probability 1/n), which is negligible here.
    assert mean_length == pytest.approx(block, rel=0.05)


def test_block_length_one_is_the_iid_bootstrap() -> None:
    idx = stationary_bootstrap_indices(1000, 1.0, 20, seed=2)
    continues = (idx[:, 1:] - idx[:, :-1]) % 1000 == 1
    assert continues.mean() < 0.01


def test_resampled_means_are_unbiased() -> None:
    x = np.random.default_rng(0).normal(0.3, 1.0, size=400)
    idx = stationary_bootstrap_indices(400, 8.0, 4000, seed=5)
    assert x[idx].mean() == pytest.approx(x.mean(), abs=0.01)


def test_bootstrap_variance_of_the_mean_tracks_serial_dependence() -> None:
    # AR(1) with phi = 0.5: the long-run variance is (1 + phi) / (1 - phi) = 3
    # times the iid one, and the block bootstrap should recover most of that
    # while resampling single days recovers none of it.
    x = ar1(0.5, 5000, seed=7)
    iid = x[stationary_bootstrap_indices(5000, 1.0, 2000, seed=1)].mean(axis=1).var()
    block = x[stationary_bootstrap_indices(5000, 25.0, 2000, seed=1)].mean(axis=1).var()
    assert block / iid == pytest.approx(3.0, rel=0.2)


def test_index_arguments() -> None:
    with pytest.raises(InsufficientDataError):
        stationary_bootstrap_indices(1, 2.0, 10)
    with pytest.raises(ValidationError):
        stationary_bootstrap_indices(10, 0.5, 10)
    with pytest.raises(ValidationError):
        stationary_bootstrap_indices(10, 2.0, 0)
    with pytest.raises(ValidationError, match="seed"):
        as_generator("abc")  # type: ignore[arg-type]
    rng = np.random.default_rng(0)
    assert as_generator(rng) is rng


@pytest.mark.parametrize(("phi", "n"), [(0.3, 10_000), (0.5, 10_000), (0.7, 20_000)])
def test_block_length_matches_the_ar1_closed_form(phi: float, n: int) -> None:
    theory = (2 * phi / (1 - phi**2)) ** (2 / 3) * n ** (1 / 3)
    estimates = [optimal_block_length(ar1(phi, n, seed)) for seed in range(10)]
    assert np.mean(estimates) == pytest.approx(theory, rel=0.1)


def test_block_length_is_short_for_white_noise_and_capped() -> None:
    noise = np.random.default_rng(3).normal(size=5000)
    assert optimal_block_length(noise) < 3.0
    walk = np.cumsum(np.random.default_rng(4).normal(size=400))
    assert optimal_block_length(walk) <= np.ceil(min(3 * np.sqrt(400), 400 / 3))
    with pytest.raises(InsufficientDataError):
        optimal_block_length(np.arange(10.0))
    with pytest.raises(ValidationError, match="zero variance"):
        optimal_block_length(np.ones(100))
