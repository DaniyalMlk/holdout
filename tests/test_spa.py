from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from holdout import InsufficientDataError, ValidationError
from holdout.spa import reality_check, superior_predictive_ability

Maker = Callable[[np.random.Generator], NDArray[np.float64]]
T, K, REPS, B = 500, 10, 150, 300


def all_zero(g: np.random.Generator) -> NDArray[np.float64]:
    return g.normal(size=(T, K))


def one_zero_nine_poor(g: np.random.Generator) -> NDArray[np.float64]:
    d = g.normal(size=(T, K))
    d[:, 1:] -= 0.2
    return d


def one_good_nine_poor(g: np.random.Generator) -> NDArray[np.float64]:
    d = one_zero_nine_poor(g)
    d[:, 0] += 0.12
    return d


def rejection_rates(make: Maker, seed: int) -> dict[str, float]:
    g = np.random.default_rng(seed)
    hits = {"rc": 0, "spa": 0}
    for _ in range(REPS):
        d = make(g)
        hits["rc"] += reality_check(d, n_bootstrap=B, block_length=5, seed=g).pvalue <= 0.05
        hits["spa"] += (
            superior_predictive_ability(d, n_bootstrap=B, block_length=5, seed=g).consistent <= 0.05
        )
    return {k: v / REPS for k, v in hits.items()}


def test_size_when_every_strategy_is_exactly_as_good_as_the_benchmark() -> None:
    # The least favourable null. Binomial s.e. at 150 runs is 1.8 points.
    rates = rejection_rates(all_zero, seed=0)
    for name, rate in rates.items():
        assert rate <= 0.05 + 0.035, name


def test_poor_strategies_make_the_reality_check_conservative_but_not_spa() -> None:
    rates = rejection_rates(one_zero_nine_poor, seed=1)
    assert rates["rc"] <= 0.02
    assert rates["spa"] <= 0.05 + 0.035
    assert rates["spa"] >= 0.02


def test_spa_has_more_power_than_the_reality_check_among_poor_alternatives() -> None:
    # Hansen's point: nine poor strategies dilute the Reality Check.
    rates = rejection_rates(one_good_nine_poor, seed=2)
    assert rates["spa"] > rates["rc"] + 0.15
    assert rates["spa"] > 0.75


def test_spa_pvalues_are_ordered() -> None:
    g = np.random.default_rng(6)
    for _ in range(20):
        d = g.normal(size=(300, 5)) + g.normal(0, 0.05, size=5)
        r = superior_predictive_ability(d, n_bootstrap=300, block_length=4, seed=g)
        assert r.lower <= r.consistent <= r.upper


def test_seeded_runs_are_reproducible_and_block_length_is_chosen_automatically() -> None:
    d = np.random.default_rng(7).normal(size=(300, 4))
    a = superior_predictive_ability(d, n_bootstrap=200, seed=11)
    b = superior_predictive_ability(d, n_bootstrap=200, seed=11)
    assert a == b
    assert a.block_length >= 1.0
    rc = reality_check(d, n_bootstrap=200, seed=11)
    assert rc.best == int(np.argmax(d.mean(axis=0)))
    assert rc.statistic == pytest.approx(np.sqrt(300) * d.mean(axis=0).max())


def test_arguments() -> None:
    d = np.random.default_rng(8).normal(size=(100, 3))
    with pytest.raises(ValidationError, match="n_bootstrap"):
        reality_check(d, n_bootstrap=10)
    with pytest.raises(InsufficientDataError):
        reality_check(d[:10])
    flat = d.copy()
    flat[:, 1] = 0.5
    with pytest.raises(ValidationError, match="column 1"):
        superior_predictive_ability(flat, n_bootstrap=100, block_length=2, seed=0)
