from __future__ import annotations

import numpy as np
import pytest

from holdout import ValidationError, kurtosis
from holdout.synthetic import crossover_sweep, garch_returns
from holdout.trials import effective_number_of_trials, trial_correlation


def test_garch_returns_have_fat_tails_and_volatility_clustering() -> None:
    x = garch_returns(20_000, seed=1)
    assert kurtosis(x) > 5.0
    sq = x**2 - np.mean(x**2)
    lag1 = float(sq[1:] @ sq[:-1] / (sq @ sq))
    assert lag1 > 0.1
    # Unconditional variance omega / (1 - alpha - beta) = 5e-5.
    assert np.var(x) == pytest.approx(5e-5, rel=0.25)


def test_garch_arguments() -> None:
    with pytest.raises(ValidationError):
        garch_returns(10, alpha=0.5, beta=0.6)
    with pytest.raises(ValidationError):
        garch_returns(10, dof=2.0)


def test_sweep_shape_and_names() -> None:
    sweep = crossover_sweep(1000, fast=(5, 10), slow=(20, 50), seed=3)
    assert sweep.table.names == ["ma5_20", "ma5_50", "ma10_20", "ma10_50"]
    assert sweep.table.values.shape == (1000, 4)
    assert sweep.market.shape == (1000,)
    assert np.all(np.isfinite(sweep.table.values))
    # Each strategy's return is plus or minus the market's (or zero on a tie).
    ratio = np.abs(sweep.table.values) - np.abs(sweep.market)[:, None]
    assert np.all((np.abs(ratio) < 1e-15) | (sweep.table.values == 0))


def test_sweep_members_are_highly_correlated() -> None:
    sweep = crossover_sweep(seed=7)
    c = trial_correlation(sweep.table.values)
    assert effective_number_of_trials(c, method="eigenvalue") < len(sweep.table.names) / 2


def test_sweep_is_reproducible_and_validates_length() -> None:
    a = crossover_sweep(600, fast=(5,), slow=(40,), seed=1)
    b = crossover_sweep(600, fast=(5,), slow=(40,), seed=1)
    np.testing.assert_array_equal(a.table.values, b.table.values)
    with pytest.raises(ValidationError, match="n_days"):
        crossover_sweep(100)
