"""The trial mean: an option on the deflated ratio that nothing ever supplied.

``deflated_sharpe_ratio`` takes ``trials_mean`` and forwards it to
:func:`expected_maximum_sharpe` as the mean of the trial Sharpe ratios. Under
the paper's null the trials have no skill and the mean is zero, which is what
``deflate_trials`` reports and what every call in the repository took, so the
parameter's one job -- moving the benchmark the selected strategy has to clear
-- had never been checked.

It is a benchmark shift and nothing else, and these tests say so in the two
ways that can come apart: that the shift is exactly the mean given, for every
trial count and both methods, and that the deflated ratio with a mean agrees
with the probabilistic ratio run against the shifted benchmark.
"""

from __future__ import annotations

import pytest

from holdout import ValidationError
from holdout.deflated import (
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
)

VARIANCE = 0.04


@pytest.mark.parametrize("method", ["approximation", "exact"])
@pytest.mark.parametrize("n_trials", [1, 2, 10, 100, 1000])
@pytest.mark.parametrize("mean", [-0.2, 0.0, 0.3])
def test_the_mean_shifts_the_expected_maximum_and_does_nothing_else(
    method: str, n_trials: int, mean: float
) -> None:
    centred = expected_maximum_sharpe(n_trials, VARIANCE, method=method)  # type: ignore[arg-type]
    shifted = expected_maximum_sharpe(n_trials, VARIANCE, mean=mean, method=method)  # type: ignore[arg-type]
    assert shifted - centred == pytest.approx(mean, rel=1e-12, abs=1e-15)


def test_one_trial_is_the_mean_itself() -> None:
    # With a single trial there is no maximum to take, so the expected maximum
    # is the mean -- and the deflated ratio becomes the probabilistic ratio
    # against it. The dispersion of the trials is irrelevant here, which is the
    # claim the second assertion makes.
    assert expected_maximum_sharpe(1, VARIANCE, mean=0.25) == 0.25
    assert expected_maximum_sharpe(1, 9.0, mean=0.25) == 0.25
    deflated = deflated_sharpe_ratio(
        0.6, 250, n_trials=1, trials_variance=VARIANCE, trials_mean=0.25
    )
    assert deflated == probabilistic_sharpe_ratio(0.6, 250, benchmark=0.25)


@pytest.mark.parametrize("mean", [-0.1, 0.0, 0.15, 0.4])
def test_the_deflated_ratio_is_the_probabilistic_ratio_against_the_shifted_benchmark(
    mean: float,
) -> None:
    benchmark = expected_maximum_sharpe(50, VARIANCE, mean=mean)
    assert deflated_sharpe_ratio(
        0.6, 250, n_trials=50, trials_variance=VARIANCE, trials_mean=mean
    ) == probabilistic_sharpe_ratio(0.6, 250, benchmark=benchmark)


def test_a_higher_trial_mean_can_only_lower_the_deflated_ratio() -> None:
    # Trials that made money on average are a higher bar than trials that did
    # not, so the selected strategy's deflated ratio falls. Read rather than
    # predicted: at a mean of zero the benchmark is 0.4553 and the ratio
    # 0.9822, and by a mean of 0.4 the benchmark has passed the strategy's own
    # 0.6 and the ratio is 0.0001.
    ratios = [
        deflated_sharpe_ratio(0.6, 250, n_trials=50, trials_variance=VARIANCE, trials_mean=m)
        for m in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[2] == pytest.approx(0.9822, abs=0.0001)
    assert ratios[-1] == pytest.approx(0.0001, abs=0.0001)
    # The spread matters: a parameter that were ignored would leave seven equal
    # numbers, which the ordering assertion above would not notice.
    assert ratios[0] - ratios[-1] > 0.9


def test_the_mean_has_to_be_a_number() -> None:
    with pytest.raises(ValidationError, match="mean must be finite"):
        deflated_sharpe_ratio(
            0.6, 250, n_trials=50, trials_variance=VARIANCE, trials_mean=float("nan")
        )
