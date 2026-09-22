"""The worked example from Bailey and López de Prado (2014), step by step.

A strategist reports a daily strategy with an annualised Sharpe ratio of 2.5
over five years (1250 days), found after 100 independent trials whose
annualised Sharpe ratios had variance 1/2. The returns have skewness -3 and
kurtosis 10. Is the strategy likely to be real?

Run with ``python examples/deflated_worked_example.py``.
"""

from __future__ import annotations

import math

from holdout import (
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)

PERIODS_PER_YEAR = 250  # the paper's convention
N_DAYS = 1250
N_TRIALS = 100


def main() -> dict[str, float]:
    # Everything is per period: de-annualise the inputs first.
    sharpe = 2.5 / math.sqrt(PERIODS_PER_YEAR)
    trials_variance = 0.5 / PERIODS_PER_YEAR
    skew, kurt = -3.0, 10.0

    benchmark = expected_maximum_sharpe(N_TRIALS, trials_variance)
    psr = probabilistic_sharpe_ratio(sharpe, N_DAYS, skewness=skew, kurtosis=kurt)
    dsr = deflated_sharpe_ratio(
        sharpe,
        N_DAYS,
        n_trials=N_TRIALS,
        trials_variance=trials_variance,
        skewness=skew,
        kurtosis=kurt,
    )
    needed = minimum_track_record_length(
        sharpe, benchmark=benchmark, confidence=0.95, skewness=skew, kurtosis=kurt
    )

    print(f"per-period Sharpe ratio:            {sharpe:.4f}")
    print(f"expected best of {N_TRIALS} unskilled trials: {benchmark:.4f}  (paper: 0.1132)")
    print(f"probabilistic Sharpe ratio, SR* = 0: {psr:.4f}")
    print(f"deflated Sharpe ratio:              {dsr:.4f}  (paper: 0.9004)")
    print(
        f"track record needed for 95% after deflation: {needed:.0f} days "
        f"({needed / PERIODS_PER_YEAR:.1f} years)"
    )
    return {"benchmark": benchmark, "psr": psr, "dsr": dsr, "needed_days": needed}


if __name__ == "__main__":
    main()
