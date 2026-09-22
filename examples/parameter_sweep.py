"""Every statistic in the library applied to one parameter sweep.

Thirty moving-average crossover rules are backtested on ten years of
simulated prices, twice: once on a market with nothing to find, and once on
a market with a slow cyclical drift that crossover rules can exploit. The
best in-sample rule looks respectable in both. The question is which
statistics can tell the two apart.

Run with ``python examples/parameter_sweep.py``.
"""

from __future__ import annotations

import math

import numpy as np

from holdout import (
    deflate_trials,
    effective_number_of_trials,
    estimate_sharpe,
    haircut_sharpe,
    probability_of_backtest_overfitting,
    superior_predictive_ability,
    trial_correlation,
)
from holdout.synthetic import crossover_sweep

ROOT = math.sqrt(252)


def assess(label: str, trend: float) -> dict[str, float]:
    sweep = crossover_sweep(trend=trend, seed=7)
    x = sweep.table.values
    best = int(np.argmax([estimate_sharpe(x[:, j]).value for j in range(x.shape[1])]))
    est = estimate_sharpe(x[:, best], periods_per_year=252)
    n_eff = effective_number_of_trials(trial_correlation(x), method="eigenvalue")
    deflated = deflate_trials(x, n_trials=n_eff)
    haircut = haircut_sharpe(est.value, est.n, n_tests=round(n_eff))
    pbo = probability_of_backtest_overfitting(x)
    spa = superior_predictive_ability(x, n_bootstrap=1000, seed=0)

    print(f"\n{label}")
    print(f"  best rule {sweep.table.names[best]}: Sharpe {est.annualised:.2f} annualised")
    print(f"  {x.shape[1]} rules, {n_eff:.1f} effective trials")
    psr, dsr = deflated.probabilistic, deflated.deflated
    print(f"  probabilistic Sharpe ratio {psr:.3f}, deflated {dsr:.3f}")
    cut = haircut.adjusted_sharpe * ROOT
    print(f"  haircut Sharpe ratio {cut:.2f} ({haircut.haircut:.0%} removed)")
    print(f"  probability of backtest overfitting {pbo.pbo:.3f}")
    print(f"  probability the selected rule loses out of sample {pbo.probability_of_loss:.3f}")
    print(f"  SPA p-value {spa.consistent:.3f}")
    return {
        "sharpe": est.annualised,
        "deflated": deflated.deflated,
        "pbo": pbo.pbo,
        "loss": pbo.probability_of_loss,
        "spa": spa.consistent,
        "haircut": haircut.haircut,
    }


def main() -> dict[str, dict[str, float]]:
    return {
        "noise": assess("market with nothing to find", 0.0),
        "trend": assess("market with a cyclical drift", 0.0006),
    }


if __name__ == "__main__":
    main()
