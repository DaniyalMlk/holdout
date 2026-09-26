"""How little thirty parameter choices can be separated, even with a real signal.

Runs the model confidence set over both synthetic sweeps and prints what each
one leaves standing. The uncomfortable result is the second: the trending sweep
has a genuine drift in it, the deflated Sharpe ratio finds the best rule
significant, and ten years of daily data still cannot rule out all but a handful
of the parameter choices.

That is not a failure of the procedure. It is the difference between "this rule
made money" and "this rule is the one that made money", and the second claim is
the one a parameter sweep is usually read as making.
"""

from __future__ import annotations

from typing import Any

from holdout import (
    Statistic,
    deflate_trials,
    effective_number_of_trials,
    model_confidence_set,
    trial_correlation,
)
from holdout.synthetic import crossover_sweep

BOOTSTRAP = 1000


def summarise(name: str, trend: float, seed: int) -> dict[str, Any]:
    sweep = crossover_sweep(trend=trend, seed=seed)
    values = sweep.table.values
    names = list(sweep.table.names)

    effective = effective_number_of_trials(trial_correlation(values), method="eigenvalue")
    deflated = deflate_trials(values, n_trials=effective)
    sets = {}
    for statistic in (Statistic.MAX, Statistic.RANGE):
        result = model_confidence_set(
            values,
            alpha=0.10,
            statistic=statistic,
            n_bootstrap=BOOTSTRAP,
            seed=7,
            names=names,
        )
        sets[statistic.value] = result

    reference = sets[Statistic.MAX.value]
    inside = reference.included
    print(f"\n{name}: {len(names)} rules over {values.shape[0]} periods")
    print(f"  best by mean:            {reference.names[reference.best]}")
    print(f"  deflated Sharpe ratio:   {deflated.deflated:.3f}")
    for statistic, result in sets.items():
        print(f"  set at 10%, {statistic:>5} statistic: {result.included.size} of {len(names)}")
    if inside.size > 1:
        spread = (reference.performance[inside[0]] - reference.performance[inside[-1]]) * 252.0
        print(f"  annualised mean across the surviving set spans {spread:.1%}")
    return {
        "deflated": deflated.deflated,
        "max": reference.included.size,
        "range": sets[Statistic.RANGE.value].included.size,
        "models": len(names),
        "best": reference.names[reference.best],
    }


def main() -> dict[str, Any]:
    noise = summarise("Nothing to find", trend=0.0, seed=7)
    trend = summarise("A real drift", trend=0.0006, seed=7)
    print(
        "\nThe second sweep has a signal — its deflated Sharpe ratio clears the search "
        f"that produced it — and {trend['max']} of {trend['models']} rules are still in "
        "the confidence set. Reporting the single best of them as the strategy is the "
        "claim this refuses to license."
    )
    return {"noise": noise, "trend": trend}


if __name__ == "__main__":
    main()
