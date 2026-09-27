"""How often each variance of a Sharpe-ratio difference rejects a true null.

Two strategies, the same Sharpe ratio, and a test asked whether they differ. It
should say no about 5% of the time at the 5% level. That is the only property of a
test that can be checked without knowing the answer, and it is the whole case for
having two variances rather than the one the literature quotes most.

Three regimes, a thousand replications each, five hundred paired periods, and the
two series correlated at 0.7 — which is what two strategies trading the same market
look like and is also where the closed form has most to gain, since correlation is
the term two separate standard errors cannot see.

The regimes differ only in serial dependence. The innovations are rescaled so an
AR(1) series keeps the unconditional volatility it was asked for; without that,
turning up the persistence would change the Sharpe ratios as well as their
dependence and the measurement would be reading two things at once.

Run it with ``python examples/sharpe_difference.py``.
"""

from __future__ import annotations

import math
import statistics

import numpy as np
from numpy.typing import NDArray

from holdout import sharpe_difference

#: Replications per regime. Enough that a 5% rate is resolved to about seven
#: tenths of a percentage point, which is finer than any of the differences below.
TRIALS = 1000
PERIODS = 500
CORRELATION = 0.7
CRITICAL = 1.959963984540054
NOMINAL = 0.05


def paired(persistence: float, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two correlated series with the same true Sharpe ratio."""
    rng = np.random.default_rng(seed)
    first = rng.normal(0.0, 1.0, PERIODS)
    second = CORRELATION * first + math.sqrt(1.0 - CORRELATION * CORRELATION) * rng.normal(
        0.0, 1.0, PERIODS
    )
    if persistence:
        for series in (first, second):
            for index in range(1, PERIODS):
                series[index] += persistence * series[index - 1]
    scale = math.sqrt(1.0 - persistence * persistence) if persistence else 1.0
    return 0.0005 + 0.01 * scale * first, 0.0005 + 0.01 * scale * second


def measure(persistence: float) -> dict[str, float]:
    closed = 0
    robust = 0
    ratios = []
    for trial in range(TRIALS):
        result = sharpe_difference(*paired(persistence, 1000 + trial))
        ratios.append(result.error_ratio)
        if abs(result.difference / result.closed_form_error) > CRITICAL:
            closed += 1
        if abs(result.difference / result.robust_error) > CRITICAL:
            robust += 1
    return {
        "persistence": persistence,
        "closed": closed / TRIALS,
        "robust": robust / TRIALS,
        "ratio": statistics.mean(ratios),
    }


def main() -> dict[str, dict[str, float]]:
    print()
    print(
        f"{TRIALS} replications, {PERIODS} paired periods, series correlated at "
        f"{CORRELATION:g}, true difference zero.\n"
    )
    print(f"    {'serial dependence':<20} {'closed form':>12} {'robust':>9} {'error ratio':>12}")
    results = {}
    for persistence, label in (
        (0.0, "none"),
        (0.3, "AR(1), rho = 0.3"),
        (0.6, "AR(1), rho = 0.6"),
    ):
        found = measure(persistence)
        results[label] = found
        print(
            f"    {label:<20} {found['closed']:>11.1%} {found['robust']:>9.1%} "
            f"{found['ratio']:>12.2f}"
        )
    print(f"\n    nominal size {NOMINAL:.0%}\n")

    print(
        "  With independent returns the closed form is right and the robust version"
        "\n  costs nothing: both land within a percentage point of nominal and the two"
        "\n  standard errors agree to within one per cent. So there is no reason to"
        "\n  prefer the closed form even where its assumptions hold.\n"
        "  Serial dependence is where they part. At a persistence of 0.3 the closed"
        "\n  form rejects a true null three times too often, and at 0.6 more than six"
        "\n  times too often — because it assumes independence and the difference of two"
        "\n  persistent series has far more sampling variability than its formula"
        "\n  allows. A third of the time it reports a difference in Sharpe ratios that"
        "\n  is not there.\n"
        "  The robust version cuts that to one in ten, which is better and is"
        "\n  not right. A Bartlett kernel truncated at a rule-of-thumb bandwidth"
        "\n  recovers only part of the long-run variance, and the part it misses is the"
        "\n  part that matters. So the honest summary is: use the robust variance, and"
        "\n  do not read its p-value as exact on data you believe is persistent."
    )
    print()

    clean = results["none"]
    mild = results["AR(1), rho = 0.3"]
    strong = results["AR(1), rho = 0.6"]
    checks = [
        abs(clean["closed"] - NOMINAL) < 0.015,
        abs(clean["robust"] - NOMINAL) < 0.015,
        abs(clean["ratio"] - 1.0) < 0.03,
        mild["closed"] > 2.0 * NOMINAL,
        strong["closed"] > 5.0 * NOMINAL,
        strong["robust"] < strong["closed"] / 2.0,
        strong["robust"] > NOMINAL,
        strong["ratio"] > mild["ratio"] > clean["ratio"],
    ]
    if all(checks):
        print("All figures reproduced.\n")
    else:
        print(f"A figure did not reproduce: {checks}\n")
        raise SystemExit(1)
    return results


if __name__ == "__main__":
    main()
