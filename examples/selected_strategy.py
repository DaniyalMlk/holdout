"""What the winning strategy's Sharpe ratio is worth, given that it won.

Three things, in order.

A committee picks the best of a set of backtests and carries its Sharpe ratio
forward. Under a null where every candidate is equally good, the ordinary 95%
interval on that number covers the selected strategy's own truth 95% of the
time out of two candidates and 31% out of fifty. The conditional interval
covers 95% throughout.

The correction is a function of how close the race was and of nothing else. A
clear winner gets the naive interval back; a photo finish gets a bound a
hundred standard errors out.

And its one assumption is independence. A sweep over neighbouring parameters
is not independent, and above a correlation of about three quarters the
ordinary interval is the better of the two -- which is measured here rather
than asserted.

Run with ``python examples/selected_strategy.py``.
"""

from __future__ import annotations

from holdout import (
    conditional_interval,
    expected_maximum_normal,
    median_unbiased_mean,
    selection_coverage,
    selection_inference,
)


def main() -> dict[str, object]:
    coverage: dict[int, dict[str, float]] = {}
    correlated: dict[float, dict[str, float]] = {}

    print("coverage of the SELECTED strategy's truth, all candidates equally good")
    print(f"{'candidates':>11}{'naive':>8}{'conditional':>13}{'bias':>8}{'E[max]':>9}")
    for candidates in (2, 5, 10, 50):
        result = selection_coverage([0.0] * candidates, [1.0] * candidates, paths=4000, seed=1)
        coverage[candidates] = result
        print(
            f"{candidates:>11}{result['naive_coverage']:>8.3f}"
            f"{result['conditional_coverage']:>13.3f}"
            f"{result['mean_selection_bias']:>8.3f}"
            f"{expected_maximum_normal(candidates):>9.3f}"
        )
    print("the bias column is a simulation; E[max] is the closed form it never calls")
    print()

    print("what it costs, against how close the race was (estimate 1.5, s.e. 1.0)")
    print(f"{'margin':>8}{'shrinkage':>11}{'lower':>10}{'upper':>8}{'naive width':>13}")
    for margin in (0.1, 0.5, 1.0, 2.0, 4.0):
        interval = conditional_interval(1.5, 1.0, 1.5 - margin)
        shrinkage = 1.5 - median_unbiased_mean(1.5, 1.0, 1.5 - margin)
        print(
            f"{margin:>8.2f}{shrinkage:>11.3f}{interval.lower:>10.2f}"
            f"{interval.upper:>8.2f}{interval.naive_width:>13.2f}"
        )
    print("four standard errors clear of the runner-up, the correction is invisible")
    print()

    print("the one assumption, tested: ten candidates at an equicorrelation")
    print(f"{'rho':>6}{'naive':>8}{'conditional':>13}{'median width':>14}")
    for rho in (0.0, 0.5, 0.75, 0.9, 0.99):
        result = selection_coverage([0.0] * 10, [1.0] * 10, paths=4000, seed=4, correlation=rho)
        correlated[rho] = result
        print(
            f"{rho:>6.2f}{result['naive_coverage']:>8.3f}"
            f"{result['conditional_coverage']:>13.3f}"
            f"{result['median_bounded_conditional_width']:>14.1f}"
        )
    print("they cross near 0.78, and past it the conditional interval is both")
    print("wider and less valid -- so it is the wrong tool for a parameter sweep")
    print()

    # One concrete committee decision.
    estimates = [0.98, 0.95, 0.80, 0.41, 0.12]
    errors = [0.31, 0.30, 0.32, 0.29, 0.33]
    inference = selection_inference(estimates, errors)
    interval = inference.interval
    print("a committee's five candidates, per-period Sharpe estimates")
    print(f"  estimates          {estimates}")
    print(f"  selected           #{inference.selected.index} at {inference.selected.estimate:.2f}")
    print(f"  margin             {inference.selected.margin:.2f} standard errors")
    print(f"  median-unbiased    {interval.median:.3f} ({inference.shrinkage:.3f} lower)")
    print(f"  conditional 95%    [{interval.lower:.2f}, {interval.upper:.2f}]")
    print(f"  naive 95%          [{interval.naive_lower:.2f}, {interval.naive_upper:.2f}]")
    print(f"  the search costs a factor of {interval.inflation:.1f} on the width")
    return {
        "coverage": coverage,
        "correlated": correlated,
        "committee": {
            "selected": inference.selected.index,
            "margin": inference.selected.margin,
            "median": interval.median,
            "shrinkage": inference.shrinkage,
            "inflation": interval.inflation,
        },
        "shrinkage_at_four_errors": 1.5 - median_unbiased_mean(1.5, 1.0, 1.5 - 4.0),
    }


if __name__ == "__main__":
    main()
