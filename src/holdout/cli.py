"""The ``holdout`` command.

Every subcommand reads a returns CSV (see :mod:`holdout.io`) and prints a
plain-text report. ``holdout sample-data`` writes synthetic parameter sweeps
so the rest can be tried without real data.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist

import numpy as np

from . import __version__
from .deflated import deflate_trials, minimum_track_record_length, probabilistic_sharpe_ratio
from .drawdown import assess_drawdown
from .exceptions import HoldoutError, ValidationError
from .io import ReturnTable, read_returns_csv, write_returns_csv
from .mcs import Statistic, model_confidence_set
from .pairwise import sharpe_difference
from .pbo import probability_of_backtest_overfitting
from .sharpe import autocorrelation_adjusted_sharpe, estimate_sharpe
from .spa import reality_check, romano_wolf, superior_predictive_ability
from .stability import DEFAULT_TRIM, sharpe_break
from .synthetic import crossover_sweep
from .trials import effective_number_of_trials, trial_correlation
from .uniqueness import (
    average_uniqueness,
    concurrency,
    effective_sample_size,
    sequential_bootstrap,
    time_decay_weights,
    uniqueness_weights,
)

__all__ = ["main"]


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = "  ".join(
        h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(header, widths, strict=True))
    )
    rule = "-" * len(line)
    body = [
        "  ".join(
            c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True))
        )
        for r in rows
    ]
    return "\n".join([line, rule, *body])


def _load(args: argparse.Namespace) -> ReturnTable:
    return read_returns_csv(args.file, index=not args.no_index)


def _years(periods: float, ppy: float) -> str:
    if not math.isfinite(periods):
        return "never"
    return f"{periods / ppy:.1f}y"


def _cmd_drawdown(args: argparse.Namespace) -> int:
    table = _load(args)
    names = [args.column] if args.column else table.names
    rows = []
    for name in names:
        got = assess_drawdown(table.column(name), drift=args.drift, volatility=args.volatility)
        worst = got.observed
        under = f"{worst.time_under_water}" + ("+" if worst.under_water_censored else "")
        rows.append(
            [
                name,
                f"{worst.proportional:.2%}",
                f"{worst.peak_index}-{worst.trough_index}",
                "never" if worst.recovery_index is None else str(worst.recovery_index),
                under,
                f"{-math.expm1(-got.expected):.2%}",
                f"{-math.expm1(-got.limit):.2%}",
                f"{got.percentile:.3f}",
                f"{got.exceedance:.4f}",
            ]
        )
    print(
        _table(
            rows,
            [
                "strategy",
                "worst",
                "peak-trough",
                "recovery",
                "under water",
                "expected",
                "1-in-20",
                "percentile",
                "p",
            ],
        )
    )
    print()
    print(
        "The drawdown is read as a fraction of the peak, on the reading that the "
        "returns are logarithmic."
    )
    print(
        "'expected' and '1-in-20' are what a Brownian motion with this record's own "
        "drift and volatility"
    )
    print(
        "would produce over a record this long. A percentile near one half is the "
        "drawdown such a strategy"
    )
    print("has anyway; it is not evidence of anything having gone wrong.")
    if any(
        assess_drawdown(table.column(name), drift=args.drift, volatility=args.volatility).estimated
        for name in names
    ):
        print()
        print(
            "The drift and volatility were estimated from the same returns, so 'p' is "
            "not a p-value: it"
        )
        print(
            "understates badly, by a factor of about thirteen at the 5% level on 252 "
            "observations. Pass"
        )
        print("--drift and --volatility to get one that is.")
    return 0


def _cmd_sharpe(args: argparse.Namespace) -> int:
    table = _load(args)
    ppy = args.periods_per_year
    names = [args.column] if args.column else table.names
    rows = []
    for name in names:
        x = table.column(name)
        est = estimate_sharpe(x, periods_per_year=ppy)
        lo, hi = est.confidence_interval(0.95, annualise=True)
        psr = probabilistic_sharpe_ratio(
            est.value, est.n, skewness=est.skewness, kurtosis=est.kurtosis
        )
        try:
            mintrl = _years(
                minimum_track_record_length(
                    est.value, skewness=est.skewness, kurtosis=est.kurtosis
                ),
                ppy,
            )
        except HoldoutError:
            mintrl = "never"
        lags = min(int(ppy) - 1, 10)
        adjusted = autocorrelation_adjusted_sharpe(x, int(ppy), max_lag=lags)
        rows.append(
            [
                name,
                f"{est.annualised:.2f}",
                f"[{lo:.2f}, {hi:.2f}]",
                f"{adjusted:.2f}",
                f"{est.skewness:.2f}",
                f"{est.kurtosis:.1f}",
                f"{psr:.3f}",
                mintrl,
            ]
        )
    header = ["strategy", "sharpe", "95% interval", "lo-adj", "skew", "kurt", "psr(0)", "min track"]
    print(f"{table.values.shape[0]} periods at {ppy:g} per year (annualised figures)\n")
    print(_table(rows, header))
    return 0


def _cmd_deflate(args: argparse.Namespace) -> int:
    table = _load(args)
    x = table.values
    correlation = trial_correlation(x)
    counts = {
        m: effective_number_of_trials(correlation, method=m)
        for m in ("average", "eigenvalue", "participation")
    }
    n_trials = float(x.shape[1]) if args.effective == "none" else counts[args.effective]
    result = deflate_trials(x, n_trials=n_trials)
    root = math.sqrt(args.periods_per_year)
    print(f"{x.shape[1]} trials over {x.shape[0]} periods")
    print(
        "effective trials: "
        + ", ".join(f"{m} {v:.1f}" for m, v in counts.items())
        + f"  (using {args.effective if args.effective != 'none' else 'all ' + str(x.shape[1])})"
    )
    print(f"selected:          {table.names[result.selected]}")
    print(f"sharpe:            {result.estimate.value * root:.2f} annualised")
    print(f"expected maximum:  {result.expected_maximum * root:.2f} annualised, if nothing works")
    print(
        f"probabilistic:     {result.probabilistic:.3f}  (P[true sharpe > 0], ignoring the search)"
    )
    print(f"deflated:          {result.deflated:.3f}  (the same, after the search)")
    return 0


def _cmd_pbo(args: argparse.Namespace) -> int:
    table = _load(args)
    result = probability_of_backtest_overfitting(table.values, n_blocks=args.blocks)
    slope, intercept, r2 = result.degradation
    freq = result.selection_frequency()
    top = np.argsort(-freq)[:5]
    splits = f"{result.n_partitions:,}"
    print(f"{result.n_strategies} strategies, {result.n_blocks} blocks, {splits} splits")
    print(f"probability of backtest overfitting: {result.pbo:.3f}")
    print(f"probability of out-of-sample loss:   {result.probability_of_loss:.3f}")
    sign = "-" if slope < 0 else "+"
    print(f"degradation: oos = {intercept:.4f} {sign} {abs(slope):.3f} * is   (r^2 {r2:.2f})")
    print("most often selected in-sample:")
    for j in top:
        if freq[j] > 0:
            print(f"  {table.names[j]:<20} {freq[j]:6.1%}")
    return 0


def _cmd_spa(args: argparse.Namespace) -> int:
    table = _load(args)
    if args.benchmark:
        bench = table.column(args.benchmark)
        table = table.without(args.benchmark)
        d = table.values - bench[:, None]
        against = args.benchmark
    else:
        d = table.values
        against = "zero"
    b, seed = args.bootstrap, args.seed
    rc = reality_check(d, n_bootstrap=b, seed=seed)
    spa = superior_predictive_ability(d, n_bootstrap=b, seed=seed)
    rw = romano_wolf(d, alpha=args.alpha, n_bootstrap=b, seed=seed)
    print(f"{d.shape[1]} strategies against {against}, {d.shape[0]} periods")
    print(
        f"block length {spa.block_length:.1f}, {args.bootstrap} bootstrap samples, seed {args.seed}"
    )
    print(f"reality check p-value:  {rc.pvalue:.3f}")
    print(
        f"spa p-value:            {spa.consistent:.3f}  (bounds {spa.lower:.3f} to {spa.upper:.3f})"
    )
    if rw.rejected.size:
        print(f"beat {against} at {args.alpha:g} (Romano-Wolf):")
        for j in rw.rejected:
            print(f"  {table.names[j]:<20} adjusted p = {rw.adjusted_pvalues[j]:.3f}")
    else:
        print(f"no strategy beats {against} at {args.alpha:g} (Romano-Wolf)")
    return 0


def _cmd_mcs(args: argparse.Namespace) -> int:
    table = _load(args)
    result = model_confidence_set(
        table.values,
        alpha=args.alpha,
        statistic=args.statistic,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        names=list(table.names),
    )
    print(f"{len(table.names)} models, {table.values.shape[0]} periods")
    print(
        f"{args.statistic} statistic, block length {result.block_length:.1f}, "
        f"{args.bootstrap} bootstrap samples, seed {args.seed}"
    )
    included = set(result.included.tolist())
    rows = [
        [
            table.names[j],
            f"{result.performance[j] * args.periods_per_year:.2%}",
            f"{result.pvalues[j]:.3f}",
            "in" if j in included else "out",
        ]
        for j in np.argsort(-result.performance)
    ]
    print()
    print(_table(rows, ["model", "mean (annualised)", "p-value", f"set at {args.alpha:g}"]))
    print()
    kept = len(included)
    if kept == len(table.names):
        print(
            f"All {kept} models are in the set. {table.values.shape[0]} periods cannot "
            "separate them, so picking the highest mean and calling it the best is a "
            "claim the data does not support."
        )
    elif kept == 1:
        print(
            f"One model survives: {result.included_names[0]}. Worth ruling out the dull "
            "explanations before believing it — a model on a different scale from the "
            "rest, or one that is an affine transform of another, both produce this."
        )
    else:
        print(
            f"{kept} of {len(table.names)} models survive at {args.alpha:g}. The set is a "
            "confidence region: its size is the result, not a shortcoming of it."
        )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    table = _load(args)
    missing = [name for name in (args.first, args.second) if name not in table.names]
    if missing:
        raise ValidationError(
            f"{args.file} has no column named {missing[0]!r}. It has "
            f"{', '.join(repr(name) for name in table.names)}."
        )
    result = sharpe_difference(
        table.column(args.first),
        table.column(args.second),
        bandwidth=args.bandwidth,
        periods_per_year=args.periods_per_year,
    )
    scale = math.sqrt(args.periods_per_year)
    print(
        f"{result.observations} paired periods at {args.periods_per_year:g} per year, "
        f"correlation {result.correlation:+.3f}\n"
    )
    rows = [
        [args.first, f"{result.first * scale:.3f}"],
        [args.second, f"{result.second * scale:.3f}"],
        ["difference (annualised)", f"{result.annualised_difference:+.3f}"],
    ]
    print(_table(rows, ["series", "sharpe"]))
    print()
    comparison = []
    for label, error in (
        ("Jobson-Korkie / Memmel", result.closed_form_error),
        (f"Ledoit-Wolf (bandwidth {result.bandwidth})", result.robust_error),
    ):
        statistic = result.difference / error
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(statistic)))
        comparison.append(
            [
                label,
                f"{error * scale:.3f}",
                f"{statistic:+.2f}",
                f"{p_value:.4f}",
                "yes" if p_value < 0.05 else "no",
            ]
        )
    print(_table(comparison, ["variance", "s.e. (ann.)", "statistic", "p", "reject at 5%"]))
    direction = (
        "so the closed form is understating the uncertainty here, which is the "
        "direction in which a difference looks real and is not"
        if result.error_ratio > 1.0
        else "so the closed form's assumptions are not costing anything on this pair"
    )
    print(
        f"\nThe robust standard error is {result.error_ratio:.2f} times the closed-form "
        f"one, {direction}. The closed form assumes independent normal returns; measured "
        "on serially dependent series with a true null it rejects about a third of the "
        "time at a persistence of 0.6, against the robust version's one in ten — better "
        "and still not 5%."
    )
    return 0


def _cmd_sample_data(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, trend in (("sweep", 0.0), ("trending", 0.0006)):
        sweep = crossover_sweep(trend=trend, seed=args.seed)
        write_returns_csv(out / f"{name}.csv", sweep.table, header="day")
        print(f"wrote {out / (name + '.csv')}: {len(sweep.table.names)} strategies")
    return 0


def _cmd_uniqueness(args: argparse.Namespace) -> int:
    """Report what overlapping labels cost, and what resampling them can recover.

    The second half is the part worth printing. The effective sample size says
    how much information is there; the bootstrap table says how much of it any
    sampling scheme could possibly reach, which is capped by arithmetic rather
    than by cleverness, and how close each scheme gets.
    """
    if args.labels is not None:
        rows = np.loadtxt(args.labels, delimiter=",", ndmin=2)
        if rows.shape[1] != 2:
            raise ValidationError(
                f"{args.labels} has {rows.shape[1]} columns; a label file holds two, "
                "the first and last bar of each window"
            )
        start, end = rows[:, 0], rows[:, 1]
    else:
        if args.count is None or args.window is None:
            raise ValidationError(
                "give --labels, or --count and --window to describe a rolling structure"
            )
        start = np.arange(0, args.count * args.step, args.step, dtype=np.int64)
        end = start + args.window - 1

    counts = concurrency(start, end)
    unique = average_uniqueness(start, end)
    weights = uniqueness_weights(start, end)
    observations = unique.size
    effective = effective_sample_size(start, end)

    print(f"{observations} labels over bars {counts.first}..{counts.last}")
    print(f"covered bars:          {counts.covered_bars}")
    print(f"peak concurrency:      {counts.peak}")
    print(
        f"average uniqueness:    {unique.mean():.4f}"
        f"  (min {unique.min():.4f}, max {unique.max():.4f})"
    )
    print(f"effective sample size: {effective:.2f} of {observations}")
    print(f"weight range:          {weights.min():.3e} to {weights.max():.3e}")
    lengths = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64) + 1.0
    print(
        f"identity check:        sum(uniqueness * length) = "
        f"{float((unique * lengths).sum()):.6f}"
        f" against {counts.covered_bars} covered bars"
    )

    print()
    print("time-decay weights over cumulative uniqueness (oldest, newest):")
    for decay in (1.0, 0.5, 0.0, -0.5):
        schedule = time_decay_weights(unique, decay=decay)
        kept = int(np.count_nonzero(schedule))
        print(
            f"  decay {decay:>5}: {schedule[0]:.4f} to {schedule[-1]:.4f}, "
            f"{kept} of {observations} keep any weight"
        )

    print()
    print("resampling, as a fraction of the cap no scheme can beat:")
    span = counts.last - counts.first + 1
    length = float(lengths.mean())
    rows_out = []
    for divisor in (1, 2, 5, 20):
        draws = max(1, observations // divisor)
        cap = min(1.0, span / (draws * length))
        achieved = []
        uniform = []
        for offset in range(args.replications):
            drawn = sequential_bootstrap(start, end, size=draws, seed=args.seed + offset)
            achieved.append(drawn.achieved)
            uniform.append(drawn.uniform)
        mean_achieved = float(np.mean(achieved))
        mean_uniform = float(np.mean(uniform))
        rows_out.append(
            [
                f"{draws}",
                f"{cap:.4f}",
                f"{mean_achieved / cap:.3f}",
                f"{mean_uniform / cap:.3f}",
                f"{mean_achieved / mean_uniform - 1.0:+.2%}",
            ]
        )
    print(_table(rows_out, ["draws", "cap", "sequential", "uniform", "gain"]))
    print(
        "The cap is span / (draws * length): total concurrency over a draw does not\n"
        "depend on which observations are drawn, so no sampling scheme can exceed it.\n"
        "Where it binds, the uniform bootstrap is already at it and drawing cleverly\n"
        "buys nothing. The remedy for heavy overlap is to draw fewer observations."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holdout", description="Judge whether a backtest is evidence of anything."
    )
    parser.add_argument("--version", action="version", version=f"holdout {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_file(p: argparse.ArgumentParser) -> None:
        p.add_argument("file", help="returns CSV: one row per period, one column per strategy")
        p.add_argument("--no-index", action="store_true", help="the CSV has no label column")
        p.add_argument("--periods-per-year", type=float, default=252.0)

    p = sub.add_parser(
        "drawdown",
        help="the worst drawdown beside what the null says to expect",
        description=(
            "Every other command here asks what a search was worth. This one asks "
            "what one realised path owes to luck: the deepest drawdown of a "
            "strategy with no edge at all is 1.2533 standard deviations of its "
            "record's length, and with an edge it grows logarithmically rather "
            "than away."
        ),
    )
    with_file(p)
    p.add_argument("--column", help="only this strategy")
    p.add_argument("--drift", type=float, help="drift per period for the null (default: estimated)")
    p.add_argument(
        "--volatility", type=float, help="volatility per period for the null (default: estimated)"
    )
    p.set_defaults(func=_cmd_drawdown)

    p = sub.add_parser("sharpe", help="Sharpe ratio inference for each column")
    with_file(p)
    p.add_argument("--column", help="only this strategy")
    p.set_defaults(func=_cmd_sharpe)

    p = sub.add_parser("deflate", help="deflated Sharpe ratio of the best column")
    with_file(p)
    p.add_argument(
        "--effective",
        choices=["eigenvalue", "average", "participation", "none"],
        default="eigenvalue",
        help="how to count correlated trials (default: eigenvalue)",
    )
    p.set_defaults(func=_cmd_deflate)

    p = sub.add_parser("pbo", help="probability of backtest overfitting")
    with_file(p)
    p.add_argument("--blocks", type=int, default=16)
    p.set_defaults(func=_cmd_pbo)

    p = sub.add_parser("spa", help="bootstrap tests against a benchmark")
    with_file(p)
    p.add_argument("--benchmark", help="column to test against (default: zero)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_spa)

    p = sub.add_parser(
        "mcs",
        help="the set of models that cannot be told apart from the best",
        description=(
            "Needs no benchmark, which is the difference from 'spa'. Columns are "
            "performance with higher better; a loss matrix must be negated first, "
            "or the set returned is the one around the worst model."
        ),
    )
    with_file(p)
    p.add_argument(
        "--statistic",
        choices=[Statistic.MAX.value, Statistic.RANGE.value],
        default=Statistic.MAX.value,
        help="max compares each model to the set average, range takes the largest "
        "studentised pair (default: max)",
    )
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="a smaller level gives a LARGER set, since this is a confidence "
        "region (default: 0.10, following the paper)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_mcs)

    p = sub.add_parser(
        "compare",
        help="test whether two strategies have different Sharpe ratios",
        description=(
            "The pairwise comparison, which the bootstrap tools are the wrong tool "
            "for: they control a family-wise error rate over a family of one. Reports "
            "both the closed-form variance and the robust one, because the "
            "disagreement between them is the finding when there is one."
        ),
    )
    with_file(p)
    p.add_argument("first", help="the strategy the difference is measured for")
    p.add_argument("second", help="the strategy it is measured against")
    p.add_argument(
        "--bandwidth",
        type=int,
        default=None,
        help="Bartlett truncation for the robust variance. Defaults to "
        "floor(4 (n/100)^(2/9)); 0 uses no autocovariances at all.",
    )
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser(
        "stability",
        help="test for a break in the Sharpe ratio at a date the data chose",
        description=(
            "Every other test here guards a maximum over strategies. This one guards "
            "a maximum over dates: it takes the largest standardised difference "
            "between the Sharpe ratio before and after a candidate break, over every "
            "candidate, and gets its null distribution from the stationary "
            "bootstrap. Splitting a sample at a date chosen by looking at the equity "
            "curve, and reading the result against a normal critical value, rejects a "
            "true null about 41% of the time; both p-values are printed so the gap is "
            "visible."
        ),
    )
    with_file(p)
    p.add_argument("--column", help="only this strategy")
    p.add_argument(
        "--trim",
        type=float,
        default=DEFAULT_TRIM,
        help=(
            "fraction of the sample ignored at each end (default: "
            f"{DEFAULT_TRIM}); it changes the answer, so it is an argument"
        ),
    )
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_stability)

    p = sub.add_parser(
        "uniqueness",
        help="what overlapping labels cost, and what resampling can recover",
        description=(
            "Reports concurrency, average uniqueness and the effective sample size "
            "for a set of label windows, both weight schedules, and how close each "
            "bootstrap gets to the cap that arithmetic places on any of them. Takes "
            "a two-column CSV of first and last bar, or --count and --window to "
            "describe a rolling structure."
        ),
    )
    p.add_argument("--labels", type=Path, default=None, help="CSV of start,end bar indices")
    p.add_argument("--count", type=int, default=None, help="number of rolling windows")
    p.add_argument("--window", type=int, default=None, help="bars per window")
    p.add_argument("--step", type=int, default=1, help="bars between window starts")
    p.add_argument("--replications", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_uniqueness)

    p = sub.add_parser("sample-data", help="write synthetic parameter sweeps")
    p.add_argument("--out", default="sample")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_cmd_sample_data)
    return parser


def _cmd_stability(args: argparse.Namespace) -> int:
    """Was the edge there throughout?

    Both p-values are printed and labelled. The naive one is not an alternative
    answer -- it is what reading this statistic against a normal critical value
    would have said, and on iid returns with no break at all it rejects 41% of the
    time. Printing it beside the bootstrap's is how a reader who has been quoting
    it finds out what it was worth.
    """
    table = _load(args)
    ppy = args.periods_per_year
    root = math.sqrt(ppy)
    names = [args.column] if args.column else table.names
    rows = []
    candidates = 0
    for name in names:
        result = sharpe_break(
            table.column(name),
            trim=args.trim,
            resamples=args.bootstrap,
            seed=args.seed,
        )
        candidates = result.candidates
        rows.append(
            [
                name,
                f"{result.at}",
                f"{result.before * root:.2f}",
                f"{result.after * root:.2f}",
                f"{result.statistic:.2f}",
                f"{result.p_value:.3f}",
                f"{result.naive_p_value:.4f}",
            ]
        )
    header = [
        "strategy",
        "break after",
        "sharpe before",
        "sharpe after",
        "sup stat",
        "p (bootstrap)",
        "p (naive)",
    ]
    # The candidate count comes out of the loop rather than from a second call.
    # Every column is the same length, so it is the same number for all of them,
    # and running the whole bootstrap again over 1,765 dates to print one integer
    # is most of the command's work done twice.
    print(
        f"{table.values.shape[0]} periods at {ppy:g} per year, "
        f"{candidates} candidate break dates, trim {args.trim:g} "
        f"(annualised Sharpe ratios)\n"
    )
    print(_table(rows, header))
    print(
        "\nThe bootstrap p-value is the one to read. The naive column is what a "
        "two-sided\nnormal critical value says about the same statistic, as if the "
        "date had been fixed\nin advance; on returns with no break at all it "
        "rejects at 5% about 41% of the time.\nA large p-value here is weak "
        "evidence either way: at 500 observations a side, a\nstrategy whose Sharpe "
        "ratio falls to zero is detected a quarter of the time."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code: int = args.func(args)
    except HoldoutError as exc:
        print(f"holdout: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # Output piped into something that stopped reading, such as head.
        # Point stdout at /dev/null so the interpreter's final flush is silent.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1
    return code
