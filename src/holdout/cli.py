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

import numpy as np

from . import __version__
from .deflated import deflate_trials, minimum_track_record_length, probabilistic_sharpe_ratio
from .exceptions import HoldoutError
from .io import ReturnTable, read_returns_csv, write_returns_csv
from .mcs import Statistic, model_confidence_set
from .pbo import probability_of_backtest_overfitting
from .sharpe import autocorrelation_adjusted_sharpe, estimate_sharpe
from .spa import reality_check, romano_wolf, superior_predictive_ability
from .synthetic import crossover_sweep
from .trials import effective_number_of_trials, trial_correlation

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


def _cmd_sample_data(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, trend in (("sweep", 0.0), ("trending", 0.0006)):
        sweep = crossover_sweep(trend=trend, seed=args.seed)
        write_returns_csv(out / f"{name}.csv", sweep.table, header="day")
        print(f"wrote {out / (name + '.csv')}: {len(sweep.table.names)} strategies")
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

    p = sub.add_parser("sample-data", help="write synthetic parameter sweeps")
    p.add_argument("--out", default="sample")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_cmd_sample_data)
    return parser


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
