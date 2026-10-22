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
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np

from . import __version__
from .deflated import deflate_trials, minimum_track_record_length, probabilistic_sharpe_ratio
from .drawdown import assess_drawdown
from .exceptions import HoldoutError, ValidationError
from .financing import (
    ExcessConvention,
    excess_columns,
    financing_impact,
    levered_returns,
    period_rate,
)
from .io import ReturnTable, read_returns_csv, write_returns_csv
from .mcs import Statistic, model_confidence_set
from .pairwise import sharpe_difference
from .pbo import probability_of_backtest_overfitting
from .selection import selection_inference
from .sequential import NORMAL_95, confidence_sequence
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
    """Read the returns CSV, in excess of financing when a rate was given.

    The conversion happens here, once, rather than inside each command, so
    that a rate supplied on the command line reaches every figure that command
    prints instead of only the ones that happened to be threaded for it. When
    a rate is in force the first line of output says so, because a Sharpe
    ratio on an excess basis and one on a total-return basis are different
    numbers and nothing else in the report distinguishes them.

    The excess-return convention is not an option here. For a constant rate it
    cannot move any ratio of a mean to a standard deviation (see
    :mod:`holdout.financing`), so offering the choice on every command would
    be offering a switch that does nothing. ``holdout financing`` exposes it,
    where it is the subject rather than an input.
    """
    table = read_returns_csv(args.file, index=not args.no_index)
    annual = float(getattr(args, "risk_free", 0.0))
    if annual == 0.0:
        return table
    rate = period_rate(annual, args.periods_per_year, compounding=args.financing_compounding)
    print(
        f"in excess of {annual:.2%} annualised financing, "
        f"{args.financing_compounding} ({rate:.6g} per period)"
    )
    return replace(table, values=excess_columns(table.values, rate))


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


def _cmd_watch(args: argparse.Namespace) -> int:
    table = _load(args)
    rows = []
    notes: list[str] = []
    for name in [args.column] if args.column else table.names:
        values = table.column(name)
        sequence = confidence_sequence(
            values,
            alpha=args.alpha,
            volatility=args.volatility,
            target=args.target,
            periods_per_year=args.periods_per_year,
        )
        crossing = sequence.crossing
        last = sequence.last
        fixed = NORMAL_95 * sequence.volatility / math.sqrt(last.observations)
        rows.append(
            [
                name,
                str(last.observations),
                f"{last.mean * args.periods_per_year:.2%}",
                f"{sequence.annualised_lower:.2%}",
                f"{last.upper * args.periods_per_year:.2%}",
                f"{last.half_width / fixed:.3f}",
                "never" if crossing is None else str(crossing),
                f"{sequence.sharpe_lower():.3f}",
            ]
        )
        if sequence.estimated_volatility:
            notes.append(name)
    print(
        _table(
            rows,
            [
                "strategy",
                "n",
                "mean",
                "lower",
                "upper",
                "width/fixed",
                "crossed at",
                "SR floor",
            ],
        )
    )
    print()
    print(
        "Every row of the sequence is valid at once, so 'crossed at' is a legitimate "
        "stopping point:"
    )
    print(
        "looking each period and stopping the first time the lower bound clears zero "
        "keeps the stated"
    )
    print(
        "level. Doing that with an ordinary interval does not. Monitoring a 95% "
        "interval and stopping"
    )
    print(
        "at the first crossing rejects 33.7% of the time at 60 observations and 61.9% "
        "at 5,000, with"
    )
    print("no edge present at all - and the rate grows with the horizon rather than")
    print("settling, so there is no multiplicity to correct for.")
    print()
    print(
        "'width/fixed' is what that costs: 1.549 at the tuning point, more before it. "
        "The sequential"
    )
    print(
        "track record length is 3.40 times the fixed-sample one, almost independently "
        "of the Sharpe ratio."
    )
    if notes:
        print()
        print(
            "The volatility was estimated from the same returns for "
            + ", ".join(notes)
            + ". That raises the"
        )
        print(
            "crossing rate from 1.47% to 1.98% at 250 observations - inside the "
            "nominal 5%, because the"
        )
        print("mixture has slack there, but not exact. Pass --volatility for a bound that is.")
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


def _cmd_financing(args: argparse.Namespace) -> int:
    """Both haircuts, side by side, which is the only reason this command exists.

    Every other command reads the CSV through ``_load`` and so works on an
    excess basis silently. This one needs the gross series as well as the
    excess one, so it reads the file itself and does the conversion in view.
    """
    if args.risk_free == 0.0:
        raise ValidationError(
            "--risk-free is what this command reports on; give a non-zero annualised "
            "rate, as a fraction (0.05 for five per cent)"
        )
    table = read_returns_csv(args.file, index=not args.no_index)
    ppy = args.periods_per_year
    root = math.sqrt(ppy)
    rate = period_rate(args.risk_free, ppy, compounding=args.financing_compounding)
    spread = period_rate(args.spread, ppy, compounding=args.financing_compounding)
    convention: ExcessConvention = args.convention
    names = [args.column] if args.column else table.names
    rows = []
    for name in names:
        series = table.column(name)
        if args.leverage != 1.0:
            series = levered_returns(series, leverage=args.leverage, rate=rate, spread=spread)
        impact = financing_impact(series, rate, convention=convention, periods_per_year=ppy)
        rows.append(
            [
                name,
                f"{impact.gross * root:+.3f}",
                f"{impact.excess * root:+.3f}",
                f"{impact.annualised_shift:.3f}",
                f"{impact.predicted_shift * root:.3f}",
                f"{impact.residual * root:+.1e}",
            ]
        )
    print(
        f"{table.values.shape[0]} periods at {ppy:g} per year, "
        f"{args.risk_free:.2%} annualised financing, {args.financing_compounding} "
        f"({rate:.6g} per period), {convention} excess"
    )
    if args.leverage != 1.0:
        print(
            f"at {args.leverage:g} turns of leverage funded {args.spread:.2%} over that rate, "
            f"which costs {(1.0 - 1.0 / args.leverage) * spread * root:.3f} of the ratio "
            f"on top of the financing itself"
        )
    print()
    print(
        _table(
            rows,
            ["strategy", "gross", "excess", "financing", "rate/vol", "residual"],
        )
    )
    print(
        "\nThe financing column is what the conversion took, annualised. rate/vol is the\n"
        "closed form for it, exact when the rate is constant, and residual is what the\n"
        "closed form missed -- zero for a constant rate, the rate's own variation\n"
        "otherwise."
    )
    if len(names) >= 2:
        # Deliberately the whole table, not the --column selection: the search
        # that earns the haircut is the one that produced every column.
        columns = table.values
        multiplicity = deflate_trials(columns).expected_maximum * root
        worst = max(float(r[3]) for r in rows)
        print(
            f"\nFor comparison, the haircut the search itself earns over {columns.shape[1]} "
            f"trials is\n{multiplicity:.3f} annualised. The largest financing haircut above is "
            f"{worst:.3f}. They are\nadjustments of the same kind to the same number; only one "
            "of them is usually reported."
        )
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


def _cmd_selected(args: argparse.Namespace) -> int:
    """Report the winner's record with the selection conditioned on.

    The margin over the runner-up is printed first because it is the number
    that decides everything else: the whole correction is a function of it,
    and of nothing about how many candidates there were.
    """
    table = _load(args)
    columns = table.values
    if columns.shape[1] < 2:
        raise ValidationError(
            f"selection inference needs at least two strategies, found {columns.shape[1]}"
        )
    root = math.sqrt(args.periods_per_year)
    estimates = []
    deviations = []
    for name in table.names:
        got = estimate_sharpe(table.column(name))
        estimates.append(got.value)
        deviations.append(got.standard_error())
    inference = selection_inference(estimates, deviations, level=args.level)
    selected = inference.selected
    interval = inference.interval
    runner_up = table.names[
        max(
            (index for index in range(len(estimates)) if index != selected.index),
            key=lambda index: estimates[index],
        )
    ]
    print(f"{columns.shape[1]} strategies over {columns.shape[0]} periods")
    print(f"selected:          {table.names[selected.index]}")
    print(f"runner-up:         {runner_up} at {selected.threshold * root:.2f} annualised")
    print(f"margin:            {selected.margin:.2f} standard errors over the runner-up")
    print(f"raw sharpe:        {selected.estimate * root:.2f} annualised")
    print(
        f"median-unbiased:   {interval.median * root:.2f} annualised"
        f"  ({inference.shrinkage * root:.2f} lower than the raw estimate)"
    )
    width = "unbounded" if not interval.bounded else f"{interval.width * root:.2f} wide"
    print(
        f"conditional {args.level:.0%}:  "
        f"[{interval.lower * root:.2f}, {interval.upper * root:.2f}]  ({width})"
    )
    print(
        f"ignoring the search: [{interval.naive_lower * root:.2f}, "
        f"{interval.naive_upper * root:.2f}]  "
        f"({interval.naive_width * root:.2f} wide)"
    )
    if interval.bounded:
        print(f"the search costs a factor of {interval.inflation:.2f} on the width")
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
        p.add_argument(
            "--periods-per-year",
            type=float,
            default=252.0,
            help="how many rows make a year, which is what annualises a "
            "Sharpe ratio. Daily by default.",
        )
        p.add_argument(
            "--risk-free",
            type=float,
            default=0.0,
            metavar="RATE",
            help="annualised financing rate to put the returns in excess of, as "
            "a fraction: 0.05 for five per cent. Zero by default, and that "
            "default is not neutral -- it overstates every Sharpe ratio below by "
            "the rate divided by the volatility, which at 5 per cent cash and 16 "
            "per cent volatility is 0.31.",
        )
        p.add_argument(
            "--financing-compounding",
            choices=("simple", "compound"),
            default="compound",
            help="how the annualised rate becomes a per-period one: 'compound' "
            "takes the period-count root so that a year of it reproduces the "
            "rate, 'simple' divides. They differ by about 2 per cent of the rate "
            "at daily frequency, so this moves the haircut and not the sign of "
            "anything.",
        )

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

    p = sub.add_parser(
        "watch",
        help="anytime-valid bounds for a record read as it accumulates",
        description=(
            "Every other command here answers a question asked once, at a record "
            "length settled before the data arrived. This one answers it at every "
            "length at once, so a caller may look whenever they like and stop at "
            "whichever look first clears zero. Doing that with an ordinary "
            "interval rejects a third of the time at sixty observations with no "
            "edge present; the cost of doing it properly is a band half again as "
            "wide and three and a half times the data."
        ),
    )
    with_file(p)
    p.add_argument("--column", help="only this strategy")
    p.add_argument("--alpha", type=float, default=0.05, help="the level the sequence holds")
    p.add_argument(
        "--volatility",
        type=float,
        help="per-period volatility (default: estimated, which is not exact)",
    )
    p.add_argument(
        "--target",
        type=int,
        help="the record length the band is narrowest at (default: the length given)",
    )
    p.set_defaults(func=_cmd_watch)

    p = sub.add_parser(
        "financing",
        help="what putting the record on an excess basis costs",
        description=(
            "Every other command here asks what a search was worth. This one asks "
            "what the record was worth before the search is considered at all: a "
            "Sharpe ratio taken on total returns rather than returns in excess of "
            "financing is too high by the rate over the volatility, which at a 5 "
            "per cent cash rate and 16 per cent volatility is 0.31 annualised -- "
            "between a third and two thirds of the haircut fifty trials earn, from "
            "a line nobody reports. Leverage is the other half: a Sharpe ratio is "
            "untouched by it only when the borrowing is at the financing rate "
            "exactly, and falls by (1 - 1/turns) times the spread over the "
            "volatility when it is not."
        ),
    )
    with_file(p)
    p.add_argument("--column", help="only this strategy")
    p.add_argument(
        "--spread",
        type=float,
        default=0.0,
        metavar="RATE",
        help="annualised spread over the financing rate that the borrowed part of a "
        "levered position pays, as a fraction. Ignored below one turn, where the "
        "position lends rather than borrows and so pays no spread.",
    )
    p.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        metavar="TURNS",
        help="turns of the asset held against one of equity. One by default, which "
        "borrows nothing.",
    )
    p.add_argument(
        "--convention",
        choices=("arithmetic", "geometric"),
        default="arithmetic",
        help="'arithmetic' subtracts the rate, 'geometric' divides by one plus it. "
        "For a constant rate the two differ by a positive constant factor and so "
        "cannot move any ratio of a mean to a standard deviation; only a rate that "
        "moves separates them, and then in the fourth decimal place.",
    )
    p.set_defaults(func=_cmd_financing)

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

    p = sub.add_parser(
        "selected",
        help="what the winning strategy's Sharpe is worth, given that it won",
        description=(
            "The deflate command asks whether the best of these records is "
            "significant. This one asks what its Sharpe ratio actually is. The "
            "winning estimate was drawn from its own sampling distribution "
            "restricted to the event that it beat every other candidate, so the "
            "honest interval inverts that truncated law rather than the "
            "unconditional one. Under a null where every strategy is equally "
            "good, the ordinary 95% interval on the winner covers its truth "
            "95% of the time out of two candidates and 31% out of fifty; this "
            "one covers 95% throughout. What it costs is width, and only when "
            "the race was close: four standard errors clear of the runner-up, "
            "the two intervals agree to within five per cent. The one "
            "assumption is that the candidates' estimates are independent. A "
            "sweep over neighbouring parameters is not, and above a "
            "correlation of about three quarters the ordinary interval is the "
            "better of the two -- so read the margin printed below, and if it "
            "is a hundredth of a standard error the candidates are telling you "
            "they are the same strategy."
        ),
    )
    with_file(p)
    p.add_argument("--level", type=float, default=0.95, help="confidence level (default: 0.95)")
    p.set_defaults(func=_cmd_selected)

    p = sub.add_parser("pbo", help="probability of backtest overfitting")
    with_file(p)
    p.add_argument(
        "--blocks",
        type=int,
        default=16,
        help="blocks the record is cut into before recombining. The number "
        "of splits is the central binomial coefficient in this, so sixteen "
        "is already 12,870 of them.",
    )
    p.set_defaults(func=_cmd_pbo)

    p = sub.add_parser("spa", help="bootstrap tests against a benchmark")
    with_file(p)
    p.add_argument("--benchmark", help="column to test against (default: zero)")
    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="stationary bootstrap resamples. The p-value cannot resolve below one over this.",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="size the test is conducted at")
    p.add_argument(
        "--seed", type=int, default=0, help="seed for the resampling, so the p-value reproduces"
    )
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
    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="stationary bootstrap resamples used to eliminate the set",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="a smaller level gives a LARGER set, since this is a confidence "
        "region (default: 0.10, following the paper)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the resampling, so the surviving set reproduces",
    )
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
    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="resamples per window, which set the width of the reported band",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="seed for the resampling, so the bands reproduce"
    )
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
    p.add_argument(
        "--replications",
        type=int,
        default=8,
        help="sequential bootstrap replications averaged over. The measured "
        "gain over uniform sampling is small, so more of them narrows a "
        "standard error rather than moving the answer.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the sampling, so the uniqueness figures reproduce",
    )
    p.set_defaults(func=_cmd_uniqueness)

    p = sub.add_parser("sample-data", help="write synthetic parameter sweeps")
    p.add_argument("--out", default="sample", help="directory to write the generated CSVs into")
    p.add_argument("--seed", type=int, default=7, help="seed for the generated sweeps")
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
