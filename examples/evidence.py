"""The front page's figures, recomputed.

The README's opening claim is that every statistic here is validated against
something outside this repository — a published worked example, a closed form,
or an independent implementation. A claim like that is worth exactly as much as
the guarantee that it still holds, and a figure pasted into a document once
stops being true quietly: the library changes, the suite stays green because
nothing in it reads the document, and the front page goes on promising a
precision the code no longer has.

So the front page quotes what this prints. Seven checks, six of them against a
reference that shares no derivation with the library:

1. The Bailey and López de Prado (2014) deflated-Sharpe worked example, run end
   to end and compared against the two figures the paper publishes.
2. The expected maximum of ``n`` standard normals, against numerical
   quadrature of its own defining integral.
3. Skewness and kurtosis, against ``scipy.stats``, in both bias conventions.
4. Lo's serial-correlation factor, against the closed form it reduces to when
   the returns are AR(1).
5. The combinatorially-purged path count, against the combinatorial identity
   ``k/N * C(N, k)``.
6. The log of the normal upper tail, against ``scipy.stats.norm.logsf``, out to
   where the tail itself has underflowed and only the log survives.
7. And the one that is not a cross-check but the reason the library exists: the
   best of fifty pure-noise strategies, scored by the probabilistic Sharpe
   ratio and then by the deflated one.

Every figure is reported as a *bound*, never as digits. Two implementations of
the same quantity agree to within something; where exactly they stop agreeing
depends on the SciPy and NumPy builds underneath, so a figure quoted to ten
significant places would be true only on the machine that measured it. The
bound is the claim, and this script fails if a bound is breached.

Run with ``python examples/evidence.py``. Exits non-zero if any figure moves
outside the band the README states.
"""

from __future__ import annotations

import runpy
import sys
from contextlib import redirect_stdout
from io import StringIO
from math import comb, sqrt
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, stats

from holdout import (
    deflate_trials,
    effective_number_of_trials,
    excess_columns,
    excess_returns,
    expected_maximum_normal,
    financing_impact,
    kurtosis,
    log_upper_tail,
    number_of_paths,
    period_rate,
    serial_correlation_factor,
    sharpe_ratio,
    skewness,
    trial_correlation,
)
from holdout.synthetic import crossover_sweep

#: The two figures Bailey and López de Prado publish for their worked example,
#: in "The Deflated Sharpe Ratio" (2014).
PAPER_BENCHMARK = 0.1132
PAPER_DEFLATED = 0.9004

#: The worked example itself lives in its own script, and this one runs that
#: rather than restating its arithmetic. Two places computing the paper's
#: figures is two places to get them wrong, and the first draft of this file
#: proved the point by de-annualising the trial variance incorrectly and
#: reporting a benchmark ten times too small.
WORKED_EXAMPLE = Path(__file__).resolve().parent / "deflated_worked_example.py"


class Check(NamedTuple):
    """One row of the front page's table."""

    label: str
    """What was checked, as the README writes it."""
    measured: float
    """The worst disagreement found, or the rate for the simulation row."""
    bound: float
    """The figure the README promises. Measured must not exceed it."""
    rendered: str
    """How the README states the result."""

    @property
    def passed(self) -> bool:
        return self.measured <= self.bound


def paper_worked_example() -> Check:
    """Run the 2014 worked example and compare it against the printed figures.

    Rounded to four places, which is the precision the paper prints, and the
    comparison is therefore exact: the benchmark has to come out at 0.1132 and
    the deflated ratio at 0.9004, not near them.

    The paper's 0.9004 is itself computed from the unrounded benchmark
    0.113172; recomputing it from the printed 0.1132 gives 0.90026. The test
    suite records both figures so that nobody bends the formula to match a
    recomputation from rounded inputs.
    """
    with redirect_stdout(StringIO()):
        result = runpy.run_path(str(WORKED_EXAMPLE))["main"]()
    benchmark = float(result["benchmark"])
    deflated = float(result["dsr"])
    worst = max(
        abs(round(benchmark, 4) - PAPER_BENCHMARK),
        abs(round(deflated, 4) - PAPER_DEFLATED),
    )
    return Check(
        label="Bailey and López de Prado (2014), their worked example end to end",
        measured=worst,
        bound=0.0,
        rendered=f"SR0 = {benchmark:.4f} and DSR = {deflated:.4f}, as published",
    )


def expected_maximum_against_quadrature() -> Check:
    """Compare the closed form against quadrature of the integral defining it.

    ``E[max of n standard normals]`` is ``∫ z n φ(z) Φ(z)^(n-1) dz``. The
    library evaluates it one way and ``scipy.integrate.quad`` another, and the
    two share no code. At ``n = 10**6`` the quadrature is the less accurate of
    the pair — ``Φ(z)**999999`` is a product of a million nearly-equal
    factors — so the disagreement there is not evidence against the closed
    form; it is the reason this is quoted as a bound over the whole range
    rather than as a figure at one ``n``.
    """
    worst = 0.0
    for n in (2, 5, 50, 100, 1000, 10**6):

        def density(z: float, n: int = n) -> float:
            return float(z * n * stats.norm.pdf(z) * stats.norm.cdf(z) ** (n - 1))

        reference, _ = integrate.quad(density, -40.0, 40.0, limit=400)
        worst = max(worst, abs(expected_maximum_normal(n) - reference))
    return Check(
        label="Expected maximum of n normals against quadrature, n = 2 to 10^6",
        measured=worst,
        bound=1e-10,
        rendered="agree to 1e-10",
    )


def moments_against_scipy() -> Check:
    """Skewness and kurtosis against ``scipy.stats``, in both conventions.

    The sample is deliberately skewed — a normal plus a scaled chi-square — so
    that the bias correction has something to correct. ``kurtosis`` here is raw
    rather than excess, which is why ``fisher=False`` is passed to SciPy.
    """
    rng = np.random.default_rng(7)
    base = rng.standard_normal(2000)
    sample = base + 0.3 * rng.standard_normal(2000) ** 2
    worst = 0.0
    for bias in (True, False):
        worst = max(
            worst,
            abs(skewness(sample, bias=bias) - float(stats.skew(sample, bias=bias))),
            abs(
                kurtosis(sample, bias=bias) - float(stats.kurtosis(sample, bias=bias, fisher=False))
            ),
        )
    return Check(
        label="Skewness and kurtosis against scipy.stats, both bias conventions",
        measured=worst,
        bound=1e-12,
        rendered="agree to 1e-12",
    )


def serial_correlation_against_closed_form() -> Check:
    """Lo's eta(q) against the AR(1) case it has a closed form for.

    Lo (2002) replaces ``sqrt(q)`` in the annualisation with
    ``q / sqrt(q + 2 Σ (q - k) ρ_k)``. For AR(1) returns ``ρ_k = φ^k``, so
    the whole factor is a formula in ``φ`` and ``q`` — including at ``φ = 0``,
    where it has to come back to ``sqrt(q)`` exactly rather than nearly.
    """
    worst = 0.0
    for phi in (0.0, 0.2, 0.4, -0.3):
        for q in (3, 12):
            autocorrelations = [phi**k for k in range(1, q)]
            reference = q / sqrt(q + 2 * sum((q - k) * phi**k for k in range(1, q)))
            worst = max(worst, abs(serial_correlation_factor(q, autocorrelations) - reference))
    return Check(
        label="Lo's eta(q) against its AR(1) closed form, eight cases",
        measured=worst,
        bound=1e-15,
        rendered="exact",
    )


def path_count_against_the_identity() -> Check:
    """The CPCV path count against ``k/N * C(N, k)``.

    Each of the ``C(N, k)`` test combinations contributes ``k`` group-sized
    pieces, and a path is built from one piece per group, so the number of
    paths is ``k C(N, k) / N``. It is an integer for every ``N`` and ``k``,
    which is itself worth asserting — a count that comes out fractional means
    the splitter is not building the paths the method describes.
    """
    worst = 0.0
    for groups in range(3, 13):
        for test_groups in range(1, min(groups, 6)):
            reference = test_groups * comb(groups, test_groups) // groups
            worst = max(worst, abs(number_of_paths(groups, test_groups) - reference))
    return Check(
        label="CPCV path count against k/N x C(N, k), every N up to 12",
        measured=worst,
        bound=0.0,
        rendered="exact",
    )


def log_tail_against_scipy() -> Check:
    """The log of the normal upper tail against ``scipy.stats.norm.logsf``.

    Checked *relatively*, not absolutely. By ``z = 400`` the log tail is about
    ``-80000``, so an absolute agreement of 1e-11 there is a relative agreement
    of 1e-16, and an absolute bound would be dominated by the largest ``z`` in
    the sweep rather than by the accuracy of either implementation. The
    relative figure says the same thing at both ends of the range.

    The range matters as much as the bound. Past ``z ≈ 38`` the tail itself is
    zero in double precision, so ``log(1 - cdf)`` is ``-inf`` and only a
    routine that works on the log scale throughout has anything to report.
    """
    sweep: NDArray[np.float64] = np.concatenate(
        [np.linspace(0.01, 10.0, 400), np.linspace(10.0, 400.0, 400)]
    )
    worst = 0.0
    for z in sweep:
        reference = float(stats.norm.logsf(z))
        worst = max(worst, abs((log_upper_tail(float(z)) - reference) / reference))
    return Check(
        label="Log normal upper tail against scipy.stats.norm.logsf, z = 0.01 to 400",
        measured=worst,
        bound=1e-14,
        rendered="agree to 1e-14 relative, past where the tail itself underflows",
    )


def noise_selected_as_a_discovery() -> tuple[Check, int, int, int]:
    """Fifty strategies with no skill at all, and the best of them selected.

    This is not a cross-check; it is the reason the library exists. The
    probabilistic Sharpe ratio asks whether the *observed* track record beats a
    benchmark, which a selected winner usually does. The deflated one raises
    the benchmark to what the best of fifty unskilled trials would be expected
    to show, which the winner of fifty unskilled trials by construction does
    not.

    Reported as the share of runs where each says "discovery", so the row is a
    rate rather than one run's verdict. The bound is on the deflated ratio: it
    must be fooled in *no* run, which is a stronger claim than a low rate and
    the one worth making.
    """
    runs = 200
    periods, variants = 500, 50
    rng = np.random.default_rng(20261018)
    psr_hits = dsr_hits = 0
    for _ in range(runs):
        trials = rng.standard_normal((periods, variants)) * 0.01
        ratios = [sharpe_ratio(trials[:, column]) for column in range(variants)]
        best = int(np.argmax(ratios))
        result = deflate_trials(trials, selected=best)
        if result.probabilistic > 0.95:
            psr_hits += 1
        if result.deflated > 0.95:
            dsr_hits += 1
    check = Check(
        label=f"{variants} pure-noise strategies, best selected, {runs} runs",
        measured=float(dsr_hits),
        bound=0.0,
        rendered=(f"PSR(0) > 95% in {psr_hits / runs:.0%} of them, DSR > 95% in none"),
    )
    return check, psr_hits, dsr_hits, runs


def the_excess_conventions_agree_for_a_constant_rate() -> Check:
    """A closed-form identity, checked as one.

    ``(1 + r) / (1 + rf) - 1`` is ``(r - rf) / (1 + rf)``: for a constant rate
    the geometric excess return is the arithmetic one times a positive
    constant, and a Sharpe ratio does not see positive constants. So the choice
    of convention cannot move any ratio of a mean to a standard deviation in
    this library, and what is left is floating-point noise rather than an
    approximation error. Only a rate that varies separates the two, and then in
    the fourth decimal place.

    Reported relatively. The absolute disagreement depends on how large the
    ratio happens to be, which is a fact about the sample rather than about the
    identity: the same five rates give 4e-16 to 1e-14 absolute while staying
    inside 3e-13 relative. For scale, implementing the geometric convention
    wrongly moves this by ten orders of magnitude or more, so a bound of 1e-12
    is loose for the arithmetic and tight for the mistake.
    """
    rng = np.random.default_rng(20261022)
    returns = rng.normal(0.0004, 0.01, 2520)
    worst = 0.0
    for annual in (0.0025, 0.01, 0.05, 0.12, 0.40):
        rate = period_rate(annual, 252.0, compounding="compound")
        arithmetic = sharpe_ratio(excess_returns(returns, rate))
        geometric = sharpe_ratio(excess_returns(returns, rate, convention="geometric"))
        worst = max(worst, abs(arithmetic - geometric) / abs(arithmetic))
    return Check(
        label="The two excess-return conventions, constant rate, five rates to 40%",
        measured=worst,
        bound=1e-12,
        rendered="agree to 1e-12 relative, which is the identity they are",
    )


def financing_against_multiplicity() -> Check:
    """The second row that is a result rather than a cross-check.

    The noise row shows the deflated Sharpe ratio refusing a winner that has no
    edge. This one shows a winner that *does* have an edge, is selected out of
    thirty trials, survives the deflation comfortably on the returns as
    reported, and does not survive it once the record is put in excess of a 5%
    cash rate. Nothing about the search changed: the benchmark the search earns
    is the same to three decimal places either way. What changed is the number
    being judged against it.

    The trials are counted by the effective number rather than by the thirty
    columns, because thirty variants of one crossover rule are not thirty
    independent trials and the library says so itself; the eigenvalue estimator
    puts them at 8.0. Counting all thirty deflates harder and gives 0.899
    against 0.402, the same conclusion from a benchmark the library would not
    defend.

    The bound is on the pair of verdicts, not on a figure: the gross deflated
    ratio must read as a discovery and the excess one must not. If either
    stops being true the row is wrong and the script says so.
    """
    sweep = crossover_sweep(trend=0.0006, seed=7)
    gross = sweep.table.values
    rate = period_rate(0.05, 252.0, compounding="compound")
    net = excess_columns(gross, rate)
    trials = effective_number_of_trials(trial_correlation(gross), method="eigenvalue")
    before = deflate_trials(gross, n_trials=trials)
    after = deflate_trials(net, n_trials=trials)
    root = sqrt(252.0)
    haircut = financing_impact(
        gross[:, before.selected], rate, periods_per_year=252.0
    ).annualised_shift
    search = before.expected_maximum * root
    # Bands rather than digits, for the usual reason and one specific to this
    # row: the effective trial count comes off an eigendecomposition, the
    # eigenvalue estimator steps when an eigenvalue crosses an integer, and
    # this sweep's eigenvalues sit on one (8.0 to fifteen places). Quoting
    # 0.971 would be quoting a figure a different BLAS could move in the third
    # decimal, which is the front page going red on somebody else's commit.
    # The bands below survive a trial count anywhere between six and ten.
    claims = (before.deflated > 0.95, after.deflated < 0.70, haircut > search)
    return Check(
        label=("30 crossover trials with a real edge, best selected, before and after financing"),
        measured=float(claims.count(False)),
        bound=0.0,
        rendered=(
            "DSR clears 0.95 gross and falls below 0.70 in excess of 5%, "
            "the financing haircut exceeding the search's own"
        ),
    )


def main() -> dict[str, object]:
    """Run every check, print the table, and return the figures."""
    noise, psr_hits, dsr_hits, runs = noise_selected_as_a_discovery()
    checks = [
        paper_worked_example(),
        expected_maximum_against_quadrature(),
        moments_against_scipy(),
        serial_correlation_against_closed_form(),
        path_count_against_the_identity(),
        log_tail_against_scipy(),
        the_excess_conventions_agree_for_a_constant_rate(),
        noise,
        financing_against_multiplicity(),
    ]

    print("What is checked, and against what")
    print()
    print("| Check | Result |")
    print("| --- | --- |")
    for check in checks:
        print(f"| {check.label} | {check.rendered} |")
    print()
    print("Measured against the bound the README states:")
    for check in checks:
        verdict = "ok" if check.passed else "BREACHED"
        print(f"  {check.label}: {check.measured:.3e} <= {check.bound:.0e}  {verdict}")
    print()
    print(f"  the noise sweep: PSR fooled in {psr_hits}/{runs}, DSR in {dsr_hits}/{runs}")

    failed = [check for check in checks if not check.passed]
    if failed:
        print()
        for check in failed:
            print(
                f"FAIL {check.label}: measured {check.measured:.6e}, "
                f"README claims {check.bound:.0e}",
                file=sys.stderr,
            )
        raise SystemExit(1)

    return {
        "checks": {check.label: check.measured for check in checks},
        "rendered": {check.label: check.rendered for check in checks},
        "psr_hits": psr_hits,
        "dsr_hits": dsr_hits,
        "runs": runs,
    }


if __name__ == "__main__":
    main()
