# holdout

A backtest overfitting toolkit: probabilistic and deflated Sharpe ratios,
probability of backtest overfitting, purged cross-validation and
multiple-testing haircuts.

Every strategy search ends with a best result. The question this library
answers is how much of that result survives once the search itself is taken
into account — the number of variants tried, the non-normality of the returns,
the way the data was split, and the fact that the winner was chosen *because*
it looked best. Each statistic is validated against a published worked example,
a closed form, or a simulation of its own sampling distribution, and each
section below says which.

See [ROADMAP.md](ROADMAP.md) for the build order.

## Install

```bash
pip install holdout-backtest
holdout sample-data --out sample
holdout pbo sample/sweep.csv
```

> **Not on the package index yet.** The `pip install holdout-backtest` line above is
> what it will be; until the first release lands, install from source and the
> commands under it work unchanged:
>
> ```bash
> pip install "git+https://github.com/DaniyalMlk/holdout.git"
> ```

**The distribution is `holdout-backtest`; the import is `holdout`.** `pip
install holdout` fetches an unrelated project that was on the index first. The
Python package keeps the short name because renaming it would have broken every
existing import to settle a registry collision, so:

```python
import holdout  # installed from holdout-backtest
```

To work on the library instead:

```bash
pip install -e ".[dev]"
python -m pytest
```

Python 3.10 or later. The only runtime dependency is NumPy; SciPy is used in
the test suite as an independent reference.

## Command line

Installing the package provides a `holdout` command. `sample-data` writes two
synthetic parameter sweeps — thirty moving-average crossover rules on ten
years of simulated GARCH prices, one market with nothing to find and one with
a slow cyclical drift — so everything below runs without real data.

```bash
holdout sample-data --out sample
holdout deflate sample/sweep.csv
holdout compare sample/trending.csv ma5_40 ma5_120
```

```
30 trials over 2520 periods
effective trials: average 11.5, eigenvalue 10.0, participation 2.2  (using eigenvalue)
selected:          ma30_120
sharpe:            0.29 annualised
expected maximum:  0.19 annualised, if nothing works
probabilistic:     0.820  (P[true sharpe > 0], ignoring the search)
deflated:          0.622  (the same, after the search)
```

```bash
holdout pbo sample/sweep.csv
```

```
30 strategies, 16 blocks, 12,870 splits
probability of backtest overfitting: 0.975
probability of out-of-sample loss:   0.638
degradation: oos = 0.0316 - 1.202 * is   (r^2 0.77)
most often selected in-sample:
  ma30_120              16.4%
  ...
```

```bash
holdout spa sample/trending.csv          # against zero; --benchmark COLUMN to compare
holdout mcs sample/trending.csv          # no benchmark needed; --statistic range
holdout sharpe sample/trending.csv --column ma10_120
```

The same statistics on the trending sweep: deflated ratio 0.971, SPA p-value
0.015, and Romano–Wolf names the rules that beat cash. Every command reads a
CSV with one row per period and one column per strategy; see
[`holdout/io.py`](src/holdout/io.py) for the layout.

## Examples

- [`examples/deflated_worked_example.py`](examples/deflated_worked_example.py)
  — the Bailey and López de Prado (2014) example step by step, ending with the
  track record the strategy would need after deflation (8.2 years).
- [`examples/parameter_sweep.py`](examples/parameter_sweep.py) — every
  statistic on both synthetic sweeps.
- [`examples/model_confidence_set.py`](examples/model_confidence_set.py) — both
  sweeps through the confidence set; the one with a real drift still keeps 29 of
  its 30 rules.
- [`examples/purged_cv.py`](examples/purged_cv.py) — a nearest-neighbour
  model with no skill scores a 0.20 forecast correlation under shuffled k-fold
  and none under purged k-fold.
- [`examples/sharpe_difference.py`](examples/sharpe_difference.py) — how often
  each variance of a Sharpe-ratio difference rejects a true null, in three
  regimes of serial dependence.

The test suite runs all five and checks the numbers they print.

## The design decision that mattered

**Every statistic states the convention it needs, and refuses inputs that
break it, instead of computing something plausible.** The formulas in this
field are short and unforgiving: an annualised Sharpe ratio passed with a
daily sample size, excess kurtosis where raw kurtosis belongs, or a
multiple-testing method applied to one ratio without the rest of its family
all produce a number of the right magnitude that is simply wrong. So:

- Sharpe ratios are per period everywhere; annualisation happens at the edge.
- (skewness, kurtosis) pairs that no distribution can have are rejected with a
  message that names the likely mix-up.
- `haircut_sharpe` refuses Holm and the FDR procedures, which need the whole
  family, and points to the function that takes it.
- The effective-trials estimator has no default, because the three estimators
  disagree and the choice should be visible at the call site.

The same principle drives the validation: every number in this README was
measured, and where a published figure did not reproduce from its own printed
inputs (the paper's DSR is computed from an unrounded benchmark) the test
suite records why rather than bending the formula to match.

## Sharpe ratio inference

```python
import numpy as np
from holdout import estimate_sharpe

returns = np.random.default_rng(0).normal(0.0006, 0.01, 1260)   # five years of days
sr = estimate_sharpe(returns, periods_per_year=252)

sr.value                          # per-period ratio
sr.annualised                     # value * sqrt(252)
sr.standard_error()               # Mertens: uses the sample skew and kurtosis
sr.standard_error("normal")       # Lo: assumes iid normal returns
sr.confidence_interval(0.95, annualise=True)
```

Two conventions are fixed once and used everywhere:

- **Sharpe ratios are per period** unless a name says otherwise. Every
  inference formula in the literature is stated for the per-period ratio and a
  sample size in periods; feeding it an annualised ratio with a daily sample
  size is off by a factor of `sqrt(252)` and nothing complains.
- **Kurtosis is raw**, 3 for a normal distribution, because the corrections
  are written in terms of `(kurtosis - 1) / 4`. Passing excess kurtosis for a
  skewed series usually produces a (skewness, kurtosis) pair that no
  distribution can have, and the library rejects it rather than computing a
  variance from it.

What was measured:

| Check | Result |
| --- | --- |
| Skewness and kurtosis against `scipy.stats`, both bias settings | agree to 1e-12 |
| Lo standard error vs simulated sampling distribution (normal, n = 500) | within Monte Carlo error |
| Same, negatively skewed returns (skew -2, kurtosis 9) | normal formula 10% too small; Mertens within 1.2% |
| 90% interval coverage, Student-t returns, 2000 samples | nominal rate within 2.5 points |
| Lo's `eta(q)` against its AR(1) closed form | exact |
| Adjusted annual ratio vs ratio of aggregated returns, AR(1) with phi = 0.4 | within 0.1%; naive `sqrt(12)` overstates by 47% |

`autocorrelation_adjusted_sharpe(returns, 12)` applies Lo's (2002)
serial-correlation factor in place of `sqrt(q)`. Smoothed or illiquid marks
produce positive autocorrelation, and positive autocorrelation is exactly the
case where the naive annualisation flatters a strategy.

## Probabilistic and deflated Sharpe ratio

```python
from holdout import (
    deflate_trials, effective_number_of_trials, minimum_track_record_length,
    probabilistic_sharpe_ratio, trial_correlation,
)

probabilistic_sharpe_ratio(0.1, 1250, benchmark=0.0, skewness=-0.5, kurtosis=6.0)
minimum_track_record_length(0.1, confidence=0.95)          # periods needed

# trials: a (periods, variants) matrix of every variant that was backtested
result = deflate_trials(trials)                             # best column, N = columns
n_eff = effective_number_of_trials(trial_correlation(trials), method="eigenvalue")
result = deflate_trials(trials, n_trials=n_eff)
result.probabilistic, result.deflated, result.expected_maximum
```

The probabilistic Sharpe ratio is the probability that the true ratio exceeds a
benchmark. The deflated Sharpe ratio raises that benchmark to the ratio the
best of `N` unskilled trials would be expected to show: the null for a
*selected* strategy is not "no skill" but "the best of `N` attempts at no
skill".

What was measured:

| Check | Result |
| --- | --- |
| Bailey and López de Prado (2014) worked example | SR0 = 0.1132 and DSR = 0.9004, as published |
| Exact expected maximum of `N` normals vs SciPy quadrature | agree to 1e-9 |
| The paper's closed-form approximation vs exact | +2.6% at 5 trials, +0.9% at 100, +0.1% at 10^6; -7.9% at 2 |
| 50 pure-noise strategies, best selected, 200 runs | PSR(0) > 95% in over 80% of runs; DSR > 95% in none |
| 49 noise strategies plus one with per-period SR 0.2 over 500 days | DSR > 95% in 53% of runs |

The last two rows are the practical point. Selecting the best of fifty
backtests makes a coin flip look like a discovery nine times out of ten; and
even an annualised Sharpe ratio above 3, hidden among forty-nine others, only
survives deflation half the time on two years of data.

One detail worth knowing: the paper's 0.9004 is computed from the unrounded
benchmark 0.113172. Recomputing from the printed 0.1132 gives 0.90026. The
test suite records both so that nobody bends the formula to match a
recomputation from rounded figures.

**Correlated trials.** The deflation assumes the `N` trials are independent;
a parameter sweep is not. `effective_number_of_trials` takes a correlation
matrix and one of three estimators, and the method is a required argument
because they disagree between the extremes. Six near-copies plus four
independent strategies count as 6.0 (eigenvalue, Li and Ji 2005), 7.0
(average correlation) or 2.5 (participation ratio). The eigenvalue method is
exact for blocks of identical trials but jumps where an eigenvalue crosses an
integer; the participation ratio is smooth but dominated by the largest block.

## Multiple testing and haircut Sharpe ratios

```python
from holdout import adjust_pvalues, haircut_sharpe, haircut_sharpe_ratios, minimum_sharpe

adjust_pvalues(pvalues, "holm")                     # also bonferroni, sidak, bh, by
haircut_sharpe(0.063, 2520, n_tests=100)            # one ratio, single-step adjustment
haircut_sharpe_ratios(sharpes, 2520, method="bh")   # a whole family, step-wise
minimum_sharpe(2520, 100, periods_per_year=252)     # 1.10: the bar after 100 tests
```

Family-wise methods (Bonferroni, Šidák, Holm) bound the chance of *any* false
discovery; the false-discovery-rate methods (Benjamini–Hochberg,
Benjamini–Yekutieli) bound the expected *share* of discoveries that are false.
Haircuts follow Harvey and Liu (2015): turn the Sharpe ratio into a
t-statistic, adjust its p-value, and turn it back. A single ratio only admits
the single-step adjustments, because Holm and the FDR procedures depend on the
rest of the family; the library refuses rather than guessing the others.

The haircut is not proportional, which is the useful thing to know about it.
On ten years of daily data, the share of the Sharpe ratio removed is:

| Annualised Sharpe ratio | 10 tests | 100 tests | 1000 tests |
| --- | --- | --- | --- |
| 0.75 | 43% | 100% | 100% |
| 1.0 | 24% | 55% | 100% |
| 1.5 | 10% | 22% | 35% |
| 2.0 | 6% | 12% | 18% |

What was measured:

| Check | Result |
| --- | --- |
| Adjusted p-values vs each procedure as usually stated, arbitrary inputs, six levels | identical rejection sets |
| Family-wise error, 20 independent nulls, 4000 runs, nominal 5% | Bonferroni 4.5%, Šidák 4.6%, Holm 4.5% |
| False discovery rate, 40 nulls and 10 alternatives, nominal 10% | BH 7.8% (bound: 8%), BY 1.9% |
| BH with equicorrelated (0.6) test statistics | at or below nominal |

## Probability of backtest overfitting

```python
from holdout import probability_of_backtest_overfitting

result = probability_of_backtest_overfitting(trials, n_blocks=16)   # (periods, variants)
result.pbo                    # share of splits where the winner lands at or below the median
result.logits                 # one per split: logit of the winner's out-of-sample rank
result.degradation            # (slope, intercept, r^2) of out-of-sample on in-sample
result.probability_of_loss    # share of splits where the winner loses money out of sample
result.selection_frequency()  # how often each variant won in-sample
```

Combinatorially symmetric cross-validation (Bailey, Borwein, López de Prado and
Zhu 2017) cuts the sample into `S` contiguous blocks and tries every one of the
`C(S, S/2)` ways of using half of them to pick a winner and the other half to
judge it. The deflated Sharpe ratio corrects one number for the size of the
search; PBO asks of the search itself how often its choice fails to generalise.

The paper's `S = 16` means 12,870 splits. For the Sharpe ratio and the mean
each split is computed from per-block sums rather than by re-slicing the data:
on a 2520 × 100 matrix that is 0.1 s against 11 s for the general path, which
remains available for any metric passed as a callable. The sums are taken over
centred returns, because the variance is a difference of sums and cancels
badly otherwise — from raw sums, a series with mean 1 and spread 1e-4 loses its
variance to a relative error of 8e-8.

What was measured:

| Check | Result |
| --- | --- |
| Fast path and general path vs a plain loop written from the paper | agree to 1e-10 (1e-12 for the large-mean case) |
| Pure noise, 20 strategies, 30 runs | mean PBO 0.51 |
| One strategy with a genuine edge among 20 | PBO below 0.01; selected in over 99% of splits |
| Strategies demeaned over the full sample (the in-sample winner must lose) | PBO exactly 1 |

## Cross-validation for overlapping labels

```python
from holdout import combinatorial_purged_cv, leakage_audit, purged_kfold, walk_forward

# observation i is made at start[i]; its label (say a 5-day forward return) is known at end[i]
splits = purged_kfold(5, start=start, end=end, embargo=10)
leakage_audit(splits, start, end)          # raises LeakageError if any label overlaps

cv = combinatorial_purged_cv(6, 2, start=start, end=end, embargo=10)
cv.n_paths                                  # 5
paths = cv.assemble_paths(per_split_oos_returns)   # (5, n): five full backtests
```

When a label spans several periods, two neighbouring observations share part
of their outcome, and ordinary k-fold keeps putting one in training and the
other in test. Purging drops every training observation whose label overlaps a
test fold; the embargo also drops the observations just after each test fold
(López de Prado 2018, chapter 7). Combinatorial purged cross-validation
(chapter 12) tests every choice of `k` of `N` groups, so each group is tested
`C(N-1, k-1)` times and the out-of-sample results stitch into that many
complete backtest paths — a distribution of Sharpe ratios rather than one.

`leakage_audit` checks every training label against every test label
directly, so it does not trust the splitter it is checking. A property test
runs purged k-fold over random label lengths, embargoes and fold counts and
requires the audit to pass; plain k-fold on five-period labels is required to
fail it. The CPCV tests check the path count against `k/N * C(N, k)` and that
every path covers each observation exactly once.

## Tests of superior predictive ability

```python
from holdout import reality_check, romano_wolf, superior_predictive_ability

# differentials: (periods, strategies) of strategy return minus benchmark return
reality_check(differentials, seed=0).pvalue
spa = superior_predictive_ability(differentials, seed=0)
spa.consistent                     # bounded by spa.lower and spa.upper
romano_wolf(differentials, alpha=0.05, seed=0).rejected   # which ones, strongest first
```

These answer "does anything here beat the benchmark?" without assuming the
strategies are independent of each other or over time. All three resample
whole rows with the stationary bootstrap (Politis and Romano 1994), so every
strategy is resampled on the same days and the correlation between them is
kept. The block length defaults to the Politis–White (2004, corrected 2009)
estimate.

- **White's Reality Check** (2000) is valid but recentres every strategy to
  mean zero, so hopeless strategies count as contenders and dilute it.
- **Hansen's SPA test** (2005) studentises and leaves clearly inferior
  strategies at their negative mean. `consistent` is the p-value to report.
- **Romano–Wolf** (2005) stepdown names the strategies that beat the
  benchmark while controlling the family-wise error rate.

What was measured:

| Check | Result |
| --- | --- |
| Block length vs the AR(1) closed form, n = 10,000–20,000 | within 10% on average for phi = 0.3, 0.5, 0.7 |
| Size at the least favourable null (6 strategies, 2000 runs, nominal 5%) | Reality Check 4.9%, SPA 5.4%, Romano–Wolf 5.5% |
| One strategy at the benchmark plus nine poor ones (300 runs) | Reality Check rejects 0.7% of the time, SPA 6.0% |
| One genuine strategy plus nine poor ones (300 runs) | SPA rejects 87% of the time, the Reality Check 56% |
| Romano–Wolf, two genuine strategies among eight | both found in every one of 300 runs |

The third and fourth rows are Hansen's argument in numbers: the Reality Check
is not wrong, but every poor strategy added to the set makes it harder for a
good one to register.

## The model confidence set, when there is no benchmark

```python
from holdout import Statistic, model_confidence_set

# performance: (periods, models), higher better — returns, or losses negated
result = model_confidence_set(returns, alpha=0.10, seed=0)
result.included            # indices in the set, best average first
result.pvalues             # one per model; in the set when above alpha
result.at(0.25)            # any other level, without rerunning the bootstrap
model_confidence_set(returns, statistic=Statistic.RANGE, seed=0)
```

Everything above needs a benchmark. This does not, which is why it answers the
question people actually arrive with: of twenty candidates with no incumbent
among them, which can be told apart from the best?

Nominating the sample-best as the benchmark and testing the rest against it is
not a repair. The benchmark is then chosen by the same data the test runs on, so
under the null it is systematically the luckiest series present and every
comparison is biased towards finding nothing. Hansen, Lunde and Nason (2011)
turn it around: test whether all the models are equally good, drop the one the
test most implicates if it is rejected, and repeat until it is not.

**The size of the set is the result.** A set holding 29 of 30 models is not the
procedure failing; it is the procedure saying that ten years of daily data cannot
separate 29 parameter choices. `examples/model_confidence_set.py` is exactly
that case, and it is the uncomfortable one:

| Sweep | deflated Sharpe | set at 10% (max) | set at 10% (range) |
| --- | --- | --- | --- |
| no drift in the market | 0.622 | 30 of 30 | 30 of 30 |
| a genuine slow drift | 0.971 | **29 of 30** | 30 of 30 |

The second row has a real signal in it — the best rule clears the search that
produced it — and the surviving set still spans 9.3% of annualised mean return.
"This rule made money" and "this rule is the one that made money" are different
claims, and a parameter sweep is usually read as making the second.

A smaller `alpha` gives a **larger** set. It is a confidence region, so the usual
asymmetry of hypothesis testing runs backwards, and the default of 0.10 follows
the paper rather than the 0.05 habit. One bootstrap produces every level, since
the p-values do not depend on it — only the reading of them does.

**Both statistics are here because neither dominates.** `MAX` studentises each
model against the average of the set; `RANGE` takes the largest studentised
difference over all pairs. Over fifteen borderline samples of ten models they
retained 8.13 and 7.07 models on average, the range set smaller on fourteen of
the fifteen and larger on one — a tendency and not a guarantee. `MAX` is the
default because a larger set is the weaker claim.

They also differ on degeneracy, and the difference is real rather than an
oversight. Two duplicated columns have no standard error between them, so `RANGE`
refuses; `MAX` never forms that pairwise error and carries on, giving both copies
the same p-value. What it cannot then do is tell the caller that three of its ten
"models" are one strategy implemented three times.

The zero it checks for is **relative, not absolute**. Two identical columns have
means that differ in the last bits of the mantissa rather than not at all, so
their bootstrap deviations come out near 1e-17 and a `scale > 0` test passes
them — after which the studentised difference is of order 1e16 and a model is
eliminated on a p-value of zero, as though the evidence against it were
overwhelming rather than absent.

## Comparing exactly two strategies

Everything above is built for families. `reality_check` and
`superior_predictive_ability` test many candidates against one benchmark with the
selection accounted for; `model_confidence_set` finds the set that cannot be told
apart from the best. Pointing any of it at two strategies is applying a
multiple-testing correction to a single test.

The pairwise comparison has its own answer:

```python
from holdout import sharpe_difference

result = sharpe_difference(strategy_a, strategy_b, periods_per_year=252)
result.annualised_difference   # +0.59 of annualised Sharpe ratio
result.correlation             # 0.656 — half of why this is not two separate tests
result.closed_form_error       # Jobson–Korkie with Memmel's correction
result.robust_error            # Ledoit–Wolf: delta method with a Newey–West covariance
result.error_ratio             # the robust error over the closed-form one
result.p_value                 # two-sided, using whichever `method` asked for
```

Differencing two `estimate_sharpe` standard errors is not a substitute, and it is
wrong twice. It ignores the correlation between the series — for two strategies on
the same market that is usually large, and it always *reduces* the variance of the
difference, so the naive interval is too wide and a real difference goes unreported.
And it treats a Sharpe ratio as if it were a mean, when it is a ratio of two
estimated moments and the delta method has terms a difference of two independent
errors does not.

Both variances always come back, whichever the `method` argument selects for the
headline statistic, because the disagreement between them is the finding when there
is one.

### What measuring them showed

`examples/sharpe_difference.py` runs a thousand replications in each of three
regimes, with the two series correlated at 0.7, the same true Sharpe ratio, and
five hundred paired periods. A test at the 5% level should reject 5% of the time.

| serial dependence | closed form | robust | robust error / closed-form error |
|---|---|---|---|
| none | 4.2% | 4.7% | 0.99 |
| AR(1), ρ = 0.3 | 14.4% | 6.4% | 1.27 |
| AR(1), ρ = 0.6 | **34.7%** | 10.0% | 1.65 |

With independent returns the closed form is right and the robust version costs
nothing: both land within a percentage point of nominal and the two standard errors
agree to within one per cent. So there is no reason to prefer the closed form even
where its assumptions hold.

Serial dependence is where they part, and both halves of that are worth stating. At
a persistence of 0.6 the closed form rejects a true null more than six times too
often — a third of the time it reports a difference in Sharpe ratios that is not
there — because it assumes independence and the difference of two persistent series
has far more sampling variability than its formula allows. The robust version gets
that to one in ten, which is better and **is not right**: a Bartlett kernel
truncated at a rule-of-thumb bandwidth recovers only part of the long-run variance.
Use the robust variance, and do not read its p-value as exact on data you believe is
persistent.

### Two refusals

Comparing a series with itself is refused rather than answered. The difference is
exactly zero and so is its variance, so the statistic is 0/0, and "these are the
same strategy" is not the same statement as "the difference between these two is not
significant" — a p-value of one would read as the second.

A series that does not move has no Sharpe ratio, and the guard for it has to be
*relative*. numpy's standard deviation of a constant array is not exactly zero:
subtracting the mean leaves rounding of order `eps × level`, which on 0.001 repeated
500 times is 4.3e-19. A guard at zero does not fire, the Sharpe ratio comes back as
2.3e15, and every number after it is arithmetic on rounding error.

## Was the edge there throughout?

Everything above guards a maximum taken over *strategies*. `spa` asks whether the
best of K beats a benchmark once K is accounted for, `romano_wolf` asks which ones
do, `deflate` raises the benchmark to what the best of N unskilled trials would
show, `pbo` asks how often the in-sample winner loses out of sample.

Nothing asked whether a strategy earned its Sharpe ratio evenly or earned all of it
in one stretch. And the obvious check has this library's own flaw, twice.

**The split point gets chosen after looking at the equity curve.** Nobody splits a
track record with a visible cliff two thirds along at the midpoint. The comparison
is made at the most damaging date available and read against a critical value for
one date fixed in advance.

**Searching honestly does not fix it.** Take the largest statistic over every
candidate date and a normal critical value is the wrong distribution, for exactly
the reason White's Reality Check exists: the maximum of several hundred correlated
statistics is not distributed like one of them.

```
$ holdout stability sweep.csv --column ma5_40
2520 periods at 252 per year, 1765 candidate break dates, trim 0.15 (annualised)

strategy  break after  sharpe before  sharpe after  sup stat  p (bootstrap)  p (naive)
--------------------------------------------------------------------------------------
ma5_40            414          -0.25          1.04      1.51          0.706     0.1298
```

So the statistic here is a supremum by construction — there is no single-date
version of it to misuse — and its null distribution comes from the stationary
bootstrap, which handles the maximum and any serial correlation in one pass. The
naive p-value is printed beside the real one because the gap is the point: a reader
who has been quoting the naive one should see what it was worth. Above, 0.13 against
0.71.

### How wrong the naive p-value is, measured

Over 500 replications of 1,000 iid normal returns with **no break at all**:

| | rejects at 5% | trim 0.02 |
|---|---|---|
| two-sided normal on the supremum | 41.4% | 58.8% |
| stationary bootstrap | 3.2% | 7.0% |

The Monte Carlo standard error is about one percentage point. The bootstrap is
conservative at the default 15% trim and liberal at an aggressive 2% one, so the
trim is an argument rather than a constant — and worth choosing before seeing the
answer rather than after.

### The most useful thing it reports is how little power it has

Over 120 replications at 500 observations either side — two four-year halves of
daily returns, a longer track record than most things get:

| per-period Sharpe | annualised | detected at 5% |
|---|---|---|
| 0.10 → 0.05 | 1.59 → 0.79 | 10% |
| 0.10 → 0.00 | 1.59 → 0.00 | 25% |
| 0.12 → −0.04 | 1.90 → −0.63 | 52% |

A strategy that loses its entire edge at the midpoint is found a quarter of the
time. So a large p-value here is close to no evidence, and reading one as
confirmation that a strategy is stable is the mistake this test makes easiest —
which is why the command says so in its own output. It is worth having anyway:
when it does reject, it rejects against the right distribution.

### The standard error is the one that accounts for shape

Two disjoint subsamples, each with its Mertens (2002) standard error from its own
skew and kurtosis, and their difference standardised by the root of the sum of
squares. The normal-returns version would be simpler and wrong in the direction
that matters: a strategy with negative skew and fat tails has a noisier Sharpe
ratio than the normal formula says, and a break test that understates the noise
finds breaks that are not there.

### Power sums, shifted, because a million Sharpe ratios cannot be a loop

A thousand resamples over a thousand days is 1.4 million subsample Sharpe ratios
with their skew and kurtosis. Cumulative power sums make that four cumulative sums
per resample.

They also cancel catastrophically on returns whose mean is a hundredth of their
standard deviation, which is every return series there is — the sum of squares and
the square of the sum agree to two significant figures and their difference loses
the rest. So the sums are taken on `x - mean(x)` and the mean added back:
subtracting a constant leaves every subsample's variance, skew and kurtosis
unchanged and moves its mean by exactly that constant.

### Two guards, and what it took to get each right

**The zero-variance check has to be relative.** A constant stretch's second moment
computed from power sums is not zero but the rounding left over from subtracting
two nearly equal sums — of order 1e-22 on daily returns. Divided into a mean of
0.001 that is a Sharpe ratio of 1e8 and a break statistic of **−2e9**: large enough
to dominate every supremum in the sample, and finite enough to pass every check for
a NaN. Only a threshold relative to the whole sample's own variance catches it, and
a genuinely quiet stretch at a tenth of the sample's volatility is nowhere near it.

**The refusal for a sample with no candidate break was nearly deleted as
unreachable.** The argument for deleting it covers even sample lengths only: with
`trim < 0.5` the edge is capped at half an even sample and a candidate always
survives. For an *odd* length the ceiling reaches `(total + 1) / 2` and takes the
last candidate away. A sweep of every length from 60 to 400 against every trim to
0.499 reaches it 389 times — every one of them odd, every one at a trim above
0.492. The branch stays, and the sweep is the test.

## The other half of the overlapping-label problem

The section above purges overlapping labels out of the training set, which fixes
the leakage. It does not fix the counting. Twenty observations whose five-day
windows cover the same week carry about one week of information between them,
and every estimator that averages over observations treats them as twenty. The
symptom is a t statistic too large for the data behind it, and purging never
touches it, because nothing has leaked.

`uniqueness` supplies the number to divide by instead of `n`. Concurrency is how
many label windows span each bar; an observation's average uniqueness is the
mean of `1 / concurrency` over its own window; the sum of those is the effective
sample size. On a purged training set of 120 observations with ten-bar rolling
windows it comes out under a quarter of the length.

One identity holds whatever the overlap looks like, and it is the check that the
arithmetic is right rather than merely self-consistent — both sides count the
same thing:

    sum over i of (uniqueness_i * length_i) == number of covered bars

### The sequential bootstrap is worth less than it is sold for

Drawing with a probability proportional to each candidate's uniqueness given
what is already drawn sounds like it fixes the redundancy at its source.
Measured, it mostly does not, and the reason is arithmetic. Total concurrency
over a draw is `size * length` however it is drawn, so achievable uniqueness is
capped at `span / (size * length)`, and attaining the cap means flattening the
concurrency — which is all any scheme can try to do. As a fraction of that cap,
over sixty seeds on two hundred observations with twenty-bar windows:

| draws | cap | sequential | uniform | gain |
| --- | --- | --- | --- | --- |
| 200 | 0.0548 | 0.998 | 0.995 | +0.44% ± 0.09% |
| 40 | 0.2737 | 0.978 | 0.946 | +3.47% ± 0.55% |
| 10 | 1.0 | 0.729 | 0.673 | +10.43% ± 2.31% |
| 5 | 1.0 | 0.865 | 0.809 | +9.37% ± 2.80% |

The heavy-overlap, full-size resample is the case the method is motivated by and
the case in which it cannot help: the uniform bootstrap is already at 99.5% of a
cap nothing can exceed. The gain is real only once the draw is small against the
span. Its largest gain of all is on non-overlapping point labels, where there is
no overlap problem and the only redundancy left is duplicate draws — which is
the clearest sign of what it is actually doing.

So the remedy for heavy overlap is to draw fewer observations, and the effective
sample size says how many.

Weights come two ways: proportional to uniqueness, and a time decay that is
linear in *cumulative uniqueness* rather than in time, so a stretch of redundant
observations ages like the information it carries. Ten redundant observations
worth one observation out of eleven get 7.8% of the weight on that scale against
26.2% on an index-linear one.

```bash
holdout uniqueness --count 200 --window 20
```

## The drawdown a track record owes to luck

Every other statistic here asks what a *search* was worth. This one asks what
one realised path owes to chance: a track record's worst drawdown is the number
that ends a mandate, and it is also a number a strategy with a real edge
produces routinely.

The drawdown of a Brownian motion is that motion reflected at its own running
peak, so the chance of never falling `h` below a high-water mark is the survival
function of a first passage with a reflecting boundary at zero and an absorbing
one at `h`. The backward equation separates, and the eigenvalues are the roots
of `k cos(kh) + b sin(kh) = 0` with `b = -mu / sigma^2` — one in each branch of
the tangent, so every root is bracketed before it is solved for.

```python
from holdout import assess_drawdown, drawdown_quantile, expected_maximum_drawdown

expected_maximum_drawdown(horizon=252.0, drift=0.0, volatility=0.15 / 252**0.5)
# 0.1879...  -- a 17.1% fall, from a strategy with no edge whatsoever
drawdown_quantile(0.95, horizon=252.0, drift=0.0, volatility=0.15 / 252**0.5)
# 0.3362...  -- the one-year-in-twenty depth for the same non-strategy
```

```bash
holdout drawdown sample/trending.csv --column ma5_40
```

### What a drawdown is worth knowing before it happens

At 15% annual volatility, the expected worst drawdown and the depth exceeded one
record in twenty, as a fraction of the peak:

| Sharpe | 1 year | 3 years | 10 years |
| --- | --- | --- | --- |
| 0.0 | 17.1% / 28.6% | 27.8% / 44.1% | 44.8% / 65.5% |
| 0.5 | 14.8% / 24.8% | 22.1% / 35.4% | 31.5% / 47.3% |
| 1.0 | 13.0% / 21.5% | 18.2% / 28.5% | 24.3% / 35.4% |
| 1.5 | 11.5% / 18.7% | 15.5% / 23.7% | 20.0% / 28.5% |

Two things are worth reading off it. Without an edge the drawdown grows like the
square root of the horizon and without bound; with one it grows logarithmically,
approaching `sigma^2 / (2 mu)` per doubling, so a drawdown limit scaled to the
length of a track record is scaled to the wrong thing. And the first row is the
answer to "is 28% too much": for a strategy with no edge, a year in twenty, no.

The waiting time is exponential in the depth, which is the same fact seen from
the other side. At a 1.0 Sharpe ratio and 15% volatility a 20% fall takes 7.8
years to arrive on average and a 40% fall takes 450.

### Three routes to the same law, and two that stop short

The expansion is checked against references that share no derivation with it.

- **The driftless case has a reflection series.** Lévy's theorem makes the
  drawdown of a driftless Brownian motion a reflected Brownian motion, whose
  maximum is distributed as the maximum absolute value of the original, and that
  has a classical image series in the normal distribution function. The two
  agree to 2e-16 over a grid.
- **The first passage has a closed-form Laplace transform at every drift**, from
  the same differential equation solved once rather than separated. Matching it
  at several arguments tests the eigenvalues and the coefficients jointly; the
  worst disagreement over 225 parameter sets is 3e-10 relative to the size of
  the coefficients, and most are at 1e-11.
- **The coefficients are inner products** and can be read off by quadrature
  against the speed measure. They agree to 1e-15.
- **The expected drawdown has one closed form**, `sigma sqrt(pi T / 2)` at zero
  drift, which the tail integral reproduces to thirteen figures.

Two things do not work and are reported rather than papered over.

The exceedance is one less a survival probability, and for a level the path will
almost certainly not reach that is a cancellation: below about `1e-13` the
answer has no significant digits, and no rearrangement of the series fixes it,
because the term-by-term complement converges only conditionally. What is
exposed instead is `final_drawdown_exceedance`, the exact law of the drawdown at
the *end* of the horizon — a rigorous lower bound on the maximum's, running at
0.36 to 0.45 of it, and accurate past `1e-300`.

And the obvious asymptote for that tail is wrong. Treating the first passage as
exponential with its own exact mean — the natural reading of "a drawdown this
deep arrives once every `E[tau]` years" — overstates a one-year 60% drawdown at
a 0.67 Sharpe ratio by a factor of 520, and the error *grows* as the level
deepens, which is the shape of a wrong exponent rather than a loose constant.
The two tails are different: exponential at rate `2 mu / sigma^2` in the
horizon, Gaussian in the level.

### Where the expansion is well conditioned, stated rather than assumed

The coefficients carry `exp(-mu h / sigma^2)`, which exceeds one only for a
*losing* strategy. There the survival probability is a cancellation of terms far
larger than the answer: the largest coefficient grows like
`0.76 exp(beta) / beta` in `beta = -mu h / sigma^2`, measured, and the absolute
error is about `eps` times that. So the domain is a property of the method and
is enforced with an error that names the reason. Nothing at or above zero drift
is affected.

The budget is deliberately split. A probability a caller reads directly is held
to an absolute error near `1e-9`; the expected drawdown and the quantiles are an
integral and an inversion, whose error budgets absorb five orders more, and
spending that carries the losing-side domain out to a total Sharpe ratio of
-3.23.

### Daily marks see less of the path than the path contains

A drawdown read off 252 daily observations is smaller than the drawdown of the
path underneath them — 7% smaller in the mean, falling to 1.4% at sixteen times
the frequency, with the shortfall dying like one over the square root of the
count. That is not an error in the estimate: the discretely observed drawdown is
a different contract, and it is the one a track record reports.

It does mean a test against the continuous law is conservative, and by how much
is worth a number. On simulated records of 252 observations with the drift and
volatility *known*, a nominal 5% test rejects 3.4% of the time. Estimating those
two parameters from the same record takes the same test to 0.27% — a further
factor of thirteen, and much the larger effect, because a path that fell a long
way also reports a larger volatility and so makes its own drawdown look
ordinary. Both effects are one-sided, so `assess_drawdown` carries a flag saying
which case it is, and the command line says in words that an estimated null does
not give a p-value.

## Layout

| Module | Contents |
| --- | --- |
| `series` | input validation for return series and matrices |
| `moments` | skewness, raw kurtosis, autocorrelation |
| `sharpe` | Sharpe ratio, Lo and Mertens standard errors, Lo's serial-correlation factor |
| `deflated` | probabilistic and deflated Sharpe ratios, minimum track record, expected maximum |
| `trials` | effective number of independent trials |
| `multiple` | adjusted p-values, haircut Sharpe ratios, minimum t-statistic |
| `pbo` | combinatorially symmetric cross-validation |
| `splits` | walk-forward, purged k-fold, combinatorial purged CV, leakage audit |
| `uniqueness` | concurrency, average uniqueness, sample weights, sequential bootstrap |
| `bootstrap` | stationary bootstrap and Politis–White block length |
| `spa` | Reality Check, SPA, Romano–Wolf |
| `mcs` | model confidence set, both statistics |
| `pairwise` | difference of two Sharpe ratios, Memmel and Ledoit–Wolf variances, Newey–West |
| `drawdown` | drawdown statistics, the exact maximum-drawdown law, calibrated limits |
| `io`, `synthetic`, `cli` | CSV input, the sample sweeps, the `holdout` command |
