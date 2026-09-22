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
pip install -e ".[dev]"
python -m pytest
```

Python 3.10 or later. The only runtime dependency is NumPy; SciPy is used in
the test suite as an independent reference.

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
