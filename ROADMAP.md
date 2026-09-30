# Roadmap

A library for deciding whether a backtest is evidence of anything. Every
strategy search produces a best result; the question is how much of that result
survives once the search itself is accounted for. The phases build from the
inference on a single Sharpe ratio outwards to the whole search: many trials,
many splits of the data, and many strategies tested against one benchmark.

## Phase 1 — Return series and Sharpe ratio inference

- [x] Return series validation with errors that name the offending observation
- [x] Sample skewness and kurtosis with the bias convention stated, checked against SciPy
- [x] Sharpe ratio with annualisation, and its standard error under normal (Lo) and non-normal (Mertens) returns
- [x] Serial-correlation-adjusted annualisation (Lo)
- [x] Packaging, type checking, linting and continuous integration

## Phase 2 — Probabilistic and deflated Sharpe ratio

- [x] Probabilistic Sharpe ratio against an arbitrary benchmark
- [x] Minimum track record length
- [x] Expected maximum Sharpe ratio across independent trials, checked against simulation
- [x] Deflated Sharpe ratio reproducing the published worked example
- [x] Effective number of independent trials from a correlation matrix of trial returns

## Phase 3 — Multiple testing

- [x] Bonferroni, Šidák, Holm, Benjamini–Hochberg and Benjamini–Yekutieli adjusted p-values, checked against their definitions
- [x] Harvey–Liu haircut Sharpe ratio under each adjustment
- [x] Minimum t-statistic for a given number of trials and error rate
- [x] Family-wise error and false discovery rates checked by simulation

## Phase 4 — Probability of backtest overfitting

- [x] Combinatorially symmetric cross-validation over a performance matrix
- [x] Probability of backtest overfitting and the logit distribution behind it
- [x] Performance degradation and probability of loss out of sample
- [x] Validation: pure noise sits near one half, a genuine edge near zero

## Phase 5 — Cross-validation for overlapping labels

- [x] Walk-forward splits, expanding and rolling
- [x] Purged k-fold with an embargo on label intervals
- [x] Combinatorial purged cross-validation with backtest path assembly
- [x] Leakage audit that fails if any training label overlaps a test label

## Phase 6 — Bootstrap tests of superior predictive ability

- [x] Stationary bootstrap with automatic block length selection
- [x] White's Reality Check
- [x] Hansen's test for superior predictive ability
- [x] Romano–Wolf stepdown to name which strategies beat the benchmark
- [x] Size and power checked by simulation

## Phase 7 — Command line and examples

- [x] CSV loading for single series and strategy matrices
- [x] `holdout` command with `sharpe`, `deflate`, `pbo`, `spa` and `sample-data`
- [x] Worked examples runnable end to end
- [x] Clean-wheel installation checked in continuous integration

## Phase 8 — Installable from a package index

- [x] Distribution name distinct from the taken one, import name unchanged
- [x] SPDX licence expression, with the licence file inside both artefacts
- [x] `--version` checked against the packaged metadata, not just printed
- [x] Metadata tests: version agreement, typing marker, entry points, licence
- [x] Tag-driven release with a version guard and no stored credential
- [ ] First release on the index

## Phase 9 — A confidence set, with no benchmark to nominate

Every test in phase 6 needs a benchmark, so none of them answers the question a
parameter sweep actually raises: of thirty candidates with no incumbent among
them, which can be told apart from the best? Nominating the sample-best as the
benchmark is not a repair — it is chosen by the data the test runs on, so under
the null it is the luckiest series present.

- [x] The equivalence test, the elimination rule, and the sequence repeated
      until it stops rejecting
- [x] Both statistics from Hansen, Lunde and Nason (2011), since neither
      dominates the other
- [x] A p-value per model, monotone in the elimination order, so every level is
      readable from one bootstrap
- [x] The stationary bootstrap already here, resampling every model on the same
      rows
- [x] Coverage of the true best model checked by simulation
- [x] Degenerate columns refused against a relative threshold, not an absolute one
- [x] An `mcs` command, and a worked example over both sweeps

Measured. On the driftless sweep all thirty rules survive under both statistics.
On the sweep with a genuine drift — deflated Sharpe ratio 0.971, so the best rule
clears the search that produced it — 29 of 30 rules are still in the 10% set, and
the surviving set spans 9.3% of annualised mean return.

Coverage: over 120 replications with five models 0.04 apart in true mean, the
true best model was in the 10% set 120 times out of 120 under both statistics.
Over-coverage is expected and is not a defect — the bound is asymptotic and
one-sided, and the sequence stops at the first failure to reject rather than
testing every subset.

The two statistics retained 8.13 and 7.07 models on average over fifteen
borderline samples, the range one smaller on fourteen of fifteen and larger on
one. So the ordering is a tendency. The default is the max statistic, because a
larger set is the weaker claim.

The degeneracy check is the part that would have been wrong the obvious way. Two
identical columns have means differing in the last bits of the mantissa rather
than not at all, so their bootstrap deviations are around 1e-17 and `scale > 0`
passes them — after which the studentised difference is of order 1e16 and the
model is eliminated on a p-value of zero, which reads as overwhelming evidence
rather than none. The threshold is relative to the largest standard error in the
same set.

## Phase 10 — Comparing exactly two strategies

The library could test one candidate against a benchmark by bootstrap, many against
one with a stepwise correction, and find the set indistinguishable from the best.
It could not do the comparison people actually make most often. Pointing the family
machinery at two strategies applies a multiple-testing correction to a single test,
and differencing two `estimate_sharpe` standard errors ignores both the correlation
between the series and the fact that a Sharpe ratio is a ratio of two estimated
moments.

- [x] Jobson-Korkie with Memmel's correction, the closed form under independent
      normal returns
- [x] Ledoit-Wolf: the delta method over the four sample moments with a
      Newey-West covariance, robust to non-normality and serial dependence
- [x] Both returned from one call, because their disagreement is the finding
- [x] The Bartlett-kernel HAC covariance and the conventional bandwidth rule,
      exposed rather than applied silently
- [x] A `compare` command printing both variances, both statistics and the
      bandwidth used
- [x] The size of both tests measured in a worked example that runs in the suite

The measurement, a thousand replications per regime on series correlated at 0.7
with the same true Sharpe ratio and 500 paired periods, against a nominal 5%:

| serial dependence | closed form | robust | error ratio |
|---|---|---|---|
| none | 4.2% | 4.7% | 0.99 |
| AR(1), rho = 0.3 | 14.4% | 6.4% | 1.27 |
| AR(1), rho = 0.6 | 34.7% | 10.0% | 1.65 |

With independent returns both hold their size and the two standard errors agree to
within one per cent, so the robust version costs nothing where it is not needed.
Under dependence the closed form rejects a true null more than six times too often
at a persistence of 0.6, and the robust version gets that to one in ten — better,
and not right, because a Bartlett kernel truncated at a rule-of-thumb bandwidth
recovers only part of the long-run variance. Both halves are in the README rather
than only the favourable one.

One defect found while writing the guards. numpy's standard deviation of a constant
array is not exactly zero: subtracting the mean leaves rounding of order
`eps * level`, 4.3e-19 on 0.001 repeated 500 times. A guard at zero therefore does
not fire, the Sharpe ratio comes back as 2.3e15, and everything downstream is
arithmetic on rounding error. The threshold is now relative to the series' own
magnitude, which also makes it mean the same thing whether returns are quoted as
fractions or in basis points.

Comparing a series with itself is refused rather than answered with a p-value of
one. The difference and its variance are both exactly zero, so the statistic is
0/0, and "these are the same strategy" is a different statement from "the
difference is not significant".

## Phase 11 — Was the edge there throughout?

- [x] The standardised Sharpe-ratio difference at every candidate break date,
      using this library's own standard error under non-normal returns rather than
      the normal one
- [x] Its supremum, and where it falls
- [x] A stationary-bootstrap null distribution for that supremum, so the maximum
      and any serial correlation are handled together
- [x] The naive p-value beside it, and the size distortion between them measured
- [x] The power measured too, because it is the finding
- [x] Vectorised over candidate dates and resamples, since 1.4 million subsample
      Sharpe ratios with their skew and kurtosis cannot be a Python loop
- [x] A command-line entry point that says what a large p-value is worth

The measurement. Over 500 replications of 1,000 iid normal returns with no break at
all, reading the supremum against a two-sided normal 5% critical value rejects
41.4% of the time and the bootstrap rejects 3.2%, against a nominal 5% and a Monte
Carlo standard error of about one point. At a 2% trim rather than the default 15%
those become 58.8% and 7.0%, so the bootstrap absorbs most of the extra and not all
of it, and the trim is an argument worth fixing before seeing the answer.

The more useful measurement is the power, and it is poor. At 500 observations
either side, a strategy whose per-period Sharpe ratio falls from 0.10 to 0.05 is
detected 10% of the time, from 0.10 to 0.00 — a complete loss of edge — 25%, and
from 0.12 to −0.04 52%. So a large p-value here is close to no evidence, which the
module docstring and the command's own output both say. The test earns its place
because when it does reject it rejects against the right distribution, not because
it finds much.

Two guards, and neither was right first time.

The zero-variance check had to become relative. A constant stretch's second moment
from power sums is not zero but the rounding left from subtracting two nearly equal
sums, of order 1e-22 on daily returns; divided into a mean of 0.001 that is a
Sharpe ratio of 1e8 and a break statistic of −2e9 — large enough to dominate every
supremum and finite enough to pass every check for a NaN.

And the refusal for a sample with no candidate break was nearly deleted as
unreachable, on an argument that only covers even sample lengths. For an odd length
the trim's ceiling can take the last candidate away; a sweep of every length from
60 to 400 against every trim to 0.499 reaches it 389 times, all odd, all above a
trim of 0.492.

Three defects in the tests. A test asserting that 70 observations at a trim of 0.45
are refused was asserting that two lower bounds on the same edge combine to be
stricter than either, which they do not — it leaves seven candidates. The
vectorised variance term was compared against the scalar one on skew and kurtosis
drawn independently, which produces impossible pairs about a third of the time and
failed inside the validator rather than on the comparison. And the constant-stretch
test asserted a statistic of zero while the code was producing −2e9, which is how
the relative guard was found.
