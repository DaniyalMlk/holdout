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
