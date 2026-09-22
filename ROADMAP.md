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

- [ ] Combinatorially symmetric cross-validation over a performance matrix
- [ ] Probability of backtest overfitting and the logit distribution behind it
- [ ] Performance degradation and probability of loss out of sample
- [ ] Validation: pure noise sits near one half, a genuine edge near zero

## Phase 5 — Cross-validation for overlapping labels

- [ ] Walk-forward splits, expanding and rolling
- [ ] Purged k-fold with an embargo on label intervals
- [ ] Combinatorial purged cross-validation with backtest path assembly
- [ ] Leakage audit that fails if any training label overlaps a test label

## Phase 6 — Bootstrap tests of superior predictive ability

- [ ] Stationary bootstrap with automatic block length selection
- [ ] White's Reality Check
- [ ] Hansen's test for superior predictive ability
- [ ] Romano–Wolf stepdown to name which strategies beat the benchmark
- [ ] Size and power checked by simulation

## Phase 7 — Command line and examples

- [ ] CSV loading for single series and strategy matrices
- [ ] `holdout` command with `sharpe`, `deflate`, `pbo`, `spa` and `sample-data`
- [ ] Worked examples runnable end to end
- [ ] Clean-wheel installation checked in continuous integration
