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

## Phase 12 — The other half of the overlapping-label problem

- [x] Concurrency per bar, from a difference array rather than by marking windows
- [x] Average uniqueness per observation, and the effective sample size it implies
- [x] The exact identity `sum(uniqueness * length) == covered bars`, asserted on
      randomised overlap structures
- [x] Sample weights from uniqueness, and a time-decay schedule over *cumulative
      uniqueness* rather than over time
- [x] A sequential bootstrap, drawing on uniqueness given what is already drawn
- [x] The cap that arithmetic places on any sampling scheme, and each scheme
      measured against it
- [x] A command-line entry point reporting all of it, including the identity

Phase 5 fixes the overlapping-label problem in the *splitter*. Purging drops
every training observation whose label window touches a test window, and the
leakage audit proves it worked. That is the right fix and it is half the
problem.

The other half is a counting error. Twenty observations whose five-day windows
cover the same week carry about one week of information between them, and every
estimator that averages over observations — a Sharpe ratio, a bootstrap, a
fitted model — treats them as twenty. Purging does not touch it, because nothing
has leaked; the observations are simply redundant. The symptom is a t statistic
too large for the data behind it, and the fix is a different number to divide
by. `effective_sample_size` is that number: `n` when nothing overlaps, one when
everything does, and in between it is what it is. On a purged training set of
120 observations with ten-bar rolling windows it comes out under a quarter of
the length, which the splitter has no way to report.

**One exact identity holds whatever the structure**, and it is the check that
the arithmetic is right rather than merely self-consistent:

    sum over i of (uniqueness_i * length_i) == number of covered bars

because both sides count `sum over covered bars of c / c`. It survives gaps,
nesting, duplicated windows and single-bar labels, and it is asserted on twelve
randomised structures rather than on an example.

**The sequential bootstrap is the part that did not survive measurement.** It
draws with a probability proportional to each candidate's uniqueness given what
is already drawn, which sounds like it should fix the redundancy at its source.
Total concurrency over a draw is `size * length` however it is drawn, so the
achievable average uniqueness is capped at `span / (size * length)` — and the
cap is attained by flattening the concurrency, which is all any scheme can try
to do. As a fraction of that cap, over sixty seeds on two hundred observations
with twenty-bar windows:

| draws | cap | sequential | uniform | gain |
| --- | --- | --- | --- | --- |
| 200 | 0.0548 | 0.998 | 0.995 | +0.44% ± 0.09% |
| 100 | 0.1095 | 0.995 | 0.984 | +1.03% ± 0.19% |
| 40 | 0.2737 | 0.978 | 0.946 | +3.47% ± 0.55% |
| 20 | 0.5475 | 0.897 | 0.830 | +8.78% ± 1.37% |
| 10 | 1.0 | 0.729 | 0.673 | +10.43% ± 2.31% |
| 5 | 1.0 | 0.865 | 0.809 | +9.37% ± 2.80% |

So the gain is real where the draw is small against the span, rising to about
ten per cent and then plateauing, and it is nothing at full size — under half a
per cent, where the uniform bootstrap already reaches 99.5% of a cap no scheme
can beat. **That is the reverse of how the method is usually described.** The
heavy-overlap, full-size resample is the case it is motivated by and the case
in which it cannot help: two hundred twenty-bar windows laid over a
219-bar span force a mean concurrency near eighteen whichever windows are
chosen, so the binding constraint is arithmetic rather than algorithmic. Its
*largest* gain of all is on non-overlapping point labels, where the overlap
problem does not exist and the only redundancy left is duplicate draws — which
is the clearest sign it is not solving the problem it is sold for.

The remedy for heavy overlap is to draw fewer observations, and the effective
sample size says how many. Drawing all `n` of them cleverly does not recover
information that is not there.

**The decay schedule runs on information, not on the calendar.** Weights are
linear in cumulative uniqueness, so a stretch in which fifty overlapping
observations say the same thing ages like the one week of information it is.
Measured against the index-linear alternative: ten redundant observations
carrying one observation's worth out of eleven — nine per cent of the
information — get 7.8% of the weight on the information scale and 26.2% on the
index scale.

Two of this phase's own test claims were wrong and were corrected rather than
loosened. The decay comparison first asserted that the newer half keeps more
*total* weight under an information scale, which is false, since the schedule
is normalised to end at one; the right measurement is the redundant pile's
share. And the bootstrap gain was asserted to be monotone down to five draws on
eight replications, where sixty show five and ten within each other's standard
errors — the plateau is now a floor rather than an ordering.

Two deliberate departures from `splits`. The labels have to be whole bar indices
here: purging asks whether two intervals intersect, which is an ordering
question that floats answer, and concurrency asks how much of a window is
shared, which is a measure question — and on a continuum a point label, the
common case where an observation is one bar, is a set of measure zero with no
uniqueness at all. And the sortedness requirement is dropped, because nothing
here is positional and insisting would mean sorting a bootstrap draw before it
could be scored.
