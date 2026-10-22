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

## Phase 13 — The drawdown a track record owes to luck

- [x] The observed statistics: running drawdown, depth, peak, trough, recovery,
      and the longest stretch under water, with censoring marked
- [x] The exact law of the maximum drawdown of a drifted Brownian motion, from
      the first-passage eigenproblem of the reflected process, every root
      bracketed in advance
- [x] The lowest eigenfunction changes family at `drift * level == variance
      rate`, and the degenerate case at the threshold itself
- [x] The expected maximum drawdown by tail integration, against its closed form
      at zero drift
- [x] Calibrated drawdown limits by inversion, and the exact mean time to a
      given depth
- [x] The exact law of the drawdown at the end of the horizon, as a rigorous
      lower bound and as the only thing that resolves in the deep tail
- [x] Validation against a reflection series, a closed-form Laplace transform at
      every drift, and the coefficients by quadrature
- [x] The domain where the expansion is well conditioned, measured and enforced
- [x] The gap between a drawdown read off daily marks and the path's own,
      measured, and the size distortion it causes
- [x] A `drawdown` command that says what an estimated null is and is not worth

### What the law says that a round-number drawdown limit does not

A strategy with no edge at all, at 15% annual volatility, expects a 17.1% worst
drawdown in its first year and 44.8% over ten, and exceeds 65.5% one decade in
twenty. So "is this drawdown too deep" has no answer that does not mention the
volatility and the length of the record.

With an edge the growth changes character rather than merely slowing: the
expected depth rises like the square root of the horizon at zero drift and
logarithmically once there is a drift, approaching `sigma^2 / (2 mu)` per
doubling. A limit scaled to the length of a track record is therefore scaled to
the wrong thing. Seen from the other side the same fact is starker: at a 1.0
Sharpe ratio a 20% fall arrives after 7.8 years on average and a 40% fall after
450.

### Two numerical points that decided the shape of the code

The rate of the lowest mode is a difference of squares whose halves agree to the
last bit when the drift is strong. Solving for the *gap* instead, through the
closed form the root condition supplies, keeps it exact: at a drift of 0.5
against a level of 2 the direct subtraction gives exactly zero where the rate is
2.5e-21.

And the first mode's root approaches zero as the drift approaches the family
threshold, where the coefficient's denominator `theta - sin(theta) cos(theta)`
is a difference of two numbers that agree to the last bit while the true value
is `2/3 theta**3`. Subtracted directly, one parameter set on a grid of drifts
returned a survival probability above one. The series is the fix, and it also
explains why the root being poorly determined there does no harm: both halves of
the coefficient are proportional to `theta**3`, so the ratio does not care.

### The asymptote that was wrong, and how it announced itself

The deep tail was first written as an exponential first passage with the exact
mean this module also computes, which is the natural reading of "a drawdown this
deep arrives once every `E[tau]` years". Against the exact law it reads 3.0
times the truth at a 30% level, 10.6 at 40%, 59.9 at 50% and 521.7 at 60%. An
error that *grows* as the level deepens is a wrong exponent, not a loose
constant, and the diagnosis is that the two tails of this law are different
things: exponential at rate `2 mu / sigma^2` in the horizon, Gaussian in the
level. What is exposed in its place is exact.

### The honest limits

The exceedance below about 1e-13 is the complement of a survival probability of
one and has no digits in it. The expansion is a cancellation for a losing
strategy, with the largest coefficient growing like `0.76 exp(beta) / beta`, so
the domain is enforced rather than hoped for. And the exceedance an estimated
null reports is not a p-value: a nominal 5% test rejects 3.4% of the time from
discrete monitoring alone and 0.27% once the drift and volatility are estimated
from the same path.

## Phase 14 — A record that is watched, not sampled once

- [x] The normal-mixture martingale, exposed so the validity can be integrated
- [x] The boundary as its level set, asserted in both directions
- [x] The mixing variance from the fixed point it is the solution of
- [x] A confidence sequence over a return series, with the first crossing
- [x] The sequential counterpart of the minimum track record length
- [x] A simulation of what monitoring a fixed-sample interval costs
- [x] The estimated-volatility case measured rather than caveated
- [x] A `watch` subcommand that prints the cost beside the band

### The test nobody was running

Everything else here is a fixed-sample test. `probabilistic_sharpe_ratio` asks
whether a record is significant at `n` observations with `n` settled before the
data arrives, and `minimum_track_record_length` inverts it. Nobody uses a track
record that way. A record accumulates, somebody looks again each month, and the
decision to call it established is taken at whichever look first makes it look
established.

Simulate that. Independent normals with a mean of exactly zero, the ordinary
two-sided 95% interval computed at every observation, stop at the first crossing:

| observations | rejection rate |
|---|---|
| 60 | 33.7% |
| 250 | 44.1% |
| 1,000 | 52.5% |
| 5,000 | 61.9% |

The rate *grows with the horizon*, which is the fact that matters. It is not a
multiplicity to be divided out, because there is no `n` to divide by: the
partial sum's fluctuations grow like `sqrt(2 n log log n)` and the interval only
like `sqrt(n)`, so a long enough look finds something with probability one. A
rule with no size cannot be corrected, only replaced.

### What replaces it, and why its validity is checkable

A confidence sequence: an interval at every `n` such that the chance any of them
ever misses the truth is at most `alpha`. The construction is the normal
mixture, and the whole of its validity is one claim — that

    M(n) = sqrt(rho / (V + rho)) exp(S**2 / (2 (V + rho)))

is a martingale. So that claim is integrated rather than cited. One step of it
against the normal density, by a Simpson rule the test file writes out itself so
that nothing is checked against its own arithmetic, reproduces its current value
to 2e-15 at five different partial sums and variance levels. Ville's inequality
then bounds the probability it ever reaches `1 / alpha`, and the boundary is
exactly that level set — asserted in both directions, because a boundary that
has drifted from its martingale has no theorem behind it.

### The tuning is a fixed point, not a guess

`rho` decides where the boundary is tightest. Differentiating the boundary in
`rho` at a target variance `V*` gives `r - log(1 + r) = 2 log(1 / alpha)` with
`r = V* / rho` — one dimension, nothing in it but the confidence level, and
`r = 8.211968` at 95%. Verified against the equation it came from and separately
by sweeping, which also shows the optimum is flat: ten per cent either way costs
three parts in ten thousand. A flat region, not a cliff, which is the useful
thing to know about a tuning choice.

### The price, stated twice

In width: 1.548560 times the fixed-sample interval at the tuning point, 1.950657
a tenth of the way there, 4.563248 a hundredth. Those are functions of
`n / target` alone — identical to six decimals at tuning points from a hundred
observations to a hundred thousand — so the horizon is the only thing a caller
has to choose.

In data: the sequential track record length is **3.40 times** the fixed-sample
one, and the multiple is a property of the mixture rather than of the strategy —
3.4047 at an annual Sharpe ratio of 0.3, 3.4040 at 0.5, 3.4047 at 1.0, 3.3977 at
2.0. Being allowed to stop whenever the record looks good costs a factor of
three and a half in observations. Self-tuning the mixture at the answer rather
than at the fixed-sample length is worth 5% of that, and is why the figure is a
property of the method and not of an arbitrary choice.

### Where the bound looks slack, and is not

Measured against the null, a sequence tuned at sixty observations crosses 1.35%
of the time by that point, against a nominal 5%. That reads as a bound three
times too conservative, and tuning it away would be a mistake: the guarantee
covers *all* `n`, and most of the budget is spent in the far future. The same
paths run longer cross 2.46% of the time by four times the tuning point, 3.48%
by twenty times and 3.64% by a hundred. The 5% is being spent; the simulation
was looking at the wrong part of the horizon.

A test that measured only the first number would have concluded the construction
was loose. The lesson is the one about unreachable guards from the other
direction: a bound whose budget is spent outside the window you measured in
looks tighter than it is, not looser.

### The estimated volatility, measured

The derivation assumes a known variance and a caller has a sample one.
Substituting it raises the crossing rate from 1.47% to 1.98% at 250
observations: a 35% increase in false alarms, still inside the nominal 5%
because the mixture has slack there. That is the whole justification for
allowing the substitution, and it is a number rather than the phrase
"approximately valid". The command line names the columns it applied to.

## Phase 15 — The record that was chosen, conditional on its having been chosen

- [x] A numerically stable log upper tail for the normal, with the crossover
      and the term count swept rather than chosen
- [x] The conditional law of the winning estimate: a normal truncated below at
      the best of the others
- [x] A confidence interval for the selected strategy's true Sharpe by
      inverting that law, and the median-unbiased point estimate from it
- [x] The naive interval reported beside it rather than replaced by it
- [x] A simulation of both coverages, so the claim is a number in the package
- [x] An equicorrelation in that simulation, to test the one assumption
- [x] A `selected` subcommand and a worked example

### The question nobody was answering

`deflated` asks whether the best of a set of records is significant and returns
a probability; `multiple` haircuts it. Neither says what the selected
strategy's Sharpe ratio *is*, and the number carried forward into the
allocation is the winning estimate itself — chosen for being large.

Condition on the selection. Hold the losers' estimates fixed and the winner's
was drawn from its own sampling distribution restricted to the event that it
exceeded the best of them: a normal truncated below at a known point. Inverting
that law in the mean gives an interval, and the inversion is exact rather than
asymptotic.

### The ordinary interval's coverage collapses in the number of candidates

Under a null where every candidate has the same true Sharpe and a unit standard
error, the naive 95% interval on the *selected* strategy's estimate covers that
strategy's own truth 95.1% of the time out of two candidates, 88.3% out of
five, 77.8% out of ten and **30.5% out of fifty**. The conditional interval
holds 94.8% to 95.2% across all four. Past a few dozen candidates the naive
interval is wrong twice as often as it is right.

The bias it is missing is the expected maximum, and that is checkable against
something already here: the simulated gap between the winning estimate and the
winner's own truth comes out 0.557, 1.156, 1.532 and 2.240 against
`expected_maximum_normal` at 0.564, 1.163, 1.539 and 2.249. Within 1.4%, from a
simulation that never calls it.

### The monotonicity was the thing to get right, and the first attempt had it backwards

The conditional tail is *increasing* in the mean — raise the mean and the
observed value stops looking like a large draw. The first implementation
assumed the opposite and returned an upper bound of 3e40, which is the kind of
wrong answer that is obvious. The useful consequence is less obvious: since the
tail runs from zero to one over the whole line, every level is attained at a
finite mean, so **the interval is finite however close the race was**. A photo
finish feels as though it should admit any true value at all. What happens
instead is that the bound runs out at `log(1 / alpha) * deviation / margin` —
123 standard errors at a margin of three hundredths of one, 111 at a thirtieth.
An exact tie is the one degenerate case and raises rather than returning an
infinity.

### The price is the width, and the mean width is the wrong way to quote it

The conditional intervals average 25.4 to 67.2 wide against the naive 3.92,
which sounds unusable. Their medians are 6.0, 9.0, 10.6 and 14.7 — between 1.5
and 3.7 times the naive width. The difference is a heavy right tail made
entirely of near-ties, so quoting the mean would describe the typical case by
the worst one. Both are reported.

And the correction costs nothing when the winner is clear. With one candidate
four standard errors ahead of nine others the median conditional width is 4.10
against 3.92, a 5% premium, and the naive coverage has itself recovered to
94.3%. The median-unbiased estimate behaves the same way: at four standard
errors of margin it is the raw estimate to four decimal places, at two it is
0.031 below, at one 0.314 below, at a tenth 6.84 below. Everything here is a
statement about how close the race was and nothing about how many candidates
there were.

### Where the one assumption fails, the two intervals change places

The conditional law treats the winner's estimate as independent of the losers'.
A sweep over neighbouring parameters is not. Imposing an equicorrelation on ten
otherwise identical candidates:

| rho | naive coverage | conditional coverage | median width |
|---|---|---|---|
| 0.00 | 78.5% | 94.9% | 10.8 |
| 0.50 | 86.1% | 94.6% | 14.1 |
| 0.75 | 90.5% | 91.4% | 19.0 |
| 0.80 | 91.4% | 90.0% | 21.0 |
| 0.90 | 93.2% | 84.1% | 28.9 |
| 0.99 | 94.3% | 55.3% | 90.0 |

They cross at about `rho = 0.78`. Correlation shrinks the winner's curse — the
mean selection bias falls from 1.53 to 0.15 — so the naive interval recovers,
while the conditional one becomes *both wider and less valid*, which is not a
trade a reader would guess. Above three quarters this phase is the wrong tool.
With fifty candidates the crossing is much further out, because the naive
interval has so much further to fall: at `rho = 0.5` it covers 68.2% against
94.6%. The synthetic sweep this library ships is exactly the bad case — thirty
moving averages over one series, with the winner clearing the runner-up by a
hundredth of a standard error — and the `selected` command prints that margin
first for that reason.

### The log tail carries the module

Every quantity here is a ratio of two normal upper tails and both can underflow.
`erfc` reaches exactly zero near 38 standard deviations, so the direct ratio is
`0.0 / 0.0` rather than something small — a missing number, not a tiny one. The
crossover to the Mills asymptotic series and the number of terms were swept
rather than chosen: at crossover 20 with 7 terms the worst relative error
against an independent implementation over `z` from 0.1 to 400 is 6.5e-16, and
the neighbouring cells in both directions are at machine precision too, so the
choice sits in a flat region rather than on a cliff.

The version-independent statement of the same thing compares the series to
`erfc` *below* the crossover, where both are accurate: 7.3e-10 at `z = 8`,
1.3e-12 at 12, 1.3e-14 at 16 and 2.8e-16 at 20. By the time the series is
used it is already exact, so anything left over against an outside reference
is that reference's error. An earlier test asserted instead that the worst
disagreement sat out in the tail, which held against one version of the
reference and failed against an older one — a fact about the reference's error
profile rather than about this code, and the one CI failure of the phase.

## Phase 16 — The command line explains itself

- [x] Help text on every option, the seeds especially — every resampling
      command here is stochastic, so a seed nobody can read is a figure nobody
      records beside the result
- [x] The replication counts described by what they buy: the bootstrap count
      is the floor on a p-value's resolution, the block count enters the split
      count as a central binomial coefficient, and more sequential
      replications narrow a standard error rather than move the answer
- [x] A parametrised sweep that renders `--help` for the root parser and every
      subcommand, since a help string is expanded only when it is displayed
- [x] Commands enumerated by walking the parser, recursively, rather than
      listed by hand
- [x] A completeness check that every argument carries help at all
- [x] The sweep verified against the regression it exists for
- [x] The same sweep in CI against the built wheel

## Phase 17 — The evidence goes above the install line

- [x] One script recomputing every cross-check the front page quotes, so the
      figures in the document and the figures in the code cannot drift apart
- [x] Six of the seven against a reference sharing no derivation with the
      library: the published worked example, quadrature of the defining
      integral, `scipy.stats`, the AR(1) closed form, a combinatorial
      identity, and `scipy.stats.norm.logsf` out past where the tail itself
      underflows
- [x] The seventh the result the library exists for — the best of fifty
      strategies with no skill at all, scored by both ratios
- [x] The published figure computed in one place: the paper row runs the
      worked example rather than restating its arithmetic, after a draft that
      restated it de-annualised the trial variance and reported a benchmark
      ten times too small
- [x] Figures quoted as bounds, never as digits, since where two
      implementations stop agreeing is a fact about the SciPy and NumPy builds
      underneath
- [x] A quadrature bound over the whole range rather than at one convenient
      `n`, because at `n = 10**6` the reference is the less accurate of the pair
- [x] The tail checked relatively: at `z = 400` the log tail is about -80000,
      so an absolute bound would describe the sweep's endpoint instead
- [x] The two directions of the pin — a figure that moves and a row that is
      deleted — each confirmed to fail before being relied on

## Phase 18 — The record is not in excess of anything

Every statistic here is stated for returns in excess of a financing rate, and
every entry point defaulted to zero. The parameter that would have changed
that was reachable from three functions out of the library's hundred and from
no command at all, and there was no conversion from a rate as it is quoted to
a rate as the formulas need it.

- [x] A per-period rate from an annualised one, simply or compounded, with the
      inverse beside it, since a conversion that cannot be undone is a
      conversion nobody can check
- [x] Computed as `expm1(log1p(a) / n)` and not as the direct root, which
      costs a small rate eleven of its digits to the cancellation in
      `1 + tiny - 1` and fails its own round trip by 4e-12
- [x] Excess returns under both conventions, against a constant rate or a rate
      series, for a single record or for a matrix of strategies sharing one
      cash rate
- [x] The conventions' difference measured rather than assumed to matter: for
      a constant rate the geometric excess is the arithmetic one times a
      positive constant, so no Sharpe ratio can tell them apart, and only a
      rate that moves separates them
- [x] The levered position, where a Sharpe ratio is untouched by leverage at
      no spread and falls by `(1 - 1/turns) * spread / sigma` otherwise, both
      asserted as identities
- [x] The borrowed fraction floored at zero below one turn, where the position
      lends and pays nobody a spread
- [x] The haircut reported with the closed form and the residual beside it, so
      that the gap between them says whether the closed form is exact here
- [x] `--risk-free` on every command that reads a return series, converted
      where the file is read rather than in each command, and announced in the
      output because an excess ratio and a total-return ratio look alike
- [x] A `financing` command putting the two haircuts in the same units on the
      same page, which is the comparison the phase exists to make
- [x] Both new rows on the front page, with the result row quoted as bands
      rather than digits — the effective trial count comes off an
      eigendecomposition whose eigenvalues sit exactly on the integer where
      the estimator steps
- [x] Seven ways of breaking the module confirmed to fail the suite before any
      of its guards were relied on

What it came to on the shipped sweep: a deflated Sharpe ratio of 0.971 on the
returns as reported and 0.644 in excess of a 5% cash rate, with the benchmark
the search earns unmoved. The financing haircut is 0.47 and the search's is
0.46, so the adjustment nobody reports is the larger of the two, and 0.971
reads as survival where 0.644 does not.

Two bugs in the new command came out of driving it over the wire rather than
calling into it. It reported the spread's cost as a rate on an annual scale
where a ratio belonged — what makes a given-up return into a Sharpe ratio is
dividing by a volatility, and the volatility is per strategy, so no single
number in a header could have carried it. And the comparison against the
search's own haircut was gated on the number of rows printed, so asking for one
strategy by name silently dropped the row the command exists for.

## Phase 19 — One cash rate is not one shift

Phase 18 measured what the conversion takes off the level and left a claim
standing under a condition nobody had stated: that the benchmark the search
earns does not move. It does, because the shift is `rate / sigma_i` and a
family sharing one cash rate therefore does not move by one amount.

- [x] The per-column shift as a quantity of its own, since the whole effect is
      made of the spread of those numbers rather than of their average
- [x] Computed as a difference of ratios rather than as a quotient, which is
      what keeps it right for a rate series, where the conversion moves each
      column's volatility as well as its mean
- [x] The invariance stated where it holds and asserted on data constructed to
      satisfy it exactly — equal *sample* standard deviations, not equal
      population ones, which is the most a family can be designed for
- [x] A constructed counterexample in exact arithmetic, and the boundary
      either side of it: shifts differing by 0.09 reverse a gross gap of 0.08
      and do not reverse 0.10
- [x] The displacement rate measured against the volatility spread that drives
      it, monotone over four spreads, asserted on shared draws so the ordering
      is not a coincidence of seeds
- [x] The dispersion effect measured in both directions, because the sign is a
      fact about the data rather than about the conversion
- [x] The benchmark read off the library's own expected maximum rather than
      recomputed, so there is not a second place to be wrong
- [x] Reported from `holdout financing` for the whole file even when `--column`
      narrowed the rows, since the comparison needs every column to have been
      a candidate
- [x] Six ways of breaking it confirmed to fail first

The measurement that mattered: a family drawn at a *single* target volatility
still has its winner displaced in about 2% of 200 draws, because the sample
standard deviations differ by order `1 / sqrt(2n)` even when the population
ones do not. Widening the span to sixteenfold takes that to about three fifths.
The algebra alone would have said the first case never happens.

What this phase corrects in the last one: the crossover sweep's benchmark goes
from 0.651 to 0.650 under the conversion, which is why phase 18 could say it
did not move. Thirty variants of one rule have nearly the same volatility.
Said as a general claim it was wrong, and the command now prints the figure
rather than leaving it to be assumed.

A latent bug came out of the same work and went in ahead of it: every entry
point that compared a standard deviation against machine epsilon did so with no
margin, so a series that does not move returned a Sharpe ratio around 1e15
instead of raising — in five of the six places that made the test. The sixth
had the diagnosis in a comment. See the commit history for the sweep that now
covers all of them.
