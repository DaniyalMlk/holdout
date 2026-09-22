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
