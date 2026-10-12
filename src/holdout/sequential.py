"""Inference on a record that is watched rather than sampled once.

Everything else here is a fixed-sample test. :func:`probabilistic_sharpe_ratio`
asks whether a record is significant at ``n`` observations, with ``n`` settled
before the data arrives, and :func:`minimum_track_record_length` inverts it.
Nobody uses a track record that way. A record accumulates, somebody looks at it
again each month, and the decision to call it established is taken at whichever
look first makes it look established.

That is a different test, and its size is not the number on the interval.

**What monitoring costs, measured.** Take a strategy with no edge at all,
compute the ordinary two-sided 95% interval on its mean at every observation,
and stop the first time the interval excludes zero. The rejection rate is 33.7%
after sixty observations, 44.1% after 250, 52.5% after 1,000 and 61.9% after
5,000 — on a test labelled 5%, against returns that are independent normals with
a mean of exactly zero. It keeps growing because the partial sum's fluctuations
grow like ``sqrt(2 n log log n)`` and the interval only like ``sqrt(n)``, so a
long enough look is certain to find something. The number is not a correction
factor, because there is no horizon to correct for: the practice has no fixed
``n`` in it.

**What is valid instead.** A *confidence sequence*: an interval at every ``n``
such that the probability any of them ever misses the truth is at most
``alpha``. The construction here is the normal mixture. For returns that are
normal with known variance, mix the one-parameter exponential martingale over
the drift with a normal prior of variance ``rho``:

    M(n) = sqrt(rho / (V + rho)) * exp(S**2 / (2 (V + rho)))

where ``S`` is the centred partial sum and ``V = n sigma**2``. That is a
martingale starting at one — checked here by integrating one step of it against
the normal density, to 2e-15, rather than cited — so Ville's inequality bounds the
probability it ever reaches ``1 / alpha`` by ``alpha``, and the level set
``M = 1 / alpha`` is the boundary

    |S| >= sqrt((V + rho) log((V + rho) / (rho alpha**2))).

**The mixing variance is derived, not guessed.** ``rho`` decides where the
boundary is tightest, and the condition for a target variance ``V*`` comes out
in closed form: differentiating the boundary in ``rho`` gives
``r - log(1 + r) = 2 log(1 / alpha)`` with ``r = V* / rho``, a one-dimensional
fixed point with nothing in it but the confidence level. At 95% it is
``r = 8.211968``. Around the optimum the boundary is flat — at 0.9 and 1.1 times
the tuned ``rho`` it is wider by three parts in ten thousand — so the choice sits
in a flat region rather than on a cliff, which is the useful thing to know about
it.

**What validity at every n costs.** At its tuning point the band is 1.548560
times the width of the fixed-sample interval, a tenth of the way there 1.950657
times, and a hundredth 4.563248. Those are functions of ``n / target`` alone —
identical to six decimals at tuning points from a hundred observations to a
hundred thousand — so the only thing a caller has to choose is the horizon that
matters to them. In observations: the sequential analogue of the minimum track
record
length is **3.40 times** the fixed-sample length, and that ratio is a property
of the mixture rather than of the strategy — 3.4047 at an annual Sharpe ratio of
0.3, 3.4040 at 0.5, 3.4047 at 1.0, 3.3977 at 2.0.

**Where the bound is slack, and why it looks slacker than it is.** Measured
against the null, a sequence tuned at sixty observations ever-crosses 1.35% of
the time by that point, not 5%. That is not conservatism to be tuned away: the
guarantee covers *all* ``n``, and most of the budget is spent in the far future.
Run the same paths longer and the crossing rate climbs to 2.46% at four times
the tuning point, 3.48% at twenty times and 3.64% at a hundred. The 5% is being
spent, just not by the horizon anyone was looking at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray, as_returns, check_periods_per_year, check_probability

__all__ = [
    "AnytimeBound",
    "ConfidenceSequence",
    "confidence_sequence",
    "mixing_variance",
    "mixture_boundary",
    "mixture_martingale",
    "peeking_rejection_rate",
    "sequential_track_record_length",
]

#: The two-sided fixed-sample multiplier at 95%, for the comparisons below.
#: Written out because the comparison is to the interval people actually use.
NORMAL_95 = 1.959963984540054


def _tightest_ratio(alpha: float) -> float:
    """The ``r`` solving ``r - log(1 + r) = 2 log(1 / alpha)``.

    The stationarity condition for the mixing variance, with ``r = V* / rho``.
    The left side is zero at ``r = 0`` and strictly increasing, so there is one
    root and bisection finds it; the bracket is doubled rather than assumed.
    """
    target = 2.0 * math.log(1.0 / alpha)
    low, high = 0.0, 1.0
    while high - math.log1p(high) < target:
        high *= 2.0
        if high > 1e300:  # pragma: no cover - needs alpha below 1e-150
            raise ValidationError(f"alpha {alpha!r} is too small to tune a mixture for")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle - math.log1p(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def mixing_variance(target: float, *, alpha: float = 0.05) -> float:
    """The mixing variance whose boundary is tightest at ``target``.

    ``target`` is a *variance process* value, ``n sigma**2``, not a count of
    observations — so a caller who knows their volatility and the horizon they
    care about passes ``horizon * sigma**2``, and one working in units of
    standard deviations passes the horizon itself.

    The answer is ``target / r`` with ``r`` the fixed point in this module's
    docstring, which depends on nothing but ``alpha``: 6.638352 at 90%,
    8.211968 at 95%, 11.756371 at 99%.
    """
    check_probability(1.0 - alpha, "1 - alpha")
    if not target > 0.0 or not math.isfinite(target):
        raise ValidationError(f"the target variance must be positive and finite, got {target!r}")
    return target / _tightest_ratio(alpha)


def mixture_martingale(total: float, variance: float, mixing: float) -> float:
    """The normal-mixture martingale at a centred partial sum.

    ``total`` is ``sum(x_i - mu)`` and ``variance`` is ``n sigma**2``. Exposed
    because the validity of everything else here is the statement that this is a
    martingale, and a claim that can be evaluated is worth more than a citation:
    the test file integrates one step of it against the normal density, with a
    Simpson rule it writes out itself, and gets agreement to 2e-15.
    """
    if mixing <= 0.0:
        raise ValidationError(f"the mixing variance must be positive, got {mixing!r}")
    if variance < 0.0:
        raise ValidationError(f"a variance process cannot be negative, got {variance!r}")
    spread = variance + mixing
    return math.sqrt(mixing / spread) * math.exp(total * total / (2.0 * spread))


def mixture_boundary(variance: float, mixing: float, *, alpha: float = 0.05) -> float:
    """The level the centred partial sum has to reach for the sequence to reject.

    ``sqrt((V + rho) log((V + rho) / (rho alpha**2)))``, which is exactly the
    level set where :func:`mixture_martingale` reaches ``1 / alpha`` — asserted
    both ways in the tests, because a boundary that drifts from its martingale
    is a boundary with no theorem behind it.
    """
    check_probability(1.0 - alpha, "1 - alpha")
    if mixing <= 0.0:
        raise ValidationError(f"the mixing variance must be positive, got {mixing!r}")
    if variance < 0.0:
        raise ValidationError(f"a variance process cannot be negative, got {variance!r}")
    spread = variance + mixing
    inside = spread / (mixing * alpha * alpha)
    if inside <= 1.0:  # pragma: no cover - needs alpha above one
        raise ValidationError("the boundary is undefined at this confidence level")
    return math.sqrt(spread * math.log(inside))


@dataclass(frozen=True)
class AnytimeBound:
    """The interval at one observation count."""

    #: One-based: the bound after this many observations.
    observations: int
    mean: float
    lower: float
    upper: float
    #: The variance process the boundary was evaluated at, ``n sigma**2``.
    variance: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0


@dataclass(frozen=True)
class ConfidenceSequence:
    """Intervals at every observation count, valid simultaneously.

    The guarantee is one statement about the whole object and not a statement
    about each row: with probability at least ``1 - alpha``, *every* interval in
    ``bounds`` contains the true mean. So a caller may read any row, or watch
    them arrive and stop at any one, without the multiplicity that makes doing
    the same thing with a fixed-sample interval reject 44% of the time at 250
    observations.
    """

    bounds: tuple[AnytimeBound, ...]
    alpha: float
    mixing: float
    #: The per-period standard deviation used, and whether it was estimated
    #: from the same returns the bound is about.
    volatility: float
    estimated_volatility: bool
    periods_per_year: float

    @property
    def last(self) -> AnytimeBound:
        return self.bounds[-1]

    @property
    def crossing(self) -> int | None:
        """The first observation count whose interval excludes zero, if any.

        This is the quantity the sequence exists to make legitimate: stopping
        here is a decision taken at a data-dependent time, and the guarantee
        still holds because it held for every row at once.
        """
        for bound in self.bounds:
            if bound.excludes_zero:
                return bound.observations
        return None

    @property
    def annualised_lower(self) -> float:
        """The last row's lower bound on the mean, scaled to a year."""
        return self.last.lower * self.periods_per_year

    def sharpe_lower(self) -> float:
        """A lower bound on the annualised Sharpe ratio at the last row.

        The mean bound divided by the volatility, scaled. This inherits the
        anytime validity for the *mean* and only approximately for the ratio,
        because the denominator is a sample quantity when
        ``estimated_volatility`` is true; the tests measure what that costs
        rather than leaving it as a caveat.
        """
        return self.last.lower / self.volatility * math.sqrt(self.periods_per_year)


def confidence_sequence(
    returns: ArrayLike,
    *,
    alpha: float = 0.05,
    volatility: float | None = None,
    target: int | None = None,
    periods_per_year: float = 252.0,
    start: int = 2,
) -> ConfidenceSequence:
    """Anytime-valid intervals for the mean of ``returns``.

    ``volatility`` is the per-period standard deviation. Left out, it is
    estimated from the whole series — which is what a caller usually has, and
    which is not free: measured against the null at 250 observations, the
    crossing rate rises from 1.47% to 1.98% when the sample standard deviation
    replaces the true one. Still inside the nominal 5%, because the mixture has
    slack at that horizon, but a 35% increase in the false-alarm rate and worth
    stating as a number rather than a warning.

    ``target`` is the observation count the band is tuned to be narrowest at,
    defaulting to the length of the series. Tuning for a horizon beyond the data
    widens the band over the data, which is the honest trade and the reason the
    parameter is exposed instead of defaulted silently.

    ``start`` is the first observation count to report. The default of two is
    where a sample standard deviation first exists; the bound itself is defined
    from one.
    """
    values = as_returns(returns, min_length=2)
    check_probability(1.0 - alpha, "1 - alpha")
    check_periods_per_year(periods_per_year)
    count = int(values.size)
    if start < 1 or start > count:
        raise ValidationError(
            f"start must be between 1 and the {count} observations given, got {start}"
        )
    horizon = count if target is None else int(target)
    if horizon < 1:
        raise ValidationError(f"the tuning target must be a positive count, got {target}")
    if volatility is None:
        estimated = True
        sigma = float(np.std(values, ddof=1))
        if not sigma > 0.0:
            raise InsufficientDataError(
                "every return is identical, so the series has no volatility and "
                "no interval can be formed from it"
            )
    else:
        estimated = False
        sigma = float(volatility)
        if not sigma > 0.0 or not math.isfinite(sigma):
            raise ValidationError(f"the volatility must be positive and finite, got {volatility!r}")
    mixing = mixing_variance(horizon * sigma * sigma, alpha=alpha)
    totals = np.cumsum(values)
    bounds: list[AnytimeBound] = []
    for index in range(start, count + 1):
        variance = index * sigma * sigma
        edge = mixture_boundary(variance, mixing, alpha=alpha)
        centre = float(totals[index - 1]) / index
        bounds.append(
            AnytimeBound(
                observations=index,
                mean=centre,
                lower=centre - edge / index,
                upper=centre + edge / index,
                variance=variance,
            )
        )
    return ConfidenceSequence(
        bounds=tuple(bounds),
        alpha=alpha,
        mixing=mixing,
        volatility=sigma,
        estimated_volatility=estimated,
        periods_per_year=periods_per_year,
    )


def sequential_track_record_length(
    sharpe: float,
    *,
    alpha: float = 0.05,
    periods_per_year: float = 252.0,
    self_tuned: bool = True,
) -> int:
    """Observations needed before an anytime-valid bound clears zero.

    ``sharpe`` is annualised. The answer is the smallest ``n`` at which the
    expected partial sum ``n * mu`` reaches the boundary, which is the sequential
    counterpart of :func:`minimum_track_record_length` and is scale-free: the
    volatility cancels, so only the Sharpe ratio and ``alpha`` matter.

    ``self_tuned`` tunes the mixture at the answer rather than at the
    fixed-sample length, by iterating the map to its fixed point. It is worth
    doing — at an annual Sharpe ratio of 0.5 the self-tuned answer is 9,286
    observations against 9,807 tuned at the fixed-sample length, a 5% saving —
    and it is what makes the comparison below a property of the method rather
    than of an arbitrary tuning choice.

    Against the one-sided fixed-sample length the ratio is **3.40**, and it
    barely moves with the strategy: 3.4047 at an annual Sharpe of 0.3, 3.4040 at
    0.5, 3.4047 at 1.0 and 3.3977 at 2.0. Watching a record and being allowed to
    stop whenever it looks good costs a factor of three and a half in data.
    """
    check_probability(1.0 - alpha, "1 - alpha")
    check_periods_per_year(periods_per_year)
    if not sharpe > 0.0:
        raise ValidationError(
            f"sharpe {sharpe:g} is not positive; no length of record makes a "
            "non-positive ratio significant"
        )
    per_period = sharpe / math.sqrt(periods_per_year)
    one_sided = _NORMAL_ONE_SIDED.get(alpha)
    if one_sided is None:
        one_sided = float(_inverse_normal(1.0 - alpha))
    guess = max(1, math.ceil((one_sided / per_period) ** 2))

    def crossing(target: int) -> int:
        mixing = mixing_variance(float(target), alpha=alpha)

        def clears(count: int) -> bool:
            return count * per_period >= mixture_boundary(float(count), mixing, alpha=alpha)

        high = 1
        while not clears(high):
            high *= 2
            if high > 2**60:  # pragma: no cover - needs a Sharpe below 1e-9
                raise ValidationError(
                    f"no reachable record length clears the boundary at sharpe {sharpe:g}"
                )
        low = 1
        while low < high:
            middle = (low + high) // 2
            if clears(middle):
                high = middle
            else:
                low = middle + 1
        return low

    if not self_tuned:
        return crossing(guess)
    target = guess
    for _ in range(100):
        found = crossing(target)
        if found == target:
            return found
        target = found
    return crossing(target)  # pragma: no cover - the map converges in a few steps


def peeking_rejection_rate(
    horizon: int,
    *,
    alpha: float = 0.05,
    paths: int = 20_000,
    seed: int = 0,
    sequence: bool = False,
) -> float:
    """Simulate the false-alarm rate of stopping at the first crossing.

    Returns, under a null of independent standard normals with mean zero, the
    fraction of paths on which the rule ever rejects by ``horizon``. With
    ``sequence`` false the rule is the ordinary two-sided fixed-sample interval
    applied at every observation — the practice this module exists to replace —
    and the answer is nothing like ``alpha``: at the default seed, 0.337 at a
    horizon of 60, 0.441 at 250, 0.525 at 1,000 and 0.619 at 5,000.

    With ``sequence`` true the rule is the normal-mixture boundary tuned at
    ``horizon``, and the answer is 0.013 to 0.019 across the same horizons —
    below ``alpha`` rather than above it, because the guarantee covers every
    ``n`` and most of its budget is spent beyond the horizon being simulated.

    Here so the two numbers in the docstring above can be reproduced from the
    installed package, which a figure quoted in prose cannot be.
    """
    check_probability(1.0 - alpha, "1 - alpha")
    if horizon < 1:
        raise ValidationError(f"the horizon must be a positive count, got {horizon}")
    if paths < 1:
        raise ValidationError(f"the path count must be positive, got {paths}")
    generator = np.random.default_rng(seed)
    counts: FloatArray = np.arange(1, horizon + 1, dtype=np.float64)
    if sequence:
        mixing = mixing_variance(float(horizon), alpha=alpha)
        level = np.asarray(
            [mixture_boundary(float(n), mixing, alpha=alpha) for n in counts],
            dtype=np.float64,
        )
    else:
        multiplier = NORMAL_95 if alpha == 0.05 else float(_inverse_normal(1.0 - alpha / 2.0))
        level = multiplier * np.sqrt(counts)
    rejected = 0
    remaining = paths
    block = max(1, min(remaining, 4_000_000 // horizon))
    while remaining > 0:
        size = min(block, remaining)
        draws = generator.standard_normal((size, horizon))
        totals = np.asarray(np.cumsum(draws, axis=1), dtype=np.float64)
        rejected += int(np.count_nonzero((np.abs(totals) >= level).any(axis=1)))
        remaining -= size
    return rejected / paths


def _inverse_normal(probability: float) -> float:
    """The standard normal quantile, by bisection on ``erf``.

    Short because the only alternative in the standard library is
    ``statistics.NormalDist``, which the rest of the package already uses; this
    exists so this module does not need the import for two call sites.
    """
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


#: Cached so the common confidence levels are exact literals rather than the
#: bisection's last bit.
_NORMAL_ONE_SIDED: dict[float, float] = {
    0.10: 1.2815515655446004,
    0.05: 1.6448536269514722,
    0.01: 2.3263478740408408,
}
