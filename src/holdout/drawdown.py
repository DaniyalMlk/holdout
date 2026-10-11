"""The maximum drawdown, and the law it is drawn from.

Every other statistic in this library asks what a *search* was worth. This one
asks what a single realised path owes to luck: a track record's worst drawdown
is the number that ends a mandate, and it is also the number a strategy with a
real edge produces routinely.

Conventions, fixed once here:

* Returns are additive. A drawdown of ``d`` means the cumulative sum of returns
  fell ``d`` below its running peak, so for log returns the proportional loss
  is ``1 - exp(-d)`` and :attr:`MaximumDrawdown.proportional` reports it.
* The equity curve carries its starting point. For ``n`` returns the curve has
  ``n + 1`` points, index ``0`` being the start, and every index this module
  reports is an index into *that* curve. A peak at index ``0`` is the money the
  strategy began with, which is a peak like any other.
* Drawdowns are non-negative, and ``0.0`` is a perfectly good answer.

The null law is the continuous-time one. The drawdown of a Brownian motion with
drift ``mu`` and volatility ``sigma`` is a Brownian motion with drift ``-mu``
reflected at zero, so ``P(MDD(T) < h)`` is the probability that the reflected
process has not yet reached ``h`` — a first-passage problem with a reflecting
boundary at one end and an absorbing boundary at the other. :func:`drawdown_survival`
solves it exactly. Discretely monitored data sees less of the path than the
path contains, which makes the continuous law an upper bound on what daily
marks can show; :func:`monitoring_ratio` measures the gap rather than ignoring
it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ValidationError
from .series import FloatArray, as_returns, check_probability

__all__ = [
    "MAX_MODES",
    "DrawdownAssessment",
    "DrawdownSpectrum",
    "MaximumDrawdown",
    "assess_drawdown",
    "drawdown_exceedance",
    "drawdown_quantile",
    "drawdown_series",
    "drawdown_spectrum",
    "drawdown_survival",
    "equity_curve",
    "expected_maximum_drawdown",
    "final_drawdown_exceedance",
    "maximum_drawdown",
    "mean_time_to_drawdown",
]

MAX_MODES = 2048
"""Most eigenmodes the expansion will sum.

The modes decay like ``exp(-(n pi)**2 tau)`` in the dimensionless time
``tau = volatility**2 * horizon / (2 * level**2)``, so the count needed grows
like ``1 / sqrt(tau)``. Hitting the cap means the level is hundreds of standard
deviations away, where :func:`drawdown_exceedance` short-circuits to zero
rather than summing thousands of terms to confirm it.
"""

_Scalar = Callable[[float], float]

_INTEGRAL_CONDITIONING = 1e12
"""A looser budget, used only where the output is an integral or an inversion.

:func:`drawdown_survival` is a probability a caller reads directly, so it is
held to an absolute error near ``1e-9``. :func:`expected_maximum_drawdown` and
:func:`drawdown_quantile` are not: the first averages the exceedance over every
level, where a ``1e-4`` error in the part of the tail that contributes nothing
is worth ``1e-4`` times a tail's width, and the second is looking for the level
at which the exceedance crosses something like ``0.05``. Spending that slack
buys domain — a losing strategy's conditioning grows like
``exp(13 total_sharpe)`` at the level where the tail is cut, so the five orders
of magnitude move the boundary from a total Sharpe ratio of -1.09 to -1.76. Both
figures are measured, and both are properties of the dimensionless groups
rather than of the horizon: the boundary sits at -1.76 for every volatility and
every horizon tried.
"""

_EXPANSION_CONDITIONING = 1e7
"""Largest coefficient magnitude the expansion is summed at.

The coefficients carry a factor ``exp(-drift * level / volatility**2)``, which
exceeds one only for a *losing* strategy. There the survival probability is a
cancellation of terms far larger than the answer, and the absolute error is
about ``eps`` times the largest of them. The cap holds that error near
``1e-9``; :func:`drawdown_spectrum` reports the realised figure as
:attr:`DrawdownSpectrum.conditioning` so a caller can see how much room is
left.
"""


def equity_curve(returns: ArrayLike) -> FloatArray:
    """Cumulative sum of ``returns`` with a leading zero.

    The leading zero is not decoration. Without it the first observation cannot
    be a drawdown, and a strategy whose very first return is its worst loss
    reports a maximum drawdown of zero.
    """
    r = as_returns(returns, min_length=1)
    curve: FloatArray = np.concatenate(([0.0], np.cumsum(r)))
    return curve


def drawdown_series(returns: ArrayLike) -> FloatArray:
    """The running drawdown of the equity curve, one value per curve point.

    ``drawdown[i] = max(curve[:i + 1]) - curve[i]``, so the result is
    non-negative, starts at zero, and returns to zero at every new high-water
    mark.
    """
    curve = equity_curve(returns)
    result: FloatArray = np.maximum.accumulate(curve) - curve
    return result


@dataclass(frozen=True)
class MaximumDrawdown:
    """The worst drawdown of a realised path, and where it happened.

    Attributes:
        depth: The drawdown, in the additive units of the returns.
        proportional: ``1 - exp(-depth)``, the loss as a fraction of the peak,
            on the reading that the returns are logarithmic.
        peak_index: Index in the equity curve of the high-water mark the fall
            started from.
        trough_index: Index in the equity curve of the low point.
        recovery_index: First index at or after ``trough_index`` where the curve
            regains the peak, or ``None`` if it never did.
        time_under_water: Longest run of consecutive curve points strictly below
            their running peak, in periods.
        under_water_censored: Whether that longest run is the one still open at
            the end of the sample. A censored run is a lower bound on the
            drawdown's true length, and reporting the two identically would
            make a record that is currently 30% down look like one that
            recovered in the same time.
        final: The drawdown at the last observation.
    """

    depth: float
    proportional: float
    peak_index: int
    trough_index: int
    recovery_index: int | None
    time_under_water: int
    under_water_censored: bool
    final: float

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        recovery = "never" if self.recovery_index is None else str(self.recovery_index)
        return (
            f"MaximumDrawdown(depth={self.depth:.6g}, peak={self.peak_index}, "
            f"trough={self.trough_index}, recovery={recovery}, "
            f"under_water={self.time_under_water})"
        )


def maximum_drawdown(returns: ArrayLike) -> MaximumDrawdown:
    """Worst drawdown of ``returns``, with the peak, trough and recovery that made it.

    The trough is the *first* curve point attaining the deepest drawdown and the
    peak is the last high-water mark at or before it, which is the only reading
    under which the pair brackets a genuine fall.
    """
    curve = equity_curve(returns)
    peaks: FloatArray = np.maximum.accumulate(curve)
    draw = peaks - curve
    trough = int(np.argmax(draw))
    depth = float(draw[trough])
    peak = int(np.argmax(curve[: trough + 1]))
    recovery: int | None = None
    if depth == 0.0:
        peak = trough
        recovery = trough
    else:
        after = np.nonzero(curve[trough:] >= curve[peak])[0]
        if after.size:
            recovery = trough + int(after[0])
    longest, censored = _longest_under_water(draw)
    return MaximumDrawdown(
        depth=depth,
        proportional=-math.expm1(-depth),
        peak_index=peak,
        trough_index=trough,
        recovery_index=recovery,
        time_under_water=longest,
        under_water_censored=censored,
        final=float(draw[-1]),
    )


def _longest_under_water(draw: FloatArray) -> tuple[int, bool]:
    """Longest run of strictly positive entries, and whether it ends the sample."""
    wet = draw > 0.0
    if not wet.any():
        return 0, False
    # Run lengths from the positions where the indicator changes.
    edges = np.nonzero(np.diff(wet.astype(np.int8)))[0] + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [wet.size]))
    lengths = ends - starts
    live = wet[starts]
    runs = lengths[live]
    best = int(runs.max())
    last_run_is_wet = bool(wet[-1])
    censored = last_run_is_wet and int(lengths[live][-1]) == best
    return best, censored


def check_parameters(horizon: float, volatility: float) -> None:
    """Reject a horizon or volatility that no drawdown law is defined for."""
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValidationError(f"horizon must be positive and finite, got {horizon}")
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise ValidationError(f"volatility must be positive and finite, got {volatility}")


def _bisect(f: _Scalar, lo: float, hi: float) -> float:
    """Bisect a sign change on ``[lo, hi]`` down to the last representable step.

    Every root this module looks for has been bracketed analytically, so there
    is no search to fail: the loop runs until the midpoint stops moving, which
    takes about 52 steps on an interval of width ``pi``.
    """
    f_lo = f(lo)
    if f_lo == 0.0:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if f(mid) * f_lo > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ground_mode(beta: float) -> tuple[float, float]:
    """The slowest-decaying mode when the drift pulls the drawdown back to zero.

    With ``beta = -drift * level / volatility**2`` and ``A = -beta``, the
    eigenfunction of the lowest mode is hyperbolic rather than trigonometric
    whenever ``A > 1``, which is to say whenever the level sits further than
    ``volatility**2 / drift`` from the peak. Its half-width ``t`` solves
    ``A tanh(t) = t`` on ``(0, A)``, and ``A == 1`` is the degenerate case where
    the two families meet and the eigenfunction is the straight line
    ``1 - x / level``.

    Returning the *gap* ``A - t`` rather than ``t`` is the whole point. The
    decay rate is ``beta**2 - t**2 = (A - t)(A + t)``, and for a strong drift
    ``t`` approaches ``A`` so closely that the difference of the squares is
    zero to the last bit of a double while the rate itself is merely small. The
    root condition gives the gap in closed form, ``A - t = 2 A / (exp(2t) + 1)``,
    which stays exact however small it is.
    """
    big_a = -beta
    if big_a == 1.0:
        t = 0.0
    else:
        start = min(1e-9, math.sqrt(3.0 * (big_a - 1.0) / big_a) * 1e-6)
        t = _bisect(lambda z: big_a * math.tanh(z) - z, start, big_a)
    decay = math.exp(-2.0 * t)
    gap = 2.0 * big_a * decay / (1.0 + decay) if t > 0.0 else big_a
    rate = gap * (big_a + t)
    if t <= 1e-6:
        # (1 - e)**3 / 4 over (1 - e**2) / 4 - t e tends to 3 as t -> 0, and both
        # halves are O(t**3) there, so the ratio is taken in the limit instead.
        shape = 3.0 * (1.0 - 0.4 * t * t)
    else:
        shape = 0.25 * (1.0 - decay) ** 3 / (0.25 * (1.0 - decay * decay) - t * decay)
    return rate, shape * math.exp(-gap)


def _trig_roots(beta: float, count: int) -> FloatArray:
    """The first ``count`` roots of ``theta cos(theta) + beta sin(theta) == 0``.

    Each root is in a bracket known in advance -- for ``beta > 0`` one in each
    ``((n - 1/2) pi, n pi)``, and for ``beta < 0`` one in each
    ``(n pi, (n + 1/2) pi)`` plus one in ``(0, pi/2)`` while ``-beta < 1``, which
    is exactly when there is no hyperbolic ground mode to take its place. A grid
    audit of the sign changes confirms the count on both sides of that
    threshold.

    Writing the root *inside* its bracket turns the equation into a contraction
    and removes the search. With ``theta = (n - 1/2) pi + delta`` the equation
    for a positive ``beta`` is ``cot(delta) == theta / beta``, so
    ``theta = (n - 1/2) pi + atan(beta / theta)``; for a negative ``beta`` and
    ``theta = n pi + delta`` it is ``theta = n pi + atan(theta / A)``. Both
    iterations contract by at most ``1 / (2 theta)``, so thirty-odd sweeps reach
    the last bit for every mode at once -- against fifty-two bisections per
    mode, one mode at a time, which was the whole cost of the law.

    The one root the iteration cannot have is the small one in ``(0, pi/2)``:
    there the map's derivative at the origin is ``1 / A > 1``, so it pushes away
    from the root rather than towards it. That single root is bisected.
    """
    if beta == 0.0:
        exact: FloatArray = (np.arange(1, count + 1, dtype=np.float64) - 0.5) * math.pi
        return exact
    roots = np.empty(count, dtype=np.float64)
    small = beta < 0.0 and -beta < 1.0
    if small:
        # Repelling fixed point: bisect the one root in (0, pi / 2).
        roots[0] = _bisect(
            lambda th: th * math.cos(th) + beta * math.sin(th),
            1e-300,
            0.5 * math.pi * (1.0 - 1e-15),
        )
        remaining = count - 1
        offsets = np.arange(1, remaining + 1, dtype=np.float64) * math.pi
    elif beta > 0.0:
        remaining = count
        offsets = (np.arange(1, count + 1, dtype=np.float64) - 0.5) * math.pi
    else:
        remaining = count
        offsets = np.arange(1, count + 1, dtype=np.float64) * math.pi
    if remaining > 0:
        theta = offsets + 0.25 * math.pi
        shift = beta if beta > 0.0 else -1.0 / beta
        for _ in range(8):
            # Newton on theta - offset - atan(shift / theta) for a positive beta,
            # and on theta - offset - atan(shift * theta) for a negative one. The
            # map is a contraction either way, so Newton from the middle of the
            # bracket converges quadratically and eight sweeps are past the last
            # bit; iterating the map itself needs eighty, and at thirty modes a
            # sweep is all numpy overhead.
            if beta > 0.0:
                residual = theta - offsets - np.arctan(shift / theta)
                slope = 1.0 + shift / (theta * theta + shift * shift)
            else:
                scaled = shift * theta
                residual = theta - offsets - np.arctan(scaled)
                slope = 1.0 - shift / (1.0 + scaled * scaled)
            theta = theta - residual / slope
        roots[count - remaining :] = theta
    return roots


def _theta_less_half_sin_two(theta: float) -> float:
    """``theta - sin(theta) cos(theta)``, which is ``theta - sin(2 theta) / 2``.

    This is the denominator of every trigonometric coefficient, and writing it
    the obvious way is wrong. The first mode's root approaches zero as the drift
    approaches the threshold where the lowest eigenfunction changes family, and
    there the expression is a difference of two numbers that agree to the last
    bit while the true value is ``2/3 theta**3``. Subtracting them gives round-off
    and the coefficient comes out arbitrary — on a grid of drifts the failure
    appears as a single parameter set whose survival probability exceeds one.

    The series is the fix, and it also explains why the root being poorly
    determined there does no harm: the coefficient is
    ``2 exp(beta) sin(theta)**3`` over this, so both halves are proportional to
    ``theta**3`` and the ratio is insensitive to ``theta`` itself.
    """
    if theta > 1.0:
        return theta - 0.5 * math.sin(2.0 * theta)
    square = (2.0 * theta) ** 2
    term = (2.0 * theta) ** 3 / 12.0
    total = 0.0
    for m in range(1, 30):
        total += term
        term *= -square / ((2 * m + 2) * (2 * m + 3))
        if abs(term) <= 1e-18 * abs(total):
            break
    return total


@dataclass(frozen=True)
class DrawdownSpectrum:
    """The eigenmodes of the first passage of the drawdown to a level.

    ``P(MDD(T) < level) == sum(weights * exp(-rates * T))`` for every ``T > 0``,
    which is why the spectrum does not mention a horizon: the level, the drift
    and the volatility fix it, and the horizon only enters as the exponent.

    Attributes:
        rates: Decay rates ``lambda_n``, per unit time, increasing.
        weights: Expansion coefficients, which sum to one and alternate in sign
            after the first.
        has_ground_mode: Whether the slowest mode is hyperbolic. It is, exactly
            when ``drift * level > volatility**2``.
        conditioning: Largest coefficient in absolute value. One for a
            driftless or profitable strategy; exponentially large for a losing
            one, where it is also the factor by which the sum loses precision.
    """

    rates: FloatArray
    weights: FloatArray
    has_ground_mode: bool
    conditioning: float

    def survival(self, horizon: float) -> float:
        """``P(MDD(horizon) < level)`` from the modes already solved for."""
        terms = self.weights * np.exp(-self.rates * horizon)
        return float(terms.sum())


def drawdown_spectrum(
    level: float,
    drift: float,
    volatility: float,
    *,
    modes: int = 64,
    max_conditioning: float = _EXPANSION_CONDITIONING,
) -> DrawdownSpectrum:
    """Solve the first-passage eigenproblem for a drawdown of ``level``.

    The drawdown of a Brownian motion with drift ``drift`` and volatility
    ``volatility`` is that motion's running peak less its value, which is a
    Brownian motion with drift ``-drift`` reflected at zero. Asking when the
    drawdown first reaches ``level`` is therefore a first-passage problem with a
    reflecting boundary at zero and an absorbing one at ``level``, and the
    backward equation separates: with ``b = -drift / volatility**2`` the
    substitution ``phi = exp(b x) w`` turns the generator into ``w'' = -k**2 w``
    with a Robin condition at zero, so the eigenvalues are the roots of
    ``k cos(k L) + b sin(k L) == 0`` and nothing has to be searched for blindly.

    Args:
        level: Drawdown depth, in the additive units of the returns. Positive.
        drift: Drift per unit time. Positive for a strategy that makes money.
        volatility: Volatility per unit time. Positive.
        modes: Number of trigonometric modes. Sixty-four is enough for any
            level inside seventy standard deviations of the horizon;
            :func:`drawdown_survival` asks for as many as the level needs.

    Raises:
        ValidationError: If an argument is out of range, or the coefficients are
            so large that the sum would be a cancellation rather than a
            calculation. The second case needs a drift far enough below zero
            that the drawdown's arrival is a foregone conclusion.
    """
    if not math.isfinite(level) or level <= 0.0:
        raise ValidationError(f"level must be positive and finite, got {level}")
    if not math.isfinite(drift):
        raise ValidationError(f"drift must be finite, got {drift}")
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise ValidationError(f"volatility must be positive and finite, got {volatility}")

    beta = -drift * level / volatility**2
    scale = volatility**2 / (2.0 * level**2)
    if modes < 1 or modes > MAX_MODES:
        raise ValidationError(f"modes must lie in [1, {MAX_MODES}], got {modes}")
    count = modes

    rates: list[float] = []
    weights: list[float] = []
    if beta < 0.0 and -beta >= 1.0:
        rate, weight = _ground_mode(beta)
        rates.append(rate * scale)
        weights.append(weight)
    has_ground = bool(rates)

    roots = _trig_roots(beta, count)
    sine = np.sin(roots)
    # theta - sin(theta) cos(theta), vectorised, with the series substituted for
    # the small roots -- there is at most one, and that is where subtracting
    # loses everything.
    denominator = roots - 0.5 * np.sin(2.0 * roots)
    for index in np.nonzero(roots <= 1.0)[0]:
        denominator[index] = _theta_less_half_sin_two(float(roots[index]))
    rates.extend(((beta * beta + roots * roots) * scale).tolist())
    weights.extend((2.0 * math.exp(beta) * sine**3 / denominator).tolist())

    weight_array = np.asarray(weights, dtype=np.float64)
    conditioning = float(np.abs(weight_array).max())
    if conditioning > max_conditioning:
        raise ValidationError(
            f"the expansion is a cancellation at these parameters: the largest "
            f"coefficient is {conditioning:.3g}, so the survival probability would "
            f"carry an absolute error near {conditioning * 2.3e-16:.1g}. This needs a "
            f"drift of {drift:g} against a level of {level:g}, which is "
            f"-drift * level / volatility**2 = {beta:.3g}; the expansion is well "
            f"conditioned for every drift at or above zero"
        )
    return DrawdownSpectrum(
        rates=np.asarray(rates, dtype=np.float64),
        weights=weight_array,
        has_ground_mode=has_ground,
        conditioning=conditioning,
    )


_DECAY_BUDGET = 45.0
"""How far into the exponential tail of the mode sum to go before stopping.

``exp(-45)`` is ``2.9e-20``, comfortably under the round-off of a sum whose
terms are of order one.
"""


def _modes_for(tau: float, beta: float) -> int:
    """Modes needed at dimensionless time ``tau = volatility**2 T / (2 level**2)``.

    The coefficients are of order ``exp(beta)``, so for a losing strategy the
    exponential has to work that much harder before a term is negligible. Taking
    the budget as ``45 + beta`` is why the truncation error stays at round-off
    rather than scaling with the coefficients.
    """
    budget = _DECAY_BUDGET + max(0.0, beta)
    return math.ceil(math.sqrt(budget / tau) / math.pi) + 4


def drawdown_survival(
    level: float,
    horizon: float,
    drift: float,
    volatility: float,
    *,
    max_conditioning: float = _EXPANSION_CONDITIONING,
) -> float:
    """``P(MDD(horizon) < level)`` for a Brownian motion with these parameters.

    Exact, in the sense that the only error is the round-off of a sum whose
    truncation has been pushed below it.

    The law depends on its four arguments through two dimensionless numbers
    only: the level in standard deviations of the horizon,
    ``level / (volatility * sqrt(horizon))``, and the horizon's Sharpe ratio,
    ``drift * sqrt(horizon) / volatility``. Doubling the volatility and
    quadrupling the horizon changes neither, and a test holds the function to
    that invariance.

    Args:
        level: Drawdown depth, positive.
        horizon: Length of the track record, in the time units of the drift.
        drift: Drift per unit time.
        volatility: Volatility per unit time, positive.

    Returns:
        The probability that the deepest drawdown over the horizon stays strictly
        below ``level``.
    """
    check_parameters(horizon, volatility)
    if not math.isfinite(level) or level <= 0.0:
        raise ValidationError(f"level must be positive and finite, got {level}")
    tau = volatility**2 * horizon / (2.0 * level**2)
    needed = _modes_for(tau, -drift * level / volatility**2)
    if needed > MAX_MODES:
        # The level is hundreds of standard deviations past anything the path
        # could reach. Rather than sum thousands of modes to confirm it, say so,
        # but only on a bound that does not depend on the expansion: the
        # drawdown is at most the whole range of the path, and a range of
        # `level` needs a move of `level / 2` in one direction.
        reachable = abs(drift) * horizon + 10.0 * volatility * math.sqrt(horizon)
        if level > 2.0 * reachable:
            return 1.0
        raise ValidationError(
            f"the level {level:g} needs {needed} eigenmodes at a dimensionless time of "
            f"{tau:.3g}, past the {MAX_MODES} this expansion will sum"
        )
    raw = drawdown_spectrum(
        level, drift, volatility, modes=needed, max_conditioning=max_conditioning
    ).survival(horizon)
    # The modes sum to one and the horizon damps them, so the sum *is* a
    # probability -- but it is a sum of up to a couple of thousand terms, and
    # round-off puts it a few ulps outside [0, 1] whenever the answer is at
    # either end. Clamping is the whole correction; a test holds the unclamped
    # sum to within 1e-12 of the clamp so that the clamp can never be hiding
    # anything larger.
    return min(1.0, max(0.0, raw))


def drawdown_exceedance(
    level: float,
    horizon: float,
    drift: float,
    volatility: float,
    *,
    max_conditioning: float = _EXPANSION_CONDITIONING,
) -> float:
    """``P(MDD(horizon) >= level)``: the chance of a drawdown at least this deep.

    This is one less the survival probability, and the subtraction is where the
    precision goes. The modes sum to one exactly, so for a level the path will
    almost certainly reach, the survival probability is a small number computed
    cleanly and the exceedance is accurate to the last digit. For a level it
    almost certainly will not, the survival probability is just under one and the
    exceedance is what is left after a cancellation: the absolute error stays at
    round-off, which makes the *relative* error of a tiny exceedance large. The
    result below about ``1e-13`` is therefore a number with no significant
    digits in it, and no rearrangement of this series fixes that — the
    term-by-term complement converges only conditionally, like an alternating
    harmonic series. Use :func:`final_drawdown_exceedance`
    there: the chance of *ending* the horizon that far down is a closed form, a
    rigorous lower bound on this, and accurate past 1e-300.
    """
    return 1.0 - drawdown_survival(
        level, horizon, drift, volatility, max_conditioning=max_conditioning
    )


def mean_time_to_drawdown(level: float, drift: float, volatility: float) -> float:
    """Expected time until the drawdown first reaches ``level``.

    Closed form, and not an integral of the law above: the same ordinary
    differential equation that the eigenproblem comes from can be solved once
    for the expected passage time directly, giving
    ``level / nu + (sigma**2 / (2 nu**2)) (exp(-2 nu level / sigma**2) - 1)``
    with ``nu = -drift``. At zero drift it reduces to ``level**2 / sigma**2``.

    The two routes agree to nine figures on a grid of parameters, which is the
    cheapest independent check there is on the expansion's coefficients and
    rates together.

    For a strategy that makes money the answer grows exponentially in the level:
    a 20% drawdown arrives eventually and a 60% one effectively never, and the
    ratio between the two is not three.
    """
    if not math.isfinite(level) or level <= 0.0:
        raise ValidationError(f"level must be positive and finite, got {level}")
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise ValidationError(f"volatility must be positive and finite, got {volatility}")
    if not math.isfinite(drift):
        raise ValidationError(f"drift must be finite, got {drift}")
    nu = -drift
    if nu == 0.0:
        return level * level / volatility**2
    return level / nu + (volatility**2 / (2.0 * nu * nu)) * math.expm1(
        -2.0 * nu * level / volatility**2
    )


def final_drawdown_exceedance(
    level: float, horizon: float, drift: float, volatility: float
) -> float:
    """``P(drawdown at the end of the horizon >= level)``, in closed form.

    The drawdown *at a single time* is a reflected Brownian motion's marginal,
    which is elementary where the law of its running maximum is not::

        P(D_T >= h) = Phi(-(h + mu T) / (sigma sqrt(T)))
                      + exp(-2 mu h / sigma**2) Phi((mu T - h) / (sigma sqrt(T)))

    Two reasons this is here rather than in a footnote.

    It is a rigorous lower bound on :func:`drawdown_exceedance`, since a record
    that *ends* ``level`` down has been ``level`` down. That makes it a check on
    the expansion that no self-consistent error inside the expansion can pass,
    and a test applies it across a grid.

    And it keeps working where the expansion runs out of digits. The exceedance
    of the maximum is the complement of a survival probability just under one,
    so it has nothing left below about ``1e-13``; this goes through ``erfc`` and
    stays accurate past ``1e-300``.

    What it is not is an asymptote for the maximum. The two tails this law has
    are different and it is worth being explicit about which is which: at a
    fixed level and a growing horizon the exceedance approaches one
    exponentially at rate ``2 drift / sigma**2``, the stationary exceedance rate
    of the reflected process, while at a fixed horizon and a growing level it
    dies like ``exp(-(level + drift * horizon)**2 / (2 sigma**2 horizon))``,
    which is Gaussian. Reading the first as though it governed the second — by
    taking the first passage to be exponential with the mean this module also
    computes — overstates a one-year 60% drawdown at a 0.67 Sharpe ratio by a
    factor of 520, and the error grows as the level deepens.
    """
    check_parameters(horizon, volatility)
    if not math.isfinite(level) or level <= 0.0:
        raise ValidationError(f"level must be positive and finite, got {level}")
    if not math.isfinite(drift):
        raise ValidationError(f"drift must be finite, got {drift}")
    scale = volatility * math.sqrt(horizon)
    drifted = drift * horizon
    first = _normal_cdf(-(level + drifted) / scale)
    second = _normal_cdf((drifted - level) / scale)
    if second == 0.0:
        return first
    log_second = math.log(second) - 2.0 * drift * level / volatility**2
    return first + (math.exp(log_second) if log_second > -745.0 else 0.0)


def _normal_cdf(x: float) -> float:
    """Standard normal distribution function, through ``erfc`` so the tail survives.

    ``NormalDist().cdf`` is fine in the body of the distribution and underflows
    to exactly zero around eight standard deviations out, which is inside the
    range this module asks about.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


_PANELS = 48
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(12)


def _crossing_exceedance(barrier: float, horizon: float, drift: float, volatility: float) -> float:
    """``P(the path reaches +barrier at some point in the horizon)``, in closed form.

    The classical first-passage result for a Brownian motion with drift, with
    the exponential folded into a logarithm so that a positive drift against a
    distant barrier does not multiply an overflow by an underflow.
    """
    scale = volatility * math.sqrt(horizon)
    drifted = drift * horizon
    first = _normal_cdf((drifted - barrier) / scale)
    second = _normal_cdf(-(barrier + drifted) / scale)
    if second == 0.0:
        return first
    log_second = math.log(second) + 2.0 * drift * barrier / volatility**2
    return min(1.0, first + (math.exp(log_second) if log_second > -745.0 else 0.0))


def _range_bound(level: float, horizon: float, drift: float, volatility: float) -> float:
    """A rigorous upper bound on ``P(MDD >= level)`` that does not use the expansion.

    A drawdown of ``level`` needs the path's whole range to be at least
    ``level``, and a range of ``level`` needs the path to reach ``a`` above its
    start or ``level - a`` below it, for any split ``a``. Splitting at
    ``drift * horizon + level / 2`` balances the two exponents, which puts the
    bound's exponent at ``level**2 / (8 sigma**2 horizon)`` — a factor of four
    short of the true one, so the bound is loose by a factor of two *in the
    level*. That is fine for its job, which is to say where the tail of an
    integral can be cut.

    The point of having it is that the expansion cannot do this. Out in the tail
    the exceedance is the complement of a survival probability of one, so what
    comes back is round-off; a ceiling chosen by waiting for *that* to fall
    below a threshold never stops, and walks the level outwards until the
    coefficients overflow their conditioning. This is in closed form and is
    monotone.
    """
    split = min(max(drift * horizon + 0.5 * level, 0.01 * level), 0.99 * level)
    upward = _crossing_exceedance(split, horizon, drift, volatility)
    downward = _crossing_exceedance(level - split, horizon, -drift, volatility)
    return min(1.0, upward + downward)


_TAIL_CUTS = (1e-12, 1e-9, 1e-7, 1e-5, 1e-4, 1e-3)
"""Bounds at which an integral's tail may be cut, tightest first.

The bound is loose by a factor of two *in the level*, so a tight cut puts the
ceiling about fifteen standard deviations out, and for a losing strategy the
coefficients at that level are ``exp(15 total_sharpe)``. Insisting on the
tightest cut therefore refuses records that ordinary sampling noise produces: a
strategy with a true Sharpe ratio of 0.6 over a year throws an estimated total
Sharpe ratio below -1.76 about once in a hundred times, and that record's
drawdown still deserves an answer.

So the cuts are tried in order and the first one whose ceiling the expansion can
reach is used, which carries the domain out to a total Sharpe ratio of -3.23.

Loosening the cut costs nothing measurable, which is worth stating because it
is not obvious. The cut is applied to a *bound* that is loose by a factor of two
in the level, so a bound of ``1e-3`` sits where the true exceedance is nearer
``1e-12``; measured against the closed form at zero drift, every cut in this
ladder gives the same relative error of ``2e-12``, and what is left is the
quadrature rather than the truncation.
"""


def _tail_ceiling(horizon: float, drift: float, volatility: float, cut: float) -> float:
    """A level whose exceedance is provably under ``cut``."""
    scale = volatility * math.sqrt(horizon)
    level = max(2.0 * scale, 2.0 * abs(drift) * horizon)
    for _ in range(200):
        if _range_bound(level, horizon, drift, volatility) < cut:
            return level
        level *= 1.5
    raise ValidationError(  # pragma: no cover - unreachable for finite parameters
        f"no level within {level:g} is out of reach of this path"
    )


def _integrate_tail(horizon: float, drift: float, volatility: float, cut: float) -> float:
    """Composite Gauss-Legendre of the exceedance over ``[0, ceiling]``."""
    ceiling = _tail_ceiling(horizon, drift, volatility, cut)
    edges = np.linspace(0.0, ceiling, _PANELS + 1)
    total = 0.0
    for left, right in pairwise(edges):
        half = 0.5 * (right - left)
        centre = 0.5 * (right + left)
        for node, weight in zip(_NODES, _WEIGHTS, strict=True):
            total += (
                half
                * weight
                * drawdown_exceedance(
                    centre + half * node,
                    horizon,
                    drift,
                    volatility,
                    max_conditioning=_INTEGRAL_CONDITIONING,
                )
            )
    return total


def _with_loosening_cut(step: Callable[[float], float]) -> float:
    """Run ``step`` at the tightest tail cut the expansion can actually reach."""
    for index, cut in enumerate(_TAIL_CUTS):
        try:
            return step(cut)
        except ValidationError:
            if index == len(_TAIL_CUTS) - 1:
                raise
    raise AssertionError  # pragma: no cover - the loop either returns or raises


def expected_maximum_drawdown(horizon: float, drift: float, volatility: float) -> float:
    """``E[MDD(horizon)]``, by integrating the exceedance over every level.

    ``E[X] = integral of P(X > h) dh`` for a non-negative ``X``, and the
    exceedance is smooth in the level — flat at one near zero, with a tail that
    dies faster than exponentially — so composite Gauss-Legendre over a ceiling
    chosen from the tail itself reaches the closed form at zero drift,
    ``sigma sqrt(pi T / 2)``, to thirteen figures.

    The number this returns is the one worth knowing before a drawdown
    happens. At zero drift it is 1.2533 standard deviations of the horizon, so a
    strategy with no edge at 15% volatility expects an 18.8% worst drawdown in
    its first year *for that reason alone*. With an edge the growth in the
    horizon is logarithmic rather than square-root, approaching
    ``sigma**2 / (2 drift)`` per doubling.
    """
    check_parameters(horizon, volatility)
    if not math.isfinite(drift):
        raise ValidationError(f"drift must be finite, got {drift}")
    return _with_loosening_cut(lambda cut: _integrate_tail(horizon, drift, volatility, cut))


def drawdown_quantile(probability: float, horizon: float, drift: float, volatility: float) -> float:
    """The level the maximum drawdown stays below with probability ``probability``.

    This is the calibrated version of a drawdown limit. A limit set at a round
    number is a limit on nothing in particular; this one answers "how deep does
    this strategy go one year in twenty, if its drift and volatility are what we
    think they are", and the answer is usually deeper than people guess.

    The exceedance is strictly decreasing in the level, so the inversion is a
    bisection on a bracket found by doubling.
    """
    check_parameters(horizon, volatility)
    check_probability(probability, "probability")
    ceiling = _with_loosening_cut(lambda cut: _tail_ceiling(horizon, drift, volatility, cut))
    target = 1.0 - probability
    low, high = 0.0, ceiling
    for _ in range(200):
        mid = 0.5 * (low + high)
        if mid <= low or mid >= high:
            break
        exceedance = drawdown_exceedance(
            mid, horizon, drift, volatility, max_conditioning=_INTEGRAL_CONDITIONING
        )
        if exceedance > target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


@dataclass(frozen=True)
class DrawdownAssessment:
    """An observed drawdown beside what the null says to expect of it.

    Attributes:
        observed: The realised worst drawdown and where it fell.
        drift: Drift per period used for the null.
        volatility: Volatility per period used for the null.
        periods: Length of the record, which is the horizon in those units.
        estimated: Whether the drift and volatility came from this same record.
        expected: ``E[MDD]`` under the null.
        median: The null's median drawdown.
        percentile: ``P(MDD < observed)``, so 0.5 means a typical drawdown for
            a strategy like this one.
        exceedance: ``P(MDD >= observed)``, the p-value against the null.
        exceedance_bound: The rigorous lower bound from
            :func:`final_drawdown_exceedance`, which still has digits when the
            exceedance does not.
        limit: The level this strategy exceeds one record in twenty.
    """

    observed: MaximumDrawdown
    drift: float
    volatility: float
    periods: int
    estimated: bool
    expected: float
    median: float
    percentile: float
    exceedance: float
    exceedance_bound: float
    limit: float


def assess_drawdown(
    returns: ArrayLike,
    *,
    drift: float | None = None,
    volatility: float | None = None,
) -> DrawdownAssessment:
    """Compare a record's worst drawdown with the law its own parameters imply.

    One period is one unit of time here, so the horizon is the number of
    returns and the drift and volatility are per period. Annualising would
    change nothing: the law depends on its arguments only through
    ``level / (volatility sqrt(horizon))`` and ``drift sqrt(horizon) / volatility``,
    both of which are unit-free.

    **What this is and is not.** With ``drift`` and ``volatility`` supplied, the
    exceedance is a p-value for the hypothesis that the record came from a
    Brownian motion with those parameters. Left to be estimated from the record,
    it is not: the null is being fitted to the same path whose drawdown is being
    judged, and the two are not independent -- a path that happened to fall a
    long way also reports a larger volatility, which makes its own drawdown look
    ordinary.

    Both effects are one-sided and both are measured. On simulated records of
    252 observations with a known drift and volatility, a nominal 5% test
    rejects 3.4% of the time, because the drawdown is read off 252 marks while
    the law describes the path underneath them -- and a discretely observed
    drawdown is smaller, by 7% in the mean at this count and 1.4% at sixteen
    times it. Re-estimating the drift and volatility from the same path takes
    the same test from 3.4% to 0.27%, a further factor of thirteen, which is
    much the larger of the two.

    So the exceedance understates, and it is still the right number to look at,
    because the alternative is comparing a drawdown against nothing. It is
    reported with :attr:`DrawdownAssessment.estimated` set so that it cannot be
    mistaken for the other thing, and a caller who has a drift and a volatility
    from somewhere else should pass them.

    Args:
        returns: Per-period returns, additive.
        drift: Drift per period for the null. Estimated from ``returns`` if
            omitted.
        volatility: Volatility per period for the null. Estimated from
            ``returns`` if omitted.
    """
    sample = as_returns(returns, min_length=2)
    estimated = drift is None or volatility is None
    mean = float(np.mean(sample)) if drift is None else drift
    sigma = float(np.std(sample, ddof=1)) if volatility is None else volatility
    periods = int(sample.size)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValidationError(
            "the returns have no variation, so no drawdown law is defined for them"
        )
    if abs(mean) * periods > 1e8 * sigma * math.sqrt(periods):
        # Constant returns do not give a sample standard deviation of exactly
        # zero -- twenty copies of 0.01 leave 1.7e-18 behind -- so the test that
        # matters is whether the record has any variation *relative to its own
        # drift*. Without this the law is asked for a drawdown of 1e-27
        # standard deviations and refuses for the wrong reason.
        raise ValidationError(
            f"the returns have no variation to speak of: a drift of {mean:g} per "
            f"period against a volatility of {sigma:g} is a deterministic record, "
            f"and its drawdown is not a question about chance"
        )
    observed = maximum_drawdown(sample)
    horizon = float(periods)
    if observed.depth == 0.0:
        percentile, exceedance, bound = 0.0, 1.0, 1.0
    else:
        percentile = drawdown_survival(observed.depth, horizon, mean, sigma)
        exceedance = 1.0 - percentile
        bound = final_drawdown_exceedance(observed.depth, horizon, mean, sigma)
    return DrawdownAssessment(
        observed=observed,
        drift=mean,
        volatility=sigma,
        periods=periods,
        estimated=estimated,
        expected=expected_maximum_drawdown(horizon, mean, sigma),
        median=drawdown_quantile(0.5, horizon, mean, sigma),
        percentile=percentile,
        exceedance=exceedance,
        exceedance_bound=bound,
        limit=drawdown_quantile(0.95, horizon, mean, sigma),
    )
