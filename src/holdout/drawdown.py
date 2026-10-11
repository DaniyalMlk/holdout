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

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ValidationError
from .series import FloatArray, as_returns

__all__ = [
    "MAX_MODES",
    "DrawdownSpectrum",
    "MaximumDrawdown",
    "drawdown_exceedance",
    "drawdown_series",
    "drawdown_spectrum",
    "drawdown_survival",
    "equity_curve",
    "maximum_drawdown",
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


def _trig_bracket(beta: float, n: int) -> tuple[float, float]:
    """The interval containing the ``n``-th trigonometric root, counting from one.

    The roots are the solutions of ``theta cos(theta) + beta sin(theta) == 0``.
    For ``beta > 0`` there is exactly one in each ``((n - 1/2) pi, n pi)``. For
    ``beta < 0`` the equation is ``tan(theta) == theta / A``, whose roots sit in
    ``(n pi, (n + 1/2) pi)``; the branch ``(0, pi/2)`` holds one as well, but
    only while ``A < 1``, which is exactly when there is no hyperbolic ground
    mode to take its place. A grid audit of the sign changes confirms the count
    on both sides of that threshold.
    """
    if beta > 0.0:
        return (n - 0.5) * math.pi, n * math.pi
    index = n if -beta >= 1.0 else n - 1
    return index * math.pi, (index + 0.5) * math.pi


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

    exp_beta = math.exp(beta)
    for n in range(1, count + 1):
        if beta == 0.0:
            theta = (n - 0.5) * math.pi
        else:
            lo, hi = _trig_bracket(beta, n)
            span = hi - lo
            left = lo + span * 1e-18 if lo > 0.0 else 1e-300
            right = hi - span * 1e-15
            theta = _bisect(lambda th: th * math.cos(th) + beta * math.sin(th), left, right)
        sine = math.sin(theta)
        rates.append((beta * beta + theta * theta) * scale)
        weights.append(2.0 * exp_beta * sine**3 / _theta_less_half_sin_two(theta))

    weight_array = np.asarray(weights, dtype=np.float64)
    conditioning = float(np.abs(weight_array).max())
    if conditioning > _EXPANSION_CONDITIONING:
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
    raw = drawdown_spectrum(level, drift, volatility, modes=needed).survival(horizon)
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
    harmonic series. Use :func:`drawdown_tail` there, which is asymptotic rather
    than exact but has the right shape.
    """
    return 1.0 - drawdown_survival(level, horizon, drift, volatility)
