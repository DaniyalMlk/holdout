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
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ValidationError
from .series import FloatArray, as_returns

__all__ = [
    "MaximumDrawdown",
    "drawdown_series",
    "equity_curve",
    "maximum_drawdown",
]


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
