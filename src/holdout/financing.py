"""Putting a return series on an excess basis, and what that costs.

Every inference formula in this library is stated for returns in excess of a
financing rate, and every function that takes a return series defaults to a
rate of zero. That default is not neutral. A Sharpe ratio taken on total
returns is too high by ``rf / sigma`` — at a 5% cash rate and 16% annualised
volatility, 0.32 of annualised Sharpe, which is larger than most of the
haircuts this library exists to compute. Deflating a number that was already
overstated in the same direction understates the deflation.

So the conversion gets a module, and both of the conventions it involves are
named rather than assumed.

**The rate, as quoted and as needed.** Cash rates are quoted annualised; the
formulas need a per-period rate. ``"simple"`` divides by the period count and
``"compound"`` takes the period-count root. At 5% over 252 periods they are
1.98413e-04 and 1.93631e-04, a difference of 4.8e-06 per period. That is small
beside the rate itself, which is the point: the convention is a rounding error
and whether the rate is subtracted at all is not.

**The excess return.** ``"arithmetic"`` is ``r - rf``; ``"geometric"`` is
``(1 + r) / (1 + rf) - 1``, the return on the funded position. They look
different and, for a constant rate, they are the same thing twice over::

    (1 + r) / (1 + rf) - 1 == (r - rf) / (1 + rf)

which is the arithmetic excess scaled by a positive constant. A Sharpe ratio is
invariant to positive scaling, so **the convention cannot change any statistic
in this library that is a ratio of a mean to a standard deviation** when the
rate is constant. It changes the mean and the volatility, both by that factor,
so it does change reported returns and drawdowns.

For a rate that moves, the scaling moves with it and the invariance goes. The
difference is still small — over a decade of daily returns against a cash rate
sweeping 0% to 6%, the two Sharpe ratios differ by about 4e-04 relative — but
it is no longer zero, and a test here holds the bound rather than the digits.

**Leverage.** The usual way a reported Sharpe ratio gets inflated after the
fact is leverage, on the argument that a Sharpe ratio does not care. It does
not, when the borrowing is at the financing rate exactly: :func:`levered_excess`
at ``spread=0`` returns ``leverage`` times the unlevered excess return, and a
test holds that to the last bit. Pay a spread ``s`` over that rate and the
Sharpe ratio falls by ``(1 - 1 / leverage) * s / sigma`` — exactly, which is
also tested, because an identity is a better check than a tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ValidationError
from .series import FloatArray, as_matrix, as_returns, check_periods_per_year

__all__ = [
    "Compounding",
    "ExcessConvention",
    "FinancingImpact",
    "annual_rate",
    "excess_columns",
    "excess_returns",
    "financing_impact",
    "levered_excess",
    "levered_returns",
    "period_rate",
]

Compounding = Literal["simple", "compound"]
ExcessConvention = Literal["arithmetic", "geometric"]


def _frequency(periods_per_year: float) -> float:
    """Validate a period count that is required rather than optional.

    :func:`holdout.series.check_periods_per_year` admits ``None`` to mean "do
    not annualise", which is a meaningful choice for a presentation step and no
    choice at all for a rate conversion: a rate quoted per year and a series
    sampled at an unstated frequency cannot be put on the same footing.
    """
    value = float(periods_per_year)
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"periods_per_year must be positive and finite, got {value!r}")
    return value


def _ratio(series: FloatArray, *, name: str) -> tuple[float, float]:
    """Mean and ``ddof=1`` standard deviation, refusing a series with no variance."""
    sd = float(np.std(series, ddof=1))
    if sd == 0.0 or sd <= np.finfo(np.float64).eps * float(np.max(np.abs(series))):
        raise ValidationError(f"{name} have zero variance, so the Sharpe ratio is undefined")
    return float(np.mean(series)), sd


def period_rate(annual: float, periods_per_year: float, *, compounding: Compounding) -> float:
    """Convert an annualised rate to a per-period one.

    ``"simple"`` divides by ``periods_per_year``; ``"compound"`` takes the
    ``periods_per_year``-th root of the growth factor, so that compounding the
    result over a year reproduces ``annual``.

    The compounded branch is ``expm1(log1p(a) / n)`` rather than
    ``(1 + a) ** (1 / n) - 1``. The two agree in real arithmetic and not in
    floating point: a per-period rate is a small number, and writing it as a
    growth factor near one and then subtracting one throws away the leading
    digits. At 0.25% over 252 periods the direct form loses eleven of them,
    and the round trip through :func:`annual_rate` comes back 4e-12 off its
    input instead of exact.

    Raises:
        ValidationError: If ``annual`` is not finite or is at or below -100%,
            which no growth factor can be, or if ``compounding`` is unknown.
    """
    ppy = _frequency(periods_per_year)
    a = float(annual)
    if not math.isfinite(a):
        raise ValidationError(f"the annual rate must be finite, got {a!r}")
    if a <= -1.0:
        raise ValidationError(
            f"the annual rate must be above -1 (-100%), got {a!r}; a growth factor of zero "
            "or less is not a rate"
        )
    if compounding == "simple":
        return a / ppy
    if compounding == "compound":
        return math.expm1(math.log1p(a) / ppy)
    raise ValidationError(f"compounding must be 'simple' or 'compound', got {compounding!r}")


def annual_rate(rate: float, periods_per_year: float, *, compounding: Compounding) -> float:
    """Convert a per-period rate to an annualised one; the inverse of :func:`period_rate`."""
    ppy = _frequency(periods_per_year)
    r = float(rate)
    if not math.isfinite(r):
        raise ValidationError(f"the period rate must be finite, got {r!r}")
    if compounding == "simple":
        return r * ppy
    if compounding == "compound":
        if r <= -1.0:
            raise ValidationError(
                f"a period rate of {r!r} cannot be compounded; the growth factor is not positive"
            )
        return math.expm1(ppy * math.log1p(r))
    raise ValidationError(f"compounding must be 'simple' or 'compound', got {compounding!r}")


def _rate_array(rate: float | ArrayLike, rows: int, *, convention: ExcessConvention) -> FloatArray:
    """Broadcast ``rate`` to ``rows`` per-period rates, validating as it goes."""
    array = np.asarray(rate, dtype=np.float64)
    if array.ndim == 0:
        value = float(array)
        if not math.isfinite(value):
            raise ValidationError(f"rate must be finite, got {value!r}")
        out = np.full(rows, value, dtype=np.float64)
    else:
        out = as_returns(array, name="rate", min_length=1).copy()
        if out.size != rows:
            raise ValidationError(
                f"rate has {out.size} observation(s) but returns has {rows}; a rate series "
                "must line up with the return series period for period"
            )
    if convention == "geometric":
        bad = np.flatnonzero(out <= -1.0)
        if bad.size:
            index = int(bad[0])
            raise ValidationError(
                f"rate[{index}] is {out[index]!r}; the geometric convention divides by "
                "(1 + rate), which is not positive here"
            )
    return out


def _check_convention(convention: ExcessConvention) -> None:
    if convention not in ("arithmetic", "geometric"):
        raise ValidationError(f"convention must be 'arithmetic' or 'geometric', got {convention!r}")


def excess_returns(
    returns: ArrayLike,
    rate: float | ArrayLike,
    *,
    convention: ExcessConvention = "arithmetic",
) -> FloatArray:
    """Returns in excess of a per-period financing ``rate``, scalar or series.

    ``rate`` is **per period**, not annualised: see :func:`period_rate`. For a
    constant rate the two conventions differ by a positive constant factor, so
    anything scale-free computed from the result is the same under both; see
    the module documentation.
    """
    x = as_returns(returns)
    _check_convention(convention)
    rf = _rate_array(rate, x.size, convention=convention)
    if convention == "arithmetic":
        return x - rf
    return (1.0 + x) / (1.0 + rf) - 1.0


def excess_columns(
    returns: ArrayLike,
    rate: float | ArrayLike,
    *,
    convention: ExcessConvention = "arithmetic",
) -> FloatArray:
    """Column-wise :func:`excess_returns` for an ``(observations, strategies)`` matrix.

    The rate is shared across columns, which is what makes this worth having as
    its own function: it is the same cash rate for every strategy, and so the
    shift it applies to each Sharpe ratio is ``rate / sigma_i``, larger for the
    low-volatility columns. Financing does not move a family of strategies
    uniformly, and it can therefore move which of them looks best.
    """
    x = as_matrix(returns)
    _check_convention(convention)
    rf = _rate_array(rate, x.shape[0], convention=convention)[:, np.newaxis]
    if convention == "arithmetic":
        return x - rf
    return (1.0 + x) / (1.0 + rf) - 1.0


def levered_returns(
    returns: ArrayLike,
    *,
    leverage: float,
    rate: float | ArrayLike,
    spread: float = 0.0,
) -> FloatArray:
    """Total return on ``leverage`` units of an asset funded at ``rate + spread``.

    Equity of one unit buys ``leverage`` of the asset; the shortfall
    ``leverage - 1`` is borrowed at the financing rate plus ``spread``, both
    per period. A ``leverage`` below one lends the remainder out at ``rate``
    itself, since a lender does not pay the borrower's spread.

    Raises:
        ValidationError: If ``leverage`` is not positive and finite, or
            ``spread`` is not finite.
    """
    x = as_returns(returns)
    lev = float(leverage)
    if not math.isfinite(lev) or lev <= 0.0:
        raise ValidationError(f"leverage must be positive and finite, got {lev!r}")
    s = float(spread)
    if not math.isfinite(s):
        raise ValidationError(f"spread must be finite, got {s!r}")
    rf = _rate_array(rate, x.size, convention="arithmetic")
    funding = rf + s if lev >= 1.0 else rf
    return lev * x - (lev - 1.0) * funding


def levered_excess(
    returns: ArrayLike,
    *,
    leverage: float,
    rate: float | ArrayLike,
    spread: float = 0.0,
    convention: ExcessConvention = "arithmetic",
) -> FloatArray:
    """Excess return on the levered position of :func:`levered_returns`.

    At ``spread=0`` this is exactly ``leverage`` times the unlevered excess
    return under the arithmetic convention, so every scale-free statistic is
    unchanged — the sense in which a Sharpe ratio "does not care" about
    leverage. At a positive spread the ratio falls by
    ``(1 - 1 / leverage) * spread / sigma``.
    """
    total = levered_returns(returns, leverage=leverage, rate=rate, spread=spread)
    return excess_returns(total, rate, convention=convention)


@dataclass(frozen=True)
class FinancingImpact:
    """What putting a record on an excess basis did to its Sharpe ratio.

    Attributes:
        gross: Per-period Sharpe ratio of the series as handed over.
        excess: Per-period Sharpe ratio after financing is subtracted.
        mean_rate: Mean per-period financing rate applied.
        volatility: Standard deviation (``ddof=1``) of the excess series.
        observations: Length of the series.
        periods_per_year: Annualisation frequency, or ``None``.
    """

    gross: float
    excess: float
    mean_rate: float
    volatility: float
    observations: int
    periods_per_year: float | None = None

    @property
    def shift(self) -> float:
        """How much Sharpe ratio financing took, per period. Positive when it took some."""
        return self.gross - self.excess

    @property
    def predicted_shift(self) -> float:
        """``mean_rate / volatility``: the shift a constant rate produces, exactly.

        A constant rate subtracted arithmetically moves the mean and leaves the
        standard deviation alone, so the shift *is* this ratio. A rate that
        moves also moves the standard deviation, and :attr:`residual` is how
        much of the shift that accounts for.
        """
        return self.mean_rate / self.volatility

    @property
    def residual(self) -> float:
        """``shift - predicted_shift``: zero for a constant rate, the rate's variation otherwise."""
        return self.shift - self.predicted_shift

    @property
    def annualised_shift(self) -> float:
        """The shift on the square-root-of-time scale the rest of the library annualises on."""
        if self.periods_per_year is None:
            raise ValidationError("periods_per_year was not given, so there is no annual scale")
        return self.shift * math.sqrt(self.periods_per_year)


def financing_impact(
    returns: ArrayLike,
    rate: float | ArrayLike,
    *,
    convention: ExcessConvention = "arithmetic",
    periods_per_year: float | None = None,
) -> FinancingImpact:
    """Sharpe ratio before and after financing, with the closed form beside it.

    Raises:
        ValidationError: If either series has no variance, which leaves the
            Sharpe ratio undefined.
    """
    x = as_returns(returns)
    ppy = check_periods_per_year(periods_per_year)
    rf = _rate_array(rate, x.size, convention=convention)
    net = excess_returns(x, rf, convention=convention)
    gross_mean, gross_sd = _ratio(x, name="returns")
    net_mean, net_sd = _ratio(net, name="excess returns")
    return FinancingImpact(
        gross=gross_mean / gross_sd,
        excess=net_mean / net_sd,
        mean_rate=float(np.mean(rf)),
        volatility=net_sd,
        observations=int(x.size),
        periods_per_year=ppy,
    )
