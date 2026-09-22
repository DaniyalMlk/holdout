"""Sample moments with their conventions stated.

Two conventions trip people up and they matter here because every correction
in this library takes skewness and kurtosis as inputs:

* **Kurtosis is raw by default.** A normal distribution has kurtosis 3, not 0.
  The probabilistic and deflated Sharpe ratio formulas are written in terms of
  raw kurtosis (they contain ``(kurtosis - 1) / 4``), so passing excess
  kurtosis there understates the correction by exactly ``3/4 * SR**2`` and
  nothing downstream would notice. Use ``excess=True`` if you want the other
  one, and say so at the call site.
* **The estimators are the plain moment ratios by default** (``bias=True`` in
  SciPy's terms): ``m3 / m2**1.5`` and ``m4 / m2**2`` with ``m_k`` the central
  moments normalised by ``n``. These are what the asymptotic Sharpe ratio
  formulas assume. ``bias=False`` gives the adjusted Fisher–Pearson estimators
  instead, identical to ``scipy.stats.skew(..., bias=False)`` and
  ``scipy.stats.kurtosis(..., bias=False)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray, as_returns

__all__ = ["Moments", "autocorrelation", "kurtosis", "moments", "skewness"]


def _central(x: FloatArray) -> tuple[float, float, float]:
    d = x - x.mean()
    m2 = float(np.mean(d**2))
    m3 = float(np.mean(d**3))
    m4 = float(np.mean(d**4))
    if m2 <= 0.0 or m2 <= (np.finfo(np.float64).eps * float(np.max(np.abs(x)))) ** 2:
        raise ValidationError("the series has zero variance, so its shape moments are undefined")
    return m2, m3, m4


def skewness(returns: ArrayLike, *, bias: bool = True) -> float:
    """Sample skewness ``m3 / m2**1.5``; ``bias=False`` applies the small-sample adjustment."""
    x = as_returns(returns, min_length=3 if not bias else 2)
    m2, m3, _ = _central(x)
    g1 = m3 / m2**1.5
    if bias:
        return float(g1)
    n = x.size
    return float(g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0))


def kurtosis(returns: ArrayLike, *, excess: bool = False, bias: bool = True) -> float:
    """Sample kurtosis ``m4 / m2**2`` (raw: 3 for a normal distribution).

    ``excess=True`` subtracts 3. ``bias=False`` applies the small-sample
    adjustment used by ``scipy.stats.kurtosis(..., bias=False)``, which needs at
    least four observations.
    """
    x = as_returns(returns, min_length=4 if not bias else 2)
    m2, _, m4 = _central(x)
    g2 = m4 / m2**2 - 3.0
    if not bias:
        n = x.size
        g2 = ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
    return g2 if excess else g2 + 3.0


@dataclass(frozen=True)
class Moments:
    """The first four moments of a series, with raw kurtosis."""

    n: int
    mean: float
    std: float
    skewness: float
    kurtosis: float

    @property
    def excess_kurtosis(self) -> float:
        return self.kurtosis - 3.0


def moments(returns: ArrayLike, *, ddof: int = 1) -> Moments:
    """Mean, standard deviation (with ``ddof``), skewness and raw kurtosis in one pass."""
    x = as_returns(returns)
    if ddof < 0 or ddof >= x.size:
        raise ValidationError(f"ddof must be in [0, {x.size - 1}], got {ddof}")
    m2, m3, m4 = _central(x)
    return Moments(
        n=int(x.size),
        mean=float(x.mean()),
        std=float(np.sqrt(m2 * x.size / (x.size - ddof))),
        skewness=m3 / m2**1.5,
        kurtosis=m4 / m2**2,
    )


def autocorrelation(returns: ArrayLike, max_lag: int) -> FloatArray:
    """Sample autocorrelations ``rho_1 .. rho_max_lag``.

    Uses the standard estimator that divides every lag's autocovariance by the
    full-sample variance (as in Box–Jenkins), which keeps the implied
    autocorrelation sequence positive semi-definite.
    """
    x = as_returns(returns)
    if max_lag < 0:
        raise ValidationError(f"max_lag must be non-negative, got {max_lag}")
    if max_lag >= x.size:
        raise InsufficientDataError(
            f"max_lag {max_lag} needs more than {max_lag} observations, got {x.size}"
        )
    d = x - x.mean()
    denominator = float(d @ d)
    if denominator <= 0.0:
        raise ValidationError("the series has zero variance, so autocorrelation is undefined")
    return np.array([float(d[k:] @ d[:-k]) / denominator for k in range(1, max_lag + 1)])
