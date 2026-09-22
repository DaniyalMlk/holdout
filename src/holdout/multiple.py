"""Multiple-testing adjustments and haircut Sharpe ratios.

A p-value is a statement about one test. Run two hundred backtests and the
best of them clears a 5% threshold whether or not anything works. The
adjustments here trade that off in the two standard ways:

* **Family-wise error rate** — the probability of *any* false discovery.
  Bonferroni and Šidák are single-step; Holm is the step-down refinement that
  is uniformly more powerful than Bonferroni under the same (no) assumptions.
* **False discovery rate** — the expected *share* of discoveries that are
  false. Benjamini–Hochberg assumes independence or positive dependence among
  the tests; Benjamini–Yekutieli holds under any dependence at the cost of a
  factor ``sum_{k<=m} 1/k``.

Harvey and Liu (2015) express the result as a *haircut*: convert a Sharpe
ratio to a t-statistic, adjust its p-value for the number of tests, convert
back. What survives is the Sharpe ratio the evidence supports.

P-values here are two-sided, from the normal approximation to the Sharpe
ratio's t-statistic ``SR * sqrt(n)`` (per-period ratio, ``n`` periods), as in
Harvey and Liu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ValidationError
from .series import FloatArray, check_probability

__all__ = [
    "Haircut",
    "adjust_pvalues",
    "haircut_sharpe",
    "haircut_sharpe_ratios",
    "minimum_sharpe",
    "minimum_t_statistic",
    "sharpe_pvalue",
]

Adjustment = Literal["bonferroni", "sidak", "holm", "bh", "by"]
SingleStep = Literal["bonferroni", "sidak"]
_METHODS = ("bonferroni", "sidak", "holm", "bh", "by")
_NORMAL = NormalDist()


def _pvalues(values: ArrayLike) -> FloatArray:
    p = np.asarray(values, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValidationError("at least one p-value is needed")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        bad = int(np.flatnonzero(~np.isfinite(p) | (p < 0.0) | (p > 1.0))[0])
        raise ValidationError(f"pvalues[{bad}] is {p[bad]!r}; p-values must lie in [0, 1]")
    return p


def adjust_pvalues(pvalues: ArrayLike, method: Adjustment) -> FloatArray:
    """Adjusted p-values, in the input order.

    A test is rejected at level ``alpha`` by the procedure exactly when its
    adjusted p-value is at most ``alpha``; the test suite checks this
    equivalence against a direct implementation of each procedure.
    """
    p = _pvalues(pvalues)
    m = p.size
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    if method == "sidak":
        # 1 - (1 - p)**m, computed without cancellation for small p.
        with np.errstate(divide="ignore"):
            return np.where(p >= 1.0, 1.0, np.minimum(-np.expm1(m * np.log1p(-p)), 1.0))
    if method not in _METHODS:
        raise ValidationError(f"method must be one of {', '.join(_METHODS)}; got {method!r}")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    if method == "holm":
        stepped = np.maximum.accumulate(np.minimum((m - ranks + 1.0) * ranked, 1.0))
    else:
        factor = float(np.sum(1.0 / ranks)) if method == "by" else 1.0
        scaled = np.minimum(ranked * m * factor / ranks, 1.0)
        stepped = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = stepped
    return out


def sharpe_pvalue(sharpe: float, n: int) -> float:
    """Two-sided p-value of a per-period Sharpe ratio over ``n`` periods (``t = SR * sqrt(n)``)."""
    if n < 2:
        raise ValidationError(f"n must be at least 2, got {n}")
    if not math.isfinite(sharpe):
        raise ValidationError(f"sharpe must be finite, got {sharpe!r}")
    t = abs(sharpe) * math.sqrt(n)
    return math.erfc(t / math.sqrt(2.0))


def _t_from_pvalue(p: float) -> float:
    if p >= 1.0:
        return 0.0
    if p <= 0.0:
        return math.inf
    return -_NORMAL.inv_cdf(p / 2.0)


@dataclass(frozen=True)
class Haircut:
    """A Sharpe ratio before and after a multiple-testing adjustment."""

    sharpe: float
    pvalue: float
    adjusted_pvalue: float
    adjusted_sharpe: float

    @property
    def haircut(self) -> float:
        """Fraction of the Sharpe ratio removed: ``1 - adjusted / original``."""
        if self.sharpe == 0.0:
            return 0.0
        return 1.0 - self.adjusted_sharpe / self.sharpe


def _haircut(sharpe: float, n: int, p: float, p_adj: float) -> Haircut:
    t_adj = _t_from_pvalue(p_adj)
    adjusted = math.copysign(min(t_adj / math.sqrt(n), abs(sharpe)), sharpe)
    return Haircut(sharpe=sharpe, pvalue=p, adjusted_pvalue=p_adj, adjusted_sharpe=adjusted)


def haircut_sharpe(
    sharpe: float, n: int, *, n_tests: int, method: SingleStep = "bonferroni"
) -> Haircut:
    """Haircut one Sharpe ratio known to be one of ``n_tests`` tests.

    Only the single-step adjustments apply here, because Holm and the FDR
    procedures depend on the other tests' p-values; use
    :func:`haircut_sharpe_ratios` with the whole family for those.
    """
    if int(n_tests) != n_tests or n_tests < 1:
        raise ValidationError(f"n_tests must be a positive integer, got {n_tests}")
    if method not in ("bonferroni", "sidak"):
        raise ValidationError(
            f"method must be 'bonferroni' or 'sidak' for a single ratio, got {method!r}; "
            "step-wise methods need the whole family"
        )
    p = sharpe_pvalue(sharpe, n)
    m = int(n_tests)
    if method == "bonferroni":
        p_adj = p * m
    else:
        p_adj = 1.0 if p >= 1.0 else -math.expm1(m * math.log1p(-p))
    return _haircut(sharpe, n, p, min(p_adj, 1.0))


def haircut_sharpe_ratios(sharpes: ArrayLike, n: int, *, method: Adjustment) -> list[Haircut]:
    """Haircut every Sharpe ratio in a family of tests, each over ``n`` periods."""
    ratios = np.asarray(sharpes, dtype=np.float64).reshape(-1)
    if ratios.size == 0 or not np.all(np.isfinite(ratios)):
        raise ValidationError("sharpes must be a non-empty sequence of finite values")
    p = np.array([sharpe_pvalue(float(s), n) for s in ratios])
    adjusted = adjust_pvalues(p, method)
    return [
        _haircut(float(s), n, float(pi), float(ai))
        for s, pi, ai in zip(ratios, p, adjusted, strict=True)
    ]


def minimum_t_statistic(
    n_tests: int, *, alpha: float = 0.05, method: SingleStep = "bonferroni"
) -> float:
    """Smallest |t| that stays significant at ``alpha`` (two-sided) after ``n_tests`` tests."""
    check_probability(alpha, "alpha")
    if int(n_tests) != n_tests or n_tests < 1:
        raise ValidationError(f"n_tests must be a positive integer, got {n_tests}")
    m = int(n_tests)
    if method == "bonferroni":
        per_test = alpha / m
    elif method == "sidak":
        per_test = -math.expm1(math.log1p(-alpha) / m)
    else:
        raise ValidationError(f"method must be 'bonferroni' or 'sidak', got {method!r}")
    return _t_from_pvalue(per_test)


def minimum_sharpe(
    n: int,
    n_tests: int,
    *,
    alpha: float = 0.05,
    method: SingleStep = "bonferroni",
    periods_per_year: float | None = None,
) -> float:
    """Smallest Sharpe ratio over ``n`` periods that survives ``n_tests`` tests.

    Per period, or annualised with ``periods_per_year``.
    """
    if n < 2:
        raise ValidationError(f"n must be at least 2, got {n}")
    sr = minimum_t_statistic(n_tests, alpha=alpha, method=method) / math.sqrt(n)
    if periods_per_year is None:
        return sr
    if not (math.isfinite(periods_per_year) and periods_per_year > 0):
        raise ValidationError(f"periods_per_year must be positive, got {periods_per_year!r}")
    return sr * math.sqrt(periods_per_year)
