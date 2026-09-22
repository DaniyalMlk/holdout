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

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ValidationError
from .series import FloatArray

__all__ = ["adjust_pvalues"]

Adjustment = Literal["bonferroni", "sidak", "holm", "bh", "by"]
_METHODS = ("bonferroni", "sidak", "holm", "bh", "by")


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
