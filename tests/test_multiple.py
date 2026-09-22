from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy import stats

from holdout import ValidationError
from holdout.multiple import adjust_pvalues

METHODS = ["bonferroni", "sidak", "holm", "bh", "by"]


def reject_directly(p: NDArray[np.float64], alpha: float, method: str) -> set[int]:
    """Each procedure as it is usually stated, independent of the implementation."""
    m = p.size
    order = list(np.argsort(p, kind="stable"))
    if method == "bonferroni":
        return {i for i in range(m) if p[i] <= alpha / m}
    if method == "sidak":
        return {i for i in range(m) if p[i] <= 1 - (1 - alpha) ** (1 / m)}
    if method == "holm":
        out: set[int] = set()
        for k, i in enumerate(order, start=1):
            if p[i] > alpha / (m - k + 1):
                break
            out.add(i)
        return out
    c = sum(1 / k for k in range(1, m + 1)) if method == "by" else 1.0
    largest = 0
    for k, i in enumerate(order, start=1):
        if p[i] <= k * alpha / (m * c):
            largest = k
    return set(order[:largest])


@settings(max_examples=150, deadline=None)
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=25),
    st.sampled_from(METHODS),
)
def test_adjusted_pvalues_reproduce_each_procedure(values: list[float], method: str) -> None:
    p = np.array(values)
    adjusted = adjust_pvalues(p, method)  # type: ignore[arg-type]
    assert np.all((adjusted >= p - 1e-15) & (adjusted <= 1.0))
    for alpha in (0.001, 0.01, 0.05, 0.1, 0.2, 0.5):
        # Tolerate the boundary: a p-value exactly on the threshold may land either side in float.
        by_adjusted = {i for i in range(p.size) if adjusted[i] <= alpha * (1 + 1e-12)}
        direct = reject_directly(p, alpha, method)
        assert direct <= by_adjusted
        borderline = by_adjusted - direct
        assert all(abs(adjusted[i] - alpha) < 1e-9 for i in borderline)


def test_textbook_values() -> None:
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    np.testing.assert_allclose(adjust_pvalues(p, "bh"), [0.05] * 5)
    np.testing.assert_allclose(adjust_pvalues(p, "holm"), [0.05, 0.08, 0.09, 0.09, 0.09])
    np.testing.assert_allclose(adjust_pvalues(p, "bonferroni"), [0.05, 0.1, 0.15, 0.2, 0.25])
    np.testing.assert_allclose(
        adjust_pvalues(p, "by"), [0.05 * (1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5)] * 5
    )
    np.testing.assert_allclose(adjust_pvalues([0.01], "sidak"), [0.01])


def test_order_is_preserved_and_ordering_of_power() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(size=40) ** 3
    shuffled = rng.permutation(40)
    for method in METHODS:
        a = adjust_pvalues(p, method)  # type: ignore[arg-type]
        np.testing.assert_allclose(adjust_pvalues(p[shuffled], method)[np.argsort(shuffled)], a)  # type: ignore[arg-type]
    bonferroni, holm, bh, by = (adjust_pvalues(p, m) for m in ("bonferroni", "holm", "bh", "by"))
    assert np.all(holm <= bonferroni + 1e-15)
    assert np.all(bh <= holm + 1e-15)
    assert np.all(bh <= by + 1e-15)


def test_sidak_is_accurate_for_tiny_pvalues() -> None:
    assert adjust_pvalues([1e-18, 0.5], "sidak")[0] == pytest.approx(2e-18, rel=1e-12)
    assert adjust_pvalues([1.0, 0.0], "sidak").tolist() == [1.0, 0.0]


def test_pvalue_validation() -> None:
    with pytest.raises(ValidationError, match=r"pvalues\[1\]"):
        adjust_pvalues([0.1, 1.2], "holm")
    with pytest.raises(ValidationError, match=r"pvalues\[0\]"):
        adjust_pvalues([math.nan], "bh")
    with pytest.raises(ValidationError, match="at least one"):
        adjust_pvalues([], "bh")
    with pytest.raises(ValidationError, match="method"):
        adjust_pvalues([0.1], "fdr")  # type: ignore[arg-type]


def test_family_wise_error_under_the_global_null() -> None:
    rng = np.random.default_rng(10)
    reps, m, alpha = 4000, 20, 0.05
    p = rng.uniform(size=(reps, m))
    for method in ("bonferroni", "sidak", "holm"):
        any_false = np.mean([np.any(adjust_pvalues(row, method) <= alpha) for row in p])
        # Sidak is exact under independence; the others are conservative. MC s.e. ~0.35pt.
        assert any_false <= alpha + 0.01
        assert any_false >= 0.035


def test_false_discovery_rate_with_true_alternatives() -> None:
    rng = np.random.default_rng(11)
    reps, m0, m1, alpha = 3000, 40, 10, 0.1
    fdp: dict[str, list[float]] = {"bh": [], "by": []}
    for _ in range(reps):
        z = np.concatenate([rng.normal(size=m0), rng.normal(3.0, 1.0, size=m1)])
        p = 2 * stats.norm.sf(np.abs(z))
        for method in ("bh", "by"):
            rejected = adjust_pvalues(p, method) <= alpha
            false = int(np.sum(rejected[:m0]))
            fdp[method].append(false / max(int(np.sum(rejected)), 1))
    # BH controls FDR at pi0 * alpha = 0.08 under independence; BY is more conservative.
    assert np.mean(fdp["bh"]) == pytest.approx(0.08, abs=0.01)
    assert np.mean(fdp["by"]) < np.mean(fdp["bh"]) / 2


def test_bh_under_positive_dependence() -> None:
    rng = np.random.default_rng(12)
    reps, m, alpha, rho = 3000, 30, 0.1, 0.6
    common = rng.normal(size=(reps, 1))
    z = math.sqrt(rho) * common + math.sqrt(1 - rho) * rng.normal(size=(reps, m))
    z[:, :5] += 3.0
    p = 2 * stats.norm.sf(np.abs(z))
    fdps: list[float] = []
    for row in p:
        rejected = adjust_pvalues(row, "bh") <= alpha
        fdps.append(float(np.sum(rejected[5:])) / max(float(np.sum(rejected)), 1.0))
    assert np.mean(fdps) <= alpha
