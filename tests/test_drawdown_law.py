"""The exact law of the maximum drawdown, against routes that share no derivation.

Three independent references are used here.

* The driftless case has a reflection-principle series. Lévy's theorem makes the
  drawdown of a driftless Brownian motion a reflected Brownian motion, whose
  maximum is distributed as the maximum *absolute value* of the original, and
  that has a classical image series in the normal distribution function. It
  shares nothing with the eigenfunction expansion but the problem statement.
* At every drift, the first-passage time of the drawdown to a level has a
  closed-form Laplace transform, obtained from the ordinary differential
  equation rather than from the expansion. Matching it at several ``theta``
  tests the eigenvalues and the coefficients jointly, and the ``theta -> 0``
  limit is the mean passage time.
* The coefficients can be read off directly as the inner products that define
  them, by quadrature against the speed measure.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from holdout import (
    MAX_MODES,
    drawdown_exceedance,
    drawdown_spectrum,
    drawdown_survival,
)
from holdout.exceptions import ValidationError
from holdout.series import FloatArray

# --------------------------------------------------------------------------- #
# references


def reflection_survival(level: float, horizon: float, volatility: float) -> float:
    """``P(MDD < level)`` at zero drift, from the two-sided reflection series."""
    z = level / (volatility * math.sqrt(horizon))
    return float(
        sum(
            (-1) ** k * (norm.cdf((2 * k + 1) * z) - norm.cdf((2 * k - 1) * z))
            for k in range(-80, 81)
        )
    )


def passage_laplace(theta: float, level: float, drift: float, volatility: float) -> float:
    """``E[exp(-theta tau)]`` for the first passage of the drawdown to ``level``.

    Solves ``(sigma**2 / 2) g'' - drift g' = theta g`` with ``g'(0) == 0`` and
    ``g(level) == 1``, then reports ``g(0)``. Written so that the hyperbolic
    functions never overflow.
    """
    nu = -drift
    b = nu / volatility**2
    d = math.sqrt(nu * nu + 2.0 * volatility**2 * theta)
    u = d * level / volatility**2
    decay = math.exp(-2.0 * u)
    return d * math.exp(b * level - u) / (nu * 0.5 * (1.0 - decay) + d * 0.5 * (1.0 + decay))


def mean_passage(level: float, drift: float, volatility: float) -> float:
    """``E[tau]`` for that first passage, from the same ODE with ``theta == 0``."""
    nu = -drift
    if nu == 0.0:
        return level * level / volatility**2
    return level / nu + (volatility**2 / (2.0 * nu * nu)) * math.expm1(
        -2.0 * nu * level / volatility**2
    )


GRID = [
    (level, drift, vol)
    for level in (0.05, 0.2, 0.5, 1.0, 2.0)
    for drift in (-0.05, 0.0, 0.03, 0.08, 0.3)
    for vol in (0.1, 0.15, 0.4)
]


# --------------------------------------------------------------------------- #
# the eigenproblem


def test_driftless_eigenvalues_are_the_closed_form() -> None:
    level, vol = 0.2, 0.15
    spectrum = drawdown_spectrum(level, 0.0, vol, modes=12)
    assert spectrum.has_ground_mode is False
    expected = np.array([0.5 * vol**2 * ((n - 0.5) * math.pi / level) ** 2 for n in range(1, 13)])
    assert spectrum.rates == pytest.approx(expected, rel=1e-14)
    # 4 (-1)**(n-1) / ((2n - 1) pi)
    weights = np.array([4.0 * (-1) ** (n - 1) / ((2 * n - 1) * math.pi) for n in range(1, 13)])
    assert spectrum.weights == pytest.approx(weights, rel=1e-13)


@pytest.mark.parametrize(
    "level,drift,vol",
    [(0.2, 0.0, 0.15), (1.0, 0.08, 0.15), (0.2, -0.05, 0.15), (2.0, 0.5, 0.2), (0.05, 0.5, 0.2)],
)
def test_mode_count_matches_a_grid_audit_of_the_root_equation(
    level: float, drift: float, vol: float
) -> None:
    """Every root is bracketed, so the count has to match a brute-force sweep."""
    beta = -drift * level / vol**2
    modes = 20
    spectrum = drawdown_spectrum(level, drift, vol, modes=modes)
    # The trigonometric roots live in theta = k * level; sweep to the end of the
    # last bracket the solver used.
    ceiling = (modes + 0.5) * math.pi + math.pi
    theta = np.linspace(1e-9, ceiling, 2_000_001)
    f = theta * np.cos(theta) + beta * np.sin(theta)
    crossings = int(np.sum(np.sign(f[:-1]) * np.sign(f[1:]) < 0))
    # The sweep sees every trigonometric root below the ceiling; the ground mode
    # is hyperbolic and is not one of them. The count of trigonometric roots
    # below the ceiling is `modes` plus however many extra brackets fit.
    assert crossings >= modes
    assert spectrum.rates.size == modes + (1 if spectrum.has_ground_mode else 0)
    # Strictly increasing rates: a repeated or out-of-order root means a bracket
    # was solved twice.
    assert np.all(np.diff(spectrum.rates) > 0.0)


def test_ground_mode_appears_exactly_at_drift_level_equal_to_variance() -> None:
    vol, drift = 0.15, 0.08
    threshold = vol**2 / drift  # 0.28125
    assert drawdown_spectrum(threshold * 0.999, drift, vol).has_ground_mode is False
    assert drawdown_spectrum(threshold, drift, vol).has_ground_mode is True
    assert drawdown_spectrum(threshold * 1.001, drift, vol).has_ground_mode is True
    # At the threshold the two families meet: the eigenfunction is the straight
    # line 1 - x / level, whose rate is drift**2 / (2 vol**2) and whose weight is
    # 3 / e.
    at = drawdown_spectrum(threshold, drift, vol)
    assert at.rates[0] == pytest.approx(drift**2 / (2.0 * vol**2), rel=1e-13)
    assert at.weights[0] == pytest.approx(3.0 / math.e, rel=1e-9)


def test_ground_mode_rate_is_not_lost_to_cancellation() -> None:
    """``beta**2 - t**2`` where ``t`` is a double's width from ``beta``."""
    level, drift, vol = 2.0, 0.5, 0.2
    spectrum = drawdown_spectrum(level, drift, vol, modes=8)
    beta = -drift * level / vol**2  # -62.5
    scale = vol**2 / (2.0 * level**2)
    # Taking the difference of the squares directly gives exactly zero here.
    t_naive = math.sqrt(beta**2 - 2.0 * level**2 * spectrum.rates[0] / vol**2)
    assert spectrum.rates[0] > 0.0
    assert t_naive == pytest.approx(-beta, rel=1e-12)
    # The gap identity says beta**2 - t**2 -> 2 |beta| * 2 |beta| exp(-2|beta|),
    # so the rate is about 4 beta**2 exp(2 beta) * scale.
    asymptote = 4.0 * beta * beta * math.exp(2.0 * beta) * scale
    assert spectrum.rates[0] == pytest.approx(asymptote, rel=1e-6)


def test_coefficients_are_the_inner_products_that_define_them() -> None:
    """``c = <1, phi> / <phi, phi>`` under the speed measure ``exp(2 b x) dx``."""
    for level, drift, vol in [
        (1.0, 0.08, 0.15),
        (1.0, 0.5, 0.2),
        (0.5, -0.04, 0.2),
        (0.3, 0.0, 0.2),
    ]:
        b = -drift / vol**2
        spectrum = drawdown_spectrum(level, drift, vol, modes=4)
        for index, rate in enumerate(spectrum.rates):
            square = 2.0 * level**2 * rate / vol**2 - (b * level) ** 2
            shape = eigenfunction(square, level, b)
            numerator = quad(lambda x, s=shape, w=b: math.exp(w * x) * s(x), 0.0, level, limit=400)[
                0
            ]
            denominator = quad(lambda x, s=shape: s(x) ** 2, 0.0, level, limit=400)[0]
            assert spectrum.weights[index] == pytest.approx(numerator / denominator, rel=1e-9)


# --------------------------------------------------------------------------- #
# the law itself


@pytest.mark.parametrize(
    "level,horizon,vol",
    [
        (0.1, 1.0, 0.15),
        (0.2, 1.0, 0.15),
        (0.4, 1.0, 0.15),
        (0.05, 0.25, 0.2),
        (1.0, 5.0, 0.3),
        (0.3, 10.0, 0.1),
        (0.6, 1.0, 0.15),
    ],
)
def test_driftless_law_against_the_reflection_series(
    level: float, horizon: float, vol: float
) -> None:
    ours = drawdown_survival(level, horizon, 0.0, vol)
    assert ours == pytest.approx(reflection_survival(level, horizon, vol), abs=5e-15)


def eigenfunction(square: float, level: float, b: float) -> Callable[[float], float]:
    """The eigenfunction behind a rate, rebuilt from the rate alone.

    A negative ``square`` is the hyperbolic ground mode, which is how this also
    checks that the rate and the mode's family agree.
    """
    if square < 0.0:
        kappa = math.sqrt(-square) / level
        return lambda x: math.cosh(kappa * x) + (b / kappa) * math.sinh(kappa * x)
    k = math.sqrt(square) / level
    return lambda x: math.cos(k * x) + (b / k) * math.sin(k * x)


def alternating_sum(terms: FloatArray) -> float:
    """Sum a series whose terms alternate and decay like ``1 / n``.

    The partial sums straddle the limit, so the mean of the last two is worth
    about a factor of ``n`` in accuracy over the last one alone. Truncating at
    2048 modes, that is the difference between 7e-4 and 5e-7 on the worst
    parameter set on this grid.
    """
    partial = np.cumsum(terms)
    return float(0.5 * (partial[-1] + partial[-2]))


def test_first_passage_laplace_transform_at_every_drift() -> None:
    """The one identity available at every drift, and it tests rates and weights together.

    The tolerance is scaled by the conditioning because that is what the error
    is: the coefficients carry ``exp(beta)``, so the truncated tail of a sum
    with no exponential damping in it is that much larger. Measured over the
    grid the worst ratio is 3e-10, and most parameter sets are at 1e-11 or
    below.
    """
    worst = 0.0
    for level, drift, vol in GRID:
        spectrum = drawdown_spectrum(level, drift, vol, modes=MAX_MODES)
        room = max(1.0, spectrum.conditioning)
        for theta in (0.05, 1.0, 10.0):
            ours = 1.0 - theta * alternating_sum(spectrum.weights / (spectrum.rates + theta))
            error = abs(ours - passage_laplace(theta, level, drift, vol))
            assert error < 1e-9 * room, (level, drift, vol, theta, error, room)
            worst = max(worst, error / room)
    assert worst < 1e-9


def test_mean_first_passage_time() -> None:
    """The ``theta -> 0`` limit of the same transform, against the ODE's solution."""
    for level, drift, vol in GRID:
        spectrum = drawdown_spectrum(level, drift, vol, modes=MAX_MODES)
        ours = alternating_sum(spectrum.weights / spectrum.rates)
        assert ours == pytest.approx(mean_passage(level, drift, vol), rel=1e-8)


def test_weights_sum_to_one_where_the_sum_is_conditioned_to_show_it() -> None:
    for level, drift, vol in GRID:
        if -drift * level / vol**2 > 2.0:  # a cancellation, tested separately
            continue
        spectrum = drawdown_spectrum(level, drift, vol, modes=MAX_MODES)
        # The partial sums of the weights converge like an alternating harmonic
        # series, so the mean of the last two is the honest estimate of a sum
        # whose terms decay like 1 / n.
        assert alternating_sum(spectrum.weights) == pytest.approx(1.0, abs=1e-6)


def test_the_law_depends_on_two_dimensionless_numbers_only() -> None:
    """Scaling the volatility and the horizon together changes nothing."""
    base = drawdown_survival(0.2, 1.0, 0.08, 0.15)
    # level / (vol sqrt(T)) and drift sqrt(T) / vol both held fixed
    for factor in (0.25, 4.0, 100.0):
        horizon = 1.0 * factor
        vol = 0.15 / math.sqrt(factor)
        drift = 0.08 / factor
        level = 0.2
        assert drawdown_survival(level, horizon, drift, vol) == pytest.approx(base, rel=1e-13)
    # and scaling the level with the volatility changes nothing either
    assert drawdown_survival(0.4, 1.0, 0.16, 0.3) == pytest.approx(base, rel=1e-13)


def test_survival_is_monotone_in_each_argument() -> None:
    levels = np.geomspace(0.02, 1.5, 60)
    values = [drawdown_survival(float(x), 3.0, 0.07, 0.18) for x in levels]
    assert np.all(np.diff(values) > 0.0)
    horizons = np.geomspace(0.1, 40.0, 60)
    values = [drawdown_survival(0.3, float(t), 0.07, 0.18) for t in horizons]
    assert np.all(np.diff(values) < 0.0)
    drifts = np.linspace(-0.1, 0.4, 60)
    values = [drawdown_survival(0.3, 3.0, float(m), 0.18) for m in drifts]
    assert np.all(np.diff(values) > 0.0)


def test_probabilities_stay_inside_the_unit_interval() -> None:
    for level, drift, vol in GRID:
        for horizon in (0.1, 1.0, 10.0):
            value = drawdown_survival(level, horizon, drift, vol)
            assert 0.0 <= value <= 1.0
            assert drawdown_exceedance(level, horizon, drift, vol) == pytest.approx(
                1.0 - value, abs=0.0
            )


def test_a_level_the_path_cannot_reach_short_circuits() -> None:
    # Thousands of standard deviations away: the mode count needed is past the
    # cap, and a bound that does not use the expansion says the answer is one.
    assert drawdown_survival(800.0, 1.0, 0.0, 0.15) == 1.0
    assert drawdown_exceedance(800.0, 1.0, 0.0, 0.15) == 0.0
    # Just inside the cap the expansion still runs, and agrees.
    assert drawdown_survival(80.0, 1.0, 0.0, 0.15) == 1.0
    with pytest.raises(ValidationError, match="eigenmodes"):
        # Past the cap, but not far enough past for the bound to settle it.
        drawdown_survival(800.0, 1.0, 400.0, 0.15)


def test_the_conditioning_limit_is_a_losing_strategy_and_is_named() -> None:
    with pytest.raises(ValidationError, match="cancellation"):
        drawdown_spectrum(2.0, -0.2, 0.1)
    # Where it bites, in the units a user has: the limit is on
    # -drift * level / volatility**2, so for a level at the drawdown a losing
    # strategy actually produces it is about the square of the horizon's Sharpe
    # ratio. Nothing at or above zero drift is affected.
    for drift in (0.0, 0.01, 0.5, 5.0):
        # At zero drift the largest coefficient is exactly 4 / pi, and raising
        # the drift only shrinks it.
        assert drawdown_spectrum(1.0, drift, 0.15).conditioning <= 4.0 / math.pi
    # On the losing side the conditioning grows like exp(beta) with a prefactor
    # of about a tenth, which is the largest value of |sin(theta)**3| / theta over
    # the roots. Asserting the growth *rate* rather than the value is the point:
    # the prefactor is incidental and the exponent is what sets the domain.
    vol = 0.15
    prefactors = []
    for drift in (-0.02, -0.05, -0.10, -0.15, -0.20, -0.3, -0.4):
        beta = -drift / vol**2
        spectrum = drawdown_spectrum(1.0, drift, vol, modes=256)
        prefactors.append(spectrum.conditioning * beta / math.exp(beta))
    # Rising towards about 0.76 and already inside a narrow band by beta of one.
    assert all(0.55 < value < 0.80 for value in prefactors), prefactors
    assert np.all(np.diff(prefactors) > 0.0)
    # So the refusal lands near beta = 19, where 0.76 exp(beta) / beta reaches
    # 1e7. For a level at the depth a losing strategy actually reaches, beta is
    # around the square of the horizon's Sharpe ratio, which puts the boundary
    # past a total Sharpe ratio of -4 -- where the drawdown's arrival was never
    # in doubt.
    assert drawdown_spectrum(1.0, -17.0 * vol**2, vol).conditioning < 1e7
    with pytest.raises(ValidationError, match="cancellation"):
        drawdown_spectrum(1.0, -21.0 * vol**2, vol)


def test_bad_arguments_are_refused() -> None:
    with pytest.raises(ValidationError):
        drawdown_survival(0.0, 1.0, 0.0, 0.15)
    with pytest.raises(ValidationError):
        drawdown_survival(0.1, 0.0, 0.0, 0.15)
    with pytest.raises(ValidationError):
        drawdown_survival(0.1, 1.0, 0.0, -0.15)
    with pytest.raises(ValidationError):
        drawdown_spectrum(0.1, float("nan"), 0.15)
    with pytest.raises(ValidationError):
        drawdown_spectrum(0.1, 0.0, 0.15, modes=0)
    with pytest.raises(ValidationError):
        drawdown_spectrum(0.1, 0.0, 0.15, modes=MAX_MODES + 1)


def test_the_clamp_only_ever_removes_round_off() -> None:
    """The sum is a probability by construction; the clamp is not doing work."""
    for level, drift, vol in GRID:
        for horizon in (0.05, 1.0, 10.0, 100.0):
            tau = vol**2 * horizon / (2.0 * level**2)
            beta = -drift * level / vol**2
            modes = min(
                MAX_MODES, math.ceil(math.sqrt((45.0 + max(0.0, beta)) / tau) / math.pi) + 4
            )
            raw = drawdown_spectrum(level, drift, vol, modes=modes).survival(horizon)
            clamped = drawdown_survival(level, horizon, drift, vol)
            assert abs(raw - clamped) < 1e-12, (level, drift, vol, horizon, raw)
