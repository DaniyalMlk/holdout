from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from holdout import InsufficientDataError, ValidationError
from holdout.deflated import (
    deflate_trials,
    deflated_sharpe_ratio,
    expected_maximum_normal,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)
from holdout.sharpe import sharpe_ratio


class TestPublishedExample:
    """Bailey and López de Prado (2014), "The Deflated Sharpe Ratio", section 6.

    A daily strategy with annualised SR = 2.5 found after N = 100 independent
    trials whose annualised Sharpe ratios have variance 1/2; T = 1250,
    skewness -3, kurtosis 10. The paper reports SR0 ~= 0.1132 (per period)
    and DSR = 0.9004.
    """

    periods = 250
    sharpe = 2.5 / math.sqrt(250)
    variance = 0.5 / 250

    def test_expected_maximum(self) -> None:
        sr0 = expected_maximum_sharpe(100, self.variance)
        assert round(sr0, 4) == 0.1132

    def test_deflated_sharpe_ratio(self) -> None:
        dsr = deflated_sharpe_ratio(
            self.sharpe,
            1250,
            n_trials=100,
            trials_variance=self.variance,
            skewness=-3.0,
            kurtosis=10.0,
        )
        assert round(dsr, 4) == 0.9004

    def test_the_printed_benchmark_is_rounded(self) -> None:
        # The paper's 0.9004 comes from the unrounded SR0 = 0.113172; plugging
        # the printed 0.1132 back in gives 0.90026. Recorded so that nobody
        # "fixes" the formula to match a recomputation from rounded figures.
        from_printed = probabilistic_sharpe_ratio(
            self.sharpe, 1250, benchmark=0.1132, skewness=-3.0, kurtosis=10.0
        )
        assert from_printed == pytest.approx(0.90026, abs=1e-5)

    def test_it_is_not_significant_at_95_percent(self) -> None:
        dsr = deflated_sharpe_ratio(
            self.sharpe,
            1250,
            n_trials=100,
            trials_variance=self.variance,
            skewness=-3.0,
            kurtosis=10.0,
        )
        psr = probabilistic_sharpe_ratio(self.sharpe, 1250, skewness=-3.0, kurtosis=10.0)
        assert psr > 0.9999
        assert dsr < 0.95


def test_psr_matches_its_formula_with_scipy() -> None:
    sr, n, bench, skew, kurt = 0.12, 600, 0.05, -0.8, 5.5
    z = (sr - bench) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    got = probabilistic_sharpe_ratio(sr, n, benchmark=bench, skewness=skew, kurtosis=kurt)
    assert got == pytest.approx(stats.norm.cdf(z), rel=1e-12)


def test_psr_is_one_half_at_the_benchmark_and_monotone() -> None:
    assert probabilistic_sharpe_ratio(0.1, 500, benchmark=0.1) == pytest.approx(0.5)
    values = [probabilistic_sharpe_ratio(0.1, n) for n in (10, 100, 1000)]
    assert values == sorted(values)
    worse = probabilistic_sharpe_ratio(0.1, 500, skewness=-2.0, kurtosis=12.0)
    assert worse < probabilistic_sharpe_ratio(0.1, 500)


def test_psr_rejects_bad_inputs() -> None:
    with pytest.raises(InsufficientDataError):
        probabilistic_sharpe_ratio(0.1, 1)
    with pytest.raises(InsufficientDataError):
        probabilistic_sharpe_ratio(0.1, 10.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        probabilistic_sharpe_ratio(0.1, 100, benchmark=math.inf)
    with pytest.raises(ValidationError, match="excess kurtosis"):
        probabilistic_sharpe_ratio(0.1, 100, skewness=-1.0, kurtosis=0.5)


def test_psr_degenerate_distribution() -> None:
    # A symmetric two-point distribution has kurtosis 1 and zero skew, so the
    # variance term at SR = 0 is exactly 1; at kurt = 1 + skew**2 with
    # SR = 2 / skew it is exactly zero.
    assert probabilistic_sharpe_ratio(1.0, 50, skewness=2.0, kurtosis=5.0) == 1.0
    assert probabilistic_sharpe_ratio(1.0, 50, benchmark=2.0, skewness=2.0, kurtosis=5.0) == 0.0


@pytest.mark.parametrize(
    ("sr", "bench", "skew", "kurt", "conf"),
    [(0.1, 0.0, 0.0, 3.0, 0.95), (0.08, 0.02, -1.2, 7.0, 0.99), (0.3, 0.1, 0.5, 4.0, 0.9)],
)
def test_minimum_track_record_length_inverts_psr(
    sr: float, bench: float, skew: float, kurt: float, conf: float
) -> None:
    n = minimum_track_record_length(
        sr, benchmark=bench, confidence=conf, skewness=skew, kurtosis=kurt
    )
    # PSR with n - 1 = MinTRL - 1 degrees reproduces the confidence exactly.
    z = (sr - bench) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    assert stats.norm.cdf(z) == pytest.approx(conf, rel=1e-12)
    assert (
        probabilistic_sharpe_ratio(sr, math.ceil(n), benchmark=bench, skewness=skew, kurtosis=kurt)
        >= conf
    )


def test_minimum_track_record_length_by_hand() -> None:
    # Annual SR 1 on daily data, normal returns, 95%: (1 + SR^2/2) * (1.645/SR)^2 + 1.
    sr = 1 / math.sqrt(252)
    expected = 1 + (1 + sr**2 / 2) * (stats.norm.ppf(0.95) / sr) ** 2
    assert minimum_track_record_length(sr) == pytest.approx(expected)
    assert minimum_track_record_length(sr) / 252 == pytest.approx(2.72, abs=0.01)


def test_minimum_track_record_length_is_undefined_below_the_benchmark() -> None:
    with pytest.raises(ValidationError, match="does not exceed"):
        minimum_track_record_length(0.05, benchmark=0.05)
    with pytest.raises(ValidationError, match="confidence"):
        minimum_track_record_length(0.1, confidence=1.0)


@pytest.mark.parametrize("n", [2, 3, 5, 10, 100, 1000, 100_000])
def test_exact_expected_maximum_matches_quadrature(n: int) -> None:
    peak = math.sqrt(2 * math.log(n))
    reference, _ = integrate.quad(
        lambda x: x * n * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1),
        -12,
        12,
        limit=500,
        points=[peak],
    )
    assert expected_maximum_normal(n) == pytest.approx(reference, abs=1e-9)


def test_exact_expected_maximum_known_values() -> None:
    assert expected_maximum_normal(1) == 0.0
    assert expected_maximum_normal(2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-12)
    assert expected_maximum_normal(3) == pytest.approx(1.5 / math.sqrt(math.pi), abs=1e-12)


@pytest.mark.parametrize(
    ("n", "error"),
    [(2, -0.0788), (5, 0.0255), (10, 0.0233), (100, 0.0092), (1000, 0.0042), (1_000_000, 0.0010)],
)
def test_approximation_error_is_what_the_docstring_says(n: int, error: float) -> None:
    approx = expected_maximum_sharpe(n, 1.0)
    exact = expected_maximum_sharpe(n, 1.0, method="exact")
    assert approx / exact - 1 == pytest.approx(error, abs=5e-4)


def test_expected_maximum_against_simulation() -> None:
    rng = np.random.default_rng(8)
    draws = rng.normal(0.02, 0.05, size=(20_000, 40)).max(axis=1)
    exact = expected_maximum_sharpe(40, 0.05**2, mean=0.02, method="exact")
    assert draws.mean() == pytest.approx(exact, abs=4 * draws.std() / math.sqrt(draws.size))


def test_expected_maximum_arguments() -> None:
    assert expected_maximum_sharpe(1, 0.3, mean=0.1) == 0.1
    assert expected_maximum_sharpe(50, 0.0, mean=0.1) == 0.1
    for bad in (0, 2.5):
        with pytest.raises(ValidationError):
            expected_maximum_sharpe(bad, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        expected_maximum_sharpe(10, -0.1)
    with pytest.raises(ValidationError):
        expected_maximum_sharpe(10, 0.1, mean=math.nan)
    with pytest.raises(ValidationError, match="method"):
        expected_maximum_sharpe(10, 0.1, method="guess")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        expected_maximum_normal(0)


def test_deflate_trials_picks_the_best_and_deflates_it() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(750, 30)) * 0.01
    x[:, 7] += 0.002
    result = deflate_trials(x)
    ratios = [sharpe_ratio(x[:, j]) for j in range(30)]
    assert result.selected == int(np.argmax(ratios))
    assert result.estimate.value == pytest.approx(max(ratios))
    assert result.trials_variance == pytest.approx(np.var(ratios, ddof=1))
    assert result.deflated < result.probabilistic
    assert result.n_trials == 30
    fewer = deflate_trials(x, n_trials=3)
    assert fewer.deflated > result.deflated
    chosen = deflate_trials(x, selected=0)
    assert chosen.selected == 0


def test_deflate_trials_arguments() -> None:
    x = np.random.default_rng(0).normal(size=(100, 4))
    with pytest.raises(ValidationError, match="selected"):
        deflate_trials(x, selected=4)
    with pytest.raises(ValidationError, match="n_trials"):
        deflate_trials(x, n_trials=0.5)
    with pytest.raises(InsufficientDataError):
        deflate_trials(x[:, :1])


def test_under_the_null_the_undeflated_ratio_is_fooled_and_the_deflated_is_not() -> None:
    # Fifty pure-noise strategies, the best one selected: PSR(0) clears 95% in
    # about nine runs out of ten; the deflated ratio should essentially never.
    rng = np.random.default_rng(0)
    reps, psr_hits, dsr_hits = 200, 0, 0
    for _ in range(reps):
        result = deflate_trials(rng.normal(size=(500, 50)) * 0.01)
        psr_hits += result.probabilistic > 0.95
        dsr_hits += result.deflated > 0.95
    assert psr_hits / reps > 0.8
    assert dsr_hits / reps <= 0.05


def test_a_genuine_edge_among_noise_survives_deflation_only_sometimes() -> None:
    # One strategy with per-period SR 0.2 (annualised ~3.2) hidden among 49
    # noise strategies over 500 periods: the README reports it clears DSR 95%
    # about half the time.
    rng = np.random.default_rng(1)
    reps, hits = 200, 0
    for _ in range(reps):
        x = rng.normal(size=(500, 50))
        x[:, 0] += 0.2
        hits += deflate_trials(x * 0.01).deflated > 0.95
    assert 0.40 < hits / reps < 0.66
