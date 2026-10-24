"""Observed drawdown statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from holdout import (
    MaximumDrawdown,
    drawdown_series,
    equity_curve,
    maximum_drawdown,
)
from holdout.exceptions import InsufficientDataError, ValidationError


def test_equity_curve_carries_its_starting_point() -> None:
    curve = equity_curve([0.1, -0.2, 0.05])
    assert curve.size == 4
    assert curve[0] == 0.0
    assert curve[-1] == pytest.approx(-0.05)


def test_first_return_can_be_the_whole_drawdown() -> None:
    # Without the leading zero this reports 0.0, which is the bug the leading
    # zero exists to prevent.
    worst = maximum_drawdown([-0.3, 0.1, 0.1])
    assert worst.depth == pytest.approx(0.3)
    assert worst.peak_index == 0
    assert worst.trough_index == 1


def test_drawdown_series_is_non_negative_and_zero_at_new_highs() -> None:
    rng = np.random.default_rng(11)
    r = rng.normal(0.001, 0.01, 500)
    draw = drawdown_series(r)
    curve = equity_curve(r)
    assert np.all(draw >= 0.0)
    highs = curve >= np.maximum.accumulate(curve)
    assert np.allclose(draw[highs], 0.0)


def test_monotone_series_have_no_drawdown() -> None:
    worst = maximum_drawdown([0.01] * 20)
    assert worst.depth == 0.0
    assert worst.proportional == 0.0
    assert worst.peak_index == worst.trough_index == 0
    assert worst.recovery_index == 0
    assert worst.time_under_water == 0
    assert worst.under_water_censored is False


def test_peak_trough_recovery_on_a_hand_checked_path() -> None:
    # curve:     0, 1, 3, 2, 0, -1, 3   (deepest fall is 3 -> -1, regained at the end)
    # drawdown:  0, 0, 0, 1, 3,  4, 0
    returns = [1.0, 2.0, -1.0, -2.0, -1.0, 4.0]
    worst = maximum_drawdown(returns)
    assert worst.depth == pytest.approx(4.0)
    assert worst.peak_index == 2
    assert worst.trough_index == 5
    assert worst.recovery_index == 6
    assert worst.time_under_water == 3  # curve points 3, 4, 5
    assert worst.under_water_censored is False
    assert worst.final == 0.0


def test_a_recovery_must_regain_the_peak_not_merely_rise() -> None:
    # curve:     0, 1, 3, 2, 0, -1, 2   — the last point is above the trough and
    # below the peak of 3, so the drawdown is still open.
    worst = maximum_drawdown([1.0, 2.0, -1.0, -2.0, -1.0, 3.0])
    assert worst.depth == pytest.approx(4.0)
    assert worst.recovery_index is None
    assert worst.time_under_water == 4
    assert worst.under_water_censored is True


def test_recovery_is_none_when_the_path_never_regains_the_peak() -> None:
    # curve: 0, 1, -1, -0.5 against a peak of 1
    worst = maximum_drawdown([1.0, -2.0, 0.5])
    assert worst.recovery_index is None
    assert worst.depth == pytest.approx(2.0)
    assert worst.final == pytest.approx(1.5)


def test_proportional_depth_reads_the_returns_as_logarithmic() -> None:
    worst = maximum_drawdown([-math.log(2.0)])
    assert worst.depth == pytest.approx(math.log(2.0))
    assert worst.proportional == pytest.approx(0.5)


def test_censoring_distinguishes_an_open_drawdown_from_a_closed_one() -> None:
    # Two runs under water of length 3; the second is still open at the end.
    open_ended = maximum_drawdown([1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0])
    assert open_ended.time_under_water == 3
    assert open_ended.under_water_censored is True

    # The open run is shorter than the closed one, so the reported length is not
    # censored even though the record ends under water.
    shorter = maximum_drawdown([1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0])
    assert shorter.time_under_water == 3
    assert shorter.under_water_censored is False

    recovered = maximum_drawdown([1.0, -1.0, -1.0, -1.0, 4.0])
    assert recovered.time_under_water == 3
    assert recovered.under_water_censored is False


def test_longest_under_water_takes_the_longest_run_not_the_deepest() -> None:
    # A short deep fall, then a long shallow one.
    returns = [-5.0, 5.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.2]
    worst = maximum_drawdown(returns)
    assert worst.depth == pytest.approx(5.0)
    assert worst.trough_index == 1
    assert worst.time_under_water == 5


def test_the_identity_between_the_series_and_the_summary() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        r = rng.normal(0.0002, 0.012, 300)
        draw = drawdown_series(r)
        worst = maximum_drawdown(r)
        assert worst.depth == pytest.approx(float(draw.max()))
        assert worst.final == pytest.approx(float(draw[-1]))
        assert draw[worst.trough_index] == pytest.approx(worst.depth)


def test_repr_is_readable() -> None:
    text = repr(maximum_drawdown([1.0, -2.0, 0.5]))
    assert "depth=" in text and "recovery=never" in text


def test_empty_and_malformed_inputs_are_refused() -> None:
    with pytest.raises(InsufficientDataError):
        maximum_drawdown([])
    with pytest.raises(ValidationError):
        maximum_drawdown([0.1, float("nan")])


def test_dataclass_is_frozen() -> None:
    worst = maximum_drawdown([1.0, -2.0])
    assert isinstance(worst, MaximumDrawdown)
    with pytest.raises(AttributeError):
        worst.depth = 0.0  # type: ignore[misc]


# The peak is the *last* high-water mark at or before the trough, which is what
# the docstring has always said and what `argmax` does not do. These assert the
# property rather than the index: that the pair brackets a fall with nothing at
# the peak's level inside it.


def bracket_holds(returns: list[float]) -> bool:
    """Whether the reported peak is the last high-water mark before the trough."""
    worst = maximum_drawdown(returns)
    curve = equity_curve(returns)
    peak, trough = worst.peak_index, worst.trough_index
    if curve[peak] != curve[: trough + 1].max():
        return False
    return all(curve[index] < curve[peak] for index in range(peak + 1, trough + 1))


def test_flat_days_at_the_high_water_mark_are_not_part_of_the_fall() -> None:
    """A strategy out of the market, then a loss. The flat days are not the fall."""
    returns = [0.01, 0.0, 0.0, -0.05, 0.02]
    worst = maximum_drawdown(returns)
    assert worst.peak_index == 3
    assert worst.trough_index == 4
    assert bracket_holds(returns)
    # Everything else is unchanged, because the peak's value is the same.
    assert worst.depth == pytest.approx(0.05)
    assert worst.recovery_index is None
    assert worst.time_under_water == 2


def test_the_first_high_water_mark_would_have_charged_the_flat_days_to_the_fall() -> None:
    """What `argmax` returns, kept here so the test says what it is defending."""
    returns = [0.01, 0.0, 0.0, -0.05, 0.02]
    curve = equity_curve(returns)
    trough = maximum_drawdown(returns).trough_index
    first = int(np.argmax(curve[: trough + 1]))
    assert first == 1
    assert trough - first == 3
    assert trough - maximum_drawdown(returns).peak_index == 1


def test_the_bracket_holds_on_records_full_of_repeated_highs() -> None:
    """Zeros make high-water marks repeat, which is the case that distinguishes them."""
    generator = np.random.default_rng(20261024)
    distinguishing = 0
    for _ in range(300):
        draws = generator.normal(0.0, 0.01, 60)
        draws[generator.random(60) < 0.4] = 0.0
        returns = [float(one) for one in draws]
        assert bracket_holds(returns)
        curve = equity_curve(returns)
        trough = maximum_drawdown(returns).trough_index
        if int(np.argmax(curve[: trough + 1])) != maximum_drawdown(returns).peak_index:
            distinguishing += 1
    # And the sweep is not vacuous: the two readings really do differ here.
    assert distinguishing > 30


def test_a_record_that_never_falls_still_reports_the_pair_together() -> None:
    worst = maximum_drawdown([0.0, 0.0, 0.0])
    assert worst.depth == 0.0
    assert worst.peak_index == worst.trough_index


def test_a_record_that_only_falls_peaks_at_the_start() -> None:
    returns = [-0.01, -0.01, -0.01]
    assert maximum_drawdown(returns).peak_index == 0
    assert bracket_holds(returns)
