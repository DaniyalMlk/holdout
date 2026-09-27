from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_example(name: str) -> Any:
    namespace = runpy.run_path(str(EXAMPLES / name))
    return namespace["main"]()


def test_deflated_worked_example_reproduces_the_paper(capsys: pytest.CaptureFixture[str]) -> None:
    result = run_example("deflated_worked_example.py")
    assert round(result["benchmark"], 4) == 0.1132
    assert round(result["dsr"], 4) == 0.9004
    assert result["needed_days"] > 1250
    assert "0.9004" in capsys.readouterr().out


def test_parameter_sweep_separates_noise_from_a_real_drift() -> None:
    result = run_example("parameter_sweep.py")
    noise, trend = result["noise"], result["trend"]
    assert noise["deflated"] < 0.95 < trend["deflated"]
    assert noise["spa"] > 0.1 > 0.05 > trend["spa"]
    assert noise["pbo"] > 0.9
    assert noise["haircut"] == 1.0
    # With a real drift most rules profit, so the winner's rank among them is
    # close to a coin flip (PBO near one half) while it rarely loses money.
    assert 0.3 < trend["pbo"] < 0.6
    assert trend["loss"] < 0.05


def test_purged_cv_example_shows_the_leak() -> None:
    result = run_example("purged_cv.py")
    assert result["leaked"] == 1.0
    assert result["shuffled"] > 0.15
    assert abs(result["purged"]) < 0.1


def test_the_model_confidence_set_example_keeps_nearly_everything(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both sweeps, and the gap between "made money" and "is the one that did".

    On the driftless sweep every rule survives under both statistics, which is the
    right answer to a set of thirty rules over a market with nothing in it. On the
    sweep with a genuine drift the deflated Sharpe ratio is 0.971 — the best rule
    clears the search that produced it — and 29 of the 30 rules are still in the
    10% set. The surviving set spans 9.3% of annualised mean return, which is the
    range the data declines to rule out.
    """
    result = run_example("model_confidence_set.py")
    noise, trend = result["noise"], result["trend"]
    assert noise["max"] == noise["models"] == 30
    assert noise["range"] == 30
    assert noise["deflated"] < 0.95 < trend["deflated"]
    assert trend["max"] >= 25
    assert trend["best"] != noise["best"]
    output = capsys.readouterr().out
    assert "29 of 30 rules are still in the confidence set" in output


def test_the_sharpe_difference_example_measures_both_tests_size(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The size measurement, which is the whole case for two variances.

    With independent returns both tests hold their size — 4.2% and 4.7% against a
    nominal 5% — and the two standard errors agree to within one per cent, so the
    robust version costs nothing where it is not needed. Serial dependence is where
    they part: at a persistence of 0.6 the closed form rejects a true null 34.7% of
    the time and the robust version 10.0%. Both halves of that are asserted,
    including that the robust one is still wrong.
    """
    result = run_example("sharpe_difference.py")
    clean = result["none"]
    strong = result["AR(1), rho = 0.6"]
    assert clean["closed"] == pytest.approx(0.042, abs=0.01)
    assert clean["robust"] == pytest.approx(0.047, abs=0.01)
    assert clean["ratio"] == pytest.approx(0.99, abs=0.02)
    assert strong["closed"] == pytest.approx(0.347, abs=0.03)
    assert strong["robust"] == pytest.approx(0.100, abs=0.02)
    assert strong["robust"] > 0.05
    assert strong["ratio"] == pytest.approx(1.65, abs=0.05)
    assert "All figures reproduced" in capsys.readouterr().out
