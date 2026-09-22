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
