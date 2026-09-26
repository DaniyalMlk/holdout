from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from holdout.cli import main


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("sample")
    assert main(["sample-data", "--out", str(out)]) == 0
    return out


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    assert main(list(argv)) == 0
    return capsys.readouterr().out


def test_sample_data_writes_both_sweeps(sample: Path) -> None:
    assert (sample / "sweep.csv").exists()
    assert (sample / "trending.csv").exists()
    header = (sample / "sweep.csv").read_text().splitlines()[0]
    assert header.startswith("day,ma5_40,")


def test_sharpe(sample: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "sharpe", str(sample / "sweep.csv"), "--column", "ma10_120")
    assert "2520 periods at 252 per year" in out
    assert "ma10_120" in out
    assert "psr(0)" in out
    # Summary, blank line, header, rule and one row.
    assert len(out.strip().splitlines()) == 5


def test_deflate_contrasts_the_two_sweeps(sample: Path, capsys: pytest.CaptureFixture[str]) -> None:
    noise = run(capsys, "deflate", str(sample / "sweep.csv"))
    trend = run(capsys, "deflate", str(sample / "trending.csv"))

    def deflated(text: str) -> float:
        line = next(x for x in text.splitlines() if x.startswith("deflated:"))
        return float(line.split()[1])

    assert "effective trials:" in noise
    assert deflated(noise) < 0.95 < deflated(trend)
    everything = run(capsys, "deflate", str(sample / "sweep.csv"), "--effective", "none")
    assert "using all 30" in everything
    assert deflated(everything) < deflated(noise)


def test_pbo(sample: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "pbo", str(sample / "sweep.csv"), "--blocks", "8")
    assert "30 strategies, 8 blocks, 70 splits" in out
    assert "probability of backtest overfitting" in out
    assert "most often selected" in out


def test_spa_against_zero_and_a_column(sample: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "spa", str(sample / "trending.csv"), "--bootstrap", "300")
    assert "30 strategies against zero" in out
    assert "(Romano-Wolf)" in out
    out = run(
        capsys, "spa", str(sample / "sweep.csv"), "--benchmark", "ma5_40", "--bootstrap", "300"
    )
    assert "29 strategies against ma5_40" in out


def test_mcs_on_pure_noise_keeps_every_model(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Thirty crossover rules over ten years of a driftless series.

    Nothing distinguishes them because nothing about them is different, and the
    output says so in the one sentence a reader needs rather than leaving them to
    infer it from thirty p-values near one.
    """
    out = run(capsys, "mcs", str(sample / "sweep.csv"), "--bootstrap", "300")
    assert "30 models, 2520 periods" in out
    assert "max statistic" in out
    assert "All 30 models are in the set" in out
    assert "claim the data does not support" in out
    assert out.count(" in") >= 30


def test_mcs_on_a_trending_series_still_keeps_nearly_everything(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The measurement worth having, because it is the uncomfortable one.

    The trending sweep has a genuine signal in it — `deflate` finds the best rule
    significant — and ten years of daily data still cannot separate 29 of the 30
    parameter choices at the 10% level. The best annualised mean is 10.8% and the
    worst survivor's is 1.5%, which is the range the data declines to rule out.
    """
    out = run(capsys, "mcs", str(sample / "trending.csv"), "--bootstrap", "300")
    assert "30 models, 2520 periods" in out
    lines = [line for line in out.splitlines() if line.endswith((" in", " out"))]
    assert len(lines) == 30
    inside = sum(line.endswith(" in") for line in lines)
    assert inside >= 25
    assert "survive at 0.1" in out or "All 30" in out


def test_mcs_reports_the_rows_in_order_of_mean_performance(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(capsys, "mcs", str(sample / "trending.csv"), "--bootstrap", "300")
    means = [
        float(line.split()[1].rstrip("%"))
        for line in out.splitlines()
        if line.endswith((" in", " out"))
    ]
    assert means == sorted(means, reverse=True)


def test_mcs_at_a_larger_level_gives_a_smaller_set(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Which is backwards from a hypothesis test and is the thing to remember."""
    sizes = []
    for alpha in ("0.05", "0.5"):
        out = run(
            capsys,
            "mcs",
            str(sample / "trending.csv"),
            "--bootstrap",
            "300",
            "--alpha",
            alpha,
        )
        sizes.append(sum(line.endswith(" in") for line in out.splitlines()))
    assert sizes[0] > sizes[1]


def test_mcs_takes_the_range_statistic_too(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(
        capsys,
        "mcs",
        str(sample / "trending.csv"),
        "--statistic",
        "range",
        "--bootstrap",
        "300",
    )
    assert "range statistic" in out


def test_errors_are_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("d,a\n1,0.1\n2,x\n")
    assert main(["sharpe", str(bad)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("holdout: ")
    assert "not a number" in err


def test_module_entry_point(sample: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "holdout", "pbo", str(sample / "sweep.csv"), "--blocks", "4"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "probability of backtest overfitting" in done.stdout
    version = subprocess.run(
        [sys.executable, "-m", "holdout", "--version"], capture_output=True, text=True, check=True
    )
    assert version.stdout.startswith("holdout ")


def test_a_closed_pipe_is_not_a_traceback(
    sample: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate "holdout sharpe ... | head -1": the reader has gone away.
    sink = (tmp_path / "sink").open("w")

    class ClosedPipe:
        def write(self, _: str) -> int:
            raise BrokenPipeError

        def flush(self) -> None:
            pass

        def fileno(self) -> int:
            return sink.fileno()

    monkeypatch.setattr(sys, "stdout", ClosedPipe())
    assert main(["sharpe", str(sample / "sweep.csv")]) == 1
    sink.close()
