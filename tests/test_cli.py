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


def test_compare_reports_both_variances_side_by_side(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of the report is the comparison, so both rows have to be there.

    A single standard error would let the reader believe the question of which
    variance to use had been settled for them, and on serially dependent data the
    two disagree by enough to change the conclusion.
    """
    text = run(capsys, "compare", str(sample / "trending.csv"), "ma5_40", "ma5_120")
    assert "Jobson-Korkie / Memmel" in text
    assert "Ledoit-Wolf" in text
    assert "correlation" in text
    assert "difference (annualised)" in text
    assert "reject at 5%" in text
    # Two data rows in the variance table, one per estimator.
    assert text.count("0.0") >= 2


def test_compare_takes_an_explicit_bandwidth(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Zero is the White case: no autocovariances, so no serial-dependence allowance.

    The report names the bandwidth it used, because the robust standard error means
    something different at each one and a figure without it cannot be reproduced.
    """
    default = run(capsys, "compare", str(sample / "trending.csv"), "ma5_40", "ma5_120")
    assert "bandwidth 8" in default
    zero = run(
        capsys,
        "compare",
        str(sample / "trending.csv"),
        "ma5_40",
        "ma5_120",
        "--bandwidth",
        "0",
    )
    assert "bandwidth 0" in zero


def test_compare_names_the_columns_it_has(sample: Path) -> None:
    assert main(["compare", str(sample / "trending.csv"), "ma5_40", "absent"]) == 2


def test_compare_refuses_a_column_against_itself(sample: Path) -> None:
    """0/0 is not a p-value of one, and the report would read as if it were."""
    assert main(["compare", str(sample / "trending.csv"), "ma5_40", "ma5_40"]) == 2


def test_stability_prints_both_p_values_and_says_which_to_read(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(
        capsys,
        "stability",
        str(sample / "sweep.csv"),
        "--bootstrap",
        "99",
        "--column",
        "ma5_40",
    )
    assert "p (bootstrap)" in out
    assert "p (naive)" in out
    assert "The bootstrap p-value is the one to read" in out
    # And it says what a large p-value is worth, because that is the reading
    # this command makes easiest to get wrong.
    assert "weak evidence" in out
    assert "candidate break dates" in out


def test_stability_reports_the_sharpe_ratio_either_side_of_the_break(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(
        capsys, "stability", str(sample / "sweep.csv"), "--bootstrap", "99", "--column", "ma5_40"
    )
    line = next(one for one in out.splitlines() if one.startswith("ma5_40"))
    fields = line.split()
    assert len(fields) == 7
    split = int(fields[1])
    assert 0 < split < 2520
    # Both Sharpe ratios are annualised, so they are of order one rather than of
    # order a hundredth.
    assert abs(float(fields[2])) < 10.0
    assert abs(float(fields[3])) < 10.0
    # The naive p-value is the smaller of the two on this series, which is the
    # whole reason both are printed.
    assert float(fields[6]) < float(fields[5])


def test_stability_takes_a_trim_and_it_changes_the_search(
    sample: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wide = run(
        capsys,
        "stability",
        str(sample / "sweep.csv"),
        "--bootstrap",
        "49",
        "--column",
        "ma5_40",
        "--trim",
        "0.02",
    )
    narrow = run(
        capsys,
        "stability",
        str(sample / "sweep.csv"),
        "--bootstrap",
        "49",
        "--column",
        "ma5_40",
        "--trim",
        "0.4",
    )

    def candidates(text: str) -> int:
        return int(text.split("per year, ")[1].split(" candidate")[0])

    assert candidates(wide) > candidates(narrow)
    assert "trim 0.02" in wide
    assert "trim 0.4" in narrow


def test_stability_refuses_a_trim_of_a_half(sample: Path) -> None:
    assert (
        main(
            [
                "stability",
                str(sample / "sweep.csv"),
                "--bootstrap",
                "9",
                "--column",
                "ma5_40",
                "--trim",
                "0.5",
            ]
        )
        == 2
    )


def test_uniqueness_reports_the_identity_and_the_cap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = run(capsys, "uniqueness", "--count", "200", "--window", "20", "--replications", "3")
    # Two hundred twenty-bar rolling windows span 219 bars and are worth about
    # eleven independent observations, which is the number the command exists
    # to print.
    assert "effective sample size: 10.95 of 200" in out
    assert "peak concurrency:      20" in out
    assert "identity check:        sum(uniqueness * length) = 219.000000" in out
    assert "against 219 covered bars" in out
    # The full-size draw sits on a cap it cannot beat; the small one does not.
    rows = {
        line.split()[0]: line.split()
        for line in out.splitlines()
        if line[:1].isdigit() and len(line.split()) == 5
    }
    assert float(rows["200"][2]) > 0.99
    assert float(rows["200"][3]) > 0.99
    assert float(rows["10"][2]) < 0.9


def test_uniqueness_reads_a_label_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("0,4\n5,9\n10,14\n", encoding="utf-8")
    out = run(capsys, "uniqueness", "--labels", str(path), "--replications", "2")
    assert "3 labels over bars 0..14" in out
    assert "average uniqueness:    1.0000" in out
    assert "effective sample size: 3.00 of 3" in out


def test_uniqueness_needs_either_a_file_or_a_shape(tmp_path: Path) -> None:
    assert main(["uniqueness"]) == 2


def test_uniqueness_rejects_a_label_file_of_the_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("0,4,9\n5,9,14\n", encoding="utf-8")
    assert main(["uniqueness", "--labels", str(path)]) == 2
