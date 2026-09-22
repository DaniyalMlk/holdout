from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from holdout import InsufficientDataError, ValidationError
from holdout.io import ReturnTable, read_returns_csv, write_returns_csv


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "r.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_round_trip_is_lossless(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    table = ReturnTable(
        names=["a", "b", "c"],
        values=rng.normal(size=(50, 3)) * 0.01,
        index=[f"t{i}" for i in range(50)],
    )
    path = tmp_path / "out.csv"
    write_returns_csv(path, table, header="date")
    back = read_returns_csv(path)
    assert back.names == table.names
    assert back.index == table.index
    np.testing.assert_array_equal(back.values, table.values)
    assert path.read_text().splitlines()[0] == "date,a,b,c"


def test_without_an_index_column(tmp_path: Path) -> None:
    p = write(tmp_path, "x,y\n0.1,0.2\n0.3,0.4\n\n")
    t = read_returns_csv(p, index=False)
    assert t.names == ["x", "y"]
    assert t.index == ["0", "1"]
    np.testing.assert_array_equal(t.column("y"), [0.2, 0.4])


def test_errors_name_the_line_and_column(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match=r"r.csv:3: column 'b' holds 'oops'"):
        read_returns_csv(write(tmp_path, "d,a,b\n1,0.1,0.2\n2,0.3,oops\n"))
    with pytest.raises(ValidationError, match=r"r.csv:2: column 'a' is 'nan'"):
        read_returns_csv(write(tmp_path, "d,a\n1,nan\n2,0.1\n"))
    with pytest.raises(ValidationError, match=r"r.csv:3: expected 3 fields, found 2"):
        read_returns_csv(write(tmp_path, "d,a,b\n1,0.1,0.2\n2,0.3\n"))
    with pytest.raises(ValidationError, match="unique"):
        read_returns_csv(write(tmp_path, "d,a,a\n1,0.1,0.2\n2,0.3,0.1\n"))
    with pytest.raises(ValidationError, match="no strategy columns"):
        read_returns_csv(write(tmp_path, "d\n1\n2\n"))
    with pytest.raises(InsufficientDataError, match="empty"):
        read_returns_csv(write(tmp_path, "\n\n"))
    with pytest.raises(InsufficientDataError, match="1 data row"):
        read_returns_csv(write(tmp_path, "d,a\n1,0.1\n"))
    with pytest.raises(ValidationError, match="cannot open"):
        read_returns_csv(tmp_path / "missing.csv")


def test_column_lookup_and_removal() -> None:
    t = ReturnTable(names=["a", "b"], values=np.array([[1.0, 2.0], [3.0, 4.0]]), index=["0", "1"])
    with pytest.raises(ValidationError, match="no column named 'z'"):
        t.column("z")
    rest = t.without("a")
    assert rest.names == ["b"]
    np.testing.assert_array_equal(rest.values, [[2.0], [4.0]])


def test_write_checks_the_shape(tmp_path: Path) -> None:
    bad = ReturnTable(names=["a"], values=np.zeros((3, 2)), index=["0", "1", "2"])
    with pytest.raises(ValidationError, match="shape"):
        write_returns_csv(tmp_path / "x.csv", bad)
