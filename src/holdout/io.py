"""CSV input and output for return series and strategy matrices.

The expected layout is one row per period and one column per strategy, with
a header row of strategy names and, by default, a first column of period
labels (dates or anything else — they are carried through, not parsed):

.. code-block:: text

    date,fast5_slow20,fast5_slow50,fast10_slow50
    2020-01-02,0.0012,-0.0004,0.0007
    2020-01-03,-0.0021,0.0009,0.0003

Parsing uses only the standard library so that the runtime dependency stays
NumPy alone. Every error names the file, the line and the column.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .exceptions import InsufficientDataError, ValidationError
from .series import FloatArray, check_labels

__all__ = ["ReturnTable", "read_returns_csv", "write_returns_csv"]


@dataclass(frozen=True)
class ReturnTable:
    """A ``(periods, strategies)`` matrix with its column names and period labels."""

    names: list[str]
    values: FloatArray
    index: list[str]

    def column(self, name: str) -> FloatArray:
        """The returns of one strategy by name."""
        try:
            j = self.names.index(name)
        except ValueError:
            known = ", ".join(self.names[:8]) + (", ..." if len(self.names) > 8 else "")
            raise ValidationError(f"no column named {name!r}; columns are {known}") from None
        result: FloatArray = self.values[:, j]
        return result

    def without(self, name: str) -> ReturnTable:
        """The table with one column removed."""
        self.column(name)
        keep = [j for j, n in enumerate(self.names) if n != name]
        return ReturnTable(
            names=[self.names[j] for j in keep], values=self.values[:, keep], index=self.index
        )


def read_returns_csv(path: str | Path, *, index: bool = True) -> ReturnTable:
    """Read a returns CSV; ``index=False`` if there is no leading label column."""
    p = Path(path)
    try:
        handle = p.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot open {p}: {exc.strerror}") from None
    with handle:
        rows = list(csv.reader(handle))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        raise InsufficientDataError(f"{p} is empty")
    header = [cell.strip() for cell in rows[0]]
    names = header[1:] if index else header
    if not names:
        raise ValidationError(f"{p}: the header has no strategy columns")
    names = check_labels(names, len(names), name=f"{p} header")
    labels: list[str] = []
    values = np.empty((len(rows) - 1, len(names)))
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValidationError(f"{p}:{line}: expected {len(header)} fields, found {len(row)}")
        cells = row[1:] if index else row
        if index:
            labels.append(row[0].strip())
        else:
            labels.append(str(line - 2))
        for j, cell in enumerate(cells):
            try:
                v = float(cell)
            except ValueError:
                raise ValidationError(
                    f"{p}:{line}: column {names[j]!r} holds {cell!r}, which is not a number"
                ) from None
            if not math.isfinite(v):
                raise ValidationError(f"{p}:{line}: column {names[j]!r} is {cell.strip()!r}")
            values[line - 2, j] = v
    if values.shape[0] < 2:
        raise InsufficientDataError(f"{p} has {values.shape[0]} data row(s); at least 2 needed")
    return ReturnTable(names=names, values=values, index=labels)


def write_returns_csv(path: str | Path, table: ReturnTable, *, header: str = "period") -> None:
    """Write ``table`` in the layout :func:`read_returns_csv` reads, losslessly."""
    p = Path(path)
    if table.values.shape != (len(table.index), len(table.names)):
        raise ValidationError(
            f"values have shape {table.values.shape} for {len(table.index)} labels "
            f"and {len(table.names)} names"
        )
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([header, *table.names])
        for label, row in zip(table.index, table.values, strict=True):
            writer.writerow([label, *(repr(float(v)) for v in row)])
