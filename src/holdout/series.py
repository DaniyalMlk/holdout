"""Validation of return series and strategy matrices.

Every public function funnels its inputs through here. The point is that a bad
input fails loudly and says where it is: a single ``NaN`` in a return series
otherwise propagates silently into a Sharpe ratio of ``nan``, and a ``nan``
compared against a threshold is simply ``False`` — which reads as "not
significant" rather than "not computed".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import InsufficientDataError, ValidationError

__all__ = [
    "CONSTANT_TOLERANCE",
    "FloatArray",
    "as_matrix",
    "as_returns",
    "check_labels",
    "check_periods_per_year",
    "check_probability",
    "constant_floor",
]

FloatArray = NDArray[np.float64]

#: Multiple of machine epsilon below which a series is treated as constant. The
#: test has to be *relative*, because the standard deviation of a constant array
#: is not exactly zero: subtracting the mean leaves rounding of order
#: ``eps * level``. On 0.001 repeated a thousand times numpy returns 4.3e-19,
#: and an absolute guard at zero does not fire — the Sharpe ratio comes back as
#: 2.3e15 and every number downstream is arithmetic on rounding error.
#:
#: Eight rather than one, because one is not enough. Over constant series at
#: ten lengths and nine magnitudes the residue reaches 1.95 times ``eps *
#: level``, so a guard at one multiple fires for some ``(length, value)`` pairs
#: and not for others — which is why this was caught in one module and missed
#: in five. Eight has margin over that and costs nothing at the other end: a
#: series whose standard deviation is 1e-13 of its own level is still accepted,
#: and no real return series is anywhere near it.
CONSTANT_TOLERANCE = 8.0


def constant_floor(values: FloatArray) -> float:
    """Standard deviation below which ``values`` is rounding rather than variation.

    Relative to the series' own magnitude, so the threshold means the same
    thing whether returns are quoted as fractions or in basis points. The
    fallback of one for an all-zero series keeps the floor positive, so that
    zeros are refused rather than compared against a floor of zero.
    """
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return CONSTANT_TOLERANCE * float(np.finfo(np.float64).eps) * (scale or 1.0)


def _first_bad(values: FloatArray) -> tuple[int, ...]:
    bad = np.argwhere(~np.isfinite(values))
    return tuple(int(i) for i in bad[0])


def as_returns(values: ArrayLike, *, name: str = "returns", min_length: int = 2) -> FloatArray:
    """Return ``values`` as a one-dimensional float array, or raise.

    Raises :class:`ValidationError` for anything that is not a finite,
    one-dimensional sequence of numbers, naming the first offending index, and
    :class:`InsufficientDataError` when there are fewer than ``min_length``
    observations.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a sequence of numbers: {exc}") from None
    if array.ndim == 0:
        raise ValidationError(f"{name} must be a sequence, got a scalar")
    if array.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        (index,) = _first_bad(array)
        raise ValidationError(f"{name}[{index}] is {array[index]!r}; every value must be finite")
    if array.size < min_length:
        raise InsufficientDataError(
            f"{name} has {array.size} observation(s); at least {min_length} are needed"
        )
    return array


def as_matrix(
    values: ArrayLike,
    *,
    name: str = "returns",
    min_rows: int = 2,
    min_columns: int = 1,
) -> FloatArray:
    """Return ``values`` as a finite ``(observations, strategies)`` float array.

    A one-dimensional input is treated as a single strategy.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a matrix of numbers: {exc}") from None
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValidationError(f"{name} must be two-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        row, column = _first_bad(array)
        raise ValidationError(
            f"{name}[{row}, {column}] is {array[row, column]!r}; every value must be finite"
        )
    rows, columns = array.shape
    if rows < min_rows:
        raise InsufficientDataError(
            f"{name} has {rows} observation(s); at least {min_rows} are needed"
        )
    if columns < min_columns:
        raise InsufficientDataError(
            f"{name} has {columns} strategy column(s); at least {min_columns} are needed"
        )
    return array


def check_periods_per_year(periods_per_year: float | None) -> float | None:
    """Validate an annualisation frequency; ``None`` means "do not annualise"."""
    if periods_per_year is None:
        return None
    value = float(periods_per_year)
    if not np.isfinite(value) or value <= 0:
        raise ValidationError(f"periods_per_year must be positive and finite, got {value!r}")
    return value


def check_probability(value: float, name: str, *, open_interval: bool = True) -> float:
    """Validate a probability, by default strictly between zero and one."""
    p = float(value)
    ok = 0.0 < p < 1.0 if open_interval else 0.0 <= p <= 1.0
    if not ok:
        bounds = "strictly between 0 and 1" if open_interval else "between 0 and 1"
        raise ValidationError(f"{name} must be {bounds}, got {p!r}")
    return p


def check_labels(labels: Sequence[str] | None, count: int, name: str = "names") -> list[str]:
    """Return ``labels`` if it has ``count`` unique entries, else default labels."""
    if labels is None:
        return [f"s{i}" for i in range(count)]
    out = [str(label) for label in labels]
    if len(out) != count:
        raise ValidationError(f"{name} has {len(out)} entries for {count} columns")
    if len(set(out)) != len(out):
        raise ValidationError(f"{name} must be unique")
    return out
