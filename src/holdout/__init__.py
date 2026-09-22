"""Statistics for judging whether a backtest is evidence of anything."""

from __future__ import annotations

from .exceptions import HoldoutError, InsufficientDataError, ValidationError
from .moments import Moments, autocorrelation, kurtosis, moments, skewness
from .series import as_matrix, as_returns

__version__ = "0.1.0"

__all__ = [
    "HoldoutError",
    "InsufficientDataError",
    "Moments",
    "ValidationError",
    "__version__",
    "as_matrix",
    "as_returns",
    "autocorrelation",
    "kurtosis",
    "moments",
    "skewness",
]
