"""Statistics for judging whether a backtest is evidence of anything."""

from __future__ import annotations

from .exceptions import HoldoutError, InsufficientDataError, ValidationError
from .moments import Moments, autocorrelation, kurtosis, moments, skewness
from .series import as_matrix, as_returns
from .sharpe import (
    SharpeRatio,
    autocorrelation_adjusted_sharpe,
    estimate_sharpe,
    serial_correlation_factor,
    sharpe_ratio,
    sharpe_standard_error,
)

__version__ = "0.1.0"

__all__ = [
    "HoldoutError",
    "InsufficientDataError",
    "Moments",
    "SharpeRatio",
    "ValidationError",
    "__version__",
    "as_matrix",
    "as_returns",
    "autocorrelation",
    "autocorrelation_adjusted_sharpe",
    "estimate_sharpe",
    "kurtosis",
    "moments",
    "serial_correlation_factor",
    "sharpe_ratio",
    "sharpe_standard_error",
    "skewness",
]
