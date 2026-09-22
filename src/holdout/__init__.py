"""Statistics for judging whether a backtest is evidence of anything."""

from __future__ import annotations

from .deflated import (
    DeflatedSharpe,
    deflate_trials,
    deflated_sharpe_ratio,
    expected_maximum_normal,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)
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
from .trials import effective_number_of_trials, trial_correlation

__version__ = "0.1.0"

__all__ = [
    "DeflatedSharpe",
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
    "deflate_trials",
    "deflated_sharpe_ratio",
    "effective_number_of_trials",
    "estimate_sharpe",
    "expected_maximum_normal",
    "expected_maximum_sharpe",
    "kurtosis",
    "minimum_track_record_length",
    "moments",
    "probabilistic_sharpe_ratio",
    "serial_correlation_factor",
    "sharpe_ratio",
    "sharpe_standard_error",
    "skewness",
    "trial_correlation",
]
