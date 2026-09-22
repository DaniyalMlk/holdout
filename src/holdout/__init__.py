"""Statistics for judging whether a backtest is evidence of anything."""

from __future__ import annotations

from .exceptions import HoldoutError, InsufficientDataError, ValidationError
from .series import as_matrix, as_returns

__version__ = "0.1.0"

__all__ = [
    "HoldoutError",
    "InsufficientDataError",
    "ValidationError",
    "__version__",
    "as_matrix",
    "as_returns",
]
