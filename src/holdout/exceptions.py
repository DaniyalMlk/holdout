"""Exception hierarchy.

Every error raised by the library derives from :class:`HoldoutError`, so a
caller can separate "the statistics cannot be computed from these inputs" from
a bug in their own code with a single ``except`` clause.
"""

from __future__ import annotations

__all__ = [
    "HoldoutError",
    "InsufficientDataError",
    "ValidationError",
]


class HoldoutError(Exception):
    """Base class for every error raised by this library."""


class ValidationError(HoldoutError, ValueError):
    """An input cannot describe what the function was asked to measure."""


class InsufficientDataError(HoldoutError, ValueError):
    """Too few observations for the requested statistic to be defined."""
