"""A synthetic strategy search, so every statistic can be run without real data.

:func:`crossover_sweep` simulates one price series and backtests a grid of
moving-average crossover rules on it — the textbook parameter sweep. The
prices follow a GARCH(1, 1) process with Student-t shocks, so returns have
fat tails and clustered volatility, and there is no drift or trend to find
unless ``trend`` is set. The resulting strategies are highly correlated with
each other, as the members of a real sweep are, which is exactly the setting
where counting trials naively goes wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bootstrap import Seed, as_generator
from .exceptions import ValidationError
from .io import ReturnTable
from .series import FloatArray

__all__ = ["Sweep", "crossover_sweep", "garch_returns"]


def garch_returns(
    n: int,
    *,
    omega: float = 1e-6,
    alpha: float = 0.08,
    beta: float = 0.9,
    dof: float = 5.0,
    drift: float = 0.0,
    seed: Seed = None,
) -> FloatArray:
    """Daily returns from a GARCH(1, 1) with unit-variance Student-t shocks."""
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1):
        raise ValidationError("GARCH parameters need omega > 0 and alpha + beta < 1")
    if not dof > 2:
        raise ValidationError(f"dof must exceed 2 for a finite variance, got {dof}")
    rng = as_generator(seed)
    shocks = rng.standard_t(dof, size=n) / math.sqrt(dof / (dof - 2.0))
    out = np.empty(n)
    variance = omega / (1.0 - alpha - beta)
    for t in range(n):
        out[t] = math.sqrt(variance) * shocks[t]
        variance = omega + alpha * out[t] ** 2 + beta * variance
    result: FloatArray = out + drift
    return result


def _moving_average(prices: FloatArray, window: int) -> FloatArray:
    c = np.cumsum(np.concatenate([[0.0], prices]))
    ma = np.full(prices.size, np.nan)
    ma[window - 1 :] = (c[window:] - c[:-window]) / window
    return ma


@dataclass(frozen=True)
class Sweep:
    """A backtested parameter grid: strategy returns plus the market they traded."""

    table: ReturnTable
    market: FloatArray
    parameters: list[tuple[int, int]]


def crossover_sweep(
    n_days: int = 2520,
    *,
    fast: tuple[int, ...] = (5, 10, 15, 20, 30),
    slow: tuple[int, ...] = (40, 60, 90, 120, 180, 250),
    trend: float = 0.0,
    seed: Seed = 7,
) -> Sweep:
    """Backtest long/short moving-average crossovers over a simulated market.

    Each strategy holds ``sign(MA_fast - MA_slow)`` of the market, decided on
    yesterday's close. The first ``max(slow)`` days are dropped so that every
    strategy is live over the same window. ``trend`` adds a slow sinusoidal
    drift that crossover rules can in principle exploit; with the default of
    zero there is nothing to find.
    """
    if n_days < 2 * max(slow):
        raise ValidationError(f"n_days must be at least {2 * max(slow)} for these windows")
    rng = as_generator(seed)
    burn = max(slow)
    returns = garch_returns(n_days + burn, seed=rng)
    if trend:
        t = np.arange(returns.size)
        returns = returns + trend * np.sin(2 * math.pi * t / 750.0)
    prices = np.cumsum(returns)  # log price
    names, columns, grid = [], [], []
    for f in fast:
        for s in slow:
            if f >= s:
                continue
            signal = np.sign(_moving_average(prices, f) - _moving_average(prices, s))
            position = np.concatenate([[0.0], signal[:-1]])
            columns.append((position * returns)[burn:])
            names.append(f"ma{f}_{s}")
            grid.append((f, s))
    values = np.column_stack(columns)
    index = [f"d{t:05d}" for t in range(values.shape[0])]
    return Sweep(
        table=ReturnTable(names=names, values=values, index=index),
        market=returns[burn:],
        parameters=grid,
    )
