"""Statistics for judging whether a backtest is evidence of anything."""

from __future__ import annotations

from .bootstrap import (
    ColumnMeans,
    bootstrap_column_means,
    optimal_block_length,
    stationary_bootstrap_indices,
)
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
from .mcs import Elimination, ModelConfidenceSet, Statistic, model_confidence_set
from .moments import Moments, autocorrelation, kurtosis, moments, skewness
from .multiple import (
    Haircut,
    adjust_pvalues,
    haircut_sharpe,
    haircut_sharpe_ratios,
    minimum_sharpe,
    minimum_t_statistic,
)
from .pbo import PBOResult, cscv_partitions, probability_of_backtest_overfitting
from .series import as_matrix, as_returns
from .sharpe import (
    SharpeRatio,
    autocorrelation_adjusted_sharpe,
    estimate_sharpe,
    serial_correlation_factor,
    sharpe_ratio,
    sharpe_standard_error,
)
from .spa import (
    RealityCheck,
    RomanoWolf,
    SPATest,
    reality_check,
    romano_wolf,
    superior_predictive_ability,
)
from .splits import (
    CombinatorialPurgedCV,
    LeakageError,
    Split,
    combinatorial_purged_cv,
    kfold,
    leakage_audit,
    number_of_paths,
    purged_kfold,
    walk_forward,
)
from .trials import effective_number_of_trials, trial_correlation

__version__ = "0.1.0"

__all__ = [
    "ColumnMeans",
    "CombinatorialPurgedCV",
    "DeflatedSharpe",
    "Elimination",
    "Haircut",
    "HoldoutError",
    "InsufficientDataError",
    "LeakageError",
    "ModelConfidenceSet",
    "Moments",
    "PBOResult",
    "RealityCheck",
    "RomanoWolf",
    "SPATest",
    "SharpeRatio",
    "Split",
    "Statistic",
    "ValidationError",
    "__version__",
    "adjust_pvalues",
    "as_matrix",
    "as_returns",
    "autocorrelation",
    "autocorrelation_adjusted_sharpe",
    "bootstrap_column_means",
    "combinatorial_purged_cv",
    "cscv_partitions",
    "deflate_trials",
    "deflated_sharpe_ratio",
    "effective_number_of_trials",
    "estimate_sharpe",
    "expected_maximum_normal",
    "expected_maximum_sharpe",
    "haircut_sharpe",
    "haircut_sharpe_ratios",
    "kfold",
    "kurtosis",
    "leakage_audit",
    "minimum_sharpe",
    "minimum_t_statistic",
    "minimum_track_record_length",
    "model_confidence_set",
    "moments",
    "number_of_paths",
    "optimal_block_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_kfold",
    "reality_check",
    "romano_wolf",
    "serial_correlation_factor",
    "sharpe_ratio",
    "sharpe_standard_error",
    "skewness",
    "stationary_bootstrap_indices",
    "superior_predictive_ability",
    "trial_correlation",
    "walk_forward",
]
