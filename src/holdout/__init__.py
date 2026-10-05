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
from .pairwise import (
    SharpeDifference,
    VarianceMethod,
    newey_west_bandwidth,
    newey_west_covariance,
    sharpe_difference,
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
from .stability import (
    DEFAULT_TRIM,
    BreakStatistics,
    SharpeBreak,
    break_statistics,
    sharpe_break,
)
from .trials import effective_number_of_trials, trial_correlation
from .uniqueness import (
    Concurrency,
    SequentialDraw,
    average_uniqueness,
    concurrency,
    drawn_uniqueness,
    effective_sample_size,
    sequential_bootstrap,
    time_decay_weights,
    uniqueness_weights,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TRIM",
    "BreakStatistics",
    "ColumnMeans",
    "CombinatorialPurgedCV",
    "Concurrency",
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
    "SequentialDraw",
    "SharpeBreak",
    "SharpeDifference",
    "SharpeRatio",
    "Split",
    "Statistic",
    "ValidationError",
    "VarianceMethod",
    "__version__",
    "adjust_pvalues",
    "as_matrix",
    "as_returns",
    "autocorrelation",
    "autocorrelation_adjusted_sharpe",
    "average_uniqueness",
    "bootstrap_column_means",
    "break_statistics",
    "combinatorial_purged_cv",
    "concurrency",
    "cscv_partitions",
    "deflate_trials",
    "deflated_sharpe_ratio",
    "drawn_uniqueness",
    "effective_number_of_trials",
    "effective_sample_size",
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
    "newey_west_bandwidth",
    "newey_west_covariance",
    "number_of_paths",
    "optimal_block_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_kfold",
    "reality_check",
    "romano_wolf",
    "sequential_bootstrap",
    "serial_correlation_factor",
    "sharpe_break",
    "sharpe_difference",
    "sharpe_ratio",
    "sharpe_standard_error",
    "skewness",
    "stationary_bootstrap_indices",
    "superior_predictive_ability",
    "time_decay_weights",
    "trial_correlation",
    "uniqueness_weights",
    "walk_forward",
]
