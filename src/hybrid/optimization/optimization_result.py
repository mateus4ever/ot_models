from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from src.hybrid.optimization import OptimizerType

@dataclass
class OptimizationResult:
    """Complete optimization run result"""

    # Metadata
    job_id: str
    optimizer_type: OptimizerType
    strategy_name: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    status: str  # 'completed', 'failed', 'cancelled'

    # Configuration
    config_snapshot: Dict  # Full config used
    parameter_space: Dict  # What was optimized
    n_combinations: int
    n_workers: int
    execution_mode: str  # 'local' or 'cloud'

    # Results
    best_params: Dict
    best_fitness: float
    best_metrics: Dict  # {total_return, sharpe, trades, etc.}

    all_evaluations: List[EvaluationResult]  # All parameter combinations tried
    valid_evaluations: List[EvaluationResult]  # Only successful ones
    failed_evaluations: List[EvaluationResult]  # Failed attempts

    # Analysis
    fitness_statistics: Dict  # {mean, std, min, max, percentiles}
    parameter_stability: Dict  # From RobustnessAnalyzer
    robust_ranges: Dict  # Recommended parameter ranges
    landscape_type: str  # 'PLATEAU_DOMINATED', 'PEAKY', 'MIXED'
    robustness_score: float

    # Checkpointing
    checkpoint_path: Optional[str]
    resumed_from: Optional[str]

    # Validation (if walk-forward used)
    train_period: Optional[Tuple[datetime, datetime]]
    test_period: Optional[Tuple[datetime, datetime]]
    train_fitness: Optional[float]
    test_fitness: Optional[float]
    degradation: Optional[float]  # (train - test) / train

@dataclass
class EvaluationResult:
    """Single parameter combination evaluation"""

    evaluation_id: int
    params: Dict
    fitness: float

    # Backtest metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    expectancy: float

    # Additional metrics
    avg_win: float
    avg_loss: float
    max_loss_streak: int
    recovery_time: Optional[int]
    market_exposure: float

    # Execution
    execution_time: float
    success: bool
    error_message: Optional[str]

    # For Bayesian
    acquisition_value: Optional[float]