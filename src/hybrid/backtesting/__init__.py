# src/hybrid/backtesting/__init__.py
# Backtesting package exports - Updated for Walk-Forward Architecture
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

# Core backtesting components
from .metrics import MetricsCalculator
from .results import ResultsFormatter
from .engine import BacktestEngine
from .executor import TradeExecutor

# NEW: Walk-forward backtesting components
from .walk_forward_engine import (
    TemporalDataGuard,
    WalkForwardRetrainingStrategy,
    WalkForwardBacktester,
    WalkForwardResultsFormatter
)

__all__ = [
    # Original components
    'MetricsCalculator',
    'ResultsFormatter',
    'BacktestEngine',
    'TradeExecutor',

    # NEW: Walk-forward components
    'TemporalDataGuard',
    'WalkForwardRetraini'
    'ngStrategy',
    'WalkForwardBacktester',
    'WalkForwardResultsFormatter'
]