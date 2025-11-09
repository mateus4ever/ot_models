# src/hybrid/results/performance_metrics.py
"""
PerformanceMetrics - Structured container for backtest performance metrics

Provides type-safe storage and serialization of calculated metrics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class PerformanceMetrics:
    """
    Structured performance metrics container

    Contains all calculated performance statistics from a backtest.
    All fields are optional to support partial metric calculation.
    """

    # === TRADE STATISTICS ===
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0

    # === P&L METRICS ===
    total_pnl: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_fees: float = 0.0

    # === RETURN METRICS ===
    total_return: float = 0.0  # Percentage return
    avg_return_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # === WIN/LOSS RATIOS ===
    win_rate: float = 0.0  # Percentage
    profit_factor: float = 0.0  # Gross profit / Gross loss

    # === RISK-ADJUSTED METRICS ===
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # === DRAWDOWN METRICS ===
    max_drawdown: float = 0.0  # Maximum peak-to-trough decline
    max_drawdown_duration: Optional[int] = None  # Bars in drawdown
    avg_drawdown: float = 0.0

    # === STREAK METRICS ===
    max_win_streak: int = 0
    max_loss_streak: int = 0
    current_streak: int = 0
    current_streak_type: Optional[str] = None  # 'win' or 'loss'

    # === HOLDING PERIOD ===
    avg_holding_period: float = 0.0  # Average bars held
    min_holding_period: Optional[int] = None
    max_holding_period: Optional[int] = None

    # === EXPOSURE ===
    market_exposure: float = 0.0  # Percentage of time in market

    # === EXPECTANCY ===
    expectancy: float = 0.0  # Expected value per trade

    def to_dict(self) -> Dict:
        """
        Convert metrics to dictionary

        Returns:
            Dictionary representation of all metrics
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PerformanceMetrics':
        """
        Create PerformanceMetrics from dictionary

        Args:
            data: Dictionary with metric values

        Returns:
            PerformanceMetrics instance
        """
        # Filter out keys that aren't in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def get_summary_dict(self) -> Dict:
        """
        Get summary of key metrics

        Returns:
            Dictionary with most important metrics only
        """
        return {
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor
        }

    def __str__(self) -> str:
        """String representation of key metrics"""
        return (
            f"PerformanceMetrics("
            f"trades={self.total_trades}, "
            f"win_rate={self.win_rate:.2%}, "
            f"return={self.total_return:.2%}, "
            f"sharpe={self.sharpe_ratio:.2f}, "
            f"max_dd={self.max_drawdown:.2%})"
        )