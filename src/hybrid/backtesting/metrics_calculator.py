# src/hybrid/backtesting/metrics_calculator.py
"""
MetricsCalculator - Calculate performance metrics from TradeHistory

Refactored to work with TradeHistory instead of raw trade dicts.
Produces structured PerformanceMetrics objects.
"""

import numpy as np
import logging
from typing import List, Optional, Dict
from datetime import datetime

from src.hybrid.backtesting.performance_metrics import PerformanceMetrics
from src.hybrid.positions.base_trade_history import BaseTradeHistory

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics from trade history

    Takes TradeHistory as input and produces PerformanceMetrics object.
    All calculation parameters are configuration-driven.
    """

    def __init__(self, config: Dict):
        """
        Initialize calculator with configuration

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache configuration values for calculations"""
        backtest_config = self.config.config.get('backtesting')

        # Risk-free rate for Sharpe calculation
        self.risk_free_rate = backtest_config.get('risk_free_rate')

        # Days per year for annualization
        self.days_per_year = backtest_config.get('days_per_year')

        # Minimum samples for statistical calculations
        calc_config = backtest_config.get('calculations')
        self.min_samples = calc_config.get('min_performance_samples')

    def calculate_metrics(
            self,
            trade_history: BaseTradeHistory,
            equity_curve: Optional[List[float]] ,
            initial_capital: float
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            trade_history: TradeHistory instance with trades
            equity_curve: Optional equity curve for drawdown calculation
            initial_capital: Starting capital

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        # Get trade statistics from TradeHistory
        stats = trade_history.get_trade_statistics(lookback_periods=0)  # All trades

        if stats.total_positions == 0:
            logger.warning("No trades to calculate metrics from")
            return PerformanceMetrics()

        # Get outcomes for detailed analysis
        outcomes = stats.outcomes

        # Calculate basic trade metrics
        total_trades = stats.total_positions
        winning_trades = stats.winning_positions
        losing_trades = stats.losing_positions
        break_even_trades = stats.break_even_positions

        # P&L metrics
        total_pnl = stats.total_pnl
        total_fees = stats.total_fees
        net_pnl = total_pnl

        # Calculate gross P&L (before fees)
        gross_pnl = sum(o.gross_pnl for o in outcomes)

        # Return metrics
        total_return = net_pnl / initial_capital if initial_capital > 0 else 0.0

        # Win/Loss analysis
        wins = [o for o in outcomes if o.outcome == 'win']
        losses = [o for o in outcomes if o.outcome == 'loss']

        avg_win = np.mean([w.net_pnl for w in wins]) if wins else 0.0
        avg_loss = np.mean([l.net_pnl for l in losses]) if losses else 0.0
        avg_return_per_trade = net_pnl / total_trades if total_trades > 0 else 0.0

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum([o.net_pnl for o in outcomes if o.net_pnl > 0])
        gross_loss = abs(sum([o.net_pnl for o in outcomes if o.net_pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate returns for risk metrics
        trade_returns = self._calculate_trade_returns(outcomes, initial_capital)

        # Risk-adjusted metrics
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        if len(trade_returns) >= self.min_samples:
            sharpe_ratio = self._calculate_sharpe_ratio(trade_returns)
            sortino_ratio = self._calculate_sortino_ratio(trade_returns)

        # Drawdown metrics
        max_drawdown = 0.0
        max_dd_duration = None
        avg_drawdown = 0.0

        if equity_curve and len(equity_curve) > 0:
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            # TODO: Calculate drawdown duration and average

        # Calmar ratio (return / max drawdown)
        calmar_ratio = 0.0
        if max_drawdown > 0:
            calmar_ratio = total_return / max_drawdown

        # Streak analysis
        max_win_streak, max_loss_streak, current_streak, current_type = self._calculate_streaks(outcomes)

        # Holding period analysis
        holding_periods = self._calculate_holding_periods(trade_history)
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        min_holding_period = int(np.min(holding_periods)) if holding_periods else None
        max_holding_period = int(np.max(holding_periods)) if holding_periods else None

        # Expectancy (average $ per trade)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            break_even_trades=break_even_trades,
            total_pnl=total_pnl,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_fees=total_fees,
            total_return=total_return,
            avg_return_per_trade=avg_return_per_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_drawdown,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            current_streak=current_streak,
            current_streak_type=current_type,
            avg_holding_period=avg_holding_period,
            min_holding_period=min_holding_period,
            max_holding_period=max_holding_period,
            expectancy=expectancy
        )

    def _calculate_trade_returns(self, outcomes: List, initial_capital: float) -> List[float]:
        """
        Calculate percentage returns for each trade

        Args:
            outcomes: List of PositionOutcome objects
            initial_capital: Starting capital

        Returns:
            List of percentage returns
        """
        returns = []
        capital = initial_capital

        for outcome in outcomes:
            if capital > 0:
                trade_return = outcome.net_pnl / capital
                returns.append(trade_return)
                capital += outcome.net_pnl

        return returns

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: List of trade returns

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < self.min_samples:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (self.risk_free_rate / self.days_per_year)

        if np.std(returns_array) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(returns_array)
        # Annualize
        sharpe_annualized = sharpe * np.sqrt(self.days_per_year)

        return float(sharpe_annualized)

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sortino ratio (penalizes downside volatility only)

        Args:
            returns: List of trade returns

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < self.min_samples:
            return 0.0

        returns_array = np.array(returns)
        downside_returns = returns_array[returns_array < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return float('inf')

        sortino = np.mean(returns_array) / downside_std
        # Annualize
        sortino_annualized = sortino * np.sqrt(self.days_per_year)

        return float(sortino_annualized)

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown from equity curve

        Args:
            equity_curve: List of equity values over time

        Returns:
            Maximum drawdown as decimal (e.g., 0.15 for 15%)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value

            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)

        return float(max_dd)

    def _calculate_streaks(self, outcomes: List) -> tuple:
        """
        Calculate win/loss streaks

        Args:
            outcomes: List of PositionOutcome objects

        Returns:
            Tuple of (max_win_streak, max_loss_streak, current_streak, current_type)
        """
        if not outcomes:
            return 0, 0, 0, None

        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for outcome in outcomes:
            if outcome.outcome == 'win':
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif outcome.outcome == 'loss':
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:  # break_even
                current_win_streak = 0
                current_loss_streak = 0

        # Determine current streak
        current_streak = current_win_streak if current_win_streak > 0 else current_loss_streak
        current_type = 'win' if current_win_streak > 0 else ('loss' if current_loss_streak > 0 else None)

        return max_win_streak, max_loss_streak, current_streak, current_type

    def _calculate_holding_periods(self, trade_history: BaseTradeHistory) -> List[int]:
        """
        Calculate holding periods for all trades

        Args:
            trade_history: TradeHistory instance

        Returns:
            List of holding periods in bars/periods
        """
        holding_periods = []

        for trade in trade_history.trades.values():
            if trade.get('status') == 'closed':
                entry_date = trade.get('entry_date')
                exit_date = trade.get('exit_date')

                if entry_date and exit_date:
                    # Parse if strings
                    if isinstance(entry_date, str):
                        entry_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
                    if isinstance(exit_date, str):
                        exit_date = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))

                    # Calculate difference (approximate bars - assumes 1 day = 1 bar)
                    holding_period = (exit_date - entry_date).days
                    if holding_period >= 0:
                        holding_periods.append(holding_period)

        return holding_periods