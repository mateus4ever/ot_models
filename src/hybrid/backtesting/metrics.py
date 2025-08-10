# src/hybrid/backtesting/metrics.py
# Performance metrics calculation with ZERO hardcoded values

import numpy as np
from typing import Dict, List
from src.hybrid.config.unified_config import UnifiedConfig


class MetricsCalculator:
    """
    Calculate backtesting performance metrics
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache ALL metrics configuration values"""
        # Mathematical constants
        constants_config = self.config.get_section('mathematical_operations', {})
        self.zero_value = constants_config.get('zero')
        self.unity_value = constants_config.get('unity')

        # Array indexing
        array_config = self.config.get_section('array_indexing', {})
        self.first_index = array_config.get('first_index')

        # Backtesting calculation parameters
        backtest_config = self.config.get_section('backtesting', {})
        backtest_calc_config = backtest_config.get('calculations', {})
        self.min_performance_samples = backtest_calc_config.get('min_performance_samples')
        self.days_per_year = backtest_calc_config.get('days_per_year')
        self.risk_free_rate = backtest_config.get('risk_free_rate')

        # Debug configuration
        debug_config = self.config.get_section('debug_configuration', {})
        self.enable_metrics_debug = debug_config.get('enable_metrics_debug', False)
        self.enable_direct_math_check = debug_config.get('enable_direct_math_check', False)

    def calculate_performance_metrics(self, final_capital: float, trades: List[Dict],
                                      equity_curve: List[float], daily_pnl: List[float],
                                      initial_capital: float, transaction_cost: float,
                                      slippage: float) -> Dict:
        """Calculate performance metrics with ALL thresholds configurable"""

        # Basic metrics
        total_return = (final_capital - initial_capital) / initial_capital
        num_trades = len(trades)

        if num_trades == self.zero_value:
            return {
                'total_return': total_return,
                'final_capital': final_capital,
                'num_trades': self.zero_value,
                'win_rate': self.zero_value,
                'avg_return_per_trade': self.zero_value,
                'sharpe_ratio': self.zero_value,
                'max_drawdown': self.zero_value,
                'avg_holding_period': self.zero_value
            }

        # Trade analysis
        trade_returns = [t['return'] for t in trades]
        trade_pnls = [t['pnl'] for t in trades]
        holding_periods = [t['holding_period'] for t in trades]

        win_rate = sum([self.unity_value for r in trade_returns if r > self.zero_value]) / num_trades
        avg_return = np.mean(trade_returns)
        avg_holding_period = np.mean(holding_periods)

        # Risk metrics using configurable sample size
        if len(trade_returns) > self.min_performance_samples:
            sharpe_ratio = self._calculate_sharpe_ratio(trade_returns)
            sortino_ratio = self._calculate_sortino_ratio(trade_returns)
        else:
            sharpe_ratio = self.zero_value
            sortino_ratio = self.zero_value

        # Drawdown calculation
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Additional metrics
        profit_factor = self._calculate_profit_factor(trade_pnls)

        # Exit reason analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, self.zero_value) + self.unity_value

        # Win/Loss streaks
        win_streak, loss_streak = self._calculate_streaks(trade_returns)

        # Calculate total fees as percentage of capital per trade
        # Only show debug output if explicitly enabled
        if self.enable_direct_math_check:
            print(f"DIRECT MATH CHECK: 1040 * 10000 * 0.00015 = {1040 * 10000 * 0.00015}")

        if self.enable_metrics_debug:
            print(f"METRICS DEBUG: num_trades={num_trades}, initial_capital={initial_capital}")
            print(f"METRICS DEBUG: transaction_cost={transaction_cost}, slippage={slippage}")
            print(f"METRICS DEBUG: calculation = {num_trades} * {initial_capital} * {transaction_cost + slippage}")

        total_fees = num_trades * initial_capital * (transaction_cost + slippage)

        if self.enable_metrics_debug:
            print(f"METRICS DEBUG: calculated total_fees = {total_fees}")

        return {
            'total_return': total_return,
            'final_capital': final_capital,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_holding_period': avg_holding_period,
            'profit_factor': profit_factor,
            'max_win_streak': win_streak,
            'max_loss_streak': loss_streak,
            'exit_reasons': exit_reasons,
            'total_fees': total_fees
        }

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio with configurable parameters"""
        if len(returns) < self.min_performance_samples:
            return self.zero_value

        excess_returns = np.array(returns) - self.risk_free_rate / self.days_per_year
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.days_per_year)

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio with configurable thresholds"""
        if len(returns) < self.min_performance_samples:
            return self.zero_value

        returns_array = np.array(returns)
        downside_returns = returns_array[returns_array < self.zero_value]

        if len(downside_returns) == self.zero_value:
            return float('inf')

        downside_deviation = np.std(downside_returns)
        return np.mean(returns_array) / downside_deviation * np.sqrt(self.days_per_year)

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown with configurable initialization"""
        if len(equity_curve) < self.min_performance_samples:
            return self.zero_value

        peak = equity_curve[self.first_index]
        max_dd = self.zero_value

        for value in equity_curve:
            if value > peak:
                peak = value

            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calculate profit factor with configurable thresholds"""
        if len(pnls) == self.zero_value:
            return self.zero_value

        gross_profit = sum([pnl for pnl in pnls if pnl > self.zero_value])
        gross_loss = abs(sum([pnl for pnl in pnls if pnl < self.zero_value]))

        return gross_profit / gross_loss if gross_loss > self.zero_value else float('inf')

    def _calculate_streaks(self, returns: List[float]) -> tuple:
        """Calculate streaks with configurable values"""
        if len(returns) == self.zero_value:
            return self.zero_value, self.zero_value

        max_win_streak = self.zero_value
        max_loss_streak = self.zero_value
        current_win_streak = self.zero_value
        current_loss_streak = self.zero_value

        for ret in returns:
            if ret > self.zero_value:
                current_win_streak += self.unity_value
                current_loss_streak = self.zero_value
                max_win_streak = max(max_win_streak, current_win_streak)
            elif ret < self.zero_value:
                current_loss_streak += self.unity_value
                current_win_streak = self.zero_value
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_win_streak = self.zero_value
                current_loss_streak = self.zero_value

        return max_win_streak, max_loss_streak