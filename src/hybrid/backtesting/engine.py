# src/hybrid/backtesting/engine.py
# Core backtesting loop with ZERO hardcoded values

import pandas as pd
from typing import Dict, List
from src.hybrid.config.unified_config import UnifiedConfig
from .executor import TradeExecutor
from .metrics import MetricsCalculator


class BacktestEngine:
    """
    Core backtesting engine
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

        # Initialize components
        self.trade_executor = TradeExecutor(config)
        self.metrics_calculator = MetricsCalculator(config)

    def _cache_config_values(self):
        """Cache ALL backtesting configuration values"""
        # Main config sections
        self.backtest_config = self.config.get_section('backtesting', {})
        self.risk_config = self.config.get_section('risk_management', {})
        self.general_config = self.config.get_section('general', {})

        # Backtesting parameters
        self.initial_capital = self.backtest_config.get('initial_capital')
        self.verbose = self.general_config.get('verbose')

        # Mathematical constants
        constants_config = self.config.get_section('mathematical_operations', {})
        self.zero_value = constants_config.get('zero')
        self.unity_value = constants_config.get('unity')

        # Backtesting calculation parameters
        backtest_calc_config = self.backtest_config.get('calculations', {})
        self.loop_start_index = backtest_calc_config.get('loop_start_index')
        self.position_long_value = backtest_calc_config.get('position_long_value')
        self.position_short_value = backtest_calc_config.get('position_short_value')
        self.position_neutral_value = backtest_calc_config.get('position_neutral_value')
        self.signal_threshold = backtest_calc_config.get('signal_threshold')
        self.size_threshold = backtest_calc_config.get('size_threshold')

        # Risk management parameters
        self.max_daily_trades = self.risk_config.get('max_daily_trades')
        self.max_daily_loss_pct = self.risk_config.get('max_daily_loss_pct')
        self.max_drawdown_pct = self.risk_config.get('max_drawdown_pct')

    def run_backtest(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> Dict:
        """Run comprehensive backtest with ALL parameters configurable"""

        if self.verbose:
            status_config = self.config.get_section('display_configuration', {}).get('status_messages', {})
            print(status_config.get('backtest_running'))

        # Initialize backtest variables from configuration
        capital = self.initial_capital
        positions = []
        trades = []
        daily_pnl = []
        equity_curve = [capital]

        # Trade tracking using configurable values
        current_position = self.position_neutral_value
        current_size = self.zero_value
        entry_price = self.zero_value
        entry_time = self.zero_value
        daily_trades = self.zero_value
        last_date = None
        daily_loss = self.zero_value

        # Risk management
        peak_capital = capital
        current_drawdown = self.zero_value

        for i in range(self.loop_start_index, len(df)):
            # Get signals with configurable column checks
            signal = signals_df['signal'].iloc[i] if 'signal' in signals_df.columns else self.zero_value
            size = signals_df['position_size'].iloc[i] if 'position_size' in signals_df.columns else self.zero_value

            price = df['close'].iloc[i]
            timestamp = df.index[i]
            current_date = timestamp.date() if hasattr(timestamp, 'date') else None

            # Reset daily counters
            if current_date != last_date:
                daily_trades = self.zero_value
                daily_loss = self.zero_value
                last_date = current_date

            # Risk management checks
            if self._check_risk_limits(capital, peak_capital, daily_loss, daily_trades):
                continue

            # Exit logic
            if current_position != self.position_neutral_value:
                should_exit, exit_reason = self.trade_executor.check_exit_conditions(
                    current_position, price, entry_price, i, entry_time
                )

                if should_exit:
                    trade_result = self.trade_executor.execute_exit(
                        current_position, current_size, entry_price, price, capital
                    )

                    capital = trade_result['new_capital']
                    daily_loss += max(self.zero_value, -trade_result['pnl'])  # Track daily losses
                    daily_pnl.append(trade_result['pnl'])

                    trades.append({
                        'entry_time': df.index[entry_time],
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': current_position,
                        'size': current_size,
                        'pnl': trade_result['pnl'],
                        'return': trade_result['return'],
                        'holding_period': i - entry_time,
                        'exit_reason': exit_reason
                    })

                    current_position = self.position_neutral_value
                    current_size = self.zero_value
                    daily_trades += self.unity_value

            # Entry logic with configurable thresholds
            if (current_position == self.position_neutral_value and
                    signal != self.signal_threshold and
                    size > self.size_threshold and
                    not self._check_risk_limits(capital, peak_capital, daily_loss, daily_trades)):
                # DEBUG: Check if position sizes changed

                current_position = self.position_long_value if signal > self.zero_value else self.position_short_value
                current_size = min(size, self.risk_config.get('max_position_size'))
                entry_price = price
                entry_time = i
                daily_trades += self.unity_value

            # Update equity curve and drawdown
            positions.append(current_position)
            equity_curve.append(capital)

            if capital > peak_capital:
                peak_capital = capital

            current_drawdown = (peak_capital - capital) / peak_capital

        # Calculate performance metrics
        performance_metrics = self.metrics_calculator.calculate_performance_metrics(
            capital, trades, equity_curve, daily_pnl, self.initial_capital,
            self.trade_executor.transaction_cost, self.trade_executor.slippage
        )

        # Add additional analysis
        performance_metrics.update({
            'trades_detail': trades,
            'equity_curve': equity_curve,
            'positions': positions,
            'daily_pnl': daily_pnl
        })

        return performance_metrics

    def _check_risk_limits(self, capital: float, peak_capital: float,
                           daily_loss: float, daily_trades: int) -> bool:
        """Check if risk limits are breached using configurable thresholds"""

        # Daily trade limit
        if daily_trades >= self.max_daily_trades:
            return True

        # Daily loss limit
        if daily_loss > capital * self.max_daily_loss_pct:
            return True

        # Maximum drawdown limit
        current_drawdown = (peak_capital - capital) / peak_capital
        if current_drawdown > self.max_drawdown_pct:
            return True

        return False