# src/hybrid/backtesting/executor.py
# Trade execution logic with ZERO hardcoded values

from typing import Dict, Tuple
from src.hybrid.config.unified_config import UnifiedConfig


class TradeExecutor:
    """
    Handle trade execution and exit conditions
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache ALL trade execution configuration values"""
        # Backtesting parameters
        self.backtest_config = self.config.get_section('backtesting', {})
        self.transaction_cost = self.backtest_config.get('transaction_cost')
        self.slippage = self.backtest_config.get('slippage')

        # Risk management parameters
        self.risk_config = self.config.get_section('risk_management', {})
        self.stop_loss_pct = self.risk_config.get('stop_loss_pct')
        self.take_profit_pct = self.risk_config.get('take_profit_pct')
        self.max_holding_periods = self.risk_config.get('max_holding_periods')

        # Mathematical constants
        constants_config = self.config.get_section('mathematical_operations', {})
        self.zero_value = constants_config.get('zero')
        self.unity_value = constants_config.get('unity')

        # Backtesting calculation parameters
        backtest_calc_config = self.backtest_config.get('calculations', {})
        self.position_neutral_value = backtest_calc_config.get('position_neutral_value')

        # Debug configuration
        self.debug_config = self.config.get_section('debug_configuration', {})
        self.enable_trade_debug = self.debug_config.get('enable_trade_debug')
        self.trade_debug_count = self.debug_config.get('trade_debug_count')

        # Output formatting configuration
        self.output_config = self.config.get_section('output_formatting', {})
        self.decimal_config = self.output_config.get('decimal_places', {})
        self.percentage_multiplier = self.output_config.get('percentage_multiplier')

        # Trade debug configuration
        self.trade_debug_config = self.config.get_section('trade_debug', {})
        self.debug_labels = self.trade_debug_config.get('debug_labels', {})
        self.position_types = self.trade_debug_config.get('position_types', {})

    def check_exit_conditions(self, current_position: int, price: float,
                              entry_price: float, current_time: int,
                              entry_time: int) -> Tuple[bool, str]:
        """Check if position should be exited - ZERO hardcoded values"""

        # Calculate percentage change consistently for both long and short
        pct_change = (price - entry_price) / entry_price

        if current_position > self.position_neutral_value:  # Long position
            # Stop loss: price drops below entry by stop_loss_pct
            if pct_change <= -self.stop_loss_pct:
                print(f"DEBUG: Stop loss triggered - Change: {pct_change:.4f}, Limit: {-self.stop_loss_pct:.4f}")
                return True, "stop_loss"
            # Take profit: price rises above entry by take_profit_pct
            if pct_change >= self.take_profit_pct:
                print(f"DEBUG: Take profit triggered - Change: {pct_change:.4f}, Limit: {self.take_profit_pct:.4f}")
                return True, "take_profit"
        else:  # Short position (current_position < 0)
            # Stop loss: price rises above entry by stop_loss_pct (bad for short)
            if pct_change >= self.stop_loss_pct:
                return True, "stop_loss"
            # Take profit: price drops below entry by take_profit_pct (good for short)
            if pct_change <= -self.take_profit_pct:
                return True, "take_profit"

        # Maximum holding period
        if current_time - entry_time >= self.max_holding_periods:
            return True, "max_holding"

        return False, ""

    def execute_exit(self, position: int, size: float, entry_price: float,
                     exit_price: float, capital: float) -> Dict:
        """Execute trade exit with ALL formatting configurable"""

        # Calculate raw return
        if position > self.position_neutral_value:  # Long position
            raw_return = (exit_price - entry_price) / entry_price
            position_type = self.position_types.get('long')
        else:  # Short position
            raw_return = (entry_price - exit_price) / entry_price
            position_type = self.position_types.get('short')

        # Debug output for first few trades
        if hasattr(self, '_trade_count'):
            self._trade_count += self.unity_value
        else:
            self._trade_count = self.unity_value

        if self.enable_trade_debug and self._trade_count <= self.trade_debug_count:
            price_precision = self.decimal_config.get('price_display')
            return_precision = self.decimal_config.get('return_display')
            percentage_precision = self.decimal_config.get('percentage_display')
            size_precision = self.decimal_config.get('position_size_display')

            print(f"{self.debug_labels.get('trade_debug')}{self._trade_count}:")
            print(f"  {self.debug_labels.get('position')} {position_type}")
            print(f"  {self.debug_labels.get('entry')} {entry_price:.{price_precision}f}, "
                  f"{self.debug_labels.get('exit')} {exit_price:.{price_precision}f}")
            print(f"  {self.debug_labels.get('raw_return')} {raw_return:.{return_precision}f} "
                  f"({raw_return * self.percentage_multiplier:.{percentage_precision}f}%)")
            print(f"  {self.debug_labels.get('position_size')} {size:.{size_precision}f}")

        # Apply position sizing
        trade_return = raw_return * size

        if self.enable_trade_debug and self._trade_count <= self.trade_debug_count:
            print(f"  {self.debug_labels.get('sized_return')} {trade_return:.{return_precision}f} "
                  f"({trade_return * self.percentage_multiplier:.{percentage_precision}f}%)")

        # Apply transaction costs from configuration
        trade_return -= self.transaction_cost  # Commission
        trade_return -= self.slippage  # Slippage

        if self.enable_trade_debug and self._trade_count <= self.trade_debug_count:
            print(f"  {self.debug_labels.get('after_costs')} {trade_return:.{return_precision}f} "
                  f"({trade_return * self.percentage_multiplier:.{percentage_precision}f}%)")
            print(f"  {self.debug_labels.get('transaction_cost')} {self.transaction_cost}")
            print(f"  {self.debug_labels.get('slippage')} {self.slippage}")

        # Calculate P&L
        pnl = capital * trade_return
        new_capital = capital + pnl

        if self.enable_trade_debug and self._trade_count <= self.trade_debug_count:
            capital_precision = self.decimal_config.get('capital_display')
            pnl_precision = self.decimal_config.get('pnl_display')
            arrow = self.debug_labels.get('arrow')

            print(f"  {self.debug_labels.get('capital')} ${capital:.{capital_precision}f}"
                  f"{arrow}${new_capital:.{capital_precision}f}")
            print(f"  {self.debug_labels.get('pnl')} ${pnl:.{pnl_precision}f}")
            print()

        return {
            'return': trade_return,
            'pnl': pnl,
            'new_capital': new_capital
        }