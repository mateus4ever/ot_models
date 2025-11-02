# money_management.py
# MoneyManager service for position sizing, risk management, and portfolio tracking
# Uses Strategy pattern internally for different algorithms
# Provides stable external interface regardless of internal implementation

import logging
from typing import Dict, Any, Optional
import pandas as pd

from .types import PortfolioState, TradingSignal, PositionDirection, Position
# Import strategy implementations
from .position_sizers.fixed_fractional_sizer import FixedFractionalSizer
from .position_sizers.kelly_criterion_sizer import KellyCriterionSizer
from .position_sizers.volatility_based_sizer import VolatilityBasedSizer
from .risk_managers.atr_based_risk_manager import ATRBasedRiskManager
from .risk_managers.volatility_based_risk_manager import VolatilityBasedRiskManager
from .risk_managers.portfolio_heat_risk_manager import PortfolioHeatRiskManager
from ..costs.transaction_costs import SimpleTransactionCostModel

logger = logging.getLogger(__name__)

class MoneyManager:
    """
    MoneyManager service for position sizing, risk management, and portfolio tracking.

    Uses Strategy pattern internally for different algorithms while providing
    a stable external interface.
    """

    def __init__(self, config, cost_model=None):
        """
        Initialize MoneyManager with configuration and cost_model

        Args:
            config: Configuration object
            cost_model: Optional external cost model (for shared instance)
        """
        try:
            logger.debug(f"Constructor called with {type(config)}")
            self.unified_config = config

            # Get money management configuration section
            mm_config = self.unified_config.get_section('money_management')
            if not mm_config:
                raise ValueError("money_management section not found in configuration")

            self.config = mm_config

            self.portfolio = PortfolioState(
                total_equity=self.config['initial_capital'],
                available_cash=self.config['initial_capital'],
                positions={}
            )
            self.portfolio.peak_equity = self.portfolio.total_equity

            # Initialize strategy components
            self.position_sizer = self._create_position_sizer()
            self.risk_manager = self._create_risk_manager()
            self.cost_model = cost_model if cost_model else SimpleTransactionCostModel(config)

            logger.info(f"MoneyManager initialized with {self.position_sizer.get_strategy_name()} "
                        f"position sizing and initial capital ${self.portfolio.total_equity:,.2f}")
        except Exception as e:
            logger.error(f"Exception in constructor: {e}")
            raise

    def _create_position_sizer(self):
        """Factory method to create position sizing strategy"""
        sizing_type = self.config['position_sizing']  # No fallback - let it fail if missing

        position_sizer = {
            'fixed_fractional': FixedFractionalSizer,
            'kelly_criterion': KellyCriterionSizer,
            'volatility_based': VolatilityBasedSizer
        }

        if sizing_type not in position_sizer:
            raise ValueError(f"Unknown position sizing type: {sizing_type}. "
                             f"Available types: {list(position_sizer.keys())}")

        position_sizer_class = position_sizer[sizing_type]
        return position_sizer_class(self.unified_config)

    def _create_risk_manager(self):
        """Factory method to create risk management strategy"""
        risk_type = self.config['risk_management']

        risk_managers = {
            'atr_based': ATRBasedRiskManager,
            'volatility_based': VolatilityBasedRiskManager,
            'portfolio_heat': PortfolioHeatRiskManager
        }

        if risk_type not in risk_managers:
            raise ValueError(
                f"Unknown risk management type: {risk_type}. "
                f"Available types: "
                f"{list(risk_managers.keys())}")

        risk_manager_class = risk_managers[risk_type]
        return risk_manager_class(self.unified_config)

    # =============================================================================
    # PUBLIC INTERFACE - STABLE METHODS
    # =============================================================================

    def calculate_position_size(self, signal: TradingSignal, market_data: pd.DataFrame) -> int:
        """
        Calculate position size for trading signal

        Args:
            signal: Trading signal with entry parameters
            market_data: Market data for risk calculations

        Returns:
            Position size in shares/units
        """
        try:
            # Calculate stop distance from risk manager
            stop_loss_price = self.risk_manager.calculate_stop_loss(signal, market_data)
            stop_distance = abs(signal.entry_price - stop_loss_price)

            # Check if risk should be reduced
            if self.risk_manager.should_reduce_risk(self.portfolio):
                logger.warning("Risk reduction triggered - reducing position size by 50%")
                base_size = self.position_sizer.calculate_size(signal, self.portfolio, stop_distance)
                return int(base_size * 0.5)

            # Normal position sizing
            position_size = self.position_sizer.calculate_size(signal, self.portfolio, stop_distance)

            # Additional safety checks
            position_size = self._apply_safety_constraints(position_size, signal)

            logger.debug(f"Position size calculated for {signal.symbol}: {position_size} shares")
            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size for {signal.symbol}: {e}")
            return 0

    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate stop loss price for position

        Args:
            signal: Trading signal with entry parameters
            market_data: Market data for stop loss calculation

        Returns:
            Stop loss price
        """
        try:
            stop_loss = self.risk_manager.calculate_stop_loss(signal, market_data)
            logger.debug(f"Stop loss calculated for {signal.symbol}: {stop_loss:.4f}")
            return stop_loss

        except Exception as e:
            logger.error(f"Error calculating stop loss for {signal.symbol}: {e}")
            return signal.entry_price  # Fallback to entry price

    def update_position(self, symbol: str, size: int, price: float,
                        direction: PositionDirection) -> bool:
        """
        Update position in portfolio tracking

        Args:
            symbol: Trading symbol
            size: Position size
            price: Execution price
            direction: Position direction

        Returns:
            True if update successful
        """
        try:
            if size == 0:
                # Close position
                if symbol in self.portfolio.positions:
                    del self.portfolio.positions[symbol]
                    logger.info(f"Position closed for {symbol}")
            else:
                # Update or create position
                position = Position(
                    symbol=symbol,
                    direction=direction,
                    size=size,
                    entry_price=price,
                    current_price=price
                )
                self.portfolio.positions[symbol] = position
                logger.info(f"Position updated for {symbol}: {size} shares at ${price:.4f}")

            # Update portfolio equity
            self._update_portfolio_equity()
            return True

        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            return False

    def update_market_prices(self, price_updates: Dict[str, float]):
        """
        Update current market prices for portfolio valuation

        Args:
            price_updates: Dictionary of symbol -> current price
        """
        try:
            for symbol, price in price_updates.items():
                if symbol in self.portfolio.positions:
                    self.portfolio.positions[symbol].current_price = price

            self._update_portfolio_equity()

        except Exception as e:
            logger.error(f"Error updating market prices: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary

        Returns:
            Dictionary with portfolio metrics
        """
        return {
            'total_equity': self.portfolio.total_equity,
            'available_cash': self.portfolio.available_cash,
            'positions_count': len(self.portfolio.positions),
            'daily_pnl': self.portfolio.daily_pnl,
            'total_pnl': self.portfolio.total_pnl,
            'max_drawdown': self.portfolio.max_drawdown,
            'position_sizing_strategy': self.position_sizer.get_strategy_name()
        }

    def get_current_positions(self) -> Dict[str, Position]:
        """Get current open positions"""
        return self.portfolio.positions.copy()

    def should_reduce_risk(self) -> bool:
        """Check if risk should be reduced based on current portfolio state"""
        return self.risk_manager.should_reduce_risk(self.portfolio)

    def calculate_position_costs(self, position, current_price, current_date):
        """
        Calculate costs for position at current state

        Args:
            position: Current position
            current_price: Current market price
            current_date: Current date

        Returns:
            Dictionary with cost breakdown
        """
        days_held = (current_date - position.entry_date).days
        is_short = (position.direction == PositionDirection.SHORT)

        return self.cost_model.calculate_total_trade_cost(
            entry_price=position.entry_price,
            exit_price=current_price,
            quantity=position.quantity,
            days_held=days_held,
            is_short=is_short
        )

    # =============================================================================
    # PRIVATE HELPER METHODS
    # =============================================================================

    def _apply_safety_constraints(self, position_size: int, signal: TradingSignal) -> int:
        """Apply additional safety constraints to position size"""
        # Minimum position size
        if position_size < 1:
            return 0

        # Check available cash
        position_value = position_size * signal.entry_price
        if position_value > self.portfolio.available_cash:
            position_size = int(self.portfolio.available_cash / signal.entry_price)

        # Maximum single position (additional safety)
        max_position_value = self.portfolio.total_equity * 0.25  # 25% hard limit
        max_shares = int(max_position_value / signal.entry_price)

        return min(position_size, max_shares)

    def _update_portfolio_equity(self):
        """Update portfolio equity based on current positions"""
        try:
            total_position_value = 0
            total_unrealized_pnl = 0

            for position in self.portfolio.positions.values():
                position_value = position.size * position.current_price
                total_position_value += position_value

                # Calculate unrealized P&L
                if position.direction == PositionDirection.LONG:
                    unrealized_pnl = position.size * (position.current_price - position.entry_price)
                else:  # SHORT
                    unrealized_pnl = position.size * (position.entry_price - position.current_price)

                position.unrealized_pnl = unrealized_pnl
                total_unrealized_pnl += unrealized_pnl

            # Update portfolio equity
            initial_equity = self.config.get('initial_capital', 100000)
            self.portfolio.total_equity = initial_equity + total_unrealized_pnl
            self.portfolio.available_cash = self.portfolio.total_equity - total_position_value

            # Update peak equity and drawdown
            if self.portfolio.total_equity > self.portfolio.peak_equity:
                self.portfolio.peak_equity = self.portfolio.total_equity

            current_drawdown = (self.portfolio.peak_equity - self.portfolio.total_equity) / self.portfolio.peak_equity
            self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, current_drawdown)

        except Exception as e:
            logger.error(f"Error updating portfolio equity: {e}")