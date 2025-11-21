# money_management.py
# MoneyManager service for position sizing, risk management, and portfolio tracking
# Uses Strategy pattern internally for different algorithms
# Provides stable external interface regardless of internal implementation

import logging
from typing import Dict, Any
import pandas as pd

from src.hybrid.positions.types import PortfolioState, TradingSignal, PositionDirection, Position
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

            # Position orchestrator will be injected via set_position_orchestrator()
            self.position_orchestrator = None

            # Initialize strategy components (stateless calculators)
            self.position_sizer = self._create_position_sizer()
            self.risk_manager = self._create_risk_manager()
            self.cost_model = cost_model if cost_model else SimpleTransactionCostModel(config)

            self.risk_reduction_factor = self.config.get('risk_reduction_factor')

            self.lock_timeout = self.config['lock_timeout_seconds']
            self.default_timeout_ms = self.config['reservation_timeout_ms']
            self.max_retries = self.config['max_retries']
            self.retry_delay_ms = self.config['retry_delay_ms']
            self.cleanup_interval_ms = self.config['cleanup_interval_ms']

            logger.info(f"MoneyManager initialized with {self.position_sizer.get_strategy_name()} "
                        f"position sizing (stateless)")
        except Exception as e:
            logger.error(f"Exception in constructor: {e}")
            raise

    def set_position_orchestrator(self, position_orchestrator):
        """Inject position orchestrator"""
        self.position_orchestrator = position_orchestrator
        logger.debug("PositionOrchestrator injected into MoneyManager")

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
            # Get portfolio state from position manager
            if not self.position_orchestrator:
                logger.error("Position orchestrator not set")
                return 0

            portfolio_state = self.position_orchestrator.get_portfolio_state()

            if portfolio_state.available_cash <= 0:
                logger.warning(f"No available capital for {signal.symbol}")
                return 0

            # Calculate actual equity from open positions with current market prices
            actual_equity = self._calculate_equity_from_portfolio_state(portfolio_state)
            portfolio_state.total_equity = actual_equity

            # Calculate stop distance from risk manager
            stop_loss_price = self.risk_manager.calculate_stop_loss(signal, market_data)
            stop_distance = abs(signal.entry_price - stop_loss_price)

            # Check if risk should be reduced
            if self.risk_manager.should_reduce_risk(portfolio_state):
                logger.warning("Risk reduction triggered - applying risk reduction factor")
                risk_reduction_factor = self.config.get('money_management', {}).get('risk_reduction_factor', 0.5)
                base_size = self.position_sizer.calculate_size(signal, portfolio_state, stop_distance)
                return int(base_size * risk_reduction_factor)

            # Normal position sizing
            position_size = self.position_sizer.calculate_size(signal, portfolio_state, stop_distance)

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

    def should_reduce_risk(self) -> bool:
        """Check if risk should be reduced based on current portfolio state"""
        portfolio_state = self.position_orchestrator.get_portfolio_state()
        return self.risk_manager.should_reduce_risk(portfolio_state)

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
        """
        Apply additional safety constraints to position size
        
        Constraints applied:
        1. Minimum position size (must be >= 1)
        2. Available cash limit (cannot exceed available capital)
        3. Maximum single position limit (25% of total equity)
        
        Args:
            position_size: Calculated position size from strategy
            signal: Trading signal with entry price
            
        Returns:
            Position size after applying safety constraints
        """
        # Validate inputs
        if position_size < 1:
            return 0
            
        if signal.entry_price <= 0:
            logger.warning(f"Invalid entry price {signal.entry_price} for {signal.symbol}")
            return 0
            
        if not self.position_orchestrator:
            logger.error("Position orchestrator not set in _apply_safety_constraints")
            return 0

        # Get portfolio state and calculate actual equity
        portfolio_state = self.position_orchestrator.get_portfolio_state()
        actual_equity = self._calculate_equity_from_portfolio_state(portfolio_state)
        
        # Constraint 1: Check available cash
        position_value = position_size * signal.entry_price
        if position_value > portfolio_state.available_cash:
            position_size = int(portfolio_state.available_cash / signal.entry_price)
            logger.debug(f"Position size limited by available cash to {position_size} shares")

        # Constraint 2: Maximum single position (25% of total equity)
        # Equity = value of open positions according to actual market price
        max_position_value = actual_equity * 0.25
        max_shares = int(max_position_value / signal.entry_price)
        
        final_size = min(position_size, max_shares)
        
        if final_size < position_size:
            logger.debug(f"Position size limited by max position constraint to {final_size} shares")
            
        return final_size

    def _calculate_equity_from_portfolio_state(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate equity from open positions in PortfolioState using actual market prices
        
        Equity = initial capital + unrealized P&L from all open positions
        Unrealized P&L is calculated using current_price from positions
        
        Args:
            portfolio_state: PortfolioState with positions
            
        Returns:
            Total equity value calculated from positions
        """

        try:
            # Get initial capital (use total_capital as base)
            initial_capital = self.position_orchestrator.position_manager.total_capital
            total_unrealized_pnl = 0.0
            
            for position in portfolio_state.positions.values():
                # Ensure current_price is set (default to entry_price if not set)
                current_price = getattr(position, 'current_price', position.entry_price)
                
                # Calculate unrealized P&L based on current market price
                if position.direction == PositionDirection.LONG:
                    unrealized_pnl = position.size * (current_price - position.entry_price)
                else:  # SHORT
                    unrealized_pnl = position.size * (position.entry_price - current_price)
                
                # Update position's unrealized_pnl
                position.unrealized_pnl = unrealized_pnl
                total_unrealized_pnl += unrealized_pnl
            
            # Equity = initial capital + unrealized P&L
            equity = initial_capital + total_unrealized_pnl
            return max(0.0, equity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating equity from portfolio state: {e}")
            return self.position_orchestrator.position_manager.total_capital
