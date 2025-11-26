# src/hybrid/positions/position_orchestrator.py
"""
PositionOrchestrator - Coordinates capital, positions, and trade history

Provides unified PortfolioState by aggregating data from:
- PositionManager: capital allocation
- PositionTracker: live position details
- TradeHistory: historical records
"""

import logging
from datetime import datetime

from src.hybrid.positions.centralized_position_manager import CentralizedPositionManager
from src.hybrid.positions.position_tracker import PositionTracker
from src.hybrid.positions.trade_history import TradeHistory
from src.hybrid.positions.types import PortfolioState
from src.hybrid.products.product_types import PositionDirection

logger = logging.getLogger(__name__)

class PositionOrchestrator:
    """Coordinates position management, tracking, and history"""

    def __init__(self, config: dict):
        """Initialize orchestrator with all components"""
        self.config = config

        # Initialize trinity
        self.position_manager = CentralizedPositionManager(config)
        self.position_tracker = PositionTracker(config)
        self.trade_history = TradeHistory(config)

        logger.info("PositionOrchestrator initialized")

    def set_initial_capital(self, capital: float):
        """Set initial capital for position manager"""
        self.position_manager.set_total_capital(capital)
        logger.info(f"Initial capital set to {capital}")

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state aggregated from all sources"""
        # Get capital allocation
        allocation = self.position_manager.get_allocation_summary()

        # Get open positions with current prices
        positions = self.position_tracker.get_all_positions()

        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        for position in positions.values():
            if position.direction == 'long':
                unrealized_pnl += (position.current_price - position.entry_price) * position.size
            else:  # short
                unrealized_pnl += (position.entry_price - position.current_price) * position.size

        # Get realized P&L from trade history
        stats = self.trade_history.get_trade_statistics()
        realized_pnl = stats.total_pnl if stats else 0.0

        # Total P&L
        total_pnl = realized_pnl + unrealized_pnl

        # Calculate current equity
        total_equity = allocation['total_capital'] + total_pnl

        # Get peak equity from trade history or current
        peak_equity = max(total_equity, allocation['total_capital'])  # Simplified

        # Calculate drawdown
        max_drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0

        # Daily P&L (would need timestamp tracking)
        daily_pnl = unrealized_pnl  # Simplified - only unrealized for now

        return PortfolioState(
            total_equity=total_equity,
            available_cash=allocation['available'],
            positions=positions,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            peak_equity=peak_equity
        )

    def open_position(
            self,
            trade_id: str,
            symbol: str,
            direction: PositionDirection ,
            quantity: int,
            entry_price: float,
            capital_required: float
    ) -> bool:
        """
        Open new position across all components

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            direction: 'long' or 'short'
            quantity: Number of shares/units
            entry_price: Entry price
            capital_required: Capital to commit

        Returns:
            True if successful
        """
        # 1. Commit capital
        if not self.position_manager.commit_position(trade_id, capital_required, "strategy"):
            logger.warning(f"Failed to commit capital for {trade_id}")
            return False

        # 2. Track position
        if not self.position_tracker.open_position(trade_id, symbol, direction, quantity, entry_price):
            # Rollback capital
            self.position_manager.release_position(trade_id)
            logger.error(f"Failed to track position {trade_id}")
            return False

        # 3. Record in history
        trade_data = {
            'uuid': trade_id,
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat() + 'Z',
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_date': datetime.now().isoformat() + 'Z',
            'status': 'open',
            'exit_price': None,
            'exit_date': None
        }
        self.trade_history.add_trade(trade_data)

        logger.info(f"Opened position {trade_id}: {quantity} {symbol} @ {entry_price}")
        return True

    def update_position_price(self, trade_id: str, current_price: float) -> bool:
        """Update position with current market price"""
        return self.position_tracker.update_position_price(trade_id, current_price)

    def close_position(self, trade_id: str, exit_price: float, exit_reason: str = 'signal') -> bool:
        """Close position across all components"""
        position = self.position_tracker.get_position(trade_id)
        if not position:
            logger.warning(f"Position {trade_id} not found")
            return False

        closed_position = self.position_tracker.close_position(trade_id)
        self.position_manager.release_position(trade_id)

        updates = {
            'exit_price': exit_price,
            'exit_date': datetime.now().isoformat() + 'Z',
            'status': 'closed',
            'exit_reason': exit_reason
        }
        self.trade_history.update_trade(trade_id, updates)

        logger.info(f"Closed position {trade_id} @ {exit_price} ({exit_reason})")
        return True

    def reset(self) -> None:
        """Reset all components - for testing"""
        self.position_manager.reset()
        self.position_tracker.reset()
        # trade_history doesn't need reset - it's persistent