# **CentralizedPositionManager:**
# - Capital allocation only
# - Knows: "trade_123 has $50k committed"
# - Doesn't know: symbol, price, shares
#
# **PositionTracker:**
# - Position details for open trades
# - Knows: "trade_123 = 1000 EURUSD @ 1.1000, current 1.1050"
# - Mutable - updates with market prices
#
# **TradeHistory:**
# - All trades (open + closed)
# - Knows: complete trade records with timestamps
# - Immutable archive - write-once, read-many
#
# **Key difference:**
# - PositionManager = money
# - PositionTracker = live positions
# - TradeHistory = historical record

# src/hybrid/position_tracking/position_tracker.py
"""
PositionTracker - Tracks live open positions with current market data

Manages state of open positions including entry details and current prices.
Does NOT calculate P&L - provides data to PerformanceMetrics for calculations.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from threading import RLock
from typing import Dict, Optional

from src.hybrid.positions.types import Position
from src.hybrid.products.product_types import PositionDirection

logger = logging.getLogger(__name__)

class PositionTracker:
    """
    Tracks open positions with current market data

    Provides position snapshots for P&L calculations by PerformanceMetrics.
    Works alongside CentralizedPositionManager (which tracks capital).
    """

    def __init__(self, config: Dict):
        """Initialize position tracker from configuration"""
        self.config = config
        self.lock = RLock()

        # Track open positions
        self.open_positions: Dict[str, Position] = {}

        logger.info("PositionTracker initialized")

    @contextmanager
    def _acquire_lock(self):
        """Helper for lock acquisition with timeout"""

        #TODO: timeout shouldn't be in money_management
        timeout = self.config.config.get('money_management', {}).get('lock_timeout_seconds', 0.5)
        if not self.lock.acquire(timeout=timeout):
            raise TimeoutError("Could not acquire lock")
        try:
            yield
        finally:
            self.lock.release()

    def get_position(self, trade_id: str) -> Optional[Position]:
        """Get current position state"""
        with self._acquire_lock():
            return self.open_positions.get(trade_id)

    def open_position(
            self,
            trade_id: str,
            symbol: str,
            direction: PositionDirection,
            quantity: int,
            entry_price: float,
            entry_time: Optional[datetime] = None
    ) -> bool:
        """Record new open position"""
        with self._acquire_lock():
            if trade_id in self.open_positions:
                logger.warning(f"Position {trade_id} already exists")
                return False

            if direction not in [PositionDirection.LONG, PositionDirection.SHORT]:
                logger.error(f"Invalid direction: {direction}")
                return False

            entry_time = entry_time or datetime.now()

            self.open_positions[trade_id] = Position(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                size=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                entry_time=entry_time,
                last_update=entry_time
            )

            logger.info(f"Opened position {trade_id}: {quantity} {symbol} @ {entry_price} ({direction})")
            return True

    def update_position_price(self, trade_id: str, current_price: float) -> bool:
        """
        Update position with current market price

        Args:
            trade_id: Position to update
            current_price: Current market price

        Returns:
            True if successful
        """
        #TODO: there should be something different than money_management
        if not self.lock.acquire(timeout=self.config.config.get('money_management', {}).get('lock_timeout_seconds', 0.5)):
            logger.error(f"Lock timeout updating position {trade_id}")
            raise TimeoutError("Could not acquire lock")

        try:
            if trade_id not in self.open_positions:
                logger.warning(f"Position {trade_id} not found")
                return False

            position = self.open_positions[trade_id]
            position.current_price = current_price
            position.last_update = datetime.now()

            logger.debug(f"Updated {trade_id} price: {current_price}")
            return True

        finally:
            self.lock.release()

    def close_position(self, trade_id: str) -> Optional[Position]:
        """
        Close position and return final state

        Args:
            trade_id: Position to close

        Returns:
            Closed position data or None if not found
        """
        #TODO: there should be something different to money_management
        if not self.lock.acquire(timeout=self.config.config.get('money_management', {}).get('lock_timeout_seconds', 0.5)):
            logger.error(f"Lock timeout closing position {trade_id}")
            raise TimeoutError("Could not acquire lock")

        try:
            if trade_id not in self.open_positions:
                logger.warning(f"Position {trade_id} not found for closing")
                return None

            position = self.open_positions.pop(trade_id)
            logger.info(f"Closed position {trade_id}")
            return position

        finally:
            self.lock.release()

    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all open positions

        Returns:
            Dictionary of trade_id -> position
        """
        #TODO: instead money_management something different.
        if not self.lock.acquire(timeout=self.config.config.get('money_management', {}).get('lock_timeout_seconds', 0.5)):
            logger.error("Lock timeout getting all positions")
            raise TimeoutError("Could not acquire lock")

        try:
            return self.open_positions.copy()
        finally:
            self.lock.release()

    def get_positions_by_symbol(self, symbol: str) -> Dict[str, Position]:
        """
        Get all positions for specific symbol

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary of trade_id -> position for symbol
        """
        if not self.lock.acquire(timeout=self.config.get('money_management', {}).get('lock_timeout_seconds', 0.5)):
            logger.error(f"Lock timeout getting positions for {symbol}")
            raise TimeoutError("Could not acquire lock")

        try:
            return {
                tid: pos for tid, pos in self.open_positions.items()
                if pos.symbol == symbol
            }
        finally:
            self.lock.release()

    def get_position_count(self) -> int:
        """Get count of open positions"""
        if not self.lock.acquire(timeout=self.config.get('money_management', {}).get('lock_timeout_seconds', 0.5)):
            logger.error("Lock timeout getting position count")
            raise TimeoutError("Could not acquire lock")

        try:
            return len(self.open_positions)
        finally:
            self.lock.release()

    def reset(self) -> None:
        """Reset tracker (clear all positions) - for testing"""
        if not self.lock.acquire(timeout=self.config.get('money_management', {}).get('lock_timeout_seconds', 0.5)):
            logger.error("Lock timeout resetting tracker")
            raise TimeoutError("Could not acquire lock")

        try:
            count = len(self.open_positions)
            self.open_positions.clear()
            logger.warning(f"PositionTracker reset: cleared {count} positions")
        finally:
            self.lock.release()

    def on_price_update(self, current_prices: Dict[str, float]):
        """Listener callback for DataManager price updates"""
        with self._acquire_lock():
            for position in self.open_positions.values():
                if position.symbol in current_prices:
                    position.current_price = current_prices[position.symbol]
                    position.last_update = datetime.now()

            logger.debug(f"Updated prices for {len(self.open_positions)} positions")