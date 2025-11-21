# src/hybrid/money_management/centralized_position_manager.py
"""
CentralizedPositionManager - Thread-safe capital allocation for parallel bot execution

Manages capital allocation across multiple concurrent trading bots with atomic reserve-on-query.
Prevents over-allocation and race conditions under high load.
"""

import logging
import threading
import time
import uuid
from threading import Lock, Thread
from typing import Dict, Optional, Tuple
from datetime import datetime

from src.hybrid.money_management import PortfolioState, Position

logger = logging.getLogger(__name__)


class CentralizedPositionManager:
    """
    Thread-safe capital allocation manager with atomic reservations

    Prevents race conditions by atomically calculating and reserving capital
    inside lock. Ensures correct capital allocation even under high load.

    Capital lifecycle:
    1. Bot requests reservation (atomic: query + reserve)
    2. Bot calculates position size
    3. Bot commits actual amount used
    4. Unused reservation auto-released on timeout
    5. Capital freed when position closes
    """

    def __init__(self, unified_config: Dict):
        """Initialize position manager from configuration"""
        mm_config = unified_config.config.get('money_management', {})

        self.total_capital = None  # Set via setter
        self.lock_timeout = mm_config['lock_timeout_seconds']
        self.default_timeout_ms = mm_config['reservation_timeout_ms']
        self.max_retries = mm_config['max_retries']
        self.retry_delay_ms = mm_config['retry_delay_ms']
        self.cleanup_interval_ms = mm_config['cleanup_interval_ms']

        self.lock = threading.RLock()
        self.reservations = {}
        self.committed = {}

        self._cleanup_running = True
        self._cleanup_thread = Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info("CentralizedPositionManager initialized from config")

    def set_total_capital(self, total_capital: float):
        """Set total capital (changeable)"""
        if total_capital <= 0:
            raise ValueError(f"Total capital must be positive, got {total_capital}")

        with self.lock:
            self.total_capital = total_capital
            logger.info(f"Total capital set to ${total_capital:,.2f}")

    def __del__(self):
        """Cleanup on deletion"""
        self._cleanup_running = False
        if hasattr(self, '_cleanup_thread') and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)

    def _get_available_capital_unlocked(self) -> float:
        """
        Calculate available capital (must be called within lock)

        Returns:
            Available capital (total - reserved - committed)
        """
        reserved_total = sum(r['amount'] for r in self.reservations.values())
        committed_total = sum(c['amount'] for c in self.committed.values())
        available = self.total_capital - reserved_total - committed_total
        return max(0.0, available)

    def get_available_capital(self) -> float:
        """
        Get currently available capital (thread-safe)

        Returns:
            Available capital amount
        """
        with self.lock:
            return self._get_available_capital_unlocked()

    def reserve_capital_percentage(
            self,
            bot_id: str,
            percentage: float,
            timeout_ms: Optional[int] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Atomically reserve percentage of current available capital

        This is the PRIMARY method for bots to request capital.
        Calculation happens inside lock to prevent race conditions.

        Args:
            bot_id: Identifier of bot requesting capital
            percentage: Percentage of available capital to reserve (0.0 to 1.0)
            timeout_ms: Reservation timeout in milliseconds (None = use default)

        Returns:
            Tuple of (reservation_id, amount_reserved) if successful, None if insufficient capital
        """
        if not (0.0 <= percentage <= 1.0):
            logger.error(f"Invalid percentage {percentage}, must be between 0 and 1")
            return None

        timeout = timeout_ms if timeout_ms is not None else self.default_timeout_ms

        with self.lock:
            available = self._get_available_capital_unlocked()

            if available <= 0:
                logger.debug(f"No available capital for {bot_id}")
                return None

            amount_to_reserve = available * percentage

            reservation_id = f"{bot_id}_{uuid.uuid4().hex[:8]}"
            self.reservations[reservation_id] = {
                'amount': amount_to_reserve,
                'bot_id': bot_id,
                'expires_at': time.time() + (timeout / 1000.0)
            }

            logger.debug(f"Reserved ${amount_to_reserve:,.2f} ({percentage * 100:.1f}%) for {bot_id}, "
                         f"reservation: {reservation_id}, available now: ${self._get_available_capital_unlocked():,.2f}")

            return (reservation_id, amount_to_reserve)

    def reserve_capital_fixed(
            self,
            bot_id: str,
            amount: float,
            timeout_ms: Optional[int] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Atomically reserve fixed amount of capital

        Args:
            bot_id: Identifier of bot requesting capital
            amount: Fixed amount to reserve
            timeout_ms: Reservation timeout in milliseconds (None = use default)

        Returns:
            Tuple of (reservation_id, amount_reserved) if successful, None if insufficient capital
        """
        if amount <= 0:
            logger.error(f"Cannot reserve non-positive amount: {amount}")
            return None

        timeout = timeout_ms if timeout_ms is not None else self.default_timeout_ms

        with self.lock:
            available = self._get_available_capital_unlocked()

            if amount > available:
                logger.debug(
                    f"Insufficient capital for {bot_id}: requested ${amount:,.2f}, available ${available:,.2f}")
                return None

            reservation_id = f"{bot_id}_{uuid.uuid4().hex[:8]}"
            self.reservations[reservation_id] = {
                'amount': amount,
                'bot_id': bot_id,
                'expires_at': time.time() + (timeout / 1000.0)
            }

            logger.debug(f"Reserved ${amount:,.2f} for {bot_id}, reservation: {reservation_id}")

            return (reservation_id, amount)

    def commit_reservation(self, reservation_id: str, trade_id: str, actual_amount: float) -> bool:
        """
        Commit reservation to active position

        Bot uses actual_amount (may be less than reserved).
        Unused portion is automatically released.

        Args:
            reservation_id: Reservation to commit
            trade_id: Trade identifier for committed position
            actual_amount: Actual capital used (must be <= reserved amount)

        Returns:
            True if successful, False if reservation not found or invalid amount
        """
        if actual_amount < 0:
            logger.error(f"Cannot commit negative amount: {actual_amount}")
            return False

        with self.lock:
            if reservation_id not in self.reservations:
                logger.warning(f"Attempted to commit unknown reservation: {reservation_id}")
                return False

            reservation = self.reservations[reservation_id]
            reserved_amount = reservation['amount']
            bot_id = reservation['bot_id']

            if actual_amount > reserved_amount:
                logger.error(f"Actual amount ${actual_amount:,.2f} exceeds reserved ${reserved_amount:,.2f}")
                return False

            # Remove reservation
            self.reservations.pop(reservation_id)

            # Commit actual amount used
            if actual_amount > 0:
                self.committed[trade_id] = {
                    'amount': actual_amount,
                    'timestamp': datetime.now(),
                    'bot_id': bot_id
                }
                logger.info(f"Committed ${actual_amount:,.2f} to trade {trade_id} (bot: {bot_id}), "
                            f"released ${reserved_amount - actual_amount:,.2f} unused")
            else:
                logger.debug(f"Released reservation {reservation_id} without committing (bot: {bot_id})")

            return True

    def release_reservation(self, reservation_id: str) -> bool:
        """
        Release reservation without committing (trade not executed)

        Args:
            reservation_id: Reservation to release

        Returns:
            True if released, False if not found
        """
        with self.lock:
            if reservation_id in self.reservations:
                reservation = self.reservations.pop(reservation_id)
                amount = reservation['amount']
                bot_id = reservation['bot_id']

                logger.debug(f"Released reservation {reservation_id}, freed ${amount:,.2f} (bot: {bot_id})")
                return True
            else:
                logger.debug(f"Attempted to release unknown reservation: {reservation_id}")
                return False

    def commit_position(self, trade_id: str, amount: float, bot_id: str = "unknown") -> bool:
        """
        Commit capital directly (without prior reservation)

        Use this for backwards compatibility or when reservation not used.
        Prefer reserve_capital_* + commit_reservation pattern for better concurrency.

        Args:
            trade_id: Unique trade identifier
            amount: Capital amount to commit
            bot_id: Identifier of bot opening position

        Returns:
            True if commitment successful, False if insufficient capital
        """
        if amount <= 0:
            logger.warning(f"Cannot commit non-positive amount: {amount}")
            return False

        with self.lock:
            # Check if already committed
            if trade_id in self.committed:
                logger.warning(f"Trade {trade_id} already committed")
                return False

            # Check available capital
            available = self._get_available_capital_unlocked()

            if amount <= available:
                self.committed[trade_id] = {
                    'amount': amount,
                    'timestamp': datetime.now(),
                    'bot_id': bot_id
                }
                logger.info(f"Committed ${amount:,.2f} for trade {trade_id} (bot: {bot_id}), "
                            f"available: ${self._get_available_capital_unlocked():,.2f}")
                return True
            else:
                logger.warning(f"Insufficient capital for {bot_id}: requested ${amount:,.2f}, "
                               f"available ${available:,.2f}")
                return False

    def release_position(self, trade_id: str) -> bool:
        """
        Release committed capital (position closed)

        Args:
            trade_id: Trade identifier to release

        Returns:
            True if released, False if trade_id not found
        """
        with self.lock:
            if trade_id in self.committed:
                position = self.committed.pop(trade_id)
                amount = position['amount']
                bot_id = position['bot_id']

                logger.info(f"Released ${amount:,.2f} from trade {trade_id} (bot: {bot_id}), "
                            f"available: ${self._get_available_capital_unlocked():,.2f}")
                return True
            else:
                logger.warning(f"Attempted to release unknown trade: {trade_id}")
                return False

    def _cleanup_loop(self):
        """Background thread to clean up expired reservations"""
        while self._cleanup_running:
            time.sleep(0.5)  # Check every 500ms
            self._cleanup_expired_reservations()

    def _cleanup_expired_reservations(self):
        """Remove expired reservations"""
        current_time = time.time()

        with self.lock:
            expired = []
            for res_id, res_data in self.reservations.items():
                if current_time > res_data['expires_at']:
                    expired.append(res_id)

            for res_id in expired:
                reservation = self.reservations.pop(res_id)
                logger.warning(f"Expired reservation {res_id} cleaned up, "
                               f"freed ${reservation['amount']:,.2f} (bot: {reservation['bot_id']})")

    def get_allocation_summary(self) -> Dict:
        """
        Get summary of capital allocation

        Returns:
            Dictionary with allocation statistics
        """
        with self.lock:
            reserved_total = sum(r['amount'] for r in self.reservations.values())
            committed_total = sum(c['amount'] for c in self.committed.values())
            available = self._get_available_capital_unlocked()

            # Group by bot
            by_bot = {}

            # Add reservations
            for res_id, res_data in self.reservations.items():
                bot_id = res_data['bot_id']
                if bot_id not in by_bot:
                    by_bot[bot_id] = {'reserved': 0.0, 'committed': 0.0, 'position_count': 0}
                by_bot[bot_id]['reserved'] += res_data['amount']

            # Add committed
            for trade_id, pos_data in self.committed.items():
                bot_id = pos_data['bot_id']
                if bot_id not in by_bot:
                    by_bot[bot_id] = {'reserved': 0.0, 'committed': 0.0, 'position_count': 0}
                by_bot[bot_id]['committed'] += pos_data['amount']
                by_bot[bot_id]['position_count'] += 1  # Now works because it's initialized

            return {
                'total_capital': self.total_capital,
                'available': available,
                'reserved': reserved_total,
                'committed': committed_total,
                'available_pct': (available / self.total_capital * 100) if self.total_capital > 0 else 0,
                'reserved_pct': (reserved_total / self.total_capital * 100) if self.total_capital > 0 else 0,
                'committed_pct': (committed_total / self.total_capital * 100) if self.total_capital > 0 else 0,
                'active_reservations': len(self.reservations),
                'active_positions': len(self.committed),
                'by_bot': by_bot
            }

    def get_committed_positions(self) -> Dict[str, Dict]:
        """
        Get all currently committed positions

        Returns:
            Dictionary of trade_id -> position info
        """
        if not self.lock.acquire(timeout=self.lock_timeout):
            logger.error(f"Lock timeout after {self.lock_timeout}s in get_committed_positions")
            raise TimeoutError(f"Could not acquire lock after {self.lock_timeout}s")

        try:
            return self.committed.copy()
        finally:
            self.lock.release()

    def get_reservations(self) -> Dict[str, Dict]:
        """
        Get all current reservations

        Returns:
            Dictionary of reservation_id -> reservation info
        """
        with self.lock:
            return self.reservations.copy()

    def reset(self) -> None:
        """
        Reset manager (clear all reservations and committed positions)

        WARNING: Only use for testing or reinitializing account
        """
        with self.lock:
            num_reservations = len(self.reservations)
            num_positions = len(self.committed)
            self.reservations.clear()
            self.committed.clear()
            logger.warning(f"Position manager reset: cleared {num_reservations} reservations "
                           f"and {num_positions} positions")
