# base_trade_history.py
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


class PositionOutcome:
    """Represents the outcome of a single position"""

    def __init__(self, outcome: str , gross_pnl: float ,
                 net_pnl: float, fees: float ):
        self.outcome = outcome
        self.gross_pnl = gross_pnl
        self.net_pnl = net_pnl
        self.fees = fees


class TradeStatistics:
    """Represents aggregated trade statistics"""

    def __init__(self, total_positions: int, winning_positions: int,
                 losing_positions: int, break_even_positions: int ,
                 total_pnl: float, total_fees: float,
                 outcomes: Optional[List[PositionOutcome]]):
        self.total_positions = total_positions
        self.winning_positions = winning_positions
        self.losing_positions = losing_positions
        self.break_even_positions = break_even_positions
        self.total_pnl = total_pnl
        self.total_fees = total_fees
        self.outcomes = outcomes or []


class BaseTradeHistory(ABC):
    """Base class for trade history management

    Provides common functionality for storage, persistence, and statistics.
    Subclasses implement validation and finalization for specific trade types.
    """

    def __init__(self, config):
        """Initialize BaseTradeHistory with configuration

        Args:
            config: UnifiedConfig instance containing trade_history and base configuration

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config

        # Get trade_history config section - required
        trade_config = self.config.get_section('trade_history')
        if trade_config is None:
            raise ValueError("trade_history section not found in config")

        # Get base currency - required
        self.base_currency = self.config.get_section('base_currency')
        if self.base_currency is None:
            raise ValueError("base_currency not found in config")

        # Configuration-driven parameters - all required
        self.default_lookback_periods = trade_config.get('default_lookback_periods')
        if self.default_lookback_periods is None:
            raise ValueError("default_lookback_periods not found in trade_history config")

        self.save_backup_on_load = trade_config.get('save_backup_on_load')
        if self.save_backup_on_load is None:
            raise ValueError("save_backup_on_load not found in trade_history config")

        self.validate_on_load = trade_config.get('validate_on_load')
        if self.validate_on_load is None:
            raise ValueError("validate_on_load not found in trade_history config")

        self.break_even_tolerance = trade_config.get('break_even_tolerance')
        if self.break_even_tolerance is None:
            raise ValueError("break_even_tolerance not found in trade_history config")

        # Timestamp-ordered trade storage
        self.trades = SortedDict()

        # Statistics cache
        self._stats_cache = {}
        self._cache_valid = False

        logger.info(f"{self.__class__.__name__} initialized with base_currency: {self.base_currency}")

    # =========================================================================
    # ABSTRACT METHODS - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _validate_trade_structure(self, trade_data: Dict[str, Any]) -> bool:
        """Validate trade data structure. Implemented by subclasses."""
        pass

    @abstractmethod
    def _finalize_closed_trade(self, trade_data: Dict[str, Any]) -> None:
        """Finalize closed trade (calculate P&L, costs). Implemented by subclasses."""
        pass

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def load_from_json(self, file_path: str) -> bool:
        """Load trade data from JSON file"""
        try:
            file_path = self._resolve_trade_file_path(file_path)

            if not file_path.exists():
                logger.error(f"Trade data file not found: {file_path}")
                return False

            logger.info(f"Loading trade data from: {file_path}")

            with open(file_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict) or 'trades' not in data:
                logger.error("Invalid JSON structure: missing 'trades' key")
                return False

            trades_data = data['trades']
            if not isinstance(trades_data, list):
                logger.error("Invalid JSON structure: 'trades' must be a list")
                return False

            self.trades.clear()
            self._invalidate_cache()

            loaded_count = 0
            for trade_data in trades_data:
                if self.add_trade(trade_data):
                    loaded_count += 1
                else:
                    logger.error(f"Failed to add trade: {trade_data.get('uuid', 'unknown')}")
                    return False

            logger.info(f"Successfully loaded {loaded_count} trades from {len(trades_data)} records")

            if self.save_backup_on_load:
                backup_path = file_path.with_suffix('.backup.json')
                self.save_to_json(str(backup_path))
                logger.debug(f"Backup saved to: {backup_path}")

            return True

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
            return False

    def save_to_json(self, file_path: str) -> bool:
        """Save trade data to JSON file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            trades_list = list(self.trades.values())
            data = {
                'trades': trades_list,
                'metadata': {
                    'base_currency': self.base_currency,
                    'total_trades': len(trades_list),
                    'trade_type': self.__class__.__name__,
                    'export_timestamp': datetime.utcnow().isoformat() + 'Z'
                }
            }

            logger.info(f"Saving {len(trades_list)} trades to: {file_path}")

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Trade data successfully saved to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving trade data: {e}")
            return False

    def _resolve_trade_file_path(self, relative_path: str) -> Path:
        """Resolve relative trade file path to absolute path"""
        file_path = Path(relative_path)

        if file_path.is_absolute():
            return file_path
        else:
            project_root = Path(__file__).resolve().parents[3]
            return project_root / relative_path

    # =========================================================================
    # TRADE OPERATIONS
    # =========================================================================

    def add_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Add new trade to history"""
        try:
            if self.validate_on_load and not self._validate_trade_structure(trade_data):
                logger.error("Invalid trade structure")
                return False

            timestamp = self._parse_timestamp(trade_data['timestamp'])
            if not timestamp:
                logger.error(f"Invalid timestamp in trade: {trade_data.get('uuid', 'unknown')}")
                return False

            if trade_data.get('status') == 'closed':
                self._finalize_closed_trade(trade_data)

            self.trades[timestamp] = trade_data
            self._invalidate_cache()

            logger.debug(f"Added trade: {trade_data.get('uuid', 'unknown')} at {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing trade by id"""
        for timestamp, trade in self.trades.items():
            if trade.get('uuid') == trade_id or trade.get('trade_id') == trade_id:
                trade.update(updates)
                if trade.get('status') == 'closed':
                    self._finalize_closed_trade(trade)
                self._invalidate_cache()
                logger.debug(f"Updated trade {trade_id}")
                return True
        logger.warning(f"Trade {trade_id} not found for update")
        return False

    def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get trade by uuid or trade_id"""
        for trade in self.trades.values():
            if trade.get('uuid') == trade_id or trade.get('trade_id') == trade_id:
                return trade
        return None

    def get_trade_count(self) -> int:
        """Get total number of trades in history"""
        return len(self.trades)

    def get_closed_positions_count(self) -> int:
        """Get number of closed positions"""
        return len(self._get_closed_positions())

    @property
    def all_positions(self) -> List[Dict[str, Any]]:
        """Get all positions/trades as a list"""
        return list(self.trades.values())

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades"""
        return [t for t in self.trades.values() if t.get('status') == 'open']

    def _get_closed_positions(self, lookback_periods: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get closed trades for statistics calculation"""
        closed_trades = []
        trades_list = list(self.trades.values())
        trades_list.reverse()

        trades_collected = 0
        max_trades = lookback_periods if lookback_periods and lookback_periods > 0 else float('inf')

        for trade in trades_list:
            if trades_collected >= max_trades:
                break

            if trade.get('status', 'closed') == 'closed' and trade.get('exit_price') is not None:
                closed_trades.append(trade)
                trades_collected += 1

        closed_trades.reverse()
        return closed_trades

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_position_outcomes(self, lookback_periods: Optional[int] = None) -> List[PositionOutcome]:
        """Get position outcomes for analysis"""
        if lookback_periods is None:
            lookback_periods = self.default_lookback_periods

        cache_key = f"outcomes_{lookback_periods}"
        if self._cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key].copy()

        try:
            closed_positions = self._get_closed_positions(lookback_periods)

            if not closed_positions:
                logger.warning("No closed positions available for outcome analysis")
                return []

            outcomes = []
            for position in closed_positions:
                outcome_obj = self._calculate_position_outcome(position)
                outcomes.append(outcome_obj)

            self._stats_cache[cache_key] = outcomes.copy()
            self._cache_valid = True

            return outcomes

        except Exception as e:
            logger.error(f"Error calculating position outcomes: {e}")
            return []

    def get_trade_statistics(self, lookback_periods: Optional[int] = None) -> TradeStatistics:
        """Get generic trade statistics"""
        outcomes = self.get_position_outcomes(lookback_periods)

        if not outcomes:
            raise ValueError("No closed positions available for statistics calculation")

        wins = [o for o in outcomes if o.outcome == 'win']
        losses = [o for o in outcomes if o.outcome == 'loss']
        break_evens = [o for o in outcomes if o.outcome == 'break_even']

        total_pnl = sum(o.net_pnl for o in outcomes)
        total_fees = sum(o.fees for o in outcomes)

        return TradeStatistics(
            total_positions=len(outcomes),
            winning_positions=len(wins),
            losing_positions=len(losses),
            break_even_positions=len(break_evens),
            total_pnl=total_pnl,
            total_fees=total_fees,
            outcomes=outcomes
        )

    def _calculate_position_outcome(self, trade: Dict[str, Any]) -> PositionOutcome:
        """Calculate outcome for a trade - can be overridden by subclasses"""
        gross_pnl = trade.get('gross_pnl')
        if gross_pnl is None:
            raise ValueError(f"gross_pnl not found in trade: {trade.get('uuid', 'unknown')}")

        net_pnl = trade.get('net_pnl')
        if net_pnl is None:
            raise ValueError(f"net_pnl not found in trade: {trade.get('uuid', 'unknown')}")

        costs = trade.get('costs', {})
        total_fees = costs.get('total', 0.0) if isinstance(costs, dict) else 0.0

        if abs(net_pnl) < self.break_even_tolerance:
            outcome = 'break_even'
        elif net_pnl > 0:
            outcome = 'win'
        else:
            outcome = 'loss'

        return PositionOutcome(
            outcome=outcome,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fees=total_fees
        )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object"""
        try:
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'

            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        except (ValueError, AttributeError) as e:
            logger.error(f"Error parsing timestamp '{timestamp_str}': {e}")
            return None

    def _invalidate_cache(self):
        """Invalidate statistics cache"""
        self._stats_cache.clear()
        self._cache_valid = False