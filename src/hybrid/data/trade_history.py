# trade_history.py
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


class PositionOutcome:
    """Represents the outcome of a single position"""

    def __init__(self, outcome: str = 'unknown', gross_pnl: float = 0.0,
                 net_pnl: float = 0.0, fees: float = 0.0):
        self.outcome = outcome
        self.gross_pnl = gross_pnl
        self.net_pnl = net_pnl
        self.fees = fees


class TradeStatistics:
    """Represents aggregated trade statistics"""

    def __init__(self, total_positions: int = 0, winning_positions: int = 0,
                 losing_positions: int = 0, break_even_positions: int = 0,
                 total_pnl: float = 0.0, total_fees: float = 0.0,
                 outcomes: Optional[List[PositionOutcome]] = None):
        self.total_positions = total_positions
        self.winning_positions = winning_positions
        self.losing_positions = losing_positions
        self.break_even_positions = break_even_positions
        self.total_pnl = total_pnl
        self.total_fees = total_fees
        self.outcomes = outcomes or []


class TradeHistory:
    """Manages trade execution history with JSON persistence and Kelly statistics interface

    Provides centralized storage and analysis of trade outcomes for position sizers.
    All parameters are configuration-driven with no hardcoded values.
    """

    def __init__(self, config):
        """Initialize TradeHistory with configuration

        Args:
            config: UnifiedConfig instance containing trade_history and base configuration
        """
        self.config = config

        # Get configuration sections
        try:
            # Try to get trade_history specific config section
            trade_config = self.config.get_section('trade_history', {})
        except (KeyError, AttributeError):
            # Fallback to empty config if section doesn't exist
            trade_config = {}
            logger.warning("No trade_history configuration section found, using defaults")

        # Get base currency from main config
        try:
            self.base_currency = self.config.get_section('base_currency', 'USD')
        except (KeyError, AttributeError):
            # If base_currency is at root level
            try:
                base_config = self.config.get_section('base', {})
                self.base_currency = base_config.get('currency', 'USD')
            except (KeyError, AttributeError):
                logger.warning("No base_currency configuration found, defaulting to USD")
                self.base_currency = 'USD'

        # Configuration-driven parameters
        self.default_lookback_periods = trade_config.get('default_lookback_periods', 0)
        self.save_backup_on_load = trade_config.get('save_backup_on_load', False)
        self.validate_on_load = trade_config.get('validate_on_load', True)

        # Timestamp-ordered trade storage
        self.trades = SortedDict()  # key: timestamp, value: trade_data

        # Statistics cache
        self._stats_cache = {}
        self._cache_valid = False

        logger.info(f"TradeHistory initialized with base_currency: {self.base_currency}")
        logger.debug(f"Configuration: lookback={self.default_lookback_periods}, "
                     f"backup={self.save_backup_on_load}, validate={self.validate_on_load}")

    def load_from_json(self, file_path: str) -> bool:
        """Load trade data from JSON file

        Args:
            file_path: Path to JSON file containing trade data

        Returns:
            True if loading successful, False otherwise
        """
        try:

            file_path = self._resolve_trade_file_path(file_path)

            if not file_path.exists():
                logger.error(f"Trade data file not found: {file_path}")
                return False

            logger.info(f"Loading trade data from: {file_path}")

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Validate JSON structure
            if not isinstance(data, dict) or 'trades' not in data:
                logger.error("Invalid JSON structure: missing 'trades' key")
                return False

            trades_data = data['trades']
            if not isinstance(trades_data, list):
                logger.error("Invalid JSON structure: 'trades' must be a list")
                return False

            # Clear existing trades
            self.trades.clear()
            self._invalidate_cache()

            # Load trades into sorted storage
            loaded_count = 0
            for trade_data in trades_data:
                if self._validate_trade_structure(trade_data):
                    timestamp = self._parse_timestamp(trade_data['timestamp'])
                    if timestamp:
                        self.trades[timestamp] = trade_data
                        loaded_count += 1
                    else:
                        logger.warning(f"Skipping trade with invalid timestamp: {trade_data.get('uuid', 'unknown')}")
                else:
                    logger.warning(f"Skipping trade with invalid structure: {trade_data.get('uuid', 'unknown')}")

            logger.info(f"Successfully loaded {loaded_count} trades from {len(trades_data)} records")

            # Optional backup save
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

    def _resolve_trade_file_path(self, relative_path: str) -> Path:
        """Resolve relative trade file path to absolute path"""
        file_path = Path(relative_path)

        if file_path.is_absolute():
            return file_path
        else:
            # Resolve relative to project root (same logic as DataManager loaders)
            project_root = Path(__file__).resolve().parents[3]
            return project_root / relative_path

    def save_to_json(self, file_path: str) -> bool:
        """Save trade data to JSON file

        Args:
            file_path: Path to save JSON file

        Returns:
            True if saving successful, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data structure
            trades_list = list(self.trades.values())
            data = {
                'trades': trades_list,
                'metadata': {
                    'base_currency': self.base_currency,
                    'total_trades': len(trades_list),
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

    def add_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Add new trade to history

        Args:
            trade_data: Dictionary containing trade information

        Returns:
            True if trade added successfully, False otherwise
        """
        try:
            if not self._validate_trade_structure(trade_data):
                logger.error("Invalid trade structure")
                return False

            timestamp = self._parse_timestamp(trade_data['timestamp'])
            if not timestamp:
                logger.error(f"Invalid timestamp in trade: {trade_data.get('uuid', 'unknown')}")
                return False

            self.trades[timestamp] = trade_data
            self._invalidate_cache()

            logger.debug(f"Added trade: {trade_data.get('uuid', 'unknown')} at {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False

    def get_position_outcomes(self, lookback_periods: Optional[int] = None) -> List[PositionOutcome]:
        """Get position outcomes for analysis

        Args:
            lookback_periods: Number of recent trades to include (0 = all trades, None = use config default)

        Returns:
            List of PositionOutcome objects
        """
        # Use configuration default if not specified
        if lookback_periods is None:
            lookback_periods = self.default_lookback_periods

        # Check cache
        cache_key = f"outcomes_{lookback_periods}"
        if self._cache_valid and cache_key in self._stats_cache:
            logger.debug(f"Returning cached position outcomes for lookback={lookback_periods}")
            return self._stats_cache[cache_key].copy()

        try:
            # Get closed positions for analysis
            closed_positions = self._get_closed_positions(lookback_periods)

            if not closed_positions:
                logger.warning("No closed positions available for outcome analysis")
                return []

            # Calculate outcomes for each position
            outcomes = []
            for position in closed_positions:
                outcome_obj = self._calculate_position_outcome(position)
                outcomes.append(outcome_obj)

            # Cache results
            self._stats_cache[cache_key] = outcomes.copy()
            self._cache_valid = True

            logger.debug(f"Position outcomes calculated: {len(outcomes)} positions")
            return outcomes

        except Exception as e:
            logger.error(f"Error calculating position outcomes: {e}")
            return []

    def get_trade_statistics(self, lookback_periods: Optional[int] = None) -> TradeStatistics:
        """Get generic trade statistics

        Args:
            lookback_periods: Number of recent trades to include (0 = all trades, None = use config default)

        Returns:
            TradeStatistics object with aggregated data
        """
        try:
            outcomes = self.get_position_outcomes(lookback_periods)

            if not outcomes:
                return TradeStatistics()  # Returns empty statistics object

            # Aggregate statistics
            wins = [o for o in outcomes if o.outcome == 'win']
            losses = [o for o in outcomes if o.outcome == 'loss']
            break_evens = [o for o in outcomes if o.outcome == 'break_even']

            total_pnl = sum(o.net_pnl for o in outcomes)
            total_fees = sum(o.fees for o in outcomes)

            stats = TradeStatistics(
                total_positions=len(outcomes),
                winning_positions=len(wins),
                losing_positions=len(losses),
                break_even_positions=len(break_evens),
                total_pnl=total_pnl,
                total_fees=total_fees,
                outcomes=outcomes
            )

            logger.debug(f"Trade statistics calculated: {len(outcomes)} positions")
            return stats

        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return TradeStatistics()  # Returns empty statistics object

    def get_trade_count(self) -> int:
        """Get total number of trades in history"""
        return len(self.trades)

    def get_closed_positions_count(self) -> int:
        """Get number of closed positions"""
        return len(self._get_closed_positions())

    def _get_closed_positions(self, lookback_periods: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get closed positions for statistics calculation

        Args:
            lookback_periods: Number of recent trades to include (0 = all)

        Returns:
            List of closed position dictionaries
        """
        closed_positions = []

        # Get all trades in reverse chronological order for lookback
        trades_list = list(self.trades.values())
        trades_list.reverse()  # Most recent first

        positions_collected = 0
        max_positions = lookback_periods if lookback_periods and lookback_periods > 0 else float('inf')

        for trade in trades_list:
            if positions_collected >= max_positions:
                break

            for position in trade.get('positions', []):
                if positions_collected >= max_positions:
                    break

                # Only include closed positions
                if (position.get('status') == 'closed' and
                        position.get('exit_value') is not None and
                        position.get('exit_timestamp') is not None):
                    closed_positions.append(position)
                    positions_collected += 1

        # Reverse to chronological order for consistent processing
        closed_positions.reverse()
        return closed_positions

    def _calculate_position_outcome(self, position: Dict[str, Any]) -> PositionOutcome:
        try:
            direction = position.get('direction', 'long')  # Default to long

            entry_value = float(position.get('entry_value', 0))
            exit_value = float(position.get('exit_value', 0))
            amount = float(position.get('amount', 0))

            # Calculate total fees from entry and exit fees
            entry_fees = float(position.get('entry_fees', 0))
            exit_fees = float(position.get('exit_fees', 0)) if position.get('exit_fees') is not None else 0
            total_fees = entry_fees + exit_fees

            if direction == 'short':
                gross_pnl = (entry_value - exit_value) * amount  # Inverted
            else:
                gross_pnl = (exit_value - entry_value) * amount  # Normal

            net_pnl = gross_pnl - total_fees

            # Determine outcome
            if net_pnl > 0:
                outcome = 'win'
            elif net_pnl < 0:
                outcome = 'loss'
            else:
                outcome = 'break_even'

            return PositionOutcome(
                outcome=outcome,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                fees=total_fees
            )

        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating position outcome: {e}")
            return PositionOutcome(outcome='error', gross_pnl=0.0, net_pnl=0.0, fees=0.0)

    def _calculate_trade_fees(self, position_data):
        """Calculate entry and exit fees from configuration"""
        transaction_config = self.config.get_section('transaction_costs', {})
        if 'commission_pct' not in transaction_config:
            raise ValueError("Missing required configuration: transaction_costs.commission_pct")

        commission_pct = transaction_config['commission_pct']

        entry_value = position_data['entry_value']
        amount = position_data['amount']
        entry_trade_value = entry_value * amount
        entry_fees = entry_trade_value * commission_pct

        if position_data.get('exit_value'):
            exit_value = position_data['exit_value']
            exit_trade_value = exit_value * amount
            exit_fees = exit_trade_value * commission_pct
        else:
            exit_fees = 0  # Open position

        return entry_fees, exit_fees

    def _validate_trade_structure(self, trade_data: Dict[str, Any]) -> bool:
        """Validate trade data structure

        Args:
            trade_data: Trade dictionary to validate

        Returns:
            True if structure is valid, False otherwise
        """
        if not self.validate_on_load:
            return True

        required_trade_fields = ['uuid', 'timestamp', 'status', 'positions']
        required_position_fields = ['name_of_position', 'amount', 'entry_value', 'currency', 'status']

        try:
            # Check trade fields
            for field in required_trade_fields:
                if field not in trade_data:
                    logger.warning(f"Missing required trade field: {field}")
                    return False

            # Check positions
            positions = trade_data.get('positions', [])
            if not isinstance(positions, list) or not positions:
                logger.warning("Trade must have at least one position")
                return False

            for position in positions:
                for field in required_position_fields:
                    if field not in position:
                        logger.warning(f"Missing required position field: {field}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade structure: {e}")
            return False

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Handle ISO format with 'Z' suffix
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

    @property
    def all_positions(self) -> List[Dict[str, Any]]:
        """All position records from all loaded trades"""
        all_positions = []
        for trade_data in self.trades.values():
            positions = trade_data.get('positions', [])
            all_positions.extend(positions)
        return all_positions