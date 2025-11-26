# trade_history.py
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from sortedcontainers import SortedDict

from src.hybrid.costs.transaction_costs import SimpleTransactionCostModel
from src.hybrid.products.product_types import PositionDirection

logger = logging.getLogger(__name__)


class PositionOutcome:
    """Represents the outcome of a single position"""

    def __init__(self, outcome: str = 'unknown', gross_pnl: float = 0.0,
                 net_pnl: float = 0.0, fees: float = 0.0, cost_model=None):
        """
        Initialize TradeHistory

        Args:
            config: Configuration dictionary
            cost_model: Optional external cost model (for shared instance)
        """


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

    def __init__(self, config, cost_model=None):
        """Initialize TradeHistory with configuration and cost_model

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
        self.default_lookback_periods = trade_config.get('default_lookback_periods')
        self.save_backup_on_load = trade_config.get('save_backup_on_load')
        self.validate_on_load = trade_config.get('validate_on_load')
        self.break_even_tolerance = trade_config.get('break_even_tolerance')

        # Timestamp-ordered trade storage
        self.trades = SortedDict()  # key: timestamp, value: trade_data

        # Statistics cache
        self._stats_cache = {}
        self._cache_valid = False

        self.cost_model = cost_model if cost_model else SimpleTransactionCostModel(config)

        logger.info(f"TradeHistory initialized with base_currency: {self.base_currency}")
        logger.debug(f"Configuration: lookback={self.default_lookback_periods}, "
                     f"backup={self.save_backup_on_load}, validate={self.validate_on_load}")

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

            # CHANGED: Use add_trade() instead of direct storage
            loaded_count = 0
            for trade_data in trades_data:
                logger.info(f"Attempting to add trade: {trade_data.get('uuid', 'unknown')}")
                result = self.add_trade(trade_data)
                logger.info(f"Result: {result}")
                if result:
                    loaded_count += 1
                else:
                    logger.error(f"Failed to add trade: {trade_data.get('uuid', 'unknown')}")
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
        """
        Add new trade to history with automatic cost calculation
        """
        try:
            if not self._validate_trade_structure(trade_data):
                logger.error("Invalid trade structure")
                return False

            timestamp = self._parse_timestamp(trade_data['timestamp'])
            if not timestamp:
                logger.error(f"Invalid timestamp in trade: {trade_data.get('uuid', 'unknown')}")
                return False

            status = trade_data.get('status')
            # Check if trade is open
            if status == 'open':
                if 'entry_costs' not in trade_data:
                    trade_data['entry_costs'] = self._calculate_entry_costs(trade_data)

            # For closed trades, finalize all costs
            self._finalize_closed_trade(trade_data)

            self.trades[timestamp] = trade_data
            self._invalidate_cache()

            logger.debug(f"Added trade: {trade_data.get('uuid', 'unknown')} at {timestamp}, status: {status}")
            return True

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False

    def _calculate_entry_costs(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate costs for opening a position"""
        if not self.cost_model:
            raise ValueError("Cost model not set - cannot calculate entry costs")

        entry_price = trade_data.get('entry_price')
        quantity = trade_data.get('quantity')

        if entry_price is None or quantity is None:
            raise ValueError(f"Missing entry_price or quantity in trade_data: {trade_data.get('uuid')}")

        return self.cost_model.calculate_entry_costs(entry_price, quantity)
    def _finalize_closed_trade(self, trade_data: Dict[str, Any]) -> None:
        """Calculate costs and P&L for closed trade"""
        if trade_data.get('status') != 'closed':
            return

        # Costs may already be set (from broker in production)
        if 'costs' not in trade_data:
            costs = self._calculate_trade_costs(trade_data)
            trade_data['costs'] = costs

        costs = trade_data['costs']
        total_costs = costs.get('total') if isinstance(costs, dict) else costs

        # Calculate P&L
        if 'gross_pnl' not in trade_data:
            trade_data['gross_pnl'] = self._calculate_gross_pnl(trade_data)

        if 'net_pnl' not in trade_data:
            trade_data['net_pnl'] = trade_data['gross_pnl'] - total_costs
    def _calculate_trade_costs(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate costs for a trade using cost model

        Args:
            trade_data: Trade information

        Returns:
            Cost breakdown dictionary
        """
        # Extract required fields
        entry_price = trade_data['entry_price']
        exit_price = trade_data['exit_price']
        quantity = trade_data['quantity']
        entry_date = trade_data['entry_date']
        exit_date = trade_data['exit_date']
        direction = trade_data['direction']

        # Parse dates if they're strings
        if isinstance(entry_date, str):
            entry_date = self._parse_timestamp(entry_date)
        if isinstance(exit_date, str):
            exit_date = self._parse_timestamp(exit_date)

        # Calculate days held
        days_held = (exit_date - entry_date).days

        # Determine if short position
        is_short = (direction == PositionDirection.SHORT)

        # Use cost model to ca0lculate
        costs = self.cost_model.calculate_total_trade_cost(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            days_held=days_held,
            is_short=is_short
        )

        return costs

    def _calculate_gross_pnl(self, trade_data: Dict[str, Any]) -> float:
        """
        Calculate gross P&L (before costs)

        Args:
            trade_data: Trade information

        Returns:
            Gross profit/loss
        """
        entry_price = trade_data['entry_price']
        exit_price = trade_data['exit_price']
        quantity = trade_data['quantity']
        direction = trade_data['direction']

        # Determine if short position
        is_short = (direction == PositionDirection.SHORT)

        if is_short:
            # Short: profit when price goes down
            gross_pnl = (entry_price - exit_price) * quantity
        else:
            # Long: profit when price goes up
            gross_pnl = (exit_price - entry_price) * quantity

        return gross_pnl

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

    @property
    def all_positions(self) -> List[Dict[str, Any]]:
        """Get all positions/trades as a list

        Returns:
            List of all trade dictionaries
        """
        return list(self.trades.values())

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades"""
        open_trades = []
        for trade in self.trades.values():
            if trade.get('status') == 'open':
                open_trades.append(trade)
        return open_trades

    def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get trade by uuid or trade_id"""
        for trade in self.trades.values():
            if trade.get('uuid') == trade_id or trade.get('trade_id') == trade_id:
                return trade
        return None

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        for timestamp, trade in self.trades.items():
            if trade.get('uuid') == trade_id or trade.get('trade_id') == trade_id:
                trade.update(updates)
                self._finalize_closed_trade(trade)
                self._invalidate_cache()
                logger.debug(f"Updated trade {trade_id}")
                return True
        logger.warning(f"Trade {trade_id} not found for update")
        return False

    def _get_closed_positions(self, lookback_periods: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get closed trades for statistics calculation

        Args:
            lookback_periods: Number of recent trades to include (0 = all)

        Returns:
            List of closed trade dictionaries
        """
        closed_trades = []

        # Get all trades in reverse chronological order for lookback
        trades_list = list(self.trades.values())
        trades_list.reverse()  # Most recent first

        trades_collected = 0
        max_trades = lookback_periods if lookback_periods and lookback_periods > 0 else float('inf')

        for trade in trades_list:
            if trades_collected >= max_trades:
                break

            # Only include closed trades
            status = trade.get('status', 'closed')
            if status == 'closed' and trade.get('exit_price') is not None:
                closed_trades.append(trade)
                trades_collected += 1

        # Reverse to chronological order for consistent processing
        closed_trades.reverse()
        return closed_trades

    def _calculate_position_outcome(self, trade: Dict[str, Any]) -> PositionOutcome:
        """Calculate outcome for a trade (new flat format)"""
        try:
            # Get values from new format
            direction = trade.get('direction')
            entry_price = float(trade.get('entry_price'))
            exit_price = float(trade.get('exit_price'))
            quantity = float(trade.get('quantity'))

            # Get calculated values if they exist
            gross_pnl = trade.get('gross_pnl')
            net_pnl = trade.get('net_pnl')
            costs = trade.get('costs', {})
            total_fees = costs.get('total') if costs else 0

            # Calculate if not already present
            if gross_pnl is None:
                if direction == PositionDirection.SHORT:
                    gross_pnl = (entry_price - exit_price) * quantity
                else:
                    gross_pnl = (exit_price - entry_price) * quantity

            if net_pnl is None:
                net_pnl = gross_pnl - total_fees

            # Determine outcome with tolerance for break-even
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

        except Exception as e:
            logger.error(f"Error calculating position outcome: {e}")
            return PositionOutcome(outcome='unknown', gross_pnl=0.0, net_pnl=0.0, fees=0.0)
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating position outcome: {e}")
            return PositionOutcome(outcome='error', gross_pnl=0.0, net_pnl=0.0, fees=0.0)

    def _validate_trade_structure(self, trade_data: Dict[str, Any]) -> bool:
        """Validate trade data structure

        Args:
            trade_data: Trade dictionary to validate

        Returns:
            True if structure is valid, False otherwise
        """
        if not self.validate_on_load:
            return True

        # Basic required fields for all trades
        basic_required = ['timestamp', 'entry_price', 'quantity',
                          'direction', 'entry_date']

        try:
            for field in basic_required:
                if field not in trade_data:
                    logger.warning(f"Missing required field: {field}")
                    return False

            # Check if closed trade
            status = trade_data.get('status', 'closed')

            if status == 'closed':
                # Closed trades need exit fields
                closed_required = ['exit_price', 'exit_date']
                for field in closed_required:
                    if field not in trade_data:
                        logger.warning(f"Closed trade missing required field: {field}")
                        return False
                    if trade_data[field] is None:
                        logger.warning(f"Closed trade has null {field}")
                        return False

            # Open trades can have null exit fields

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
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
