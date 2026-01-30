# spread_trade_history.py
import logging
from typing import Dict, Any, List

from src.hybrid.positions.base_trade_history import BaseTradeHistory, PositionOutcome

logger = logging.getLogger(__name__)


class SpreadTradeHistory(BaseTradeHistory):
    """Trade history for spread trades (triangular arbitrage, pairs, etc.)

    P&L is pre-calculated from spread convergence. No cost calculation needed.
    """

    def __init__(self, config):
        """Initialize SpreadTradeHistory

        Args:
            config: UnifiedConfig instance

        Raises:
            ValueError: If required configuration is missing
        """
        super().__init__(config)
        logger.info("SpreadTradeHistory initialized")

    def _validate_trade_structure(self, trade_data: Dict[str, Any]) -> bool:
        """Validate spread trade structure"""
        required = ['timestamp', 'entry_price', 'exit_price', 'quantity',
                    'direction', 'leg_trades', 'gross_pnl', 'status']

        try:
            for field in required:
                if field not in trade_data:
                    logger.warning(f"Spread trade missing required field: {field}")
                    return False

            leg_trades = trade_data.get('leg_trades')
            if not isinstance(leg_trades, list):
                logger.warning("Spread trade leg_trades must be a list")
                return False

            if len(leg_trades) == 0:
                logger.warning("Spread trade must have at least one leg_trade")
                return False

            status = trade_data.get('status')
            if status == 'closed':
                closed_required = ['entry_date', 'exit_date']
                for field in closed_required:
                    if field not in trade_data:
                        logger.warning(f"Closed spread trade missing required field: {field}")
                        return False
                    if trade_data[field] is None:
                        logger.warning(f"Closed spread trade has null {field}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Spread trade validation error: {e}")
            return False

    def _finalize_closed_trade(self, trade_data: Dict[str, Any]) -> None:
        """Finalize spread trade - P&L already calculated, just copy gross to net if missing"""
        if trade_data.get('status') != 'closed':
            return

        if 'net_pnl' not in trade_data:
            trade_data['net_pnl'] = trade_data.get('gross_pnl', 0.0)

    def _calculate_position_outcome(self, trade: Dict[str, Any]) -> PositionOutcome:
        """Calculate outcome for spread trade - no fees"""
        try:
            gross_pnl = trade.get('gross_pnl', 0.0)
            net_pnl = trade.get('net_pnl', gross_pnl)

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
                fees=0.0
            )

        except Exception as e:
            logger.error(f"Error calculating spread position outcome: {e}")
            return PositionOutcome(outcome='unknown')

    def add_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Add spread trade to history"""
        try:
            if self.validate_on_load and not self._validate_trade_structure(trade_data):
                logger.error("Invalid spread trade structure")
                return False

            timestamp = self._parse_timestamp(trade_data['timestamp'])
            if not timestamp:
                logger.error(f"Invalid timestamp in spread trade: {trade_data.get('uuid', 'unknown')}")
                return False

            if trade_data.get('status') == 'closed':
                self._finalize_closed_trade(trade_data)

            # Ensure trade_type is set
            trade_data['trade_type'] = 'spread'

            self.trades[timestamp] = trade_data
            self._invalidate_cache()

            logger.debug(f"Added spread trade: {trade_data.get('uuid', 'unknown')} at {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Error adding spread trade: {e}")
            return False

    def get_leg_trade_ids(self, spread_trade_id: str) -> List[str]:
        """Get leg trade IDs for a spread trade"""
        trade = self.get_trade_by_id(spread_trade_id)
        if trade:
            return trade.get('leg_trades', [])
        return []

    def get_spread_statistics(self) -> Dict[str, Any]:
        """Get spread-specific statistics"""
        stats = self.get_trade_statistics()

        closed = self._get_closed_positions()
        if not closed:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl_pips': 0.0,
                'long_spread_count': 0,
                'short_spread_count': 0
            }

        long_spreads = [t for t in closed if t.get('direction') == 'LONG_SPREAD']
        short_spreads = [t for t in closed if t.get('direction') == 'SHORT_SPREAD']

        total_pnl_pips = sum(t.get('pnl_pips', 0.0) for t in closed)
        avg_pnl_pips = total_pnl_pips / len(closed) if closed else 0.0

        return {
            'total_trades': stats.total_positions,
            'winning_trades': stats.winning_positions,
            'losing_trades': stats.losing_positions,
            'total_pnl': stats.total_pnl,
            'win_rate': stats.winning_positions / stats.total_positions if stats.total_positions > 0 else 0.0,
            'avg_pnl_pips': avg_pnl_pips,
            'long_spread_count': len(long_spreads),
            'short_spread_count': len(short_spreads)
        }