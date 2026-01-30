# leg_trade_history.py
import logging
from typing import Dict, Any

from src.hybrid.costs.transaction_costs import SimpleTransactionCostModel
from src.hybrid.positions.base_trade_history import BaseTradeHistory
from src.hybrid.products.product_types import PositionDirection

logger = logging.getLogger(__name__)


class LegTradeHistory(BaseTradeHistory):
    """Trade history for single-leg trades (stocks, forex pairs, etc.)

    Handles cost calculation and P&L computation from entry/exit prices.
    """

    def __init__(self, config, cost_model=None):
        """Initialize LegTradeHistory

        Args:
            config: UnifiedConfig instance
            cost_model: Optional transaction cost model (creates default if not provided)

        Raises:
            ValueError: If required configuration is missing
        """
        super().__init__(config)

        if cost_model is None:
            self.cost_model = SimpleTransactionCostModel(config)
        else:
            self.cost_model = cost_model

        logger.info("LegTradeHistory initialized with cost model")

    def _validate_trade_structure(self, trade_data: Dict[str, Any]) -> bool:
        """Validate leg trade structure"""
        basic_required = ['timestamp', 'entry_price', 'quantity', 'direction', 'entry_date']

        try:
            for field in basic_required:
                if field not in trade_data:
                    logger.warning(f"Leg trade missing required field: {field}")
                    return False

            status = trade_data.get('status', 'closed')

            if status == 'closed':
                closed_required = ['exit_price', 'exit_date']
                for field in closed_required:
                    if field not in trade_data:
                        logger.warning(f"Closed leg trade missing required field: {field}")
                        return False
                    if trade_data[field] is None:
                        logger.warning(f"Closed leg trade has null {field}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Leg trade validation error: {e}")
            return False

    def _finalize_closed_trade(self, trade_data: Dict[str, Any]) -> None:
        """Calculate costs and P&L for closed leg trade"""
        if trade_data.get('status') != 'closed':
            return

        if 'costs' not in trade_data:
            trade_data['costs'] = self._calculate_trade_costs(trade_data)

        costs = trade_data['costs']
        total_costs = costs.get('total', 0.0) if isinstance(costs, dict) else costs

        if 'gross_pnl' not in trade_data:
            trade_data['gross_pnl'] = self._calculate_gross_pnl(trade_data)

        if 'net_pnl' not in trade_data:
            trade_data['net_pnl'] = trade_data['gross_pnl'] - total_costs

    def _calculate_entry_costs(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate costs for opening a position"""
        entry_price = trade_data.get('entry_price')
        quantity = trade_data.get('quantity')

        if entry_price is None:
            raise ValueError(f"Missing entry_price in trade_data: {trade_data.get('uuid')}")
        if quantity is None:
            raise ValueError(f"Missing quantity in trade_data: {trade_data.get('uuid')}")

        return self.cost_model.calculate_entry_costs(entry_price, quantity)

    def _calculate_trade_costs(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate costs for a trade using cost model"""
        entry_price = trade_data['entry_price']
        exit_price = trade_data['exit_price']
        quantity = trade_data['quantity']
        entry_date = trade_data['entry_date']
        exit_date = trade_data['exit_date']
        direction = trade_data['direction']

        if isinstance(entry_date, str):
            entry_date = self._parse_timestamp(entry_date)
        if isinstance(exit_date, str):
            exit_date = self._parse_timestamp(exit_date)

        days_held = (exit_date - entry_date).days

        is_short = self._is_short_direction(direction)

        return self.cost_model.calculate_total_trade_cost(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            days_held=days_held,
            is_short=is_short
        )

    def _calculate_gross_pnl(self, trade_data: Dict[str, Any]) -> float:
        """Calculate gross P&L from entry/exit prices"""
        entry_price = trade_data['entry_price']
        exit_price = trade_data['exit_price']
        quantity = trade_data['quantity']
        direction = trade_data['direction']

        is_short = self._is_short_direction(direction)

        if is_short:
            return (entry_price - exit_price) * quantity
        else:
            return (exit_price - entry_price) * quantity

    def _is_short_direction(self, direction) -> bool:
        """Check if direction is short"""
        if isinstance(direction, PositionDirection):
            return direction == PositionDirection.SHORT
        else:
            return str(direction).upper() == 'SHORT'

    def add_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Add leg trade with entry cost calculation for open trades"""
        try:
            if self.validate_on_load and not self._validate_trade_structure(trade_data):
                logger.error("Invalid trade structure")
                return False

            timestamp = self._parse_timestamp(trade_data['timestamp'])
            if not timestamp:
                logger.error(f"Invalid timestamp in trade: {trade_data.get('uuid', 'unknown')}")
                return False

            status = trade_data.get('status')

            if status == 'open':
                if 'entry_costs' not in trade_data:
                    trade_data['entry_costs'] = self._calculate_entry_costs(trade_data)

            if status == 'closed':
                self._finalize_closed_trade(trade_data)

            self.trades[timestamp] = trade_data
            self._invalidate_cache()

            logger.debug(f"Added leg trade: {trade_data.get('uuid', 'unknown')} at {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Error adding leg trade: {e}")
            return False