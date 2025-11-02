"""
Transaction cost model for backtesting
Calculates entry, exit, and holding costs
"""

from typing import Dict


class SimpleTransactionCostModel:
    """
    Approximate transaction costs for backtesting
    Configuration-driven, no hardcoded values
    """

    def __init__(self, config):
        """
        Initialize cost model from configuration

        Args:
            config: UnifiedConfig object

        Raises:
            ValueError: If required cost parameters missing
        """
        # Extract cost config from UnifiedConfig
        cost_config = config.get_section('transaction_costs')

        if not cost_config:
            raise ValueError("Missing 'transaction_costs' section in configuration")

        # Get values from config - FAIL HARD if missing
        self.commission_per_trade = cost_config.get('commission_per_trade')
        self.slippage_pct = cost_config.get('slippage_pct')
        self.spread_pct = cost_config.get('spread_pct')
        self.short_borrow_rate_annual = cost_config.get('short_borrow_rate')

        # Validate all required fields present
        if self.commission_per_trade is None:
            raise ValueError("Missing 'commission_per_trade' in transaction_costs configuration")
        if self.slippage_pct is None:
            raise ValueError("Missing 'slippage_pct' in transaction_costs configuration")
        if self.spread_pct is None:
            raise ValueError("Missing 'spread_pct' in transaction_costs configuration")
        if self.short_borrow_rate_annual is None:
            raise ValueError("Missing 'short_borrow_rate' in transaction_costs configuration")

    def calculate_entry_costs(self, entry_price: float, quantity: float) -> Dict[str, float]:
        """
        Calculate costs to enter position

        Args:
            entry_price: Entry price per unit
            quantity: Number of units

        Returns:
            Dictionary with cost breakdown
        """
        position_value = entry_price * quantity

        commission = self.commission_per_trade
        slippage = position_value * self.slippage_pct
        spread = position_value * self.spread_pct

        return {
            'commission': commission,
            'slippage': slippage,
            'spread': spread,
            'total': commission + slippage + spread
        }

    def calculate_exit_costs(self, exit_price: float, quantity: float) -> Dict[str, float]:
        """
        Calculate costs to exit position

        Args:
            exit_price: Exit price per unit
            quantity: Number of units

        Returns:
            Dictionary with cost breakdown
        """
        position_value = exit_price * quantity

        commission = self.commission_per_trade
        slippage = position_value * self.slippage_pct
        spread = position_value * self.spread_pct

        return {
            'commission': commission,
            'slippage': slippage,
            'spread': spread,
            'total': commission + slippage + spread
        }

    def calculate_holding_costs(self, position_value: float, days_held: int,
                                is_short: bool) -> float:
        """
        Calculate costs to hold position over time

        Args:
            position_value: Dollar value of position
            days_held: Number of days position held
            is_short: True if short position

        Returns:
            Total holding cost (positive number = cost)
        """
        if not is_short:
            return 0.0

        # Short borrow costs (daily accrual)
        daily_rate = self.short_borrow_rate_annual / 365.0
        return position_value * daily_rate * days_held

    def calculate_total_trade_cost(self, entry_price: float, exit_price: float,
                                   quantity: float, days_held: int,
                                   is_short: bool) -> Dict[str, float]:
        """
        Calculate total costs for complete trade lifecycle

        Args:
            entry_price: Entry price per unit
            exit_price: Exit price per unit
            quantity: Number of units
            days_held: Number of days position held
            is_short: True if short position

        Returns:
            Dictionary with complete cost breakdown
        """
        entry_costs = self.calculate_entry_costs(entry_price, quantity)
        exit_costs = self.calculate_exit_costs(exit_price, quantity)

        # Use average position value for holding costs
        avg_position_value = ((entry_price + exit_price) / 2.0) * quantity
        holding_costs = self.calculate_holding_costs(avg_position_value, days_held, is_short)

        total = entry_costs['total'] + exit_costs['total'] + holding_costs

        return {
            'entry': entry_costs['total'],
            'exit': exit_costs['total'],
            'holding': holding_costs,
            'total': total,
            'breakdown': {
                'commission': entry_costs['commission'] + exit_costs['commission'],
                'slippage': entry_costs['slippage'] + exit_costs['slippage'],
                'spread': entry_costs['spread'] + exit_costs['spread'],
                'borrow': holding_costs
            }
        }

    def update_costs(self, new_cost_config: dict) -> None:
        """
        Update cost parameters (for time-varying costs)

        Args:
            new_cost_config: New transaction_costs configuration
        """
        if 'commission_per_trade' in new_cost_config:
            self.commission_per_trade = new_cost_config['commission_per_trade']
        if 'slippage_pct' in new_cost_config:
            self.slippage_pct = new_cost_config['slippage_pct']
        if 'spread_pct' in new_cost_config:
            self.spread_pct = new_cost_config['spread_pct']
        if 'short_borrow_rate' in new_cost_config:
            self.short_borrow_rate_annual = new_cost_config['short_borrow_rate']