# money_management.py
# MoneyManager for capital allocation and position sizing
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class MoneyManager:
    """Manages position sizing, risk management, and capital allocation"""

    def __init__(self, config):
        self.config = config
        self.initial_capital = self._get_initial_capital()
        self.risk_config = self.config.get_section('risk_management', {})
        self.allocated_capital = {}

    def _get_initial_capital(self):
        """Get initial capital from config"""
        backtest_config = self.config.get_section('backtesting', {})
        return backtest_config.get('initial_capital', 10000)

    def allocate_capital(self, strategies):
        """Allocate capital among strategies"""
        # TODO: Implement capital allocation logic
        pass

    def calculate_position_size(self, strategy_name: str, signal: Dict) -> float:
        """Calculate position size for a trading signal"""
        # TODO: Implement position sizing logic
        pass

    def get_capital_allocation(self, strategy_name: str) -> float:
        """Get allocated capital for a strategy"""
        return self.allocated_capital.get(strategy_name, 0)