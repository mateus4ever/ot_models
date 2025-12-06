# position_sizers/fixed_fractional_sizer.py
import logging
from typing import TYPE_CHECKING
from .sizer_interface import PositionSizingStrategy

if TYPE_CHECKING:
    from src.hybrid.positions.types import TradingSignal, PortfolioState

logger = logging.getLogger(__name__)


class FixedFractionalSizer(PositionSizingStrategy):
    """Fixed fractional position sizing - risk fixed percentage per trade"""

    def __init__(self, config):
        """
        Initialize FixedFractionalSizer with configuration

        Args:
            config: UnifiedConfig instance
        """
        super().__init__(config)
        mm_config = self.unified_config.get_section('money_management')
        sizing_type = mm_config['position_sizing']  # Gets "fixed_fractional"
        sizer_config = mm_config['position_sizers'][sizing_type]  # Gets the nested section

        # TODO: check how to insert optimized parameters
        self.risk_per_trade = sizer_config['parameters']['risk_per_trade']
        self.max_position_size = sizer_config['parameters']['max_position_pct']

    def calculate_size(self, signal: 'TradingSignal', portfolio: 'PortfolioState',
                       stop_distance: float) -> int:
        risk_amount = portfolio.total_equity * self.risk_per_trade

        if stop_distance <= 0:
            return 0

        # Position size based on risk
        position_value = risk_amount / (stop_distance / signal.entry_price)
        position_size = int(position_value / signal.entry_price)

        # Apply maximum position size constraint
        max_position_value = portfolio.total_equity * self.max_position_size
        max_shares = int(max_position_value / signal.entry_price)

        return min(position_size, max_shares)


    def get_strategy_name(self) -> str:
        return "FixedFractional"