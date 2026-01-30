# position_sizers/margin_based_sizer.py
import logging
from typing import TYPE_CHECKING

from .sizer_interface import PositionSizingStrategy

if TYPE_CHECKING:
    from src.hybrid.positions.types import TradingSignal, PortfolioState

logger = logging.getLogger(__name__)


class MarginBasedSizer(PositionSizingStrategy):
    """
    Margin-based position sizing for forex/triangular arbitrage

    Returns position size in MICRO LOTS (1 micro lot = 1,000 units)
    - 100 micro lots = 1 standard lot (100,000 units)
    - 50 micro lots = 0.5 standard lots (50,000 units)
    """

    def __init__(self, config):
        super().__init__(config)

        params = self.config['parameters']

        self.margin_per_lot = params['margin_per_lot']
        self.max_margin_usage = params['max_margin_usage']
        self.max_lots = params['max_lots']

        logger.info(f"MarginBasedSizer initialized: margin_per_lot={self.margin_per_lot}, "
                    f"max_margin_usage={self.max_margin_usage}, max_lots={self.max_lots}")

    def calculate_size(self, signal: 'TradingSignal', portfolio: 'PortfolioState',
                       stop_distance: float) -> int:
        """
        Calculate position size based on margin requirements

        Returns:
            Position size in MICRO LOTS (int)
            100 = 1.0 standard lot, 50 = 0.5 standard lots
        """
        available_capital = portfolio.available_cash

        max_lot = (available_capital * self.max_margin_usage) / self.margin_per_lot
        lot_size = min(self.max_lots, max_lot)

        # Convert to micro lots (1 standard lot = 100 micro lots)
        micro_lots = int(lot_size * 100)

        logger.debug(f"MarginBasedSizer: capital={available_capital}, "
                     f"lot_size={lot_size:.2f}, micro_lots={micro_lots}")

        return max(0, micro_lots)

    def get_strategy_name(self) -> str:
        return "margin_based"