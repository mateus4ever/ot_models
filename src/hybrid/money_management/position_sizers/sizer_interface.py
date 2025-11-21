# position_sizers/sizer_interface.py
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.hybrid.positions.types import TradingSignal, PortfolioState

logger = logging.getLogger(__name__)


class PositionSizingStrategy(ABC):
    """Abstract base class for position sizing algorithms"""

    def __init__(self, config):
        self.unified_config = config
        mm_config = self.unified_config.get_section('money_management')
        if not mm_config:
            raise ValueError("money_management section not found in configuration")

        # Set self.config to the strategy-specific section
        sizing_type = mm_config['position_sizing']
        self.config = mm_config['position_sizers'][sizing_type]

    @abstractmethod
    def calculate_size(self, signal: 'TradingSignal', portfolio: 'PortfolioState',
                       stop_distance: float) -> int:
        """Calculate position size for given signal and stop distance"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging"""
        pass

    def update_trade_result(self, trade_result: Any) -> None:
        """
        Update sizer with completed trade result for dynamic statistics

        Override this method for sizers that need trade history (e.g., Kelly Criterion)
        Default implementation does nothing (for static sizers)

        Args:
            trade_result: Completed trade result with win/loss information
        """
        pass

    def update_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Update sizer with recent market data for volatility calculations

        Override this method for sizers that need market volatility (e.g., Volatility-based)
        Default implementation does nothing (for static sizers)

        Args:
            market_data: Recent price data for volatility calculations
        """
        pass