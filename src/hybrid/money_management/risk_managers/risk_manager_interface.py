import logging
import pandas as pd
from abc import ABC, abstractmethod

from src.hybrid.positions.types import TradingSignal, PortfolioState

logger = logging.getLogger(__name__)

class RiskManagementStrategy(ABC):
    """Abstract base class for risk management algorithms"""

    def __init__(self, config):
        """
        Initialize RiskManagementStrategy with configuration

        Args:
            config: UnifiedConfig instance
        """
        self.unified_config = config

        # Get money management configuration section
        mm_config = self.unified_config.get_section('money_management')
        if not mm_config:
            raise ValueError("money_management section not found in configuration")

        self.config = mm_config

    @abstractmethod
    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """Calculate stop loss price"""
        pass

    @abstractmethod
    def should_reduce_risk(self, portfolio: PortfolioState) -> bool:
        """Check if risk should be reduced based on portfolio state"""
        pass