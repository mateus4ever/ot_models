"""
Predictor Interface - Common contract for all predictors
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class PredictorInterface(ABC):
    """
    Base interface for all predictors.

    Predictors forecast market state (volatility regime, trend duration, spread signals, etc.)
    """

    @abstractmethod
    def train(self, data: Any) -> Dict:
        """
        Train predictor on historical data.

        Args:
            data: Historical market data (DataFrame or DataManager)

        Returns:
            Training metrics dict. Should include 'success': bool for validation.
        """
        pass

    @abstractmethod
    def predict(self, data: Any) -> Dict:
        """
        Generate prediction.

        Args:
            data: Market data (DataFrame or DataManager)

        Returns:
            Prediction dict (structure varies by predictor type)
        """
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether predictor is ready to predict"""
        pass

    def get_required_markets(self) -> List[str]:
        """
        Markets required by this predictor.

        Override for multi-market predictors (e.g., triangular arbitrage).
        Single-market predictors return empty list.
        """
        return []