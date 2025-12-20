"""
Predictor Interface - Common contract for all predictors
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
import pandas as pd


class PredictorInterface(ABC):
    """
    Base interface for all predictors.

    Predictors forecast market state (volatility regime, trend duration, etc.)
    Signals use predictions to decide BUY/SELL/HOLD.
    """

    @abstractmethod
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train predictor on historical data.

        Args:
            df: Historical market data

        Returns:
            Training metrics (accuracy, samples, etc.)
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.

        Args:
            df: Market data

        Returns:
            Tuple of (predictions, confidence)
        """
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether predictor is ready to predict"""
        pass