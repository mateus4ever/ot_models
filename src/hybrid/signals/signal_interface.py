"""
SignalInterface Protocol for all trading signal implementations
Defines contract for both rule-based and ML-based signals
"""

from typing import Protocol, Dict, Any
import pandas as pd

from src.hybrid.signals.market_signal_enum import MarketSignal


class SignalInterface(Protocol):
    """
    Protocol defining interface for all trading signal generators

    Supports both rule-based signals (Bollinger, RSI, MA crossover) and
    ML-based signals (neural networks, decision trees) with unified interface
    """

    def train(self, training_data: pd.DataFrame) -> None:
        """
        Initial training on historical data

        Args:
            training_data: Historical market data for training

        Note:
            - Rule-based signals may ignore this method
            - ML signals use this for initial model training
            - Should prevent data leakage by only using historical data
        """
        ...

    def update_with_new_data(self, data_point: pd.Series) -> None:
        """
        Update signal with new data point

        Args:
            data_point: New market data point to incorporate

        Behavior:
            - Adds data to internal historical buffer
            - Updates/retrains model if needed (ML signals)
            - Updates indicators (rule-based signals)
        """
        ...

    def generate_signal(self) -> MarketSignal:
        """
        Generate trading signal based on current market data

        Args:
            current_data: Current market data point

        Returns:
            Trading signal: "BUY", "SELL", or "HOLD"

        Note:
            - Should only use current_data and internal historical buffer
            - Must not peek at future data to prevent leakage
            - Should raise exception if not properly trained (ML signals)
        """
        ...

    def getMetrics(self) -> Dict[str, float]:
        """
        Return signal-specific metrics and measurements

        Returns:
            Dictionary of signal-specific metrics

        Examples:
            BollingerSignal: {"band_distance": 0.02, "volatility": 0.15}
            RSISignal: {"rsi_value": 28.5, "oversold_degree": 0.8}
            MLSignal: {"prediction_probability": 0.73, "uncertainty": 0.12}

        Note:
            - Metrics should be informative for strategy decision-making
            - No standardized metrics - each signal returns relevant data
            - Strategy can use metrics for position sizing, filtering, etc.
        """
        ...

    @staticmethod
    def get_required_parameters(self) -> list[str]:
        "Returns a list of required paramenters"