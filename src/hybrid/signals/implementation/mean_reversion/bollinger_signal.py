"""
BollingerSignal - Rule-based signal using Bollinger Bands
Simple implementation for BaseStrategy - discipline noob benchmark
All parameters configurable - no hardcoded values
"""

import pandas as pd
from typing import Dict, Any
import logging
from src.hybrid.signals.signal_interface import SignalInterface

logger = logging.getLogger(__name__)


class BollingerSignal(SignalInterface):
    """
    Simple Bollinger Bands signal implementation

    Trading Logic:
    - BUY when price touches or goes below lower Bollinger band
    - SELL when price touches or goes above upper Bollinger band
    - HOLD otherwise

    Simple discipline noob approach - no complex filtering or confirmations
    All parameters configurable through config object
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Bollinger Bands signal with configurable parameters

        Args:
            config: Configuration dictionary with signal parameters
                   Expected keys:
                   - period: Moving average period (default from config)
                   - std_dev: Standard deviation multiplier (default from config)
                   - buffer_multiplier: Historical data buffer size multiplier (default from config)
        """
        config = config or {}

        # Load configurable parameters
        self.period = config.get('period', 20)  # TODO: Load from unified config system
        self.std_dev = config.get('std_dev', 2.0)  # TODO: Load from unified config system
        self.buffer_multiplier = config.get('buffer_multiplier', 2)  # TODO: Load from unified config system

        self.historical_data = pd.DataFrame()
        self.is_ready = False

        logger.debug(
            f"BollingerSignal initialized: period={self.period}, std_dev={self.std_dev}, buffer_multiplier={self.buffer_multiplier}")

    def train(self, training_data: pd.DataFrame) -> None:
        """
        Initialize with historical data - rule-based signal doesn't need training

        Args:
            training_data: Historical market data with 'close' column
        """
        if 'close' not in training_data.columns:
            raise ValueError("Training data must contain 'close' column")

        self.historical_data = training_data[['close']].copy()
        self.is_ready = len(self.historical_data) >= self.period

        logger.info(f"BollingerSignal trained with {len(training_data)} data points")

    def update_with_new_data(self, data_point: pd.Series) -> None:
        """
        Add new data point to historical buffer

        Args:
            data_point: New market data point with 'close' price
        """
        if 'close' not in data_point:
            raise ValueError("Data point must contain 'close' price")

        # Add new data point to historical buffer
        new_row = pd.DataFrame({'close': [data_point['close']]},
                               index=[data_point.name])
        self.historical_data = pd.concat([self.historical_data, new_row])

        # Keep only necessary data (configurable buffer size)
        buffer_size = self.period * self.buffer_multiplier
        if len(self.historical_data) > buffer_size:
            self.historical_data = self.historical_data.tail(buffer_size)

        self.is_ready = len(self.historical_data) >= self.period

        logger.debug(f"Updated historical data, buffer size: {len(self.historical_data)}")

    def generate_signal(self, current_data: pd.Series) -> str:
        """
        Generate Bollinger Bands trading signal

        Args:
            current_data: Current market data point with 'close' price

        Returns:
            "BUY", "SELL", or "HOLD"

        Raises:
            ValueError: If insufficient historical data or missing 'close' price
        """
        if not self.is_ready:
            raise ValueError(
                f"Insufficient historical data. Need {self.period} points, have {len(self.historical_data)}")

        if 'close' not in current_data:
            raise ValueError("Current data must contain 'close' price")

        current_price = current_data['close']

        # Calculate Bollinger Bands using historical data
        recent_prices = self.historical_data['close'].tail(self.period)

        # Moving average (center line)
        sma = recent_prices.mean()

        # Standard deviation
        std = recent_prices.std()

        # Bollinger Bands
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)

        # Generate signal based on current price vs bands
        if current_price <= lower_band:
            signal = "BUY"
            logger.debug(f"BUY signal: price={current_price:.4f} <= lower_band={lower_band:.4f}")
        elif current_price >= upper_band:
            signal = "SELL"
            logger.debug(f"SELL signal: price={current_price:.4f} >= upper_band={upper_band:.4f}")
        else:
            signal = "HOLD"
            logger.debug(f"HOLD signal: {lower_band:.4f} < price={current_price:.4f} < {upper_band:.4f}")

        return signal

    def getMetrics(self) -> Dict[str, float]:
        """
        Return Bollinger Bands specific metrics

        Returns:
            Dictionary with band positions, volatility, and band width
        """
        if not self.is_ready:
            return {"error": -1.0, "message": "insufficient_data"}

        # Calculate current Bollinger Bands
        recent_prices = self.historical_data['close'].tail(self.period)
        current_price = recent_prices.iloc[-1]

        sma = recent_prices.mean()
        std = recent_prices.std()

        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)

        # Calculate useful metrics
        band_width = (upper_band - lower_band) / sma  # Normalized band width
        price_position = (current_price - lower_band) / (upper_band - lower_band)  # 0.0-1.0 position
        distance_to_upper = (upper_band - current_price) / current_price
        distance_to_lower = (current_price - lower_band) / current_price
        volatility = std / sma  # Normalized volatility

        return {
            "sma": float(sma),
            "upper_band": float(upper_band),
            "lower_band": float(lower_band),
            "current_price": float(current_price),
            "band_width": float(band_width),
            "price_position": float(price_position),  # 0.0 = at lower band, 1.0 = at upper band
            "distance_to_upper": float(distance_to_upper),
            "distance_to_lower": float(distance_to_lower),
            "volatility": float(volatility),
            "period": float(self.period),
            "std_dev_multiplier": float(self.std_dev)
        }

    @staticmethod
    def get_required_parameters() -> list[str]:
        return ['period', 'std_dev', 'buffer_multiplier']