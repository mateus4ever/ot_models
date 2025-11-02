"""
RSISignal - Rule-based signal using Relative Strength Index
Simple RSI implementation for momentum confirmation
All parameters configurable - no hardcoded values
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from src.hybrid.signals.market_signal_enum import MarketSignal
from src.hybrid.signals.signal_interface import SignalInterface

logger = logging.getLogger(__name__)


class RSISignal(SignalInterface):
    """
    Simple RSI (Relative Strength Index) signal implementation

    Trading Logic:
    - BUY when RSI <= oversold threshold (typically 30)
    - SELL when RSI >= overbought threshold (typically 70)
    - HOLD when RSI is in neutral zone

    Simple discipline noob approach - classic overbought/oversold levels
    All parameters configurable through config object
    """

    def __init__(self, config):
        """
        Initialize RSI signal with configurable parameters

        Args:
            config: Configuration dictionary with signal parameters
                   Expected keys:
                   - period: RSI calculation period (default 14)
                   - oversold_threshold: RSI level for oversold condition (default 30)
                   - overbought_threshold: RSI level for overbought condition (default 70)
                   - buffer_multiplier: Historical data buffer size multiplier (default 3)
        """
        self.rsi_config = config.get_section('signals')['momentum']['rsi']

        # Load configurable parameters
        self.period = self.rsi_config['period']
        self.oversold_threshold = self.rsi_config['oversold_threshold']
        self.overbought_threshold = self.rsi_config['overbought_threshold']
        self.buffer_multiplier = self.rsi_config['buffer_multiplier']

        self.historical_data = pd.DataFrame()
        self.is_ready = False

        logger.debug(
            f"RSISignal initialized: period={self.period}, oversold={self.oversold_threshold}, overbought={self.overbought_threshold}")

    def train(self, training_data: pd.DataFrame) -> None:
        """
        Initialize with historical data - rule-based signal doesn't need training

        Args:
            training_data: Historical market data with 'close' column
        """
        if 'close' not in training_data.columns:
            raise ValueError("Training data must contain 'close' column")

        self.historical_data = training_data[['close']].copy()
        self.is_ready = len(self.historical_data) >= self.period + 1  # Need period + 1 for RSI calculation

        logger.info(f"RSISignal trained with {len(training_data)} data points")

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

        # Keep configurable buffer size
        buffer_size = self.period * self.buffer_multiplier
        if len(self.historical_data) > buffer_size:
            self.historical_data = self.historical_data.tail(buffer_size)

        self.is_ready = len(self.historical_data) >= self.period + 1

        logger.debug(f"Updated historical data, buffer size: {len(self.historical_data)}")

    def generate_signal(self) -> MarketSignal:
        """
        Generate RSI trading signal based on current historical data

        Returns:
            "BULLISH", "BEARISH", or "NEUTRAL"

        Raises:
            ValueError: If insufficient historical data
        """
        if not self.is_ready:
            raise ValueError(
                f"Insufficient historical data. Need {self.period + 1} points, have {len(self.historical_data)}")

        # Calculate RSI from historical data only
        rsi_value = self._calculate_rsi(self.historical_data['close'])

        # Generate signal based on RSI thresholds (product-agnostic)
        if rsi_value <= self.oversold_threshold:
            signal = MarketSignal.BULLISH
            logger.debug(f"BULLISH signal: RSI={rsi_value:.2f} <= oversold_threshold={self.oversold_threshold}")
        elif rsi_value >= self.overbought_threshold:
            signal = MarketSignal.BEARISH
            logger.debug(f"BEARISH signal: RSI={rsi_value:.2f} >= overbought_threshold={self.overbought_threshold}")
        else:
            signal = MarketSignal.NEUTRAL
            logger.debug(
                f"NEUTRAL signal: {self.oversold_threshold} < RSI={rsi_value:.2f} < {self.overbought_threshold}")

        return signal

    def _calculate_rsi(self, prices: pd.Series) -> float:
        """
        Calculate Relative Strength Index

        Args:
            prices: Series of closing prices

        Returns:
            RSI value (0-100)
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.rolling(window=self.period, min_periods=self.period).mean()
        avg_losses = losses.rolling(window=self.period, min_periods=self.period).mean()

        # Calculate relative strength and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Return the most recent RSI value
        return rsi.iloc[-1]

    def getMetrics(self) -> Dict[str, float]:
        """
        Return RSI specific metrics

        Returns:
            Dictionary with RSI value, momentum indicators, and threshold distances
        """
        if not self.is_ready:
            return {"error": -1.0, "message": "insufficient_data"}

        # Calculate current RSI
        all_prices = self.historical_data['close']
        current_rsi = self._calculate_rsi(all_prices)

        # Calculate useful metrics
        oversold_distance = abs(current_rsi - self.oversold_threshold)
        overbought_distance = abs(current_rsi - self.overbought_threshold)
        neutral_position = (current_rsi - self.oversold_threshold) / (
                    self.overbought_threshold - self.oversold_threshold)

        # Momentum indicators
        recent_prices = all_prices.tail(self.period)
        price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

        # RSI momentum (change in RSI over last few periods)
        if len(all_prices) >= self.period + 5:
            prev_rsi = self._calculate_rsi(all_prices.iloc[:-5])
            rsi_momentum = current_rsi - prev_rsi
        else:
            rsi_momentum = 0.0

        return {
            "rsi_value": float(current_rsi),
            "oversold_threshold": float(self.oversold_threshold),
            "overbought_threshold": float(self.overbought_threshold),
            "oversold_distance": float(oversold_distance),
            "overbought_distance": float(overbought_distance),
            "neutral_position": float(neutral_position),  # 0.0 = oversold, 1.0 = overbought
            "price_momentum": float(price_momentum),
            "rsi_momentum": float(rsi_momentum),
            "period": float(self.period),
            "is_oversold": current_rsi <= self.oversold_threshold,
            "is_overbought": current_rsi >= self.overbought_threshold
        }

    @staticmethod
    def get_required_parameters() -> list[str]:
        return ['period', 'oversold_threshold', 'overbought_threshold', 'buffer_mulitplier']
