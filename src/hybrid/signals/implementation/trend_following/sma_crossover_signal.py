"""
SimpleMovingAverageCrossover - Rule-based signal using SMA crossover
Classic trend-following signal for direction confirmation
All parameters configurable - no hardcoded values
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from src.hybrid.signals.signal_interface import SignalInterface

logger = logging.getLogger(__name__)


class SimpleMovingAverageCrossover(SignalInterface):
    """
    Simple Moving Average Crossover signal implementation

    Trading Logic:
    - BUY when fast SMA crosses above slow SMA (golden cross)
    - SELL when fast SMA crosses below slow SMA (death cross)
    - HOLD when no crossover occurs

    Classic trend-following approach for discipline noobs
    All parameters configurable through config object
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SMA Crossover signal with configurable parameters

        Args:
            config: Configuration dictionary with signal parameters
                   Expected keys:
                   - fast_period: Fast moving average period (default 10)
                   - slow_period: Slow moving average period (default 30)
                   - buffer_multiplier: Historical data buffer size multiplier (default 3)
                   - crossover_confirmation: Periods to confirm crossover (default 1)
        """
        config = config or {}

        # Load configurable parameters
        self.fast_period = config.get('fast_period', 10)  # TODO: Load from unified config system
        self.slow_period = config.get('slow_period', 30)
        self.buffer_multiplier = config.get('buffer_multiplier', 3)
        self.crossover_confirmation = config.get('crossover_confirmation', 1)

        # Validation
        if self.fast_period >= self.slow_period:
            raise ValueError(f"Fast period ({self.fast_period}) must be less than slow period ({self.slow_period})")

        self.historical_data = pd.DataFrame()
        self.is_ready = False
        self.last_crossover_signal = "HOLD"

        logger.debug(
            f"SMASignal initialized: fast={self.fast_period}, slow={self.slow_period}, confirmation={self.crossover_confirmation}")

    def train(self, training_data: pd.DataFrame) -> None:
        """
        Initialize with historical data - rule-based signal doesn't need training

        Args:
            training_data: Historical market data with 'close' column
        """
        if 'close' not in training_data.columns:
            raise ValueError("Training data must contain 'close' column")

        self.historical_data = training_data[['close']].copy()
        self.is_ready = len(self.historical_data) >= self.slow_period + self.crossover_confirmation

        logger.info(f"SMASignal trained with {len(training_data)} data points")

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
        buffer_size = self.slow_period * self.buffer_multiplier
        if len(self.historical_data) > buffer_size:
            self.historical_data = self.historical_data.tail(buffer_size)

        self.is_ready = len(self.historical_data) >= self.slow_period + self.crossover_confirmation

        logger.debug(f"Updated historical data, buffer size: {len(self.historical_data)}")

    def generate_signal(self, current_data: pd.Series) -> str:
        """
        Generate SMA crossover trading signal

        Args:
            current_data: Current market data point with 'close' price

        Returns:
            "BUY", "SELL", or "HOLD"

        Raises:
            ValueError: If insufficient historical data or missing 'close' price
        """
        if not self.is_ready:
            required_points = self.slow_period + self.crossover_confirmation
            raise ValueError(
                f"Insufficient historical data. Need {required_points} points, have {len(self.historical_data)}")

        if 'close' not in current_data:
            raise ValueError("Current data must contain 'close' price")

        # Calculate moving averages using historical data + current price
        current_price = current_data['close']
        all_prices = pd.concat([self.historical_data['close'], pd.Series([current_price])])

        fast_sma = all_prices.rolling(window=self.fast_period).mean()
        slow_sma = all_prices.rolling(window=self.slow_period).mean()

        # Check for crossover
        signal = self._detect_crossover(fast_sma, slow_sma)

        logger.debug(f"SMA Signal: {signal}, Fast SMA: {fast_sma.iloc[-1]:.4f}, Slow SMA: {slow_sma.iloc[-1]:.4f}")
        return signal

    def _detect_crossover(self, fast_sma: pd.Series, slow_sma: pd.Series) -> str:
        """
        Detect crossover events between fast and slow moving averages

        Args:
            fast_sma: Fast moving average series
            slow_sma: Slow moving average series

        Returns:
            Signal based on crossover detection
        """
        # Need at least confirmation periods + 1 to detect crossover
        if len(fast_sma) < self.crossover_confirmation + 1:
            return "HOLD"

        # Current and previous values
        current_fast = fast_sma.iloc[-1]
        current_slow = slow_sma.iloc[-1]
        prev_fast = fast_sma.iloc[-(1 + self.crossover_confirmation)]
        prev_slow = slow_sma.iloc[-(1 + self.crossover_confirmation)]

        # Detect crossover with confirmation
        if prev_fast <= prev_slow and current_fast > current_slow:
            # Golden cross - fast MA crossed above slow MA
            signal = "BUY"
            self.last_crossover_signal = signal
            logger.debug(f"Golden cross detected: fast crossed above slow")
        elif prev_fast >= prev_slow and current_fast < current_slow:
            # Death cross - fast MA crossed below slow MA
            signal = "SELL"
            self.last_crossover_signal = signal
            logger.debug(f"Death cross detected: fast crossed below slow")
        else:
            # No crossover - maintain trend direction or hold
            if current_fast > current_slow:
                signal = "HOLD"  # Uptrend but no fresh signal
            elif current_fast < current_slow:
                signal = "HOLD"  # Downtrend but no fresh signal
            else:
                signal = "HOLD"  # MAs are equal

        return signal

    def getMetrics(self) -> Dict[str, float]:
        """
        Return SMA crossover specific metrics

        Returns:
            Dictionary with MA values, spreads, and trend information
        """
        if not self.is_ready:
            return {"error": -1.0, "message": "insufficient_data"}

        # Calculate current moving averages
        prices = self.historical_data['close']
        current_price = prices.iloc[-1]

        fast_sma = prices.rolling(window=self.fast_period).mean().iloc[-1]
        slow_sma = prices.rolling(window=self.slow_period).mean().iloc[-1]

        # Calculate useful metrics
        ma_spread = fast_sma - slow_sma
        ma_spread_percent = (ma_spread / slow_sma) * 100
        price_vs_fast = (current_price - fast_sma) / fast_sma
        price_vs_slow = (current_price - slow_sma) / slow_sma

        # Trend strength indicators
        fast_slope = self._calculate_slope(prices.rolling(window=self.fast_period).mean(), periods=5)
        slow_slope = self._calculate_slope(prices.rolling(window=self.slow_period).mean(), periods=5)

        # Distance from crossover point
        crossover_distance = abs(ma_spread_percent)

        return {
            "fast_sma": float(fast_sma),
            "slow_sma": float(slow_sma),
            "current_price": float(current_price),
            "ma_spread": float(ma_spread),
            "ma_spread_percent": float(ma_spread_percent),
            "price_vs_fast_percent": float(price_vs_fast * 100),
            "price_vs_slow_percent": float(price_vs_slow * 100),
            "fast_slope": float(fast_slope),
            "slow_slope": float(slow_slope),
            "crossover_distance": float(crossover_distance),
            "is_uptrend": fast_sma > slow_sma,
            "is_downtrend": fast_sma < slow_sma,
            "last_crossover_signal": self.last_crossover_signal,
            "fast_period": float(self.fast_period),
            "slow_period": float(self.slow_period)
        }

    def _calculate_slope(self, ma_series: pd.Series, periods: int) -> float:
        """
        Calculate slope of moving average over specified periods

        Args:
            ma_series: Moving average series
            periods: Number of periods to calculate slope over

        Returns:
            Slope value (positive = upward trend, negative = downward trend)
        """
        if len(ma_series) < periods:
            return 0.0

        recent_values = ma_series.tail(periods)
        if len(recent_values) < 2:
            return 0.0

        # Simple slope calculation: (end - start) / periods
        slope = (recent_values.iloc[-1] - recent_values.iloc[0]) / periods
        return slope

    @staticmethod
    def get_required_parameters() -> list[str]:
        return ['fast_period', 'slow_period', 'buffer_multiplier',
                'crossover_confirmation']