# technical_trading.py
# Technical analysis indicators and signal generation

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging
from src.hybrid.config.unified_config import UnifiedConfig, get_config

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Collection of technical analysis indicators

    ALL VALUES CONFIGURABLE - NO HARDCODED NUMBERS
    """

    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        self._cache_config_values()
        self._validate_config()

    def _cache_config_values(self):
        """Cache ALL technical analysis config values"""
        tech_config = self.config.config.get('technical_analysis', {})

        # KAMA parameters
        kama_config = tech_config.get('kama', {})
        self.kama_period = kama_config.get('period')
        self.kama_fast_sc = kama_config.get('fast_sc')
        self.kama_slow_sc = kama_config.get('slow_sc')

        # Kalman parameters
        kalman_config = tech_config.get('kalman', {})
        self.kalman_process_noise = kalman_config.get('process_noise')
        self.kalman_measurement_noise = kalman_config.get('measurement_noise')

        # Technical indicator parameters
        indicators_config = tech_config.get('indicators', {})

        # RSI parameters
        self.rsi_period = indicators_config.get('rsi_period')
        self.rsi_oversold_threshold = indicators_config.get('rsi_oversold_threshold')
        self.rsi_overbought_threshold = indicators_config.get('rsi_overbought_threshold')

        # ATR parameters
        self.atr_period = indicators_config.get('atr_period')

        # Bollinger Bands parameters
        self.bb_period = indicators_config.get('bb_period')
        self.bb_std_multiplier = indicators_config.get('bb_std_multiplier')

        # MACD parameters
        self.macd_fast = indicators_config.get('macd_fast')
        self.macd_slow = indicators_config.get('macd_slow')
        self.macd_signal = indicators_config.get('macd_signal')

        # Stochastic parameters
        self.stoch_k_period = indicators_config.get('stoch_k_period')
        self.stoch_d_period = indicators_config.get('stoch_d_period')

        # Moving average parameters
        self.sma_periods = indicators_config.get('sma_periods')
        self.ema_periods = indicators_config.get('ema_periods')

        # Signal generation parameters
        signal_config = tech_config.get('signal_generation', {})
        self.kama_signal_threshold = signal_config.get('kama_signal_threshold')
        self.kalman_threshold_multiplier = signal_config.get('kalman_threshold_multiplier')
        self.kalman_volatility_window = signal_config.get('kalman_volatility_window')
        self.trend_alignment_required = signal_config.get('trend_alignment_required')

        # Mathematical constants - ALL configurable
        self.zero_threshold = signal_config.get('zero_threshold')
        self.positive_signal_value = signal_config.get('positive_signal_value')
        self.negative_signal_value = signal_config.get('negative_signal_value')
        self.neutral_signal_value = signal_config.get('neutral_signal_value')
        self.percentage_multiplier = signal_config.get('percentage_multiplier')
        self.unity_value = signal_config.get('unity_value')
        self.array_shift_periods = signal_config.get('array_shift_periods')
        self.first_array_index = signal_config.get('first_array_index')
        self.second_array_index = signal_config.get('second_array_index')

    def _validate_config(self):
        """Validate that ALL required config values are present"""
        required_values = [
            ('kama_period', self.kama_period),
            ('kama_fast_sc', self.kama_fast_sc),
            ('kama_slow_sc', self.kama_slow_sc),
            ('kalman_process_noise', self.kalman_process_noise),
            ('kalman_measurement_noise', self.kalman_measurement_noise),
            ('rsi_period', self.rsi_period),
            ('rsi_oversold_threshold', self.rsi_oversold_threshold),
            ('rsi_overbought_threshold', self.rsi_overbought_threshold),
            ('atr_period', self.atr_period),
            ('bb_period', self.bb_period),
            ('bb_std_multiplier', self.bb_std_multiplier),
            ('macd_fast', self.macd_fast),
            ('macd_slow', self.macd_slow),
            ('macd_signal', self.macd_signal),
            ('stoch_k_period', self.stoch_k_period),
            ('stoch_d_period', self.stoch_d_period),
            ('sma_periods', self.sma_periods),
            ('ema_periods', self.ema_periods),
            ('kama_signal_threshold', self.kama_signal_threshold),
            ('kalman_threshold_multiplier', self.kalman_threshold_multiplier),
            ('kalman_volatility_window', self.kalman_volatility_window),
            ('trend_alignment_required', self.trend_alignment_required),
            ('zero_threshold', self.zero_threshold),
            ('positive_signal_value', self.positive_signal_value),
            ('negative_signal_value', self.negative_signal_value),
            ('neutral_signal_value', self.neutral_signal_value),
            ('percentage_multiplier', self.percentage_multiplier),
            ('unity_value', self.unity_value),
            ('array_shift_periods', self.array_shift_periods),
            ('first_array_index', self.first_array_index),
            ('second_array_index', self.second_array_index)
        ]

        missing_values = [name for name, value in required_values if value is None]
        if missing_values:
            raise ValueError(f"Missing required technical analysis config values: {missing_values}")

    def compute_kama(self, series: pd.Series) -> pd.Series:
        """Kaufman's Adaptive Moving Average with ALL values configurable"""
        if len(series) < self.kama_period + self.unity_value:
            return pd.Series(index=series.index, dtype=float)

        # Convert to numpy for speed
        values = series.values
        n = len(values)

        fast_sc = self.unity_value * self.unity_value / (self.kama_fast_sc + self.unity_value)
        slow_sc = self.unity_value * self.unity_value / (self.kama_slow_sc + self.unity_value)

        # Vectorized calculations
        change = np.abs(values[self.kama_period:] - values[:-self.kama_period])

        # Rolling sum of absolute differences
        abs_diff = np.abs(np.diff(values))
        volatility = np.convolve(abs_diff, np.ones(self.kama_period), mode='valid')

        # Efficiency Ratio with configurable thresholds
        er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility != self.zero_threshold)
        er = np.nan_to_num(er, nan=self.zero_threshold, posinf=self.zero_threshold, neginf=self.zero_threshold)

        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** self.unity_value * self.unity_value

        # Initialize KAMA array
        kama = np.full(n, np.nan)
        kama[self.kama_period] = values[self.kama_period]

        # Optimized loop
        for i in range(self.kama_period + self.unity_value, n):
            kama[i] = kama[i - self.unity_value] + sc[i - self.kama_period] * (values[i] - kama[i - self.unity_value])

        return pd.Series(kama, index=series.index)

    def kalman_filter(self, series: pd.Series) -> pd.Series:
        """Kalman filter with ALL parameters configurable"""
        if len(series) == self.zero_threshold:
            return pd.Series(dtype=float)

        # Convert to numpy
        values = series.values
        n = len(values)

        # Pre-allocate arrays
        xhat = np.zeros(n)
        P = np.zeros(n)

        # Handle first value with configurable defaults
        xhat[self.zero_threshold] = values[self.zero_threshold] if not np.isnan(
            values[self.zero_threshold]) else self.zero_threshold
        P[self.zero_threshold] = self.unity_value

        # Vectorized NaN mask
        nan_mask = np.isnan(values)

        # Main loop with configurable parameters
        for k in range(self.unity_value, n):
            if nan_mask[k]:
                xhat[k] = xhat[k - self.unity_value]
                P[k] = P[k - self.unity_value] + self.kalman_process_noise
                continue

            # Prediction
            xhatminus = xhat[k - self.unity_value]
            Pminus = P[k - self.unity_value] + self.kalman_process_noise

            # Update
            K = Pminus / (Pminus + self.kalman_measurement_noise)
            xhat[k] = xhatminus + K * (values[k] - xhatminus)
            P[k] = (self.unity_value - K) * Pminus

        return pd.Series(xhat, index=series.index)

    def rsi(self, series: pd.Series) -> pd.Series:
        """Relative Strength Index with configurable period"""
        delta = series.diff()
        gain = (delta.where(delta > self.zero_threshold, self.zero_threshold)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < self.zero_threshold, self.zero_threshold)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return self.percentage_multiplier - (self.percentage_multiplier / (self.unity_value + rs))

    def atr(self, df: pd.DataFrame) -> pd.Series:
        """Average True Range with configurable period"""
        high_low = df["high"] - df["low"]
        high_close_prev = np.abs(df["high"] - df["close"].shift(self.array_shift_periods))
        low_close_prev = np.abs(df["low"] - df["close"].shift(self.array_shift_periods))

        tr = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
        return pd.Series(tr, index=df.index).rolling(self.atr_period).mean()

    def bollinger_bands(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands with configurable parameters"""
        sma = series.rolling(window=self.bb_period).mean()
        std = series.rolling(window=self.bb_period).std()
        upper_band = sma + (std * self.bb_std_multiplier)
        lower_band = sma - (std * self.bb_std_multiplier)
        return upper_band, sma, lower_band

    def macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator with configurable parameters"""
        ema_fast = series.ewm(span=self.macd_fast).mean()
        ema_slow = series.ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator with configurable parameters"""
        lowest_low = low.rolling(window=self.stoch_k_period).min()
        highest_high = high.rolling(window=self.stoch_k_period).max()
        k_percent = self.percentage_multiplier * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=self.stoch_d_period).mean()
        return k_percent, d_percent


class TechnicalSignalGenerator:
    """
    Technical Analysis component: Generate trading signals

    ALL VALUES CONFIGURABLE - NO HARDCODED NUMBERS
    """

    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        self.indicators = TechnicalIndicators(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical trading signals with ALL parameters configurable"""
        signals = pd.DataFrame(index=df.index)

        # Compute trend indicators
        signals['kama'] = self.indicators.compute_kama(df['close'])
        signals['kalman'] = self.indicators.kalman_filter(df['close'])

        # Compute additional indicators
        signals['rsi'] = self.indicators.rsi(df['close'])
        signals['atr'] = self.indicators.atr(df)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(df['close'])
        signals['bb_upper'] = bb_upper
        signals['bb_middle'] = bb_middle
        signals['bb_lower'] = bb_lower
        signals['bb_width'] = (bb_upper - bb_lower) / bb_middle
        signals['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # MACD
        macd, macd_signal, macd_hist = self.indicators.macd(df['close'])
        signals['macd'] = macd
        signals['macd_signal'] = macd_signal
        signals['macd_histogram'] = macd_hist

        # Stochastic
        stoch_k, stoch_d = self.indicators.stochastic_oscillator(df['high'], df['low'], df['close'])
        signals['stoch_k'] = stoch_k
        signals['stoch_d'] = stoch_d

        # Generate moving averages with configurable periods
        for period in self.indicators.sma_periods:
            signals[f'sma_{period}'] = df['close'].rolling(period).mean()

        for period in self.indicators.ema_periods:
            signals[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # Generate trading signals
        signals = self._generate_trading_signals(df, signals)

        return signals

    def _generate_trading_signals(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals with ALL thresholds configurable"""

        # KAMA signals with configurable thresholds
        kama_slope = signals['kama'].diff()
        signals['kama_signal'] = self.indicators.neutral_signal_value
        kama_long = (df['close'] > signals['kama']) & (kama_slope > self.indicators.kama_signal_threshold)
        kama_short = (df['close'] < signals['kama']) & (kama_slope < -self.indicators.kama_signal_threshold)
        signals.loc[kama_long, 'kama_signal'] = self.indicators.positive_signal_value
        signals.loc[kama_short, 'kama_signal'] = self.indicators.negative_signal_value

        # Kalman signals with configurable parameters
        price_vs_kalman = df['close'] / signals['kalman'] - self.indicators.unity_value
        volatility = df['close'].pct_change().rolling(self.indicators.kalman_volatility_window).std()
        threshold = volatility * self.indicators.kalman_threshold_multiplier

        signals['kalman_signal'] = self.indicators.neutral_signal_value
        kalman_long = price_vs_kalman > threshold
        kalman_short = price_vs_kalman < -threshold
        signals.loc[kalman_long, 'kalman_signal'] = self.indicators.positive_signal_value
        signals.loc[kalman_short, 'kalman_signal'] = self.indicators.negative_signal_value

        # Trend alignment signals with configurable requirements
        signals['trend_signal'] = self.indicators.neutral_signal_value

        if len(self.indicators.sma_periods) >= self.indicators.unity_value * self.indicators.unity_value:
            short_sma = signals[f'sma_{self.indicators.sma_periods[self.indicators.first_array_index]}']
            medium_sma = signals[f'sma_{self.indicators.sma_periods[self.indicators.second_array_index]}']

            if self.indicators.trend_alignment_required:
                long_trend = (df['close'] > short_sma) & (short_sma > medium_sma)
                short_trend = (df['close'] < short_sma) & (short_sma < medium_sma)
            else:
                long_trend = df['close'] > short_sma
                short_trend = df['close'] < short_sma

            signals.loc[long_trend, 'trend_signal'] = self.indicators.positive_signal_value
            signals.loc[short_trend, 'trend_signal'] = self.indicators.negative_signal_value

        # RSI signals with configurable thresholds
        signals['rsi_signal'] = self.indicators.neutral_signal_value
        rsi_oversold = signals['rsi'] < self.indicators.rsi_oversold_threshold
        rsi_overbought = signals['rsi'] > self.indicators.rsi_overbought_threshold
        signals.loc[rsi_oversold, 'rsi_signal'] = self.indicators.positive_signal_value
        signals.loc[rsi_overbought, 'rsi_signal'] = self.indicators.negative_signal_value

        # Bollinger Bands signals
        signals['bb_signal'] = self.indicators.neutral_signal_value
        bb_oversold = df['close'] < signals['bb_lower']
        bb_overbought = df['close'] > signals['bb_upper']
        signals.loc[bb_oversold, 'bb_signal'] = self.indicators.positive_signal_value
        signals.loc[bb_overbought, 'bb_signal'] = self.indicators.negative_signal_value

        # MACD signals with configurable shift
        signals['macd_crossover_signal'] = self.indicators.neutral_signal_value
        macd_bullish = (signals['macd'] > signals['macd_signal']) & (
                signals['macd'].shift(self.indicators.array_shift_periods) <= signals['macd_signal'].shift(
            self.indicators.array_shift_periods))
        macd_bearish = (signals['macd'] < signals['macd_signal']) & (
                signals['macd'].shift(self.indicators.array_shift_periods) >= signals['macd_signal'].shift(
            self.indicators.array_shift_periods))
        signals.loc[macd_bullish, 'macd_crossover_signal'] = self.indicators.positive_signal_value
        signals.loc[macd_bearish, 'macd_crossover_signal'] = self.indicators.negative_signal_value

        return signals

    def get_signal_summary(self, signals: pd.DataFrame) -> dict:
        """Get summary statistics with configurable thresholds"""
        signal_cols = [col for col in signals.columns if col.endswith('_signal')]
        summary = {}

        for col in signal_cols:
            total_signals = (signals[col] != self.indicators.neutral_signal_value).sum()
            long_signals = (signals[col] == self.indicators.positive_signal_value).sum()
            short_signals = (signals[col] == self.indicators.negative_signal_value).sum()

            summary[col] = {
                'total': total_signals,
                'long': long_signals,
                'short': short_signals,
                'long_pct': long_signals / total_signals if total_signals > self.indicators.zero_threshold else self.indicators.zero_threshold,
                'short_pct': short_signals / total_signals if total_signals > self.indicators.zero_threshold else self.indicators.zero_threshold,
                'frequency': total_signals / len(signals) if len(
                    signals) > self.indicators.zero_threshold else self.indicators.zero_threshold
            }

        return summary


def calculate_technical_indicators(df: pd.DataFrame, config: UnifiedConfig = None) -> pd.DataFrame:
    """
    Convenience function to calculate all technical indicators
    """
    generator = TechnicalSignalGenerator(config)
    signals_df = generator.generate_signals(df)
    return signals_df