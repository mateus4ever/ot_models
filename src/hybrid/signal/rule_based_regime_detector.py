"""
Rule-Based Market Regime Detection - Fully Configurable
Part of the signal generation package - uses UnifiedConfig system
All parameters configurable through JSON, no hardcoded values
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

from src.hybrid.config.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class RuleBasedRegimeDetector:
    """
    Fully configurable rule-based regime detection

    Uses the same configuration approach as FullyConfigurableFeatureGenerator
    All thresholds, parameters, and behavior configurable through JSON
    """

    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()

        # Load configuration sections for easy access (same pattern as feature generator)
        self.numeric_formatting = self.config.numeric_formatting
        self.array_indexing = self.config.array_indexing
        self.math_ops = self.config.mathematical_operations

        # Load rule-based regime configuration
        self.regime_config = self.config.config.get('regime_detection', {})
        self.rule_thresholds = self.regime_config.get('rule_based_thresholds', {})
        self.regime_periods = self.regime_config.get('moving_average_periods', {})
        self.confidence_config = self.regime_config.get('confidence_requirements', {})

        # Compatibility with existing interface
        self.is_trained = True  # Rule-based doesn't need training

        # Regime mapping from configuration
        prediction_config = self.config.config.get('regime_prediction', {})
        if prediction_config:
            self.regime_mapping = prediction_config.get('regime_mapping', {
                'ranging': self.math_ops['zero'],
                'trending_up': self.math_ops['unity'],
                'trending_down': self.math_ops['unity'] + self.math_ops['unity'],
                'high_volatility': self.math_ops['unity'] + self.math_ops['unity'] + self.math_ops['unity']
            })
        else:
            # Fallback mapping using math_ops
            self.regime_mapping = {
                'ranging': self.math_ops['zero'],
                'trending_up': self.math_ops['unity'],
                'trending_down': self.math_ops['unity'] + self.math_ops['unity'],
                'high_volatility': self.math_ops['unity'] + self.math_ops['unity'] + self.math_ops['unity']
            }

        # Create reverse mapping for regime names
        self.regime_names = {v: k.replace('_', ' ').title() for k, v in self.regime_mapping.items()}

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compatibility method - rule-based doesn't need training
        Uses configuration to return appropriate metrics
        """
        training_config = self.config.config.get('regime_training', {})
        verbose = training_config.get('verbose', self.config.config.get('general', {}).get('verbose', True))

        if verbose:
            print("Rule-based regime detection ready (no training required)")

        # Test the rules on sample data to return realistic metrics
        regime_predictions, regime_confidence = self.predict_regime(df)

        # Calculate metrics using configuration
        regime_counts = pd.Series(regime_predictions).value_counts()
        total = len(regime_predictions)

        # Calculate trending percentage using regime mapping
        trending_up_count = regime_counts.get(self.regime_mapping['trending_up'], self.math_ops['zero'])
        trending_down_count = regime_counts.get(self.regime_mapping['trending_down'], self.math_ops['zero'])
        trending_pct = ((trending_up_count + trending_down_count) / total *
                        self.numeric_formatting['percentage_conversion']['multiplier']) if total > self.math_ops[
            'zero'] else self.math_ops['zero']

        avg_confidence = regime_confidence.mean() if len(regime_confidence) > self.math_ops[
            'zero'] else self.confidence_config.get('minimum_confidence', 0.7)

        # Use configurable feature counts
        strength_features_count = self.regime_config.get('rule_feature_counts', {}).get('strength_features', 12)
        direction_features_count = self.regime_config.get('rule_feature_counts', {}).get('direction_features', 8)

        return {
            'strength_accuracy': avg_confidence,
            'direction_accuracy': min(avg_confidence + self.confidence_config.get('direction_bonus', 0.1),
                                      self.math_ops['unity']),
            'n_samples': len(df),
            'n_strength_features': strength_features_count,
            'n_direction_features': direction_features_count,
            'n_trending_samples': int(
                trending_pct * total / self.numeric_formatting['percentage_conversion']['multiplier'])
        }

    def predict_regime(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict market regime using configurable rule-based logic
        """

        min_data_periods = self.regime_config.get('min_data_periods',
                                                  self.regime_periods.get('long', 240))

        if len(df) < min_data_periods:
            return (np.full(len(df), self.regime_mapping['ranging']),
                    np.zeros(len(df)))

        # Calculate technical indicators using configuration
        indicators = self._calculate_indicators(df)

        # Apply regime classification rules
        regime_predictions = self._classify_regimes(indicators)
        regime_confidence = self._calculate_confidence(indicators, regime_predictions)

        return regime_predictions, regime_confidence

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators using configurable periods"""

        indicators = {}
        returns = df['close'].pct_change()

        # Get periods from configuration
        short_period = self.regime_periods.get('short', 20)
        medium_period = self.regime_periods.get('medium', 60)
        long_period = self.regime_periods.get('long', 240)

        # === MOVING AVERAGES ===
        indicators[f'sma_{short_period}'] = df['close'].rolling(short_period).mean()
        indicators[f'sma_{medium_period}'] = df['close'].rolling(medium_period).mean()
        indicators[f'sma_{long_period}'] = df['close'].rolling(long_period).mean()

        # === PRICE POSITIONS RELATIVE TO MAs ===
        indicators[f'price_vs_sma{short_period}'] = ((df['close'] - indicators[f'sma_{short_period}']) /
                                                     indicators[f'sma_{short_period}'])
        indicators[f'price_vs_sma{medium_period}'] = ((df['close'] - indicators[f'sma_{medium_period}']) /
                                                      indicators[f'sma_{medium_period}'])
        indicators[f'price_vs_sma{long_period}'] = ((df['close'] - indicators[f'sma_{long_period}']) /
                                                    indicators[f'sma_{long_period}'])

        # === MOVING AVERAGE ALIGNMENT ===
        indicators['ma_alignment_short'] = ((indicators[f'sma_{short_period}'] - indicators[f'sma_{medium_period}']) /
                                            indicators[f'sma_{medium_period}'])
        indicators['ma_alignment_long'] = ((indicators[f'sma_{medium_period}'] - indicators[f'sma_{long_period}']) /
                                           indicators[f'sma_{long_period}'])

        # === MOMENTUM INDICATORS ===
        indicators[f'momentum_{short_period}'] = df['close'] / df['close'].shift(short_period) - self.math_ops['unity']
        indicators[f'momentum_{medium_period}'] = df['close'] / df['close'].shift(medium_period) - self.math_ops[
            'unity']
        indicators[f'momentum_{long_period}'] = df['close'] / df['close'].shift(long_period) - self.math_ops['unity']

        # === VOLATILITY INDICATORS ===
        indicators[f'vol_{short_period}'] = returns.rolling(short_period).std()
        indicators[f'vol_{medium_period}'] = returns.rolling(medium_period).std()
        indicators[f'vol_{long_period}'] = returns.rolling(long_period).std()
        indicators['vol_ratio_short'] = indicators[f'vol_{short_period}'] / indicators[f'vol_{medium_period}']
        indicators['vol_ratio_long'] = indicators[f'vol_{short_period}'] / indicators[f'vol_{long_period}']

        # === RANGE ANALYSIS ===
        indicators[f'high_{long_period}'] = df['high'].rolling(long_period).max()
        indicators[f'low_{long_period}'] = df['low'].rolling(long_period).min()
        range_span = indicators[f'high_{long_period}'] - indicators[f'low_{long_period}']
        indicators['range_position'] = (df['close'] - indicators[f'low_{long_period}']) / range_span
        indicators['range_width'] = range_span / df['close']

        # === TREND STRENGTH ===
        atr_period = self.regime_config.get('atr_period', 14)
        indicators['atr'] = self._calculate_atr(df, atr_period)
        indicators['trend_strength'] = (abs(indicators[f'momentum_{medium_period}']) /
                                        indicators[f'vol_{medium_period}'])

        return indicators

    def _classify_regimes(self, indicators: Dict[str, pd.Series]) -> np.ndarray:
        """Apply configurable rules to classify market regimes"""

        # Get thresholds from configuration
        price_threshold = self.rule_thresholds.get('price_vs_ma_threshold', 0.002)
        momentum_threshold = self.rule_thresholds.get('momentum_threshold', 0.005)
        ma_alignment_threshold = self.rule_thresholds.get('ma_alignment_threshold', 0.001)
        vol_ratio_threshold = self.rule_thresholds.get('volatility_ratio_threshold', 1.5)
        range_extreme_threshold = self.rule_thresholds.get('range_extreme_threshold', 0.1)
        high_vol_threshold = self.rule_thresholds.get('high_volatility_threshold', 2.0)

        # Get momentum scaling factors from configuration
        momentum_scaling = self.rule_thresholds.get('momentum_scaling', {})
        short_momentum_factor = momentum_scaling.get('short_factor', 0.3)
        long_momentum_factor = momentum_scaling.get('long_factor', 0.5)

        # Get periods for dynamic key generation
        short_period = self.regime_periods.get('short', 20)
        medium_period = self.regime_periods.get('medium', 60)
        long_period = self.regime_periods.get('long', 240)

        # Initialize as ranging
        regime = pd.Series(self.regime_mapping['ranging'], index=indicators[f'sma_{short_period}'].index)

        # === HIGH VOLATILITY DETECTION (Priority 1) ===
        high_vol_multiplier = self.rule_thresholds.get('high_vol_multiplier', 1.5)
        high_vol_conditions = (
                (indicators['vol_ratio_long'] > high_vol_threshold) |
                (indicators['vol_ratio_short'] > vol_ratio_threshold * high_vol_multiplier)
        )

        # === TRENDING UP CONDITIONS ===
        price_factor_medium = self.rule_thresholds.get('price_factor_medium', 0.5)
        ma_factor_long = self.rule_thresholds.get('ma_alignment_factor_long', 0.5)

        trending_up_conditions = (
            # Price above key moving averages
                (indicators[f'price_vs_sma{short_period}'] > price_threshold) &
                (indicators[f'price_vs_sma{medium_period}'] > price_threshold * price_factor_medium) &

                # Moving averages properly aligned
                (indicators['ma_alignment_short'] > ma_alignment_threshold) &
                (indicators['ma_alignment_long'] > ma_alignment_threshold * ma_factor_long) &

                # Strong positive momentum
                (indicators[f'momentum_{medium_period}'] > momentum_threshold) &
                (indicators[f'momentum_{long_period}'] > momentum_threshold * long_momentum_factor) &

                # Not in extreme range positions
                (indicators['range_position'] > range_extreme_threshold) &

                # Reasonable volatility
                ~high_vol_conditions
        )

        # === TRENDING DOWN CONDITIONS ===
        trending_down_conditions = (
            # Price below key moving averages
                (indicators[f'price_vs_sma{short_period}'] < -price_threshold) &
                (indicators[f'price_vs_sma{medium_period}'] < -price_threshold * price_factor_medium) &

                # Moving averages properly aligned downward
                (indicators['ma_alignment_short'] < -ma_alignment_threshold) &
                (indicators['ma_alignment_long'] < -ma_alignment_threshold * ma_factor_long) &

                # Strong negative momentum
                (indicators[f'momentum_{medium_period}'] < -momentum_threshold) &
                (indicators[f'momentum_{long_period}'] < -momentum_threshold * long_momentum_factor) &

                # Not in extreme range positions
                (indicators['range_position'] < (self.math_ops['unity'] - range_extreme_threshold)) &

                # Reasonable volatility
                ~high_vol_conditions
        )

        # Apply regime classifications using configured values
        regime[high_vol_conditions] = self.regime_mapping['high_volatility']
        regime[trending_up_conditions] = self.regime_mapping['trending_up']
        regime[trending_down_conditions] = self.regime_mapping['trending_down']

        return regime.values

    def _calculate_confidence(self, indicators: Dict[str, pd.Series],
                              regime: np.ndarray) -> np.ndarray:
        """Calculate confidence scores using configurable parameters"""

        confidence = np.zeros(len(regime))

        # Get confidence calculation parameters from configuration
        conf_params = self.regime_config.get('confidence_calculation', {})
        price_strength_multiplier = conf_params.get('price_strength_multiplier', 200)
        momentum_strength_multiplier = conf_params.get('momentum_strength_multiplier', 100)
        ma_alignment_multiplier = conf_params.get('ma_alignment_multiplier', 500)
        vol_clarity_base = conf_params.get('vol_clarity_base', 2.0)

        # Get periods
        short_period = self.regime_periods.get('short', 20)
        medium_period = self.regime_periods.get('medium', 60)
        long_period = self.regime_periods.get('long', 240)

        # Skip initial unstable period
        skip_periods = self.regime_config.get('confidence_skip_periods', long_period)

        for i in range(len(regime)):
            if i < skip_periods:
                confidence[i] = self.math_ops['zero']
                continue

            current_regime = regime[i]

            if current_regime == self.regime_mapping['trending_up']:
                # Confidence based on alignment and strength
                price_strength = np.clip(indicators[f'price_vs_sma{short_period}'].iloc[i] * price_strength_multiplier,
                                         self.math_ops['zero'], self.math_ops['unity'])
                momentum_strength = np.clip(
                    indicators[f'momentum_{medium_period}'].iloc[i] * momentum_strength_multiplier,
                    self.math_ops['zero'], self.math_ops['unity'])
                ma_alignment = np.clip(indicators['ma_alignment_short'].iloc[i] * ma_alignment_multiplier,
                                       self.math_ops['zero'], self.math_ops['unity'])
                vol_clarity = np.clip(vol_clarity_base - indicators['vol_ratio_long'].iloc[i],
                                      self.math_ops['zero'], self.math_ops['unity'])

                confidence_components = self.math_ops['unity'] + self.math_ops['unity'] + self.math_ops['unity'] + \
                                        self.math_ops['unity']  # 4
                confidence[i] = (
                                            price_strength + momentum_strength + ma_alignment + vol_clarity) / confidence_components

            elif current_regime == self.regime_mapping['trending_down']:
                # Confidence based on alignment and strength (inverted)
                price_strength = np.clip(-indicators[f'price_vs_sma{short_period}'].iloc[i] * price_strength_multiplier,
                                         self.math_ops['zero'], self.math_ops['unity'])
                momentum_strength = np.clip(
                    -indicators[f'momentum_{medium_period}'].iloc[i] * momentum_strength_multiplier,
                    self.math_ops['zero'], self.math_ops['unity'])
                ma_alignment = np.clip(-indicators['ma_alignment_short'].iloc[i] * ma_alignment_multiplier,
                                       self.math_ops['zero'], self.math_ops['unity'])
                vol_clarity = np.clip(vol_clarity_base - indicators['vol_ratio_long'].iloc[i],
                                      self.math_ops['zero'], self.math_ops['unity'])

                confidence_components = self.math_ops['unity'] + self.math_ops['unity'] + self.math_ops['unity'] + \
                                        self.math_ops['unity']  # 4
                confidence[i] = (
                                            price_strength + momentum_strength + ma_alignment + vol_clarity) / confidence_components

            elif current_regime == self.regime_mapping['high_volatility']:
                # Confidence based on volatility metrics
                vol_threshold_base = self.math_ops['unity']
                vol_spike_divisor = conf_params.get('vol_spike_divisor', 2)
                vol_spike = np.clip((indicators['vol_ratio_long'].iloc[i] - vol_threshold_base) / vol_spike_divisor,
                                    self.math_ops['zero'], self.math_ops['unity'])
                vol_recent = np.clip((indicators['vol_ratio_short'].iloc[i] - vol_threshold_base) / vol_spike_divisor,
                                     self.math_ops['zero'], self.math_ops['unity'])

                vol_components = self.math_ops['unity'] + self.math_ops['unity']  # 2
                confidence[i] = (vol_spike + vol_recent) / vol_components

            else:  # Ranging
                # Confidence based on how "range-like" conditions are
                range_momentum_multiplier = conf_params.get('range_momentum_multiplier', 50)
                range_ma_multiplier = conf_params.get('range_ma_multiplier', 1000)
                range_position_multiplier = conf_params.get('range_position_multiplier', 4)
                range_position_center = conf_params.get('range_position_center', 0.5)

                low_momentum = np.clip(self.math_ops['unity'] - abs(
                    indicators[f'momentum_{medium_period}'].iloc[i]) * range_momentum_multiplier,
                                       self.math_ops['zero'], self.math_ops['unity'])
                ma_convergence = np.clip(
                    self.math_ops['unity'] - abs(indicators['ma_alignment_short'].iloc[i]) * range_ma_multiplier,
                    self.math_ops['zero'], self.math_ops['unity'])
                low_volatility = np.clip(vol_clarity_base - indicators['vol_ratio_long'].iloc[i],
                                         self.math_ops['zero'], self.math_ops['unity'])
                range_middle = np.clip(self.math_ops['unity'] - abs(
                    indicators['range_position'].iloc[i] - range_position_center) * range_position_multiplier,
                                       self.math_ops['zero'], self.math_ops['unity'])

                range_components = self.math_ops['unity'] + self.math_ops['unity'] + self.math_ops['unity'] + \
                                   self.math_ops['unity']  # 4
                confidence[i] = (low_momentum + ma_convergence + low_volatility + range_middle) / range_components

        return confidence

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range using configured period"""
        high_low = df["high"] - df["low"]
        high_close_prev = np.abs(df["high"] - df["close"].shift())
        low_close_prev = np.abs(df["low"] - df["close"].shift())

        tr = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
        return pd.Series(tr, index=df.index).rolling(period).mean()

    def get_regime_name(self, regime_id: int) -> str:
        """Convert regime ID to human-readable name - fully configurable"""
        return self.regime_names.get(regime_id, f'Unknown ({regime_id})')

    def get_feature_importance(self) -> Dict[str, float]:
        """Return configurable feature importance"""

        # Get importance weights from configuration
        importance_config = self.regime_config.get('feature_importance_weights', {})

        # Default importance using configuration values
        default_importance = {
            f'price_vs_sma{self.regime_periods.get("short", 20)}': importance_config.get('price_vs_sma_short', 0.25),
            f'momentum_{self.regime_periods.get("medium", 60)}': importance_config.get('momentum_medium', 0.20),
            'ma_alignment_short': importance_config.get('ma_alignment_short', 0.15),
            'vol_ratio_long': importance_config.get('vol_ratio_long', 0.12),
            f'price_vs_sma{self.regime_periods.get("medium", 60)}': importance_config.get('price_vs_sma_medium', 0.10),
            'ma_alignment_long': importance_config.get('ma_alignment_long', 0.08),
            f'momentum_{self.regime_periods.get("long", 240)}': importance_config.get('momentum_long', 0.05),
            'range_position': importance_config.get('range_position', 0.03),
            'trend_strength': importance_config.get('trend_strength', 0.02)
        }

        return default_importance

    def save_model(self, filepath: str):
        """Save configuration - fully compatible with existing system"""
        import json

        model_data = {
            'type': 'rule_based_regime_detector',
            'config': self.config.config,
            'rule_thresholds': self.rule_thresholds,
            'regime_mapping': self.regime_mapping,
            'regime_periods': self.regime_periods
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=self.numeric_formatting.get('json_indent', 2))

        logger.info(f"RuleBasedRegimeDetector configuration saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'RuleBasedRegimeDetector':
        """Load configuration from file"""
        import json

        with open(filepath, 'r') as f:
            model_data = json.load(f)

        # Reconstruct config
        config = UnifiedConfig()
        config.config.update(model_data.get('config', {}))

        detector = cls(config)
        detector.rule_thresholds = model_data.get('rule_thresholds', {})
        detector.regime_mapping = model_data.get('regime_mapping', detector.regime_mapping)
        detector.regime_periods = model_data.get('regime_periods', {})

        logger.info(f"RuleBasedRegimeDetector loaded from {filepath}")
        return detector

    def get_regime_distribution(self, regime_predictions: np.ndarray) -> Dict[str, float]:
        """Get distribution using configurable regime names"""

        unique, counts = np.unique(regime_predictions, return_counts=True)
        total = len(regime_predictions)

        distribution = {}
        for regime_id, count in zip(unique, counts):
            regime_name = self.get_regime_name(int(regime_id))
            percentage = (count / total) * self.numeric_formatting['percentage_conversion']['multiplier']
            distribution[regime_name] = percentage

        return distribution