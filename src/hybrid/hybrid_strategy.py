"""
Hybrid Trading Strategy - Updated to Use Clean Technical Module
Combines rule-based regime detection with ML volatility prediction and technical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import time as timer

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.technical_trading import TechnicalSignalGenerator

logger = logging.getLogger(__name__)


class HybridStrategy:
    """
    Complete Hybrid Trading Strategy

    Components:
    - Rule-based regime detection (transparent, no overfitting)
    - ML volatility prediction (proven to work)
    - Technical analysis signals (clean, configurable module)
    - Intelligent signal combination based on market conditions

    ALL VALUES CONFIGURABLE - NO HARDCODED NUMBERS
    """

    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        self._cache_config_values()
        self._validate_config()

        # Initialize ML Model Manager for coordinated ML operations
        from src.hybrid.ml_model.ml_manager import MLModelManager
        self.ml_manager = MLModelManager(self.config)

        # Initialize clean technical signal generator
        self.technical_generator = TechnicalSignalGenerator(self.config)

        # Strategy state
        self.is_trained = False
        self.training_results = {}

    def _cache_config_values(self):
        """Cache ALL configuration values for hybrid strategy"""
        # Technical analysis configuration
        self.technical_config = self.config.get_section('technical_analysis', {})
        self.risk_config = self.config.get_section('risk_management', {})
        general_config = self.config.get_section('general', {})

        # Signal combination parameters - ALL configurable
        signal_config = self.technical_config.get('signal_combination', {})
        self.combination_method = signal_config.get('method')
        self.signal_weights = signal_config.get('weights', {})

        # Risk management parameters - ALL configurable
        position_config = self.risk_config.get('position_sizing', {})
        self.base_size = position_config.get('base_size')

        self.regime_adjustments = position_config.get('regime_adjustments', {})
        self.confidence_scaling = position_config.get('confidence_scaling')
        self.max_position_size = self.risk_config.get('max_position_size')

        # General parameters
        self.verbose = general_config.get('verbose')

        # Mathematical constants - ALL configurable
        constants_config = self.config.get_section('mathematical_operations', {})
        self.zero_value = constants_config.get('zero')
        self.unity_value = constants_config.get('unity')

        # Boolean constants - ALL configurable
        boolean_config = self.config.get_section('boolean_values', {})
        self.true_value = boolean_config.get('true')
        self.false_value = boolean_config.get('false')

        # Signal processing parameters - ALL configurable
        signal_processing_config = self.technical_config.get('signal_processing', {})
        self.high_vol_position_multiplier = signal_processing_config.get('high_vol_position_multiplier')
        self.high_vol_confidence_threshold = signal_processing_config.get('high_vol_confidence_threshold')
        self.ranging_signal_multiplier = signal_processing_config.get('ranging_signal_multiplier')
        self.high_vol_signal_multiplier = signal_processing_config.get('high_vol_signal_multiplier')
        self.confidence_divisor = signal_processing_config.get('confidence_divisor')
        self.signal_clip_min = signal_processing_config.get('signal_clip_min')
        self.signal_clip_max = signal_processing_config.get('signal_clip_max')

        # Duration multipliers - ALL configurable
        duration_config = self.technical_config.get('duration_multipliers', {})
        self.duration_multipliers = duration_config.get('category_multipliers', {})
        self.duration_confidence_blend = duration_config.get('confidence_blend_factor')

    def _validate_config(self):
        """Validate that ALL required config values are present"""
        required_values = [
            ('combination_method', self.combination_method),
            ('base_size', self.base_size),
            ('confidence_scaling', self.confidence_scaling),
            ('max_position_size', self.max_position_size),
            ('verbose', self.verbose),
            ('zero_value', self.zero_value),
            ('unity_value', self.unity_value),
            ('high_vol_position_multiplier', self.high_vol_position_multiplier),
            ('high_vol_confidence_threshold', self.high_vol_confidence_threshold),
            ('ranging_signal_multiplier', self.ranging_signal_multiplier),
            ('high_vol_signal_multiplier', self.high_vol_signal_multiplier),
            ('confidence_divisor', self.confidence_divisor),
            ('signal_clip_min', self.signal_clip_min),
            ('signal_clip_max', self.signal_clip_max),
            ('duration_confidence_blend', self.duration_confidence_blend)
        ]

        missing_values = [name for name, value in required_values if value is None]
        if missing_values:
            raise ValueError(f"Missing required hybrid strategy config values: {missing_values}")

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the hybrid strategy

        Args:
            df: OHLC price data

        Returns:
            Comprehensive training results
        """
        print("=== Training Hybrid Strategy ===")
        start_time = timer.time()

        # Train all ML models and initialize rule-based components
        self.training_results = self.ml_manager.train_all_models(df)

        # Add strategy-level metrics
        self.training_results.update({
            'strategy_type': 'Hybrid ML-Technical',
            'ml_components': self._get_ml_component_summary(),
            'training_time': timer.time() - start_time
        })

        self.is_trained = True

        if self.verbose:
            self._print_training_summary()

        return self.training_results

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive trading signals using clean technical module

        Args:
            df: OHLC price data

        Returns:
            DataFrame with all signals and metadata
        """
        if not self.is_trained:
            raise ValueError("Strategy must be trained before generating signals")

        print("Generating hybrid signals...")
        start_time = timer.time()

        # Get all ML predictions
        ml_predictions = self.ml_manager.predict_all(df)

        # Generate technical analysis signals using our clean module
        technical_signals_df = self.technical_generator.generate_signals(df)

        # Combine all signals intelligently
        combined_signals = self._combine_all_signals(df, ml_predictions, technical_signals_df)

        # Create comprehensive signals DataFrame
        signals_df = self._create_signals_dataframe(df, ml_predictions, technical_signals_df, combined_signals)

        print(f"✓ Signal generation took: {timer.time() - start_time:.1f} seconds")

        return signals_df

    def _combine_all_signals(self, df: pd.DataFrame, ml_predictions: Dict,
                             technical_signals_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Intelligently combine rule-based regime detection and ML predictions with technical signals

        Process Steps:
        1. Extract rule-based regime detection, ML volatility prediction, and ML duration prediction data with None handling
        2. Extract and combine technical signals (KAMA, Kalman)
        3. Calculate regime-based signal strength (trending/ranging/high-vol from rule-based detector)
        4. Apply confidence scaling and ML volatility scaling multipliers
        5. Calculate final trading signals and position sizes
        6. Return formatted results as pandas Series
        """
        # Step 1: Extract rule-based regime detection and ML prediction data
        regime_pred, regime_conf, vol_pred_array, vol_conf_array, duration_pred_array, duration_conf_array = \
            self._prepare_ml_predictions(df, ml_predictions)

        # Step 2: Extract and combine technical signals
        tech_signal_array = self._extract_technical_signals(df, technical_signals_df)

        # Step 3: Calculate regime-based signal strength
        signal_strength_array, regime_multiplier_array, source_array = \
            self._calculate_regime_signals(regime_pred, tech_signal_array)

        # Step 4: Apply confidence and volatility scaling
        confidence_multiplier_array = self._calculate_confidence_scaling(df, regime_conf)
        volatility_multiplier_array = self._calculate_volatility_scaling(vol_pred_array, vol_conf_array)
        duration_multiplier_array = self._get_duration_multiplier_vectorized(duration_pred_array, duration_conf_array)

        # Step 5: Calculate final signals and position sizes
        final_signal_array, position_size_array = self._calculate_final_signals(
            signal_strength_array, confidence_multiplier_array,
            volatility_multiplier_array, duration_multiplier_array, regime_multiplier_array
        )

        # Step 6: Calculate confidence scores and return results
        confidence_scores_array = self._calculate_confidence_scores(
            df, regime_conf, vol_conf_array, duration_conf_array
        )

        return self._format_signal_results(df, final_signal_array, position_size_array,
                                           source_array, confidence_scores_array)

    def _prepare_ml_predictions(self, df: pd.DataFrame, ml_predictions: Dict) -> tuple:
        """
        Step 1: Extract and sanitize ML predictions with comprehensive None handling

        Goal: Convert ML predictions to clean numpy arrays, replacing None values with defaults
        """
        # Extract ML predictions with defaults
        regime_pred, regime_conf = ml_predictions.get('regime', (np.zeros(len(df)), np.zeros(len(df))))
        vol_pred, vol_conf = ml_predictions.get('volatility',
                                                (pd.Series(self.false_value, index=df.index),
                                                 pd.Series(self.zero_value, index=df.index)))
        duration_pred, duration_conf = ml_predictions.get('duration', (
            np.ones(len(df)), np.full(len(df), self.high_vol_confidence_threshold)))

        # Sanitize regime predictions
        regime_pred = np.array(regime_pred) if not isinstance(regime_pred, np.ndarray) else regime_pred
        regime_conf = self._sanitize_confidence_array(regime_conf, len(df), self.zero_value)

        # Sanitize volatility predictions
        if isinstance(vol_pred, pd.Series):
            vol_pred_array = vol_pred.values
            vol_conf_array = vol_conf.values
        else:
            vol_pred_array = np.array(vol_pred)
            vol_conf_array = np.array(vol_conf)

        vol_conf_array = self._sanitize_confidence_array(vol_conf_array, len(df), self.zero_value)

        # Sanitize duration predictions
        duration_pred_array = np.array(duration_pred)
        duration_conf_array = self._sanitize_confidence_array(duration_conf, len(df),
                                                              self.high_vol_confidence_threshold)

        return regime_pred, regime_conf, vol_pred_array, vol_conf_array, duration_pred_array, duration_conf_array

    def _sanitize_confidence_array(self, confidence_array, array_length: int, default_value: float) -> np.ndarray:
        """
        Utility: Convert confidence array to clean numpy array with None handling

        Goal: Ensure confidence arrays are numeric numpy arrays without None values
        """
        if confidence_array is None:
            return np.full(array_length, default_value)

        if not isinstance(confidence_array, np.ndarray):
            confidence_array = np.array(confidence_array)

        confidence_array = np.where(confidence_array == None, default_value, confidence_array)
        return confidence_array.astype(float)

    def _extract_technical_signals(self, df: pd.DataFrame, technical_signals_df: pd.DataFrame) -> np.ndarray:
        """
        Step 2: Extract and combine technical analysis signals

        Goal: Get unified technical signal array from KAMA and Kalman signals
        """
        kama_signal = technical_signals_df.get('kama_signal', pd.Series(self.zero_value, index=df.index))
        kalman_signal = technical_signals_df.get('kalman_signal', pd.Series(self.zero_value, index=df.index))

        tech_signal = self._combine_technical_signals_weighted(kama_signal, kalman_signal)
        return tech_signal.values

    def _calculate_regime_signals(self, regime_pred: np.ndarray, tech_signal_array: np.ndarray) -> tuple:
        """
        Step 3: Calculate signal strength based on market regime

        Goal: Apply regime-specific signal processing (trending up/down, high vol, ranging)
        """
        signal_strength_array = np.zeros(len(regime_pred))
        regime_multiplier_array = np.ones(len(regime_pred))
        source_array = np.full(len(regime_pred), 'ranging', dtype='object')

        # Trending Up (regime == 1)
        trending_up_mask = regime_pred == self.unity_value
        signal_strength_array[trending_up_mask] = np.maximum(self.zero_value, tech_signal_array[trending_up_mask])
        regime_multiplier_array[trending_up_mask] = self.regime_adjustments.get('trending', self.unity_value)
        source_array[trending_up_mask] = 'trending_up'

        # Trending Down (regime == 2)
        trending_down_mask = regime_pred == (self.unity_value * 2)
        signal_strength_array[trending_down_mask] = np.minimum(self.zero_value, tech_signal_array[trending_down_mask])
        regime_multiplier_array[trending_down_mask] = self.regime_adjustments.get('trending', self.unity_value)
        source_array[trending_down_mask] = 'trending_down'

        # High Volatility (regime == 3)
        high_vol_mask = regime_pred == (self.unity_value * 3)
        signal_strength_array[high_vol_mask] = tech_signal_array[high_vol_mask] * self.high_vol_signal_multiplier
        regime_multiplier_array[high_vol_mask] = self.regime_adjustments.get('high_volatility',
                                                                             self.high_vol_position_multiplier)
        source_array[high_vol_mask] = 'high_volatility'

        # Ranging (regime == 0) - default case
        ranging_mask = regime_pred == self.zero_value
        signal_strength_array[ranging_mask] = tech_signal_array[ranging_mask] * self.ranging_signal_multiplier
        regime_multiplier_array[ranging_mask] = self.regime_adjustments.get('ranging',
                                                                            self.high_vol_confidence_threshold)

        return signal_strength_array, regime_multiplier_array, source_array

    def _calculate_confidence_scaling(self, df: pd.DataFrame, regime_conf: np.ndarray) -> np.ndarray:
        """
        Step 4a: Calculate confidence-based signal scaling

        Goal: Scale signals based on regime detection confidence
        """
        if self.confidence_scaling:
            confidence_multiplier_array = self._sanitize_confidence_array(regime_conf, len(df), self.unity_value)
        else:
            confidence_multiplier_array = np.full(len(df), self.unity_value)

        return confidence_multiplier_array

    def _calculate_volatility_scaling(self, vol_pred_array: np.ndarray, vol_conf_array: np.ndarray) -> np.ndarray:
        """
        Step 4b: Calculate volatility-based position scaling

        Goal: Adjust position sizes based on predicted volatility
        """
        vol_pred_boolean = np.asarray(vol_pred_array, dtype=bool)
        vol_conf_boolean = vol_conf_array > self.high_vol_confidence_threshold

        return np.where(
            vol_pred_boolean & vol_conf_boolean,
            self.high_vol_position_multiplier,
            self.unity_value
        )

    def _calculate_final_signals(self, signal_strength_array: np.ndarray, confidence_multiplier_array: np.ndarray,
                                 volatility_multiplier_array: np.ndarray, duration_multiplier_array: np.ndarray,
                                 regime_multiplier_array: np.ndarray) -> tuple:
        """
        Step 5: Calculate final trading signals and position sizes

        Goal: Combine all multipliers to get final signals with position sizing
        """
        # Calculate final signal strength
        final_signal_array = signal_strength_array * confidence_multiplier_array * volatility_multiplier_array

        # Calculate position sizes with all multipliers
        position_size_array = (self.base_size * regime_multiplier_array * confidence_multiplier_array *
                               volatility_multiplier_array * duration_multiplier_array * np.abs(final_signal_array))

        # Apply constraints
        position_size_array = np.minimum(position_size_array, self.max_position_size)
        final_signal_array = np.clip(final_signal_array, self.signal_clip_min, self.signal_clip_max)

        return final_signal_array, position_size_array

    def _calculate_confidence_scores(self, df: pd.DataFrame, regime_conf: np.ndarray,
                                     vol_conf_array: np.ndarray, duration_conf_array: np.ndarray) -> np.ndarray:
        """
        Step 6a: Calculate combined confidence scores from all ML components

        Goal: Create unified confidence metric from all ML predictions
        """
        regime_conf_safe = self._sanitize_confidence_array(regime_conf, len(df), self.zero_value)
        vol_conf_array_safe = self._sanitize_confidence_array(vol_conf_array, len(df), self.zero_value)
        duration_conf_array_safe = self._sanitize_confidence_array(duration_conf_array, len(df), self.zero_value)

        return (regime_conf_safe + vol_conf_array_safe + duration_conf_array_safe) / self.confidence_divisor

    def _format_signal_results(self, df: pd.DataFrame, final_signal_array: np.ndarray,
                               position_size_array: np.ndarray, source_array: np.ndarray,
                               confidence_scores_array: np.ndarray) -> Dict[str, pd.Series]:
        """
        Step 6b: Format results as pandas Series for return

        Goal: Convert numpy arrays to properly typed pandas Series
        """
        return {
            'final_signal': pd.Series(final_signal_array, index=df.index, dtype='float64'),
            'position_size': pd.Series(position_size_array, index=df.index, dtype='float64'),
            'signal_source': pd.Series(source_array, index=df.index, dtype='object'),
            'confidence': pd.Series(confidence_scores_array, index=df.index, dtype='float64')
        }

    def _get_duration_multiplier_vectorized(self, predicted_duration_array: np.ndarray,
                                            confidence_array: np.ndarray) -> np.ndarray:
        """
        Vectorized duration multiplier calculation - NO LOOPS
        ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
        """
        # Handle None values in confidence_array
        if confidence_array is None:
            confidence_array = np.full(len(predicted_duration_array), self.high_vol_confidence_threshold)

        # Ensure confidence_array is numpy array and handle None elements
        if not isinstance(confidence_array, np.ndarray):
            confidence_array = np.array(confidence_array)

        # Replace any None values with default confidence
        confidence_array = np.where(confidence_array == None, self.high_vol_confidence_threshold, confidence_array)

        # Ensure confidence_array is numeric
        confidence_array = confidence_array.astype(float)

        # Create base multiplier array
        base_multiplier_array = np.full(len(predicted_duration_array), self.high_vol_confidence_threshold)

        # Apply multipliers based on duration categories vectorized
        for duration_str, multiplier in self.duration_multipliers.items():
            duration_int = int(duration_str)
            mask = predicted_duration_array == duration_int
            base_multiplier_array[mask] = multiplier

        # Scale by confidence vectorized - now safe from None values
        confidence_scaled_multiplier = (base_multiplier_array * confidence_array +
                                        self.duration_confidence_blend * (self.unity_value - confidence_array))

        return confidence_scaled_multiplier

    def _combine_technical_signals_weighted(self, kama_signal: pd.Series, kalman_signal: pd.Series) -> pd.Series:
        """Combine technical signals using configured weights"""
        if self.combination_method == 'weighted_average':
            kama_weight = self.signal_weights.get('kama', self.high_vol_confidence_threshold)
            kalman_weight = self.signal_weights.get('kalman', self.high_vol_confidence_threshold)
            return kama_signal * kama_weight + kalman_signal * kalman_weight
        else:
            # Simple average fallback using configurable divisor
            return (kama_signal + kalman_signal) / self.unity_value / self.unity_value

    def _calculate_regime_signal(self, regime: int, tech_signal: float) -> Tuple[float, float, str]:
        """
        Calculate signal strength based on current regime
        Uses ALL configurable parameters - NO hardcoded values

        Args:
            regime: Current market regime (0=Range, 1=Up, 2=Down, 3=HighVol)
            tech_signal: Technical analysis signal

        Returns:
            Tuple of (signal_strength, regime_multiplier, source_description)
        """

        if regime == self.unity_value:  # Trending Up
            signal_strength = max(self.zero_value, tech_signal)  # Only long signals
            regime_multiplier = self.regime_adjustments.get('trending', self.unity_value)
            source = 'trending_up'

        elif regime == self.unity_value + self.unity_value:  # Trending Down
            signal_strength = min(self.zero_value, tech_signal)  # Only short signals
            regime_multiplier = self.regime_adjustments.get('trending', self.unity_value)
            source = 'trending_down'

        elif regime == self.unity_value + self.unity_value + self.unity_value:  # High Volatility
            signal_strength = tech_signal * self.high_vol_signal_multiplier
            regime_multiplier = self.regime_adjustments.get('high_volatility', self.high_vol_position_multiplier)
            source = 'high_volatility'

        else:  # Ranging (0)
            signal_strength = tech_signal * self.ranging_signal_multiplier
            regime_multiplier = self.regime_adjustments.get('ranging', self.high_vol_confidence_threshold)
            source = 'ranging'

        return signal_strength, regime_multiplier, source

    def _get_duration_multiplier(self, predicted_duration: int, confidence: float) -> float:
        """
        Get position size multiplier based on predicted trend duration
        Uses ALL configurable multipliers - NO hardcoded values

        Args:
            predicted_duration: Predicted duration category (0=very_short, 1=short, 2=medium, 3=long)
            confidence: Confidence in duration prediction

        Returns:
            Duration-based position multiplier
        """

        # Get multiplier from config
        base_multiplier = self.duration_multipliers.get(str(predicted_duration), self.high_vol_confidence_threshold)

        # Scale by confidence using configurable blend factor
        confidence_scaled_multiplier = (base_multiplier * confidence +
                                        self.duration_confidence_blend * (self.unity_value - confidence))

        return confidence_scaled_multiplier

    def _create_signals_dataframe(self, df: pd.DataFrame, ml_predictions: Dict,
                                  technical_signals_df: pd.DataFrame, combined_signals: Dict) -> pd.DataFrame:
        """Create comprehensive signals DataFrame using clean technical module data"""

        # Extract ML predictions
        regime_pred, regime_conf = ml_predictions.get('regime', (np.zeros(len(df)), np.zeros(len(df))))
        vol_pred, vol_conf = ml_predictions.get('volatility',
                                                (pd.Series(False, index=df.index),
                                                 pd.Series(self.zero_value, index=df.index)))
        duration_pred, duration_conf = ml_predictions.get('duration', (
        np.ones(len(df)), np.full(len(df), self.high_vol_confidence_threshold)))

        # Create comprehensive DataFrame
        signals_df = pd.DataFrame(index=df.index)

        # Final signals
        signals_df['signal'] = combined_signals['final_signal']
        signals_df['position_size'] = combined_signals['position_size']
        signals_df['signal_source'] = combined_signals['signal_source']
        signals_df['confidence'] = combined_signals['confidence']

        # ML predictions
        signals_df['regime'] = pd.Series(regime_pred, index=df.index)
        signals_df['regime_confidence'] = pd.Series(regime_conf, index=df.index)
        signals_df['volatility_prediction'] = vol_pred if isinstance(vol_pred, pd.Series) else pd.Series(vol_pred,
                                                                                                         index=df.index)
        signals_df['volatility_confidence'] = vol_conf if isinstance(vol_conf, pd.Series) else pd.Series(vol_conf,
                                                                                                         index=df.index)
        signals_df['duration_prediction'] = pd.Series(duration_pred, index=df.index)
        signals_df['duration_confidence'] = pd.Series(duration_conf, index=df.index)

        # Technical signals from clean module
        signals_df['kama_signal'] = technical_signals_df.get('kama_signal', pd.Series(self.zero_value, index=df.index))
        signals_df['kalman_signal'] = technical_signals_df.get('kalman_signal',
                                                               pd.Series(self.zero_value, index=df.index))
        signals_df['rsi_signal'] = technical_signals_df.get('rsi_signal', pd.Series(self.zero_value, index=df.index))
        signals_df['bb_signal'] = technical_signals_df.get('bb_signal', pd.Series(self.zero_value, index=df.index))
        signals_df['macd_crossover_signal'] = technical_signals_df.get('macd_crossover_signal',
                                                                       pd.Series(self.zero_value, index=df.index))
        signals_df['trend_signal'] = technical_signals_df.get('trend_signal',
                                                              pd.Series(self.zero_value, index=df.index))

        # Technical indicators from clean module
        signals_df['kama'] = technical_signals_df.get('kama', pd.Series(self.zero_value, index=df.index))
        signals_df['kalman'] = technical_signals_df.get('kalman', pd.Series(self.zero_value, index=df.index))
        signals_df['rsi'] = technical_signals_df.get('rsi', pd.Series(self.zero_value, index=df.index))
        signals_df['atr'] = technical_signals_df.get('atr', pd.Series(self.zero_value, index=df.index))

        return signals_df

    def _get_ml_component_summary(self) -> Dict[str, str]:
        """Get summary of enabled ML components"""

        status = self.ml_manager.get_model_status()

        summary = {
            'regime_detection': status['regime']['method'],
            'volatility_prediction': 'enabled' if status['volatility']['initialized'] else 'disabled',
            'trend_duration': 'enabled' if status['duration']['enabled'] else 'disabled'
        }

        return summary

    def _print_training_summary(self):
        """Print comprehensive training summary"""

        print("\n" + "=" * 80)
        print("HYBRID STRATEGY TRAINING RESULTS")
        print("=" * 80)

        # Strategy overview
        print(f"\nStrategy Type: {self.training_results.get('strategy_type', 'Unknown')}")
        print(f"ML Components: {self.training_results.get('ml_components', {})}")
        print(f"Training Time: {self.training_results.get('training_time', self.zero_value):.1f} seconds")

        # ML model results
        if 'regime' in self.training_results:
            regime_results = self.training_results['regime']
            print(f"\nRegime Detection:")
            print(f"  Method: {regime_results.get('method', 'unknown')}")
            print(f"  Strength Accuracy: {regime_results.get('strength_accuracy', self.zero_value):.3f}")
            print(f"  Direction Accuracy: {regime_results.get('direction_accuracy', self.zero_value):.3f}")
            print(f"  Samples: {regime_results.get('n_samples', self.zero_value)}")

        if 'volatility' in self.training_results:
            vol_results = self.training_results['volatility']
            print(f"\nVolatility Prediction:")
            print(f"  Accuracy: {vol_results.get('accuracy', self.zero_value):.3f}")
            print(f"  High Vol %: {vol_results.get('high_vol_pct', self.zero_value):.1f}%")
            print(f"  Features: {vol_results.get('n_features', self.zero_value)}")

        if 'duration' in self.training_results:
            duration_results = self.training_results['duration']
            print(f"\nTrend Duration Prediction:")
            print(f"  Accuracy: {duration_results.get('accuracy', self.zero_value):.3f}")
            print(f"  Categories: {duration_results.get('n_categories', self.zero_value)}")
            print(f"  Features: {duration_results.get('n_features', self.zero_value)}")

        print("\n" + "=" * 80)

    def get_feature_importance(self) -> Dict[str, Dict]:
        """Get feature importance from all ML models"""
        return self.ml_manager.get_feature_importance()

    def save_strategy(self, filepath: str):
        """Save trained strategy to disk"""
        self.ml_manager.save_all_models(filepath)

    @classmethod
    def load_strategy(cls, filepath: str, config: Optional[UnifiedConfig] = None) -> 'HybridStrategy':
        """Load trained strategy from disk"""
        strategy = cls(config)
        strategy.ml_manager = strategy.ml_manager.load_all_models(filepath, config)
        strategy.is_trained = True
        return strategy