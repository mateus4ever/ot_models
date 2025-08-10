# ml_models/volatility_predictor.py
"""
Volatility Prediction ML Model

This module contains the VolatilityPredictor class which uses machine learning
to predict periods of high vs low volatility. Volatility is much more predictable
than price direction, making this a key component of the hybrid strategy.

High volatility predictions are used for:
- Position sizing (reduce size in high vol periods)
- Risk management (adjust stop losses)
- Signal filtering (be more conservative in volatile markets)

ABSOLUTE ZERO HARDCODED VALUES - EVERYTHING CONFIGURABLE
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityPredictor:
    """
    ML component for predicting volatility regimes

    ALL PARAMETERS MUST BE PROVIDED VIA CONFIG - ZERO HARDCODED VALUES
    """

    def __init__(self, config: Optional[object] = None):
        """
        Initialize the Volatility Predictor

        Args:
            config: UnifiedConfig object with model parameters
        """
        from src.hybrid.config.unified_config import get_config

        self.config = config or get_config()

        # Get model parameters - NO hardcoded defaults
        model_params = self.config.config.get('volatility_prediction', {}).get('model_params', {})
        if not model_params:
            raise ValueError("volatility_prediction.model_params must be configured in JSON config")

        self.model = RandomForestClassifier(**model_params)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None

        # Cache ALL config values - enforce configuration
        self._cache_config_values()
        self._validate_config()

    def _cache_config_values(self):
        """Cache ALL configuration values - EVERYTHING must come from config"""
        vol_config = self.config.config.get('volatility_prediction', {})
        general_config = self.config.config.get('general', {})

        # Core volatility settings
        self.forward_window = vol_config.get('forward_window')
        self.high_vol_percentile = vol_config.get('high_vol_percentile')
        self.feature_periods = vol_config.get('feature_periods')

        # General settings
        self.verbose = general_config.get('verbose')
        self.train_test_split = general_config.get('train_test_split')
        self.random_state = general_config.get('random_state')

        # Feature generation settings
        feature_config = vol_config.get('feature_generation', {})
        self.vol_window = feature_config.get('vol_window')
        self.vol_threshold_multiplier = feature_config.get('threshold_multiplier')
        self.min_samples = feature_config.get('min_samples')
        self.momentum_period = feature_config.get('momentum_period')
        self.volume_ma_period = feature_config.get('volume_ma_period')
        self.return_ma_period = feature_config.get('return_ma_period')
        self.skip_initial_rows = feature_config.get('skip_initial_rows')
        self.default_volume = feature_config.get('default_volume')

        # Previously hardcoded values - ALL configurable
        self.nan_replacement_value = feature_config.get('nan_replacement_value')
        self.gap_shift_periods = feature_config.get('gap_shift_periods')
        self.default_close_position = feature_config.get('default_close_position')
        self.zero_threshold = feature_config.get('zero_threshold')
        self.default_consecutive_value = feature_config.get('default_consecutive_value')
        self.consecutive_reset_value = feature_config.get('consecutive_reset_value')
        self.volume_default_ratio = feature_config.get('volume_default_ratio')
        self.default_fill_value = feature_config.get('default_fill_value')
        self.consecutive_window = feature_config.get('consecutive_window')
        self.consecutive_loop_start = feature_config.get('consecutive_loop_start')

        # Array indices and thresholds - ALL configurable
        self.min_periods_for_ratio = feature_config.get('min_periods_for_ratio')
        self.min_periods_for_long_ratio = feature_config.get('min_periods_for_long_ratio')
        self.mid_period_index = feature_config.get('mid_period_index')
        self.percentage_multiplier = feature_config.get('percentage_multiplier')
        self.astype_conversion_value = feature_config.get('astype_conversion_value')
        self.last_element_index = feature_config.get('last_element_index')
        self.axis_parameter = feature_config.get('axis_parameter')
        self.reverse_sort_flag = feature_config.get('reverse_sort_flag')

    def _validate_config(self):
        """Validate that ALL required config values are present"""
        required_values = [
            ('forward_window', self.forward_window),
            ('high_vol_percentile', self.high_vol_percentile),
            ('feature_periods', self.feature_periods),
            ('verbose', self.verbose),
            ('train_test_split', self.train_test_split),
            ('random_state', self.random_state),
            ('vol_window', self.vol_window),
            ('vol_threshold_multiplier', self.vol_threshold_multiplier),
            ('min_samples', self.min_samples),
            ('momentum_period', self.momentum_period),
            ('volume_ma_period', self.volume_ma_period),
            ('return_ma_period', self.return_ma_period),
            ('skip_initial_rows', self.skip_initial_rows),
            ('default_volume', self.default_volume),
            ('nan_replacement_value', self.nan_replacement_value),
            ('gap_shift_periods', self.gap_shift_periods),
            ('default_close_position', self.default_close_position),
            ('zero_threshold', self.zero_threshold),
            ('default_consecutive_value', self.default_consecutive_value),
            ('consecutive_reset_value', self.consecutive_reset_value),
            ('volume_default_ratio', self.volume_default_ratio),
            ('default_fill_value', self.default_fill_value),
            ('consecutive_window', self.consecutive_window),
            ('consecutive_loop_start', self.consecutive_loop_start),
            ('min_periods_for_ratio', self.min_periods_for_ratio),
            ('min_periods_for_long_ratio', self.min_periods_for_long_ratio),
            ('mid_period_index', self.mid_period_index),
            ('percentage_multiplier', self.percentage_multiplier),
            ('astype_conversion_value', self.astype_conversion_value),
            ('last_element_index', self.last_element_index),
            ('axis_parameter', self.axis_parameter),
            ('reverse_sort_flag', self.reverse_sort_flag)
        ]

        missing_values = [name for name, value in required_values if value is None]
        if missing_values:
            raise ValueError(f"Missing required config values: {missing_values}")

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features with ALL values from config"""
        features = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()

        # Use configured feature periods
        for period in self.feature_periods:
            features[f'vol_{period}'] = returns.rolling(period).std()

        # Safe ratio calculations with configurable thresholds
        if len(self.feature_periods) >= self.min_periods_for_ratio:
            short_period = min(self.feature_periods)
            long_period = max(self.feature_periods)
            features['vol_ratio_short'] = features[f'vol_{short_period}'] / features[f'vol_{long_period}'].replace(
                self.zero_threshold, np.nan)

        if len(self.feature_periods) >= self.min_periods_for_long_ratio:
            mid_period = sorted(self.feature_periods)[self.mid_period_index]
            long_period = max(self.feature_periods)
            features['vol_ratio_long'] = features[f'vol_{mid_period}'] / features[f'vol_{long_period}'].replace(
                self.zero_threshold, np.nan)

        # Price range features
        price_range = df['high'] - df['low']
        features['high_low'] = price_range / df['close']

        # Close position calculation with configurable defaults
        range_size = df['high'] - df['low']
        features['close_position'] = np.where(
            range_size > self.zero_threshold,
            (df['close'] - df['low']) / range_size,
            self.default_close_position
        )

        # Return features with configurable periods
        features['abs_return'] = returns.abs()
        features['return_magnitude_ma'] = features['abs_return'].rolling(self.return_ma_period).mean()

        # Gap features with configurable shift
        gap = (df['open'] - df['close'].shift(self.gap_shift_periods)) / df['close'].shift(self.gap_shift_periods)
        features['overnight_gap'] = gap.fillna(self.default_fill_value)
        features['gap_magnitude'] = features['overnight_gap'].abs()

        # Momentum with configurable period
        features[f'momentum_{self.momentum_period}'] = (
                    df['close'] / df['close'].shift(self.momentum_period) - self.default_consecutive_value).fillna(
            self.default_fill_value)

        # Intraday range with configurable threshold
        features['intraday_range'] = np.where(
            df['open'] > self.zero_threshold,
            price_range / df['open'],
            self.default_fill_value
        )

        # Volume with configurable defaults
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(self.volume_ma_period).mean()
            features['volume_ratio'] = (df['volume'] / vol_ma).fillna(self.volume_default_ratio)
        else:
            features['volume_ratio'] = self.default_volume

        # Store feature names
        self.feature_names = features.columns.tolist()

        # Clean NaN values with configurable fill
        features_clean = features.ffill().bfill()
        features_clean = features_clean.fillna(self.default_fill_value)

        # Skip initial rows (configurable)
        features_final = features_clean.iloc[self.skip_initial_rows:]

        if self.verbose:
            print(f"Final features shape: {features_final.shape}")
            print(f"NaN count: {features_final.isna().sum().sum()}")

        return features_final

    def _calculate_consecutive_moves(self, returns: pd.Series, window: int = None) -> pd.Series:
        """Calculate consecutive moves with ALL values configurable"""
        if window is None:
            window = self.consecutive_window

        direction = np.sign(returns)
        consecutive = pd.Series(self.consecutive_reset_value, index=returns.index)

        for i in range(self.consecutive_loop_start, len(returns)):
            if direction.iloc[i] == direction.iloc[i - self.default_consecutive_value] and direction.iloc[
                i] != self.zero_threshold:
                consecutive.iloc[i] = consecutive.iloc[
                                          i - self.default_consecutive_value] + self.default_consecutive_value
            else:
                consecutive.iloc[i] = self.default_consecutive_value if direction.iloc[
                                                                            i] != self.zero_threshold else self.consecutive_reset_value

        return consecutive.rolling(window).mean()

    def create_volatility_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Create labels with ALL parameters from config"""
        returns = df['close'].pct_change()

        # Calculate future volatility
        future_vol = returns.rolling(self.forward_window).std().shift(-self.forward_window)

        # Calculate historical volatility
        hist_vol = returns.rolling(self.vol_window).std()

        # Compare with configurable threshold
        labels = (future_vol > hist_vol * self.vol_threshold_multiplier).astype(self.astype_conversion_value)

        # Remove NaN values
        labels = labels.dropna()
        return labels.values

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train with ALL parameters configurable"""
        if self.verbose:
            print("Training Volatility Predictor...")

        features_df = self.create_volatility_features(df)
        labels = self.create_volatility_labels(df)

        # Align features and labels
        min_len = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_len]
        labels = labels[:min_len]

        if len(features_df) < self.min_samples:
            if self.verbose:
                print(f"Insufficient data for volatility prediction (need {self.min_samples}, got {len(features_df)})")
            return {}

        # Train/test split with configurable ratio
        split_idx = int(len(features_df) * self.train_test_split)

        X_train = features_df.iloc[:split_idx]
        X_test = features_df.iloc[split_idx:]
        y_train = labels[:split_idx]
        y_test = labels[split_idx:]

        # Scale and train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate high_vol_pct regardless of verbose setting - FIXED
        high_vol_pct = np.mean(y_test) * self.percentage_multiplier

        if self.verbose:
            print(f"Volatility Prediction Accuracy: {accuracy:.3f}")
            print(f"High volatility periods: {high_vol_pct:.1f}% of test data")

        self.is_trained = True
        return {
            'accuracy': accuracy,
            'n_samples': len(X_test),
            'high_vol_pct': high_vol_pct,
            'n_features': len(self.feature_names)
        }

    def predict_volatility(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ALL values configurable"""
        if not self.is_trained:
            return np.zeros(len(df)), np.zeros(len(df))

        features_df = self.create_volatility_features(df)
        if len(features_df) == self.zero_threshold:
            return np.zeros(len(df)), np.zeros(len(df))

        X_scaled = self.scaler.transform(features_df)
        predictions = self.model.predict(X_scaled)
        confidences = np.max(self.model.predict_proba(X_scaled), axis=self.axis_parameter)

        # Align with original dataframe
        full_predictions = np.zeros(len(df))
        full_confidences = np.zeros(len(df))

        start_idx = len(df) - len(predictions)
        full_predictions[start_idx:] = predictions
        full_confidences[start_idx:] = confidences

        return full_predictions, full_confidences

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance with configurable sorting"""
        if not self.is_trained or self.feature_names is None:
            return {}

        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(
            sorted(importance_dict.items(), key=lambda x: x[self.axis_parameter], reverse=self.reverse_sort_flag))

    def save_model(self, filepath: str):
        """Save the trained model"""
        import joblib

        if self.is_trained:
            joblib.dump(self, filepath)
            logger.info(f"VolatilityPredictor saved to {filepath}")
        else:
            logger.warning("Cannot save untrained model")

    @classmethod
    def load_model(cls, filepath: str) -> 'VolatilityPredictor':
        """Load a trained model"""
        import joblib

        model = joblib.load(filepath)
        logger.info(f"VolatilityPredictor loaded from {filepath}")
        return model

    def analyze_volatility_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze patterns with ALL thresholds configurable"""
        if not self.is_trained:
            return {}

        features_df = self.create_volatility_features(df)
        predictions, confidences = self.predict_volatility(df)

        returns = df['close'].pct_change()
        actual_vol = returns.rolling(self.forward_window).std()

        # Calculate metrics using configurable values only
        high_vol_periods = np.sum(predictions)
        total_periods = len(predictions[predictions > self.zero_threshold])

        if total_periods > self.zero_threshold:
            high_vol_frequency = high_vol_periods / total_periods
            avg_confidence = np.mean(confidences[
                                         predictions == self.default_consecutive_value]) if high_vol_periods > self.zero_threshold else self.zero_threshold
        else:
            high_vol_frequency = self.zero_threshold
            avg_confidence = self.zero_threshold

        return {
            'high_vol_frequency': high_vol_frequency,
            'avg_confidence_high_vol': avg_confidence,
            'total_predictions': total_periods,
            'current_vol_regime': predictions[-self.last_element_index] if len(
                predictions) > self.zero_threshold else self.zero_threshold
        }