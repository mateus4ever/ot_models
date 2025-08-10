"""
Trend Duration Predictor - ML Model for Predicting How Long Trends Will Last
Uses the same configurable approach as the feature generator system
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import logging
import time as timer

from src.hybrid.config.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class TrendDurationPredictor:
    """
    ML model to predict how long current trends will continue

    Uses configurable feature generation and fully configurable parameters
    More valuable than predicting price direction
    """

    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()

        # Load configuration sections (same pattern as feature generator)
        self.numeric_formatting = self.config.numeric_formatting
        self.array_indexing = self.config.array_indexing
        self.math_ops = self.config.mathematical_operations

        # Load trend duration specific configuration
        self.duration_config = self.config.config.get('trend_duration_prediction', {})
        self.model_params = self.duration_config.get('model_params', {})
        self.feature_config = self.duration_config.get('feature_generation', {})
        self.label_config = self.duration_config.get('label_generation', {})

        # Initialize model with configurable parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': self.config.config.get('general', {}).get('random_state', 42)
        }
        model_config = {**default_params, **self.model_params}

        self.model = RandomForestClassifier(**model_config)
        self.scaler = StandardScaler()

        self.is_trained = False
        self.feature_names = None

        # Duration categories from configuration
        self.duration_categories = self.label_config.get('duration_categories', {
            'very_short': {'min_periods': 0, 'max_periods': 15, 'label': 0},
            'short': {'min_periods': 15, 'max_periods': 60, 'label': 1},
            'medium': {'min_periods': 60, 'max_periods': 180, 'label': 2},
            'long': {'min_periods': 180, 'max_periods': 999999, 'label': 3}
        })

    def create_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that predict trend exhaustion using configurable parameters
        """
        features = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()

        # Get feature configuration
        periods = self.feature_config.get('momentum_periods', [5, 20, 60, 240])
        volatility_periods = self.feature_config.get('volatility_periods', [5, 20, 60])
        ma_periods = self.feature_config.get('moving_average_periods', [20, 60, 240])

        print(f"Generating trend duration features with {len(periods)} momentum periods...")

        # === TREND MOMENTUM DECAY FEATURES ===
        # How is momentum changing over time?
        momentum_features = self._create_momentum_decay_features(df, returns, periods)
        features = pd.concat([features, momentum_features], axis=1)

        # === VOLATILITY PATTERN FEATURES ===
        # Trends often end with volatility spikes
        volatility_features = self._create_volatility_pattern_features(df, returns, volatility_periods)
        features = pd.concat([features, volatility_features], axis=1)

        # === TREND MATURITY FEATURES ===
        # How long has current trend been running?
        maturity_features = self._create_trend_maturity_features(df, ma_periods)
        features = pd.concat([features, maturity_features], axis=1)

        # === EXHAUSTION SIGNAL FEATURES ===
        # RSI divergence, volume patterns, etc.
        exhaustion_features = self._create_exhaustion_features(df, returns)
        features = pd.concat([features, exhaustion_features], axis=1)

        # === REVERSION PRESSURE FEATURES ===
        # Distance from moving averages (overextension)
        reversion_features = self._create_reversion_pressure_features(df, ma_periods)
        features = pd.concat([features, reversion_features], axis=1)

        # Clean features using same pattern as feature generator
        features = self._clean_features(features)

        # Apply buffer periods
        buffer_periods = self.feature_config.get('buffer_periods', 240)
        features_final = features.iloc[buffer_periods:]

        print(f"Trend duration features: {features_final.shape}")
        return features_final

    def _create_momentum_decay_features(self, df: pd.DataFrame, returns: pd.Series, periods: list) -> pd.DataFrame:
        """Create momentum decay features using configurable periods"""
        features = pd.DataFrame(index=df.index)

        # Calculate momentum for each period
        momentums = {}
        for period in periods:
            col_name = f'momentum_{period}'
            momentums[period] = df['close'] / df['close'].shift(period) - self.math_ops['unity']
            features[col_name] = momentums[period]

        # Calculate momentum acceleration/deceleration
        if len(periods) >= self.math_ops['unity'] + self.math_ops['unity']:  # >= 2
            first_idx = self.array_indexing['first_index']
            second_idx = self.array_indexing['second_index']

            features['momentum_acceleration'] = (momentums[periods[first_idx]] /
                                                 momentums[periods[second_idx]])

            if len(periods) >= self.math_ops['unity'] + self.math_ops['unity'] + self.math_ops['unity']:  # >= 3
                third_idx = self.array_indexing['third_index']
                features['momentum_decay'] = (momentums[periods[second_idx]] /
                                              momentums[periods[third_idx]])

        return features

    def _create_volatility_pattern_features(self, df: pd.DataFrame, returns: pd.Series, periods: list) -> pd.DataFrame:
        """Create volatility pattern features using configurable periods"""
        features = pd.DataFrame(index=df.index)

        # Calculate volatility for each period
        volatilities = {}
        for period in periods:
            col_name = f'volatility_{period}'
            volatilities[period] = returns.rolling(period).std()
            features[col_name] = volatilities[period]

        # Calculate volatility expansion (trend exhaustion signal)
        if len(periods) >= self.math_ops['unity'] + self.math_ops['unity']:  # >= 2
            first_idx = self.array_indexing['first_index']
            second_idx = self.array_indexing['second_index']

            features['volatility_expansion'] = (volatilities[periods[first_idx]] /
                                                volatilities[periods[second_idx]])

            # Volatility regime shift
            regime_period = self.feature_config.get('volatility_regime_period', 60)
            mid_vol = volatilities[periods[second_idx]]
            features['volatility_regime'] = (mid_vol /
                                             mid_vol.shift(self.math_ops['unity']).rolling(regime_period).mean())

        return features

    def _create_trend_maturity_features(self, df: pd.DataFrame, ma_periods: list) -> pd.DataFrame:
        """Create trend maturity features using configurable MA periods"""
        features = pd.DataFrame(index=df.index)

        # Calculate trend direction based on shortest MA
        short_ma_period = ma_periods[self.array_indexing['first_index']]
        sma_short = df['close'].rolling(short_ma_period).mean()
        trend_direction = np.where(df['close'] > sma_short, self.math_ops['unity'], -self.math_ops['unity'])

        # Count consecutive periods in same direction (trend age)
        trend_age = pd.Series(index=df.index, dtype=int)
        current_streak = self.math_ops['zero']
        last_direction = self.math_ops['zero']

        for i in range(len(trend_direction)):
            if trend_direction[i] == last_direction:
                current_streak += self.math_ops['unity']
            else:
                current_streak = self.math_ops['unity']
                last_direction = trend_direction[i]
            trend_age.iloc[i] = current_streak

        features['trend_age'] = trend_age

        # Normalize trend age by configurable window
        normalization_window = self.feature_config.get('trend_age_normalization_window', 240)
        features['trend_age_normalized'] = trend_age / normalization_window

        # Trend acceleration (is trend getting stronger or weaker?)
        if len(ma_periods) >= self.math_ops['unity'] + self.math_ops['unity']:  # >= 2
            medium_ma_period = ma_periods[self.array_indexing['second_index']]
            sma_medium = df['close'].rolling(medium_ma_period).mean()

            # Trend strength relative to different timeframes
            features['trend_strength_short'] = (df['close'] - sma_short) / sma_short
            features['trend_strength_medium'] = (df['close'] - sma_medium) / sma_medium

            # Trend consistency across timeframes
            features['trend_consistency'] = (
                        features['trend_strength_short'] * features['trend_strength_medium']).apply(np.sign)

        return features

    def _create_exhaustion_features(self, df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Create trend exhaustion signal features using configurable parameters"""
        features = pd.DataFrame(index=df.index)

        # RSI calculation and divergence
        rsi_period = self.feature_config.get('rsi_period', 14)
        rsi = self._calculate_rsi(df['close'], rsi_period)
        features['rsi'] = rsi

        # RSI divergence detection
        divergence_periods = self.feature_config.get('rsi_divergence_periods', [5, 10])
        for period in divergence_periods:
            # Price vs RSI divergence
            price_change = df['close'] > df['close'].shift(period)
            rsi_change = rsi > rsi.shift(period)
            features[f'rsi_divergence_{period}'] = (price_change != rsi_change).astype(int)

        # Volume exhaustion (if volume data available)
        if 'volume' in df.columns:
            volume_ma_period = self.feature_config.get('volume_ma_period', 20)
            volume_sma = df['volume'].rolling(volume_ma_period).mean()
            volume_threshold = self.feature_config.get('volume_exhaustion_threshold', 0.8)
            features['volume_exhaustion'] = (df['volume'] < volume_sma * volume_threshold).astype(int)
        else:
            # Create dummy volume exhaustion feature
            features['volume_exhaustion'] = self.math_ops['zero']

        # Price momentum exhaustion
        momentum_exhaustion_period = self.feature_config.get('momentum_exhaustion_period', 10)
        price_momentum = returns.rolling(momentum_exhaustion_period).mean()
        momentum_threshold = self.feature_config.get('momentum_exhaustion_threshold', 0.001)
        features['momentum_exhaustion'] = (abs(price_momentum) < momentum_threshold).astype(int)

        return features

    def _create_reversion_pressure_features(self, df: pd.DataFrame, ma_periods: list) -> pd.DataFrame:
        """Create mean reversion pressure features using configurable MA periods"""
        features = pd.DataFrame(index=df.index)

        # Distance from moving averages (overextension)
        for period in ma_periods:
            sma = df['close'].rolling(period).mean()
            features[f'ma_distance_{period}'] = (df['close'] - sma) / sma
            features[f'overextension_{period}'] = abs(features[f'ma_distance_{period}'])

        # Bollinger Band-like overextension
        if len(ma_periods) >= self.math_ops['unity']:
            base_period = ma_periods[self.array_indexing['first_index']]
            bb_period = self.feature_config.get('bollinger_period', base_period)
            bb_std_multiplier = self.feature_config.get('bollinger_std_multiplier', 2.0)

            bb_sma = df['close'].rolling(bb_period).mean()
            bb_std = df['close'].rolling(bb_period).std()

            upper_band = bb_sma + (bb_std * bb_std_multiplier)
            lower_band = bb_sma - (bb_std * bb_std_multiplier)

            features['bb_position'] = (df['close'] - bb_sma) / (upper_band - lower_band)
            features['bb_overextension'] = (
                    (df['close'] > upper_band).astype(int) +
                    (df['close'] < lower_band).astype(int) * -self.math_ops['unity']
            )

        return features

    def create_duration_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create labels for how long trends will continue using configurable categories
        """

        # Get label generation parameters
        trend_detection_period = self.label_config.get('trend_detection_period', 20)
        forward_window = self.label_config.get('forward_window', 240)

        # Calculate trend direction
        sma = df['close'].rolling(trend_detection_period).mean()
        current_trend = np.where(df['close'] > sma, self.math_ops['unity'], -self.math_ops['unity'])

        duration_labels = []

        # Calculate buffer for safe prediction
        buffer_periods = self.feature_config.get('buffer_periods', 240)

        for i in range(len(df) - forward_window - buffer_periods):
            # Current trend direction at feature time
            feature_index = i + buffer_periods
            current_direction = current_trend[feature_index]

            # Look forward to see how long trend continues
            continuation_periods = self.math_ops['zero']
            for j in range(self.math_ops['unity'], forward_window + self.math_ops['unity']):
                future_index = feature_index + j
                if future_index >= len(current_trend):
                    break
                if current_trend[future_index] == current_direction:
                    continuation_periods += self.math_ops['unity']
                else:
                    break

            # Convert to duration category using configuration
            duration_label = self._periods_to_category(continuation_periods)
            duration_labels.append(duration_label)

        print(f"Duration labels: {len(duration_labels)} samples")
        return np.array(duration_labels)

    def _periods_to_category(self, periods: int) -> int:
        """Convert continuation periods to duration category using configuration"""

        for category_name, category_config in self.duration_categories.items():
            min_periods = category_config['min_periods']
            max_periods = category_config['max_periods']

            if min_periods <= periods <= max_periods:
                return category_config['label']

        # Default to longest category if no match
        return max(cat['label'] for cat in self.duration_categories.values())

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the trend duration predictor using configurable parameters
        """
        training_config = self.duration_config.get('training', {})
        verbose = training_config.get('verbose', True)

        if verbose:
            print("Training Trend Duration Predictor...")

        # Generate features and labels
        start_time = timer.time()
        features_df = self.create_duration_features(df)
        print(f"Feature generation took: {timer.time() - start_time:.1f} seconds")

        start_time = timer.time()
        duration_labels = self.create_duration_labels(df)
        print(f"Label generation took: {timer.time() - start_time:.1f} seconds")

        # Align features and labels
        min_len = min(len(features_df), len(duration_labels))
        features_df = features_df.iloc[:min_len]
        duration_labels = duration_labels[:min_len]

        # Check minimum samples
        min_samples = training_config.get('min_samples', 1000)
        if len(features_df) < min_samples:
            if verbose:
                print(f"Insufficient data for duration prediction (need {min_samples}, got {len(features_df)})")
            return {}

        # Train/test split using configuration
        train_test_ratio = training_config.get('train_test_split', 0.7)
        random_state = training_config.get('random_state', 42)
        use_stratify = training_config.get('stratify', True)

        stratify_param = duration_labels if use_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            features_df.values, duration_labels,
            train_size=train_test_ratio,
            random_state=random_state,
            stratify=stratify_param
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Store feature names
        self.feature_names = features_df.columns.tolist()
        self.is_trained = True

        if verbose:
            print(f"\n=== TREND DURATION RESULTS ===")
            precision = self.numeric_formatting['decimal_precision'].get('price', 3)
            print(f"Duration Prediction Accuracy: {accuracy:.{precision}f}")

            # Category distribution
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set distribution:")
            for cat_id, count in zip(unique, counts):
                cat_name = self._get_category_name(cat_id)
                pct = count / len(y_test) * self.numeric_formatting['percentage_conversion']['multiplier']
                print(f"  {cat_name}: {pct:.1f}%")

        return {
            'accuracy': accuracy,
            'n_samples': len(X_test),
            'n_features': len(self.feature_names),
            'n_categories': len(self.duration_categories)
        }

    def predict_duration(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict trend duration for new data
        """
        if not self.is_trained:
            return np.zeros(len(df))

        features_df = self.create_duration_features(df)
        if len(features_df) == self.math_ops['zero']:
            return np.zeros(len(df))

        X_scaled = self.scaler.transform(features_df.values)
        predictions = self.model.predict(X_scaled)

        # Align with original dataframe length
        full_predictions = np.zeros(len(df))
        start_idx = len(df) - len(predictions)
        full_predictions[start_idx:] = predictions

        return full_predictions

    def _get_category_name(self, category_id: int) -> str:
        """Get human-readable category name from ID"""
        for name, config in self.duration_categories.items():
            if config['label'] == category_id:
                return name.replace('_', ' ').title()
        return f"Category_{category_id}"

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using configurable period"""
        delta = prices.diff()
        gain = (delta.where(delta > self.math_ops['zero'], self.math_ops['zero'])).rolling(window=period).mean()
        loss = (-delta.where(delta < self.math_ops['zero'], self.math_ops['zero'])).rolling(window=period).mean()
        rs = gain / loss
        rsi_multiplier = self.numeric_formatting.get('rsi_multiplier', 100)
        return rsi_multiplier - (rsi_multiplier / (self.math_ops['unity'] + rs))

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean features using same pattern as feature generator"""
        features_clean = features.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.ffill().bfill()
        features_clean = features_clean.fillna(self.math_ops['zero'])
        return features_clean

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or self.feature_names is None:
            return {}

        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance_dict.items(),
                           key=lambda x: x[self.array_indexing['second_index']],
                           reverse=True))

    def save_model(self, filepath: str):
        """Save trained model using same pattern as other components"""
        import joblib

        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'duration_categories': self.duration_categories,
                'config': self.config.config
            }
            joblib.dump(model_data, filepath)
            logger.info(f"TrendDurationPredictor saved to {filepath}")
        else:
            logger.warning("Cannot save untrained model")

    @classmethod
    def load_model(cls, filepath: str) -> 'TrendDurationPredictor':
        """Load trained model from disk"""
        import joblib

        model_data = joblib.load(filepath)

        # Reconstruct config
        config = UnifiedConfig()
        config.config.update(model_data.get('config', {}))

        predictor = cls(config)
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.duration_categories = model_data['duration_categories']
        predictor.is_trained = True

        logger.info(f"TrendDurationPredictor loaded from {filepath}")
        return predictor