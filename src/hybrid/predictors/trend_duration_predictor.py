import logging
import time as timer
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.predictors.predictor_interface import PredictorInterface

logger = logging.getLogger(__name__)


class TrendDurationPredictor(PredictorInterface):
    """
    Trend Duration Predictor - ML model for predicting how long trends will last.

    Purpose:
        Predicts whether current price trends will continue for a short, medium,
        or long duration. Used to inform position sizing and exit timing.

    How it works:
        1. Detects current trend direction (price above/below short MA)
        2. Extracts features that signal trend exhaustion:
           - Momentum decay (is trend slowing down?)
           - Volatility expansion (often precedes reversals)
           - RSI divergence (price vs momentum disagreement)
           - Mean reversion pressure (distance from moving averages)
        3. Classifies expected duration into categories:
           - very_short: 0-5 periods
           - short: 6-15 periods
           - medium: 16-30 periods
           - long: 31+ periods

    Uses incremental feature caching to avoid recomputing features
    for unchanged historical data.
    """

    def __init__(self, config: UnifiedConfig):
        if config is None:
            raise ValueError("UnifiedConfig is required")

        self.config = config

        # Load trend duration specific configuration
        self.duration_config = self.config.get_section('trend_duration_prediction')
        if not self.duration_config:
            raise ValueError("trend_duration_prediction section must be configured in JSON config")

        ml_config = self.duration_config['ml']['parameters']
        self.feature_config = ml_config['feature_generation']
        self.label_config = ml_config['label_generation']
        self.training_config = ml_config['training']

        # Initialize model
        model_params = ml_config['model_params']
        self.model = RandomForestClassifier(**model_params)
        self.scaler = StandardScaler()

        self._is_trained = False
        self.feature_names = None

        # Feature caching
        self.feature_cache = None
        self.last_processed_length = 0
        self.cache_index_reference = None

        # Duration categories from config
        self.duration_categories = self.label_config['duration_categories']

    def _get_incremental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get features using incremental computation where possible."""
        if self._needs_full_recomputation(df):
            return self._compute_all_features(df)

        if len(df) <= self.last_processed_length:
            return self._get_cached_features(len(df))

        return self._compute_incremental_features(df)

    def _needs_full_recomputation(self, df: pd.DataFrame) -> bool:
        """Check if full feature recomputation is needed."""
        return (self.feature_cache is None or
                self.last_processed_length == 0 or
                not self._is_cache_valid(df))

    def _compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features from scratch."""
        if self.feature_cache is not None:
            logger.debug('Cache invalidated, recomputing all features')

        self.feature_cache = self.create_duration_features(df)
        self.last_processed_length = len(df)
        self.cache_index_reference = df.index.copy()
        return self.feature_cache

    def _get_cached_features(self, current_length: int) -> pd.DataFrame:
        """Return cached features, truncated if needed."""
        if current_length < self.last_processed_length:
            cache_end_position = current_length - self._get_buffer_periods()
            return self.feature_cache.iloc[:cache_end_position]
        return self.feature_cache

    def _compute_incremental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features only for new rows, merge with cache."""
        current_length = len(df)
        new_rows_count = current_length - self.last_processed_length
        logger.debug(f'Computing features for {new_rows_count} new rows (total: {current_length})')

        lookback_periods = self._get_max_lookback_period()
        computation_start = max(0, self.last_processed_length - lookback_periods)
        extended_df = df.iloc[computation_start:]

        extended_features = self.create_duration_features(extended_df)

        buffer_periods = self._get_buffer_periods()
        cache_end_index = self.last_processed_length - buffer_periods
        new_feature_start = cache_end_index - computation_start

        if 0 <= new_feature_start < len(extended_features):
            return self._merge_new_features(df, extended_features, new_feature_start)

        logger.warning('Incremental computation failed, falling back to full recomputation')
        return self._compute_all_features(df)

    def _merge_new_features(self, df: pd.DataFrame, extended_features: pd.DataFrame,
                            new_feature_start: int) -> pd.DataFrame:
        """Merge new features with cache and enforce size limit."""
        new_features = extended_features.iloc[new_feature_start:]
        self.feature_cache = pd.concat([self.feature_cache, new_features], axis=0)

        max_cache_size = self.duration_config['ml']['parameters'].get('max_cache_size', 100000)
        if len(self.feature_cache) > max_cache_size:
            self.feature_cache = self.feature_cache.iloc[-max_cache_size:]

        self.feature_cache = self.feature_cache.sort_index()
        self.last_processed_length = len(df)
        self.cache_index_reference = df.index.copy()

        return self.feature_cache

    def _is_cache_valid(self, df: pd.DataFrame) -> bool:
        """Check if cached features are still valid for the current dataset."""
        if self.cache_index_reference is None:
            return False

        cache_length = len(self.cache_index_reference)
        if cache_length > len(df):
            return False

        return df.index[:cache_length].equals(self.cache_index_reference)

    def _get_max_lookback_period(self) -> int:
        """Get maximum lookback period needed for feature computation."""
        momentum_periods = self.feature_config['momentum_periods']
        volatility_periods = self.feature_config['volatility_periods']
        ma_periods = self.feature_config['moving_average_periods']
        rsi_period = self.feature_config['rsi_period']
        bollinger_period = self.feature_config['bollinger_period']
        normalization_window = self.feature_config['trend_age_normalization_window']

        max_period = max(
            max(momentum_periods),
            max(volatility_periods),
            max(ma_periods),
            rsi_period,
            bollinger_period,
            normalization_window
        )

        safety_multiplier = self.feature_config.get('lookback_safety_multiplier', 2)
        return max_period * safety_multiplier

    def _get_buffer_periods(self) -> int:
        """Get buffer periods from configuration."""
        return self.feature_config['buffer_periods']
    def _fallback_full_computation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback to full feature computation when incremental fails - ALL VALUES CONFIGURABLE"""
        self.feature_cache = self.create_duration_features(df)
        self.last_processed_length = len(df)
        self.cache_index_reference = df.index.copy()
        return self.feature_cache

    def create_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that predict trend exhaustion."""
        features = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()

        periods = self.feature_config['momentum_periods']
        volatility_periods = self.feature_config['volatility_periods']
        ma_periods = self.feature_config['moving_average_periods']

        # === TREND MOMENTUM DECAY FEATURES ===
        momentum_features = self._create_momentum_decay_features(df, returns, periods)
        features = pd.concat([features, momentum_features], axis=1)

        # === VOLATILITY PATTERN FEATURES ===
        volatility_features = self._create_volatility_pattern_features(df, returns, volatility_periods)
        features = pd.concat([features, volatility_features], axis=1)

        # === TREND MATURITY FEATURES ===
        maturity_features = self._create_trend_maturity_features(df, ma_periods)
        features = pd.concat([features, maturity_features], axis=1)

        # === EXHAUSTION SIGNAL FEATURES ===
        exhaustion_features = self._create_exhaustion_features(df, returns)
        features = pd.concat([features, exhaustion_features], axis=1)

        # === REVERSION PRESSURE FEATURES ===
        reversion_features = self._create_reversion_pressure_features(df, ma_periods)
        features = pd.concat([features, reversion_features], axis=1)

        features = self._clean_features(features)

        buffer_periods = self._get_buffer_periods()
        features_final = features.iloc[buffer_periods:]

        logger.debug(f'Trend duration features: {features_final.shape}')
        return features_final

    def predict(self, df: pd.DataFrame) -> Dict:
        """Predict trend duration using incremental feature computation."""
        if not self._is_trained:
            return {
                'predictions': np.zeros(len(df)),
                'success': False,
                'reason': 'Model not trained'
            }

        features_df = self._get_incremental_features(df)

        if len(features_df) == 0:
            return {
                'predictions': np.zeros(len(df)),
                'success': False,
                'reason': 'No features generated'
            }

        X_scaled = self.scaler.transform(features_df.values)
        predictions = self.model.predict(X_scaled)

        full_predictions = np.zeros(len(df))
        start_idx = len(df) - len(predictions)
        full_predictions[start_idx:] = predictions

        return {
            'predictions': full_predictions,
            'success': True
        }

    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache = None
        self.last_processed_length = 0
        self.cache_index_reference = None
        logger.debug('Feature cache cleared')

    def get_cache_info(self) -> Dict[str, any]:
        """Get information about current cache state."""
        cache_exists = self.feature_cache is not None
        cache_length = len(self.feature_cache) if cache_exists else 0
        cache_memory = self.feature_cache.memory_usage(deep=True).sum() if cache_exists else 0

        return {
            'cache_exists': cache_exists,
            'cache_length': cache_length,
            'last_processed_length': self.last_processed_length,
            'cache_memory_usage': cache_memory
        }

    def _create_momentum_decay_features(self, df: pd.DataFrame, returns: pd.Series, periods: list) -> pd.DataFrame:
        """Create momentum decay features."""
        features = pd.DataFrame(index=df.index)

        momentums = {}
        for period in periods:
            col_name = f'momentum_{period}'
            momentums[period] = df['close'] / df['close'].shift(period) - 1
            features[col_name] = momentums[period]

        # Need at least 2 periods to calculate acceleration (short/medium momentum ratio)
        if len(periods) >= 2:
            features['momentum_acceleration'] = momentums[periods[0]] / momentums[periods[1]]

            # Need at least 3 periods to calculate decay (medium/long momentum ratio)
            if len(periods) >= 3:
                features['momentum_decay'] = momentums[periods[1]] / momentums[periods[2]]

        return features

    def _create_volatility_pattern_features(self, df: pd.DataFrame, returns: pd.Series, periods: list) -> pd.DataFrame:
        """Create volatility pattern features."""
        features = pd.DataFrame(index=df.index)

        volatilities = {}
        for period in periods:
            col_name = f'volatility_{period}'
            volatilities[period] = returns.rolling(period).std()
            features[col_name] = volatilities[period]

        # Need at least 2 periods to calculate expansion ratio (short/long)
        if len(periods) >= 2:
            features['volatility_expansion'] = volatilities[periods[0]] / volatilities[periods[1]]

            regime_period = self.feature_config['volatility_regime_period']
            mid_vol = volatilities[periods[1]]
            features['volatility_regime'] = mid_vol / mid_vol.shift(1).rolling(regime_period).mean()

        return features

    def _create_trend_maturity_features(self, df: pd.DataFrame, ma_periods: list) -> pd.DataFrame:
        """Create trend maturity features - how old is the trend and is it strengthening or weakening."""
        features = pd.DataFrame(index=df.index)

        # Calculate trend direction based on shortest MA
        short_ma_period = ma_periods[0]
        sma_short = df['close'].rolling(short_ma_period).mean()
        trend_direction = np.where(df['close'] > sma_short, 1, -1)

        # Count consecutive periods in same direction (trend age)
        trend_age = pd.Series(index=df.index, dtype=int)
        current_streak = 0
        last_direction = None

        for i in range(len(trend_direction)):
            if trend_direction[i] == last_direction:
                current_streak += 1
            else:
                current_streak = 1
                last_direction = trend_direction[i]
            trend_age.iloc[i] = current_streak

        features['trend_age'] = trend_age

        # Normalize trend age for model input (raw count would dominate other features)
        normalization_window = self.feature_config['trend_age_normalization_window']
        features['trend_age_normalized'] = trend_age / normalization_window

        # Need at least 2 MA periods to compare short vs medium term strength
        if len(ma_periods) >= 2:
            medium_ma_period = ma_periods[1]
            sma_medium = df['close'].rolling(medium_ma_period).mean()

            # Trend strength: how far price is from MA (as percentage)
            features['trend_strength_short'] = (df['close'] - sma_short) / sma_short
            features['trend_strength_medium'] = (df['close'] - sma_medium) / sma_medium

            # Trend consistency: +1 if both agree, -1 if diverging
            features['trend_consistency'] = (
                    features['trend_strength_short'] * features['trend_strength_medium']
            ).apply(np.sign)

        return features

    def _create_exhaustion_features(self, df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Create trend exhaustion signals - indicators that a trend may be ending."""
        features = pd.DataFrame(index=df.index)

        # RSI - overbought/oversold indicator
        rsi_period = self.feature_config['rsi_period']
        rsi = self._calculate_rsi(df['close'], rsi_period)
        features['rsi'] = rsi

        # RSI divergence: price makes new high but RSI doesn't (or vice versa)
        # Strong signal that momentum is fading
        divergence_periods = self.feature_config['rsi_divergence_periods']
        for period in divergence_periods:
            price_change = df['close'] > df['close'].shift(period)
            rsi_change = rsi > rsi.shift(period)
            features[f'rsi_divergence_{period}'] = (price_change != rsi_change).astype(int)

        # Volume exhaustion: declining volume often precedes trend reversal
        if 'volume' in df.columns:
            volume_ma_period = self.feature_config['volume_ma_period']
            volume_sma = df['volume'].rolling(volume_ma_period).mean()
            volume_threshold = self.feature_config['volume_exhaustion_threshold']
            features['volume_exhaustion'] = (df['volume'] < volume_sma * volume_threshold).astype(int)
        else:
            features['volume_exhaustion'] = 0

        # Momentum exhaustion: trend slowing down
        momentum_period = self.feature_config['momentum_exhaustion_period']
        momentum_threshold = self.feature_config['momentum_exhaustion_threshold']
        price_momentum = returns.rolling(momentum_period).mean()
        features['momentum_exhaustion'] = (abs(price_momentum) < momentum_threshold).astype(int)

        return features

    def _create_reversion_pressure_features(self, df: pd.DataFrame, ma_periods: list) -> pd.DataFrame:
        """Create mean reversion pressure features - how far price has stretched from equilibrium."""
        features = pd.DataFrame(index=df.index)

        # Distance from moving averages: overextended prices tend to revert
        for period in ma_periods:
            sma = df['close'].rolling(period).mean()
            features[f'ma_distance_{period}'] = (df['close'] - sma) / sma
            features[f'overextension_{period}'] = abs(features[f'ma_distance_{period}'])

        # Bollinger Band position: where is price within the volatility envelope
        bb_period = self.feature_config['bollinger_period']
        bb_std_multiplier = self.feature_config['bollinger_std_multiplier']

        bb_sma = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()

        upper_band = bb_sma + (bb_std * bb_std_multiplier)
        lower_band = bb_sma - (bb_std * bb_std_multiplier)

        # Position within bands: -1 to +1 range typically
        features['bb_position'] = (df['close'] - bb_sma) / (upper_band - lower_band)

        # Binary signal: +1 if above upper band, -1 if below lower band, 0 if inside
        above_upper = (df['close'] > upper_band).astype(int)
        below_lower = (df['close'] < lower_band).astype(int)
        features['bb_overextension'] = above_upper - below_lower

        return features

    def create_duration_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create labels for how long trends will continue.

        Looks forward from each point to count how many periods the trend
        continues before reversing, then maps to duration category.
        """
        trend_detection_period = self.label_config['trend_detection_period']
        forward_window = self.label_config['forward_window']

        # Determine trend direction: above MA = uptrend (+1), below = downtrend (-1)
        sma = df['close'].rolling(trend_detection_period).mean()
        current_trend = np.where(df['close'] > sma, 1, -1)

        duration_labels = []
        buffer_periods = self._get_buffer_periods()

        for i in range(len(df) - forward_window - buffer_periods):
            feature_index = i + buffer_periods
            current_direction = current_trend[feature_index]

            # Count how many periods trend continues before reversing
            continuation_periods = 0
            for j in range(1, forward_window + 1):
                future_index = feature_index + j
                if future_index >= len(current_trend):
                    break
                if current_trend[future_index] == current_direction:
                    continuation_periods += 1
                else:
                    break

            duration_label = self._periods_to_category(continuation_periods)
            duration_labels.append(duration_label)

        logger.debug(f'Duration labels: {len(duration_labels)} samples')
        return np.array(duration_labels)

    def _periods_to_category(self, periods: int) -> int:
        """Convert continuation periods to duration category using configuration - ZERO HARDCODED VALUES"""
        for category_name, category_config in self.duration_categories.items():
            min_periods = category_config['min_periods']
            max_periods = category_config['max_periods']

            if min_periods <= periods <= max_periods:
                return category_config['label']

        # Default to longest category if no match
        return max(cat['label'] for cat in self.duration_categories.values())

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the trend duration predictor."""

        min_samples = self.training_config['min_samples']
        train_test_ratio = self.training_config['train_test_split']
        random_state = self.training_config.get('random_state')

        # Generate features and labels
        features_df = self.create_duration_features(df)
        duration_labels = self.create_duration_labels(df)

        # Align features and labels
        min_len = min(len(features_df), len(duration_labels))
        features_df = features_df.iloc[:min_len]
        duration_labels = duration_labels[:min_len]

        if len(features_df) < min_samples:
            logger.warning(f'Insufficient data: need {min_samples}, got {len(features_df)}')
            return {}

        X_train, X_test, y_train, y_test = train_test_split(
            features_df.values, duration_labels,
            train_size=train_test_ratio,
            random_state=random_state,
            stratify=duration_labels
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        self.feature_names = features_df.columns.tolist()
        self._is_trained = True
        self.clear_cache()

        # Log results
        logger.info(f'Duration Prediction Accuracy: {accuracy:.1%}')
        unique, counts = np.unique(y_test, return_counts=True)
        for cat_id, count in zip(unique, counts):
            cat_name = self._get_category_name(cat_id)
            pct = count / len(y_test) * 100
            logger.info(f'  {cat_name}: {pct:.1f}%')

        return {
            'accuracy': accuracy,
            'n_samples': len(X_test),
            'n_features': len(self.feature_names),
            'n_categories': len(self.duration_categories)
        }

    def _get_category_name(self, category_id: int) -> str:
        """Get human-readable category name from ID."""
        for name, config in self.duration_categories.items():
            if config['label'] == category_id:
                return name.replace('_', ' ').title()

        return f'Category_{category_id}'

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean features: handle inf and NaN values."""
        features_clean = features.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.ffill().bfill()
        features_clean = features_clean.fillna(0)
        return features_clean

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model, sorted by importance."""
        if not self._is_trained or self.feature_names is None:
            return {}

        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, filepath: str):
        """Save trained model to disk."""
        import joblib

        if not self._is_trained:
            logger.warning('Cannot save untrained model')
            return

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'duration_categories': self.duration_categories,
            'config': self.config.config
        }
        joblib.dump(model_data, filepath)
        logger.info(f'TrendDurationPredictor saved to {filepath}')

    @classmethod
    def load_model(cls, filepath: str) -> 'TrendDurationPredictor':
        """Load trained model from disk."""
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
        predictor._is_trained = True

        logger.info(f'TrendDurationPredictor loaded from {filepath}')
        return predictor

    @property
    def is_trained(self) -> bool:
        """Whether predictor is ready to predict"""
        return self._is_trained