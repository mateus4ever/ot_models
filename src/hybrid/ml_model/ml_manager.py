"""
ML Model Manager - Pure Coordination Only
Updated for clean architecture: coordinates ML models without providing utilities
Each ML model is self-contained and handles its own feature generation
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from src.hybrid.config.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class MLModelManager:
    """
    Pure coordination manager for ML models

    Responsibilities:
    - Initialize ML components based on configuration
    - Coordinate training across models
    - Collect predictions from all models
    - Manage model persistence
    - Provide status and metadata

    Does NOT provide:
    - Shared utilities (each model is self-contained)
    - Feature generation (each model handles its own)
    - Complex coordination logic (simple delegation)
    """

    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or UnifiedConfig()

        # Load configuration sections
        self.regime_config = self.config.get_section('regime_detection', {})
        self.volatility_config = self.config.get_section('volatility_prediction', {})
        self.duration_config = self.config.get_section('trend_duration_prediction', {})
        self.general_config = self.config.get_section('general', {})

        # Initialize components
        self.regime_detector = None
        self.volatility_predictor = None
        self.duration_predictor = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize components based on configuration"""

        verbose = self.general_config.get('verbose', True)

        # === REGIME DETECTION ===
        regime_method = self.regime_config.get('method', 'rule_based')

        if regime_method == 'rule_based':
            try:
                from src.hybrid.signal.rule_based_regime_detector import RuleBasedRegimeDetector
                self.regime_detector = RuleBasedRegimeDetector(self.config)
                if verbose:
                    print("✓ Rule-based regime detection initialized")
            except ImportError as e:
                logger.error(f"Failed to import RuleBasedRegimeDetector: {e}")
                self.regime_detector = None
        else:
            # Legacy ML-based regime detection (if still needed)
            try:
                from src.hybrid.market_regime_detector import MarketRegimeDetector
                self.regime_detector = MarketRegimeDetector(self.config)
                if verbose:
                    print("✓ ML-based regime detection initialized")
            except ImportError as e:
                logger.error(f"Failed to import MarketRegimeDetector: {e}")
                self.regime_detector = None

        # === VOLATILITY PREDICTION ===
        if self.volatility_config.get('use_volatility_ml', True):
            try:
                from src.hybrid.ml_model.volatility_predictor import VolatilityPredictor
                self.volatility_predictor = VolatilityPredictor(self.config)
                if verbose:
                    print("✓ ML volatility prediction initialized")
            except ImportError as e:
                logger.error(f"Failed to import VolatilityPredictor: {e}")
                self.volatility_predictor = None

        # === TREND DURATION PREDICTION ===
        if self.duration_config.get('enabled', False):
            try:
                from src.hybrid.ml_model.trend_duration_predictor import TrendDurationPredictor
                self.duration_predictor = TrendDurationPredictor(self.config)
                if verbose:
                    print("✓ Trend duration prediction initialized")
            except ImportError as e:
                logger.error(f"Failed to import TrendDurationPredictor: {e}")
                self.duration_predictor = None

    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Coordinate training of all ML models
        Pure delegation - each model handles its own training
        """
        results = {}
        verbose = self.general_config.get('verbose', True)

        if verbose:
            print("=== Training ML Models ===")

        # Train regime detector
        if self.regime_detector:
            try:
                regime_results = self.regime_detector.train(df)
                results['regime'] = regime_results
                if regime_results:
                    results['regime']['method'] = self.regime_config.get('method', 'rule_based')
                if verbose and regime_results:
                    method = results['regime'].get('method', 'unknown')
                    print(f"✓ Regime detector ({method}) ready")
            except Exception as e:
                logger.error(f"Error training regime detector: {e}")
                results['regime'] = {'error': str(e)}

        # Train volatility predictor
        if self.volatility_predictor:
            try:
                vol_results = self.volatility_predictor.train(df)
                results['volatility'] = vol_results
                if verbose and vol_results:
                    print(f"✓ Volatility predictor trained")
            except Exception as e:
                logger.error(f"Error training volatility predictor: {e}")
                results['volatility'] = {'error': str(e)}

        # Train duration predictor
        if self.duration_predictor:
            try:
                duration_results = self.duration_predictor.train(df)
                results['duration'] = duration_results
                if verbose and duration_results:
                    print(f"✓ Duration predictor trained")
            except Exception as e:
                logger.error(f"Error training duration predictor: {e}")
                results['duration'] = {'error': str(e)}

        return results

    def predict_all(self, df: pd.DataFrame) -> Dict[str, Tuple]:
        """
        Collect predictions from all models
        Pure delegation - each model handles its own prediction logic
        """
        predictions = {}
        verbose = self.general_config.get('verbose', True)

        if verbose:
            print("=== Getting ML Predictions ===")

        # Regime predictions
        if self.regime_detector and hasattr(self.regime_detector, 'predict_regime'):
            try:
                regime_pred, regime_conf = self.regime_detector.predict_regime(df)
                predictions['regime'] = (regime_pred, regime_conf)
                if verbose:
                    print(f"✓ Regime predictions generated")
            except Exception as e:
                logger.error(f"Error getting regime predictions: {e}")
                predictions['regime'] = self._get_default_regime_prediction(df)
        else:
            predictions['regime'] = self._get_default_regime_prediction(df)

        # Volatility predictions
        if self.volatility_predictor and hasattr(self.volatility_predictor, 'predict_volatility'):
            try:
                vol_pred = self.volatility_predictor.predict_volatility(df)

                # Handle different return formats
                if isinstance(vol_pred, tuple):
                    vol_predictions, vol_confidence = vol_pred
                else:
                    vol_predictions = vol_pred
                    vol_confidence = pd.Series(0.7, index=df.index)

                predictions['volatility'] = (vol_predictions, vol_confidence)
                if verbose:
                    print(f"✓ Volatility predictions generated")
            except Exception as e:
                logger.error(f"Error getting volatility predictions: {e}")
                predictions['volatility'] = self._get_default_volatility_prediction(df)
        else:
            predictions['volatility'] = self._get_default_volatility_prediction(df)

        # Duration predictions
        if self.duration_predictor and hasattr(self.duration_predictor, 'predict_duration'):
            try:
                duration_pred = self.duration_predictor.predict_duration(df)
                duration_conf = self._calculate_duration_confidence(duration_pred)
                predictions['duration'] = (duration_pred, duration_conf)
                if verbose:
                    print(f"✓ Duration predictions generated")
            except Exception as e:
                logger.error(f"Error getting duration predictions: {e}")
                predictions['duration'] = self._get_default_duration_prediction(df)
        else:
            predictions['duration'] = self._get_default_duration_prediction(df)

        return predictions

    def _get_default_regime_prediction(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get default regime prediction when no detector available"""
        default_regime = self.regime_config.get('default_regime', 0)  # Ranging
        regime_pred = np.full(len(df), default_regime)
        regime_conf = np.full(len(df), 0.5)
        return regime_pred, regime_conf

    def _get_default_volatility_prediction(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Get default volatility prediction when no predictor available"""
        vol_pred = pd.Series(False, index=df.index)  # Low volatility
        vol_conf = pd.Series(0.0, index=df.index)
        return vol_pred, vol_conf

    def _get_default_duration_prediction(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get default duration prediction when no predictor available"""
        default_duration = self.duration_config.get('default_duration_category', 1)  # Short
        duration_pred = np.full(len(df), default_duration)
        duration_conf = np.full(len(df), 0.5)
        return duration_pred, duration_conf

    def _calculate_duration_confidence(self, duration_predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence for duration predictions based on consistency"""
        confidence = np.full(len(duration_predictions), 0.5)
        window = self.duration_config.get('confidence_window', 10)

        for i in range(window, len(duration_predictions)):
            recent_preds = duration_predictions[i - window:i]
            current_pred = duration_predictions[i]
            consistency = np.mean(recent_preds == current_pred)
            confidence[i] = consistency

        return confidence

    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all initialized models"""
        status = {}

        # Regime detection status
        if self.regime_detector:
            status['regime'] = {
                'initialized': True,
                'trained': getattr(self.regime_detector, 'is_trained', False),
                'method': self.regime_config.get('method', 'rule_based'),
                'type': type(self.regime_detector).__name__
            }
        else:
            status['regime'] = {
                'initialized': False,
                'trained': False,
                'method': 'none',
                'type': 'none'
            }

        # Volatility prediction status
        if self.volatility_predictor:
            status['volatility'] = {
                'initialized': True,
                'trained': getattr(self.volatility_predictor, 'is_trained', False),
                'type': type(self.volatility_predictor).__name__
            }
        else:
            status['volatility'] = {
                'initialized': False,
                'trained': False,
                'type': 'none'
            }

        # Duration prediction status
        if self.duration_predictor:
            status['duration'] = {
                'initialized': True,
                'trained': getattr(self.duration_predictor, 'is_trained', False),
                'enabled': self.duration_config.get('enabled', False),
                'type': type(self.duration_predictor).__name__
            }
        else:
            status['duration'] = {
                'initialized': False,
                'trained': False,
                'enabled': False,
                'type': 'none'
            }

        return status

    def get_feature_importance(self) -> Dict[str, Dict]:
        """Collect feature importance from all models that support it"""
        importance = {}

        # Each model handles its own feature importance
        models = [
            ('regime', self.regime_detector),
            ('volatility', self.volatility_predictor),
            ('duration', self.duration_predictor)
        ]

        for name, model in models:
            if model and hasattr(model, 'get_feature_importance'):
                try:
                    model_importance = model.get_feature_importance()
                    importance[name] = model_importance
                except Exception as e:
                    logger.error(f"Error getting {name} feature importance: {e}")
                    importance[name] = {}
            else:
                importance[name] = {}

        return importance

    def save_all_models(self, base_filepath: str) -> list:
        """Save all trained models"""
        saved_models = []

        models = [
            ('regime', self.regime_detector),
            ('volatility', self.volatility_predictor),
            ('duration', self.duration_predictor)
        ]

        for name, model in models:
            if model and getattr(model, 'is_trained', False):
                model_path = f"{base_filepath}_{name}.pkl"
                try:
                    model.save_model(model_path)
                    saved_models.append(f"{name} -> {model_path}")
                except Exception as e:
                    logger.error(f"Error saving {name} model: {e}")

        if self.general_config.get('verbose', True) and saved_models:
            print(f"Saved models: {saved_models}")

        return saved_models

    @classmethod
    def load_all_models(cls, base_filepath: str, config: Optional[UnifiedConfig] = None) -> 'MLModelManager':
        """Load all models from disk"""
        manager = cls(config)

        # Try to load each model type
        model_types = ['regime', 'volatility', 'duration']

        for model_type in model_types:
            model_path = f"{base_filepath}_{model_type}.pkl"

            try:
                if model_type == 'regime' and manager.regime_detector:
                    manager.regime_detector = manager.regime_detector.load_model(model_path)
                    logger.info(f"Loaded regime detector from {model_path}")

                elif model_type == 'volatility' and manager.volatility_predictor:
                    manager.volatility_predictor = manager.volatility_predictor.load_model(model_path)
                    logger.info(f"Loaded volatility predictor from {model_path}")

                elif model_type == 'duration' and manager.duration_predictor:
                    manager.duration_predictor = manager.duration_predictor.load_model(model_path)
                    logger.info(f"Loaded duration predictor from {model_path}")

            except Exception as e:
                logger.error(f"Error loading {model_type} model: {e}")

        return manager

    def get_predictions_summary(self, predictions: Dict[str, Tuple]) -> Dict[str, any]:
        """Generate summary of predictions for analysis"""
        summary = {}

        # Regime prediction summary
        if 'regime' in predictions:
            regime_pred, regime_conf = predictions['regime']
            unique, counts = np.unique(regime_pred, return_counts=True)
            regime_dist = dict(zip(unique, counts))

            summary['regime'] = {
                'distribution': regime_dist,
                'avg_confidence': np.mean(regime_conf),
                'total_predictions': len(regime_pred)
            }

        # Volatility prediction summary
        if 'volatility' in predictions:
            vol_pred, vol_conf = predictions['volatility']
            if isinstance(vol_pred, pd.Series):
                high_vol_count = vol_pred.sum() if vol_pred.dtype == bool else (vol_pred > 0.5).sum()
                summary['volatility'] = {
                    'high_vol_predictions': high_vol_count,
                    'high_vol_percentage': (high_vol_count / len(vol_pred)) * 100,
                    'avg_confidence': vol_conf.mean() if hasattr(vol_conf, 'mean') else np.mean(vol_conf)
                }

        # Duration prediction summary
        if 'duration' in predictions:
            duration_pred, duration_conf = predictions['duration']
            unique, counts = np.unique(duration_pred, return_counts=True)
            duration_dist = dict(zip(unique, counts))

            summary['duration'] = {
                'distribution': duration_dist,
                'avg_confidence': np.mean(duration_conf),
                'avg_predicted_duration': np.mean(duration_pred)
            }

        return summary

    def __repr__(self) -> str:
        """String representation"""
        status = self.get_model_status()
        initialized_count = sum(1 for s in status.values() if s['initialized'])
        trained_count = sum(1 for s in status.values() if s['trained'])

        return (f"MLModelManager("
                f"initialized={initialized_count}/3, "
                f"trained={trained_count}/3, "
                f"config={self.config.config_path})")


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def create_ml_manager(config: Optional[UnifiedConfig] = None) -> MLModelManager:
    """Convenience function to create ML manager"""
    return MLModelManager(config)


def get_available_ml_models() -> Dict[str, bool]:
    """Check which ML models are available for import"""
    available = {}

    # Check regime detectors
    try:
        from src.hybrid.signal.rule_based_regime_detector import RuleBasedRegimeDetector
        available['rule_based_regime'] = True
    except ImportError:
        available['rule_based_regime'] = False

    try:
        from src.hybrid.market_regime_detector import MarketRegimeDetector
        available['ml_regime'] = True
    except ImportError:
        available['ml_regime'] = False

    # Check other ML models
    try:
        from src.hybrid.ml_model.volatility_predictor import VolatilityPredictor
        available['volatility_predictor'] = True
    except ImportError:
        available['volatility_predictor'] = False

    try:
        from src.hybrid.ml_model.trend_duration_predictor import TrendDurationPredictor
        available['duration_predictor'] = True
    except ImportError:
        available['duration_predictor'] = False

    return available