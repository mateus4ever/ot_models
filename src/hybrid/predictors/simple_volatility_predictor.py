"""
Simple Volatility Predictor - Fast rule-based predictor for base_strategy
No ML, no training delay, instant calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.hybrid.predictors.predictor_interface import PredictorInterface


class SimpleVolatilityPredictor(PredictorInterface):
    """
    Fast rule-based volatility predictor.

    Compares recent volatility to historical volatility.
    If recent > historical * threshold → HIGH_VOL

    For use in base_strategy where speed matters.
    """

    def __init__(self, config):
        self.config = config

        params = config.get_section('volatility_prediction')['simple']['parameters']

        self.lookback_period = params['lookback_period']
        self.threshold_multiplier = params['threshold_multiplier']

        self._is_trained = True  # No training needed

    def train(self, df: pd.DataFrame) -> Dict:
        """No training needed - returns empty metrics"""
        return {
            'method': 'rule_based',
            'training_required': False
        }

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict volatility regime for each bar.

        Returns:
            Dict with predictions and confidence
        """
        if len(df) < self.lookback_period:
            return {
                'predictions': np.zeros(len(df)),
                'confidences': np.ones(len(df)),
                'success': True
            }

        returns = df['close'].pct_change()

        # Rolling recent volatility
        recent_vol = returns.rolling(self.lookback_period).std()

        # Expanding historical volatility (all data up to that point)
        historical_vol = returns.expanding(min_periods=self.lookback_period).std()

        # Compare: recent > historical * threshold → HIGH_VOL
        predictions = (recent_vol > historical_vol * self.threshold_multiplier).astype(int)
        predictions = predictions.fillna(0).values

        # Confidence is always 1.0 for rule-based
        confidences = np.ones(len(df))

        return {
            'predictions': predictions,
            'confidences': confidences,
            'success': True
        }

    @property
    def is_trained(self) -> bool:
        return self._is_trained