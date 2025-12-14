# src/hybrid/optimization/sensitivity.py
# Parameter sensitivity analysis for optimization results

import numpy as np
from typing import Dict, List
from src.hybrid.config.unified_config import UnifiedConfig
import logging

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Analyze which parameters have the most impact on fitness.

    Use case:
    - High sensitivity parameters → invest effort to optimize carefully
    - Low sensitivity parameters → pick reasonable value and move on
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache sensitivity analysis configuration values"""
        optimization_config = self.config.get_section('optimization')
        self.sensitivity_config = optimization_config.get('sensitivity', {})

        # Thresholds for classification
        self.high_sensitivity_threshold = self.sensitivity_config.get('high_threshold', 0.5)
        self.low_sensitivity_threshold = self.sensitivity_config.get('low_threshold', 0.2)
        self.min_samples = self.sensitivity_config.get('min_samples', 10)

    def analyze(self, results: List[Dict]) -> Dict:
        """
        Analyze parameter sensitivity from optimization results.

        Args:
            results: List of optimization results with 'params' and 'fitness'

        Returns:
            Dictionary with sensitivity analysis per parameter
        """
        if len(results) < self.min_samples:
            logger.warning(
                f"Insufficient results for sensitivity analysis: need {self.min_samples}, got {len(results)}")
            return {}

        # Filter invalid fitness values
        valid_results = [r for r in results if r.get('fitness') not in (None, -999, float('inf'), float('-inf'))]

        if len(valid_results) < self.min_samples:
            logger.warning("Not enough valid fitness values for sensitivity analysis")
            return {}

        fitness_scores = np.array([r['fitness'] for r in valid_results])

        # Get parameter names from first result
        param_names = list(valid_results[0]['params'].keys())

        sensitivity = {}
        for param_name in param_names:
            sensitivity[param_name] = self._analyze_parameter(param_name, valid_results, fitness_scores)

        return {
            'parameters': sensitivity,
            'summary': self._generate_summary(sensitivity)
        }

    def _analyze_parameter(self, param_name: str, results: List[Dict], fitness_scores: np.ndarray) -> Dict:
        """Analyze sensitivity of a single parameter"""
        param_values = np.array([r['params'][param_name] for r in results])

        # Calculate correlation with fitness
        correlation = np.corrcoef(param_values, fitness_scores)[0, 1]

        # Handle NaN correlation (constant parameter)
        if np.isnan(correlation):
            return {
                'correlation': 0.0,
                'sensitivity_class': 'CONSTANT',
                'recommendation': 'Parameter has no variation - cannot assess sensitivity'
            }

        abs_correlation = abs(correlation)

        # Classify sensitivity
        if abs_correlation >= self.high_sensitivity_threshold:
            sensitivity_class = 'HIGH'
            recommendation = 'Optimize carefully - significant impact on performance'
        elif abs_correlation >= self.low_sensitivity_threshold:
            sensitivity_class = 'MEDIUM'
            recommendation = 'Worth optimizing but not critical'
        else:
            sensitivity_class = 'LOW'
            recommendation = 'Pick reasonable value and move on'

        return {
            'correlation': correlation,
            'abs_correlation': abs_correlation,
            'direction': 'POSITIVE' if correlation > 0 else 'NEGATIVE',
            'sensitivity_class': sensitivity_class,
            'recommendation': recommendation
        }

    def _generate_summary(self, sensitivity: Dict) -> Dict:
        """Generate summary of sensitivity analysis"""
        high_sensitivity = [name for name, data in sensitivity.items() if data.get('sensitivity_class') == 'HIGH']
        medium_sensitivity = [name for name, data in sensitivity.items() if data.get('sensitivity_class') == 'MEDIUM']
        low_sensitivity = [name for name, data in sensitivity.items() if data.get('sensitivity_class') == 'LOW']

        return {
            'high_sensitivity_params': high_sensitivity,
            'medium_sensitivity_params': medium_sensitivity,
            'low_sensitivity_params': low_sensitivity,
            'focus_on': high_sensitivity if high_sensitivity else medium_sensitivity,
            'can_simplify': low_sensitivity
        }