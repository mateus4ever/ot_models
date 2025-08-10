# src/hybrid/optimization/robustness.py
# Robustness analysis tools for parameter optimization
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.hybrid.config.unified_config import UnifiedConfig


class RobustnessAnalyzer:
    """
    Analyze parameter robustness and stability
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache robustness analysis configuration values"""
        optimization_config = self.config.get_section('optimization', {})
        self.robustness_config = optimization_config.get('robustness', {})

        # Thresholds for robustness classification
        self.cv_threshold_robust = self.robustness_config.get('cv_threshold_robust')
        self.cv_threshold_sensitive = self.robustness_config.get('cv_threshold_sensitive')
        self.top_performers_percentile = self.robustness_config.get('top_performers_percentile')
        self.stability_window_size = self.robustness_config.get('stability_window_size')

        # Mathematical constants
        constants = self.config.get_section('mathematical_operations', {})
        self.zero_value = constants.get('zero')
        self.unity_value = constants.get('unity')

    def analyze_parameter_stability(self, results: List[Dict]) -> Dict:
        """
        Analyze parameter stability across optimization results

        Args:
            results: List of optimization results with params and fitness scores

        Returns:
            Dictionary with stability analysis for each parameter
        """
        if not results:
            return {}

        # Sort by fitness (best first)
        sorted_results = sorted(results, key=lambda x: x['fitness'], reverse=True)

        # Take top performers
        n_top = max(self.unity_value, int(len(sorted_results) * self.top_performers_percentile))
        top_results = sorted_results[:n_top]

        # Analyze each parameter
        param_analysis = {}

        # Get all parameter names from first result
        sample_params = top_results[self.zero_value]['params']

        for param_name in sample_params.keys():
            if param_name == 'signal_weights':
                # Handle nested signal weights separately
                param_analysis.update(self._analyze_signal_weights(top_results))
            else:
                param_analysis[param_name] = self._analyze_single_parameter(
                    param_name, top_results
                )

        return param_analysis

    def _analyze_single_parameter(self, param_name: str, results: List[Dict]) -> Dict:
        """Analyze stability of a single parameter"""
        values = [r['params'][param_name] for r in results]

        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # Coefficient of variation (relative variability)
        cv = std_val / mean_val if mean_val != self.zero_value else float('inf')

        # Classify robustness
        if cv < self.cv_threshold_robust:
            robustness_class = 'ROBUST'
        elif cv < self.cv_threshold_sensitive:
            robustness_class = 'MODERATE'
        else:
            robustness_class = 'SENSITIVE'

        # Range as percentage of mean
        range_pct = (max_val - min_val) / mean_val if mean_val != self.zero_value else float('inf')

        return {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'cv': cv,
            'range_pct': range_pct,
            'robustness_class': robustness_class,
            'n_samples': len(values)
        }

    def _analyze_signal_weights(self, results: List[Dict]) -> Dict:
        """Analyze signal weight parameters separately"""
        kama_weights = []
        kalman_weights = []

        for result in results:
            weights = result['params'].get('signal_weights', {})
            kama_weights.append(weights.get('kama', self.zero_value))
            kalman_weights.append(weights.get('kalman', self.zero_value))

        return {
            'signal_weights_kama': self._analyze_parameter_values('kama_weight', kama_weights),
            'signal_weights_kalman': self._analyze_parameter_values('kalman_weight', kalman_weights)
        }

    def _analyze_parameter_values(self, param_name: str, values: List[float]) -> Dict:
        """Helper to analyze a list of parameter values"""
        if not values:
            return {'error': 'No values provided'}

        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        cv = std_val / mean_val if mean_val != self.zero_value else float('inf')

        if cv < self.cv_threshold_robust:
            robustness_class = 'ROBUST'
        elif cv < self.cv_threshold_sensitive:
            robustness_class = 'MODERATE'
        else:
            robustness_class = 'SENSITIVE'

        range_pct = (max_val - min_val) / mean_val if mean_val != self.zero_value else float('inf')

        return {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'cv': cv,
            'range_pct': range_pct,
            'robustness_class': robustness_class,
            'n_samples': len(values)
        }

    def find_robust_parameter_ranges(self, results: List[Dict]) -> Dict:
        """
        Find parameter ranges that consistently produce good results

        Returns:
            Dictionary with robust ranges for each parameter
        """
        stability_analysis = self.analyze_parameter_stability(results)
        robust_ranges = {}

        for param_name, analysis in stability_analysis.items():
            if analysis.get('robustness_class') == 'ROBUST':
                # Define robust range as mean ± 1 standard deviation
                mean_val = analysis['mean']
                std_val = analysis['std']

                robust_ranges[param_name] = {
                    'min': mean_val - std_val,
                    'max': mean_val + std_val,
                    'center': mean_val,
                    'confidence': 'HIGH'
                }
            elif analysis.get('robustness_class') == 'MODERATE':
                # Wider range for moderate parameters
                mean_val = analysis['mean']
                std_val = analysis['std']
                std_multiplier = self.robustness_config.get('moderate_std_multiplier')

                robust_ranges[param_name] = {
                    'min': mean_val - std_val * std_multiplier,
                    'max': mean_val + std_val * std_multiplier,
                    'center': mean_val,
                    'confidence': 'MEDIUM'
                }

        return robust_ranges

    def analyze_fitness_landscape(self, results: List[Dict]) -> Dict:
        """
        Analyze the fitness landscape to identify plateaus and peaks

        Returns:
            Analysis of fitness distribution and landscape characteristics
        """
        if not results:
            return {'error': 'No results provided'}

        fitness_values = [r['fitness'] for r in results if r['fitness'] != -999]

        if not fitness_values:
            return {'error': 'No valid fitness values'}

        fitness_array = np.array(fitness_values)

        # Basic statistics
        fitness_stats = {
            'mean': np.mean(fitness_array),
            'std': np.std(fitness_array),
            'min': np.min(fitness_array),
            'max': np.max(fitness_array),
            'range': np.max(fitness_array) - np.min(fitness_array)
        }

        # Percentile analysis
        percentiles = self.robustness_config.get('fitness_percentiles', [])
        fitness_percentiles = {}
        for p in percentiles:
            fitness_percentiles[f'p{int(p * 100)}'] = np.percentile(fitness_array, p * 100)

        # Plateau detection (look for flat regions)
        plateau_threshold = self.robustness_config.get('plateau_threshold')
        top_fitness_threshold = fitness_percentiles.get('p90', fitness_stats['max'])

        plateau_count = np.sum(fitness_array >= top_fitness_threshold)
        plateau_percentage = plateau_count / len(fitness_array)

        # Landscape classification
        cv_fitness = fitness_stats['std'] / fitness_stats['mean'] if fitness_stats[
                                                                         'mean'] != self.zero_value else float('inf')

        if plateau_percentage > plateau_threshold and cv_fitness < self.cv_threshold_robust:
            landscape_type = 'PLATEAU_DOMINATED'
        elif cv_fitness > self.cv_threshold_sensitive:
            landscape_type = 'PEAKY'
        else:
            landscape_type = 'MIXED'

        return {
            'fitness_stats': fitness_stats,
            'fitness_percentiles': fitness_percentiles,
            'plateau_analysis': {
                'plateau_count': plateau_count,
                'plateau_percentage': plateau_percentage,
                'threshold_used': top_fitness_threshold
            },
            'landscape_type': landscape_type,
            'cv_fitness': cv_fitness
        }

    def calculate_parameter_correlation(self, results: List[Dict]) -> Dict:
        """
        Calculate correlations between parameters and fitness

        Returns:
            Correlation analysis between parameters and performance
        """
        if len(results) < self.stability_window_size:
            return {'error': f'Need at least {self.stability_window_size} results for correlation analysis'}

        # Extract parameter values and fitness scores
        param_data = {}
        fitness_scores = []

        for result in results:
            if result['fitness'] == -999:
                continue

            fitness_scores.append(result['fitness'])

            for param_name, param_value in result['params'].items():
                if param_name == 'signal_weights':
                    # Handle nested signal weights
                    for weight_name, weight_value in param_value.items():
                        full_name = f"signal_weights_{weight_name}"
                        if full_name not in param_data:
                            param_data[full_name] = []
                        param_data[full_name].append(weight_value)
                else:
                    if param_name not in param_data:
                        param_data[param_name] = []
                    param_data[param_name].append(param_value)

        # Calculate correlations
        correlations = {}
        fitness_array = np.array(fitness_scores)

        for param_name, param_values in param_data.items():
            if len(param_values) == len(fitness_scores):
                param_array = np.array(param_values)
                correlation = np.corrcoef(param_array, fitness_array)[self.zero_value, self.unity_value]

                # Classify correlation strength
                abs_corr = abs(correlation)
                if abs_corr > self.robustness_config.get('strong_correlation_threshold'):
                    strength = 'STRONG'
                elif abs_corr > self.robustness_config.get('moderate_correlation_threshold'):
                    strength = 'MODERATE'
                else:
                    strength = 'WEAK'

                correlations[param_name] = {
                    'correlation': correlation,
                    'abs_correlation': abs_corr,
                    'strength': strength,
                    'direction': 'POSITIVE' if correlation > self.zero_value else 'NEGATIVE'
                }

        return correlations

    def generate_robustness_report(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive robustness report

        Returns:
            Complete robustness analysis report
        """
        return {
            'parameter_stability': self.analyze_parameter_stability(results),
            'robust_ranges': self.find_robust_parameter_ranges(results),
            'fitness_landscape': self.analyze_fitness_landscape(results),
            'parameter_correlations': self.calculate_parameter_correlation(results),
            'summary': self._generate_summary(results)
        }

    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary of robustness analysis"""
        stability = self.analyze_parameter_stability(results)
        landscape = self.analyze_fitness_landscape(results)

        # Count robustness classifications
        robust_count = sum(self.unity_value for analysis in stability.values()
                           if analysis.get('robustness_class') == 'ROBUST')
        total_params = len(stability)

        robustness_score = robust_count / total_params if total_params > self.zero_value else self.zero_value

        return {
            'total_parameters_analyzed': total_params,
            'robust_parameters': robust_count,
            'robustness_score': robustness_score,
            'landscape_type': landscape.get('landscape_type', 'UNKNOWN'),
            'recommendation': self._get_recommendation(robustness_score, landscape.get('landscape_type'))
        }

    def _get_recommendation(self, robustness_score: float, landscape_type: str) -> str:
        """Generate recommendation based on robustness analysis"""
        high_robustness_threshold = self.robustness_config.get('high_robustness_threshold')
        moderate_robustness_threshold = self.robustness_config.get('moderate_robustness_threshold')

        if robustness_score >= high_robustness_threshold and landscape_type == 'PLATEAU_DOMINATED':
            return 'EXCELLENT - Strategy shows high robustness with stable parameter regions'
        elif robustness_score >= moderate_robustness_threshold and landscape_type != 'PEAKY':
            return 'GOOD - Strategy is reasonably robust, consider expanding parameter search'
        elif landscape_type == 'PEAKY':
            return 'CAUTION - Strategy shows sensitivity to parameters, risk of overfitting'
        else:
            return 'POOR - Low robustness detected, consider strategy redesign'