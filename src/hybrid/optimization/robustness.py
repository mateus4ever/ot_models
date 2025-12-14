# src/hybrid/optimization/robustness.py
# Robustness analysis tools for parameter optimization

import numpy as np
from typing import Dict, List
from src.hybrid.config.unified_config import UnifiedConfig
import logging

logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """
    Detect if optimized parameters are reliable or fragile.

    Purpose:
        Optimizer finds "best" parameters, but best on historical data
        might be overfitted noise. RobustnessAnalyzer checks if those
        parameters are in a stable region (plateau) or a fragile peak.

    Problem it solves:
        - PLATEAU: Parameters 18-22 all perform similarly well → safe to use
        - PEAK: Only parameter 20 works, 19 and 21 fail → overfitting risk

    How it works:
        - Analyzes top-performing parameter combinations
        - Measures variation (do top performers have similar parameter values?)
        - Classifies landscape as PLATEAU_DOMINATED, PEAKY, or MIXED
        - Returns recommendation: EXCELLENT, GOOD, CAUTION, or POOR

    Example:
        results = optimizer.run()  # List of {params, fitness} dicts
        analysis = robustness.analyze_parameter_stability(results)
        # Returns: {'atr_period': {'robustness_class': 'ROBUST', ...}}

        report = robustness.generate_robustness_report(results)
        # Returns: {'landscape_type': 'PLATEAU_DOMINATED', 'recommendation': 'EXCELLENT'}
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache robustness analysis configuration values"""
        optimization_config = self.config.get_section('optimization')
        self.robustness_config = optimization_config['robustness']

        # Direct access - fail hard if missing
        self.cv_threshold_robust = self.robustness_config['cv_threshold_robust']
        self.cv_threshold_sensitive = self.robustness_config['cv_threshold_sensitive']
        self.top_performers_percentile = self.robustness_config['top_performers_percentile']
        self.stability_window_size = self.robustness_config['stability_window_size']

        # Fitness penalty threshold
        fitness_config = optimization_config['fitness']
        self.severe_penalty = fitness_config['penalties']['severe_penalty']

    def analyze_parameter_stability(self, results: List[Dict]) -> Dict:
        if not results:
            return {}

        try:
            sorted_results = sorted(results, key=lambda x: x['fitness'], reverse=True)
            n_top = max(1, int(len(sorted_results) * self.top_performers_percentile))
            top_results = sorted_results[:n_top]

            if n_top < self.stability_window_size:
                raise ValueError(f"Insufficient samples: {n_top}, need {self.stability_window_size}")

            param_analysis = {}
            sample_params = top_results[0]['params']

            for param_name in sample_params.keys():
                param_analysis[param_name] = self._analyze_single_parameter(
                    param_name, top_results
                )

            return param_analysis

        except Exception as e:
            logger.error(f"Error analyzing parameter stability: {e}")
            return {}

    def _analyze_single_parameter(self, param_name: str, results: List[Dict]) -> Dict:
        """Analyze stability of a single parameter"""
        values = [r['params'][param_name] for r in results]

        mean_val = np.mean(values)

        if mean_val == 0:
            raise ValueError(f"Parameter '{param_name}' has mean of zero - invalid parameter values")

        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        cv = std_val / mean_val
        range_pct = (max_val - min_val) / mean_val

        # Classify robustness
        if cv < self.cv_threshold_robust:
            robustness_class = 'ROBUST'
        elif cv < self.cv_threshold_sensitive:
            robustness_class = 'MODERATE'
        else:
            robustness_class = 'SENSITIVE'

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

    def _analyze_parameter_values(self, param_name: str, values: List[float]) -> Dict:
        """Helper to analyze a list of parameter values"""
        if not values:
            raise ValueError(f"Parameter '{param_name}' has no values to analyze")

        mean_val = np.mean(values)

        if mean_val == 0:
            raise ValueError(f"Parameter '{param_name}' has mean of zero - invalid parameter values")

        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        cv = std_val / mean_val
        range_pct = (max_val - min_val) / mean_val

        if cv < self.cv_threshold_robust:
            robustness_class = 'ROBUST'
        elif cv < self.cv_threshold_sensitive:
            robustness_class = 'MODERATE'
        else:
            robustness_class = 'SENSITIVE'

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
            logger.warning("Cannot analyze fitness landscape: no results provided")
            return {}

        fitness_values = [r['fitness'] for r in results if r['fitness'] != self.severe_penalty]

        if not fitness_values:
            logger.warning("Cannot analyze fitness landscape: no valid fitness values")
            return {}

        fitness_array = np.array(fitness_values)

        try:
            # Basic statistics
            fitness_stats = {
                'mean': np.mean(fitness_array),
                'std': np.std(fitness_array),
                'min': np.min(fitness_array),
                'max': np.max(fitness_array),
                'range': np.max(fitness_array) - np.min(fitness_array)
            }

            # Gradient analysis for plateau detection
            sorted_fitness = np.sort(fitness_array)[::-1]
            top_n_count = self.robustness_config['top_n_for_gradient']
            top_n = sorted_fitness[:top_n_count]
            differences = np.abs(np.diff(top_n))
            mean_gradient = np.mean(differences)

            # Normalize gradient relative to fitness scale
            gradient_ratio = mean_gradient / fitness_stats['mean'] if fitness_stats['mean'] != 0 else float('inf')

            # Landscape classification
            gradient_threshold_flat = self.robustness_config['gradient_threshold_flat']
            gradient_threshold_steep = self.robustness_config['gradient_threshold_steep']

            if mean_gradient < gradient_threshold_flat:
                landscape_type = 'PLATEAU_DOMINATED'
            elif mean_gradient > gradient_threshold_steep:
                landscape_type = 'PEAKY'
            else:
                landscape_type = 'MIXED'

            return {
                'fitness_stats': fitness_stats,
                'gradient_analysis': {
                    'mean_gradient': mean_gradient,
                    'gradient_ratio': gradient_ratio,
                    'top_n_analyzed': top_n_count
                },
                'landscape_type': landscape_type
            }

        except Exception as e:
            logger.error(f"Error analyzing fitness landscape: {e}")
            return {}

    def calculate_parameter_correlation(self, results: List[Dict]) -> Dict:
        """
        Calculate correlations between parameters and fitness

        Returns:
            Correlation analysis between parameters and performance
        """
        if len(results) < self.stability_window_size:
            logger.warning(
                f"Insufficient results for correlation analysis: need {self.stability_window_size}, got {len(results)}")
            return {}

        try:
            # Extract parameter values and fitness scores
            param_data = {}
            fitness_scores = []

            for result in results:
                if result['fitness'] == self.severe_penalty:
                    continue

                fitness_scores.append(result['fitness'])

                for param_name, param_value in result['params'].items():
                    if param_name not in param_data:
                        param_data[param_name] = []
                    param_data[param_name].append(param_value)

            if not fitness_scores:
                logger.warning("No valid fitness scores for correlation analysis")
                return {}

            # Calculate correlations
            correlations = {}
            fitness_array = np.array(fitness_scores)

            for param_name, param_values in param_data.items():
                if len(param_values) == len(fitness_scores):
                    param_array = np.array(param_values)
                    correlation = np.corrcoef(param_array, fitness_array)[0, 1]

                    # Classify correlation strength
                    abs_corr = abs(correlation)
                    strong_threshold = self.robustness_config['strong_correlation_threshold']
                    moderate_threshold = self.robustness_config['moderate_correlation_threshold']

                    if abs_corr > strong_threshold:
                        strength = 'STRONG'
                    elif abs_corr > moderate_threshold:
                        strength = 'MODERATE'
                    else:
                        strength = 'WEAK'

                    correlations[param_name] = {
                        'correlation': correlation,
                        'abs_correlation': abs_corr,
                        'strength': strength,
                        'direction': 'POSITIVE' if correlation > 0 else 'NEGATIVE'
                    }

            return correlations

        except Exception as e:
            logger.error(f"Error calculating parameter correlations: {e}")
            return {}

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
        robust_count = sum(1 for analysis in stability.values()
                           if analysis.get('robustness_class') == 'ROBUST')
        total_params = len(stability)

        robustness_score = robust_count / total_params if total_params > 0 else 0

        return {
            'total_parameters_analyzed': total_params,
            'robust_parameters': robust_count,
            'robustness_score': robustness_score,
            'landscape_type': landscape.get('landscape_type', 'UNKNOWN'),
            'recommendation': self._get_recommendation(robustness_score, landscape.get('landscape_type'))
        }
    def _get_recommendation(self, robustness_score: float, landscape_type: str) -> str:
        high_robustness_threshold = self.robustness_config['high_robustness_threshold']
        moderate_robustness_threshold = self.robustness_config['moderate_robustness_threshold']
        poor_robustness_threshold = self.robustness_config['poor_robustness_threshold']

        if robustness_score >= high_robustness_threshold and landscape_type == 'PLATEAU_DOMINATED':
            return 'EXCELLENT - Strategy shows high robustness with stable parameter regions'
        elif robustness_score >= moderate_robustness_threshold and landscape_type != 'PEAKY':
            return 'GOOD - Strategy is reasonably robust, consider expanding parameter search'
        elif robustness_score < poor_robustness_threshold:
            return 'POOR - Low robustness detected, consider strategy redesign'
        elif landscape_type == 'PEAKY':
            return 'CAUTION - Strategy shows sensitivity to parameters, risk of overfitting'
        else:
            return 'POOR - Low robustness detected, consider strategy redesign'