# src/hybrid/optimization/fitness.py
"""
Fitness Calculator for optimization

Converts backtest metrics into a single score for comparing parameter combinations.
The optimizer uses this score to decide which parameters are "better".
"""

from typing import Dict
from src.hybrid.config.unified_config import UnifiedConfig


class FitnessCalculator:
    """
    Calculate fitness scores for optimization.

    Purpose:
        Backtest produces many metrics (return, drawdown, win_rate, etc.)
        Optimizer needs ONE number to compare parameter combinations.
        FitnessCalculator combines metrics into that single score.

    How it works:
        - Each metric has a weight (importance)
        - Each metric has a direction (maximize profit, minimize drawdown)
        - Penalty conditions punish unacceptable results (e.g., >50% drawdown)

    Example:
        metrics = {'total_return': 0.15, 'max_drawdown': 0.25, 'win_rate': 0.55}
        fitness = calculator.calculate(metrics)  # Returns single score like 7.3
    """

    def __init__(self, config: UnifiedConfig):
        """
        Initialize fitness calculator with configuration

        Args:
            config: System configuration containing fitness parameters
        """
        self.config = config
        self._load_config()

    def _load_config(self):
        """Load fitness calculation configuration"""
        optimization_config = self.config.get_section('optimization', {})
        fitness_config = optimization_config.get('fitness')

        # Load configured metrics (list of {name, weight, direction})
        self.metrics_config = fitness_config.get('metrics')

        # Load penalty configuration
        penalties_config = fitness_config.get('penalties')
        self.severe_penalty = penalties_config.get('severe_penalty')
        self.penalty_conditions = penalties_config.get('conditions')

    def calculate_fitness(self, metrics: Dict) -> float:
        """
        Calculate fitness score from metrics

        Args:
            metrics: Dictionary of metric values from backtest/metrics calculator

        Returns:
            Fitness score (higher is better)

        Example:
            metrics = {
                'total_return': 0.20,
                'sharpe_ratio': 2.0,
                'max_drawdown': 0.10,
                'num_trades': 50
            }

            Config specifies:
            - total_return: weight=1.0, maximize
            - sharpe_ratio: weight=0.5, maximize
            - max_drawdown: weight=0.3, minimize

            Result: 0.20*1.0 + 2.0*0.5 - 0.10*0.3 = 1.17
        """
        # Check penalty conditions first
        if self._check_penalties(metrics):
            return self.severe_penalty

        # Calculate fitness from configured metrics
        fitness = 0.0

        for metric_config in self.metrics_config:
            name = metric_config['name']
            weight = metric_config['weight']
            direction = metric_config['direction']

            # Get metric value (default to 0 if missing)
            value = metrics.get(name)

            # Apply direction (maximize = positive, minimize = negative)
            multiplier = 1 if direction == 'maximize' else -1

            # Add weighted contribution to fitness
            fitness += value * weight * multiplier

        return fitness

    def _check_penalties(self, metrics: Dict) -> bool:
        """Check if any penalty conditions are violated"""
        for condition in self.penalty_conditions:
            metric_name = condition['metric']
            operator = condition['operator']
            threshold = condition['threshold']

            # Metric must exist for penalty checking
            if metric_name not in metrics:
                raise ValueError(f"Penalty condition requires metric '{metric_name}' but it's missing from results")

            value = metrics[metric_name]

            # Check condition
            if operator == '<' and value < threshold:
                return True
            elif operator == '>' and value > threshold:
                return True
            elif operator == '<=' and value <= threshold:
                return True
            elif operator == '>=' and value >= threshold:
                return True
            elif operator == '==' and value == threshold:
                return True

        return False