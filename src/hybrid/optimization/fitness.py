# src/hybrid/optimization/fitness.py
# Fitness function definitions for parameter optimization
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

import numpy as np
from typing import Dict, Optional, List
from enum import Enum
from src.hybrid.config.unified_config import UnifiedConfig


class PerformanceClassification(Enum):
    """Performance classification for validation"""
    GOOD = "GOOD"
    AVERAGE = "AVERAGE"
    BAD = "BAD"


class FitnessCalculator:
    """
    Calculate fitness scores for optimization
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache fitness calculation configuration values"""
        optimization_config = self.config.get_section('optimization', {})
        self.fitness_config = optimization_config.get('fitness', {})

        # Fitness function weights
        self.weights = self.fitness_config.get('weights', {})

        # Penalty thresholds
        self.penalties = self.fitness_config.get('penalties', {})

        # Minimum requirements
        self.requirements = self.fitness_config.get('requirements', {})

        # Scaling factors
        self.scaling = self.fitness_config.get('scaling', {})

        # Mathematical constants
        constants = self.config.get_section('mathematical_operations', {})
        self.zero_value = constants.get('zero')
        self.unity_value = constants.get('unity')

        # Severe penalty value
        self.severe_penalty = self.penalties.get('severe_penalty_value')

    def calculate_fitness(self, backtest_results: Dict, fitness_function: str = 'default') -> float:
        """
        Calculate fitness score based on backtest results

        Args:
            backtest_results: Results from backtesting
            fitness_function: Type of fitness function to use

        Returns:
            Fitness score (higher is better)
        """
        if fitness_function == 'sharpe_focused':
            return self._calculate_sharpe_focused_fitness(backtest_results)
        elif fitness_function == 'return_focused':
            return self._calculate_return_focused_fitness(backtest_results)
        elif fitness_function == 'balanced':
            return self._calculate_balanced_fitness(backtest_results)
        elif fitness_function == 'risk_adjusted':
            return self._calculate_risk_adjusted_fitness(backtest_results)
        else:
            return self._calculate_default_fitness(backtest_results)

    def _calculate_default_fitness(self, results: Dict) -> float:
        """Default fitness function balancing return, risk, and stability"""

        # Extract metrics
        total_return = results.get('total_return', self.zero_value)
        sharpe_ratio = results.get('sharpe_ratio', self.zero_value)
        max_drawdown = results.get('max_drawdown', self.unity_value)
        num_trades = results.get('num_trades', self.zero_value)
        win_rate = results.get('win_rate', self.zero_value)
        profit_factor = results.get('profit_factor', self.zero_value)

        # Apply penalties for inadequate results
        penalty_score = self._calculate_penalties(results)
        if penalty_score < self.zero_value:
            return penalty_score

        # Calculate component scores
        return_component = total_return * self.weights.get('return_weight')
        sharpe_component = sharpe_ratio * self.weights.get('sharpe_weight')
        drawdown_component = (self.unity_value - max_drawdown) * self.weights.get('drawdown_weight')
        trade_count_component = self._normalize_trade_count(num_trades) * self.weights.get('trade_count_weight')
        win_rate_component = win_rate * self.weights.get('win_rate_weight')
        profit_factor_component = self._normalize_profit_factor(profit_factor) * self.weights.get(
            'profit_factor_weight')

        # Combine components
        fitness = (
                return_component +
                sharpe_component +
                drawdown_component +
                trade_count_component +
                win_rate_component +
                profit_factor_component
        )

        return fitness

    def _calculate_sharpe_focused_fitness(self, results: Dict) -> float:
        """Fitness function focused on risk-adjusted returns (Sharpe ratio)"""

        sharpe_ratio = results.get('sharpe_ratio', self.zero_value)
        max_drawdown = results.get('max_drawdown', self.unity_value)
        num_trades = results.get('num_trades', self.zero_value)

        # Apply basic penalties
        penalty_score = self._calculate_penalties(results)
        if penalty_score < self.zero_value:
            return penalty_score

        # Sharpe-focused scoring
        sharpe_component = sharpe_ratio * self.weights.get('sharpe_focused_sharpe_weight')
        drawdown_penalty = max_drawdown * self.weights.get('sharpe_focused_drawdown_penalty')
        trade_bonus = self._normalize_trade_count(num_trades) * self.weights.get('sharpe_focused_trade_bonus')

        return sharpe_component - drawdown_penalty + trade_bonus

    def _calculate_return_focused_fitness(self, results: Dict) -> float:
        """Fitness function focused on absolute returns"""

        total_return = results.get('total_return', self.zero_value)
        max_drawdown = results.get('max_drawdown', self.unity_value)
        num_trades = results.get('num_trades', self.zero_value)

        # Apply basic penalties
        penalty_score = self._calculate_penalties(results)
        if penalty_score < self.zero_value:
            return penalty_score

        # Return-focused scoring
        return_component = total_return * self.weights.get('return_focused_return_weight')
        drawdown_penalty = max_drawdown * self.weights.get('return_focused_drawdown_penalty')
        trade_requirement = self._normalize_trade_count(num_trades) * self.weights.get('return_focused_trade_weight')

        return return_component - drawdown_penalty + trade_requirement

    def _calculate_balanced_fitness(self, results: Dict) -> float:
        """Balanced fitness function considering multiple factors equally"""

        total_return = results.get('total_return', self.zero_value)
        sharpe_ratio = results.get('sharpe_ratio', self.zero_value)
        max_drawdown = results.get('max_drawdown', self.unity_value)
        win_rate = results.get('win_rate', self.zero_value)
        profit_factor = results.get('profit_factor', self.zero_value)

        # Apply penalties
        penalty_score = self._calculate_penalties(results)
        if penalty_score < self.zero_value:
            return penalty_score

        # Equal weighting approach
        equal_weight = self.weights.get('balanced_equal_weight')

        normalized_return = self._normalize_return(total_return)
        normalized_sharpe = self._normalize_sharpe(sharpe_ratio)
        normalized_drawdown = self.unity_value - max_drawdown
        normalized_win_rate = win_rate
        normalized_profit_factor = self._normalize_profit_factor(profit_factor)

        fitness = equal_weight * (
                normalized_return +
                normalized_sharpe +
                normalized_drawdown +
                normalized_win_rate +
                normalized_profit_factor
        )

        return fitness

    def _calculate_risk_adjusted_fitness(self, results: Dict) -> float:
        """Risk-adjusted fitness with emphasis on stability"""

        total_return = results.get('total_return', self.zero_value)
        sharpe_ratio = results.get('sharpe_ratio', self.zero_value)
        sortino_ratio = results.get('sortino_ratio', self.zero_value)
        max_drawdown = results.get('max_drawdown', self.unity_value)
        max_loss_streak = results.get('max_loss_streak', self.zero_value)

        # Apply penalties
        penalty_score = self._calculate_penalties(results)
        if penalty_score < self.zero_value:
            return penalty_score

        # Risk-adjusted components
        return_component = total_return * self.weights.get('risk_adjusted_return_weight')
        sharpe_component = sharpe_ratio * self.weights.get('risk_adjusted_sharpe_weight')
        sortino_component = sortino_ratio * self.weights.get('risk_adjusted_sortino_weight')
        drawdown_penalty = max_drawdown * self.weights.get('risk_adjusted_drawdown_penalty')
        streak_penalty = self._normalize_loss_streak(max_loss_streak) * self.weights.get('risk_adjusted_streak_penalty')

        return (
                return_component +
                sharpe_component +
                sortino_component -
                drawdown_penalty -
                streak_penalty
        )

    def _calculate_penalties(self, results: Dict) -> float:
        """Calculate penalties for inadequate performance"""

        num_trades = results.get('num_trades', self.zero_value)
        max_drawdown = results.get('max_drawdown', self.zero_value)
        total_return = results.get('total_return', self.zero_value)

        # Minimum trade requirement
        min_trades = self.requirements.get('min_trades')
        if num_trades < min_trades:
            return self.severe_penalty

        # Maximum drawdown limit
        max_drawdown_limit = self.penalties.get('max_drawdown_limit')
        if max_drawdown > max_drawdown_limit:
            return self.severe_penalty

        # Minimum return requirement
        min_return = self.requirements.get('min_return')
        if total_return < min_return:
            return self.severe_penalty

        return self.zero_value  # No penalties applied

    def _normalize_trade_count(self, num_trades: int) -> float:
        """Normalize trade count for fitness calculation"""
        optimal_trades = self.scaling.get('optimal_trade_count')
        max_trade_bonus = self.scaling.get('max_trade_bonus')

        if num_trades <= optimal_trades:
            return (num_trades / optimal_trades) * max_trade_bonus
        else:
            # Diminishing returns for excessive trading
            excess_factor = self.scaling.get('excess_trade_penalty_factor')
            penalty = (num_trades - optimal_trades) * excess_factor
            return max_trade_bonus - penalty

    def _normalize_return(self, total_return: float) -> float:
        """Normalize return for balanced scoring"""
        return_scale = self.scaling.get('return_normalization_scale')
        return total_return * return_scale

    def _normalize_sharpe(self, sharpe_ratio: float) -> float:
        """Normalize Sharpe ratio for balanced scoring"""
        max_sharpe_score = self.scaling.get('max_sharpe_score')
        sharpe_scale = self.scaling.get('sharpe_normalization_scale')

        normalized = sharpe_ratio / sharpe_scale
        return min(normalized, max_sharpe_score)

    def _normalize_profit_factor(self, profit_factor: float) -> float:
        """Normalize profit factor for fitness calculation"""
        if profit_factor <= self.unity_value:
            return self.zero_value

        max_pf_score = self.scaling.get('max_profit_factor_score')
        pf_scale = self.scaling.get('profit_factor_scale')

        normalized = (profit_factor - self.unity_value) / pf_scale
        return min(normalized, max_pf_score)

    def _normalize_loss_streak(self, max_loss_streak: int) -> float:
        """Normalize loss streak for penalty calculation"""
        acceptable_streak = self.penalties.get('acceptable_loss_streak')
        streak_penalty_scale = self.penalties.get('streak_penalty_scale')

        if max_loss_streak <= acceptable_streak:
            return self.zero_value

        excess_streak = max_loss_streak - acceptable_streak
        return excess_streak * streak_penalty_scale

    def compare_fitness_methods(self, results: Dict) -> Dict:
        """Compare different fitness calculation methods"""

        fitness_methods = ['default', 'sharpe_focused', 'return_focused', 'balanced', 'risk_adjusted']
        fitness_scores = {}

        for method in fitness_methods:
            fitness_scores[method] = self.calculate_fitness(results, method)

        # Find best and worst methods
        best_method = max(fitness_scores.keys(), key=lambda k: fitness_scores[k])
        worst_method = min(fitness_scores.keys(), key=lambda k: fitness_scores[k])

        # Calculate consistency (standard deviation of scores)
        scores = list(fitness_scores.values())
        consistency = np.std(scores) if len(scores) > self.unity_value else self.zero_value

        return {
            'fitness_scores': fitness_scores,
            'best_method': best_method,
            'worst_method': worst_method,
            'score_range': max(scores) - min(scores),
            'consistency': consistency,
            'recommendation': self._recommend_fitness_method(fitness_scores, consistency)
        }

    def _recommend_fitness_method(self, fitness_scores: Dict, consistency: float) -> str:
        """Recommend best fitness method based on scores and consistency"""

        consistency_threshold = self.fitness_config.get('consistency_threshold')

        if consistency < consistency_threshold:
            return f"All methods agree - use any (recommended: {max(fitness_scores.keys(), key=lambda k: fitness_scores[k])})"
        else:
            # High disagreement between methods
            sorted_methods = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
            return f"Methods disagree - consider strategy revision. Best: {sorted_methods[0][0]}"

    def calculate_multi_objective_fitness(self, results: Dict, objectives: List[str]) -> Dict:
        """
        Calculate multi-objective fitness scores

        Args:
            results: Backtest results
            objectives: List of objectives to optimize

        Returns:
            Dictionary with individual objective scores and combined score
        """

        objective_scores = {}

        for objective in objectives:
            if objective == 'return':
                objective_scores[objective] = results.get('total_return', self.zero_value)
            elif objective == 'sharpe':
                objective_scores[objective] = results.get('sharpe_ratio', self.zero_value)
            elif objective == 'drawdown':
                objective_scores[objective] = self.unity_value - results.get('max_drawdown', self.unity_value)
            elif objective == 'win_rate':
                objective_scores[objective] = results.get('win_rate', self.zero_value)
            elif objective == 'profit_factor':
                pf = results.get('profit_factor', self.zero_value)
                objective_scores[objective] = pf if pf > self.unity_value else self.zero_value
            elif objective == 'trade_efficiency':
                num_trades = results.get('num_trades', self.zero_value)
                total_return = results.get('total_return', self.zero_value)
                if num_trades > self.zero_value:
                    objective_scores[objective] = total_return / num_trades
                else:
                    objective_scores[objective] = self.zero_value

        # Apply penalties
        penalty_score = self._calculate_penalties(results)
        if penalty_score < self.zero_value:
            return {
                'objective_scores': objective_scores,
                'combined_score': penalty_score,
                'penalty_applied': True
            }

        # Calculate combined score (equal weighting)
        if objective_scores:
            combined_score = sum(objective_scores.values()) / len(objective_scores)
        else:
            combined_score = self.zero_value

        return {
            'objective_scores': objective_scores,
            'combined_score': combined_score,
            'penalty_applied': False,
            'objectives_optimized': objectives
        }

    def validate_fitness_function(self, test_results: List[Dict]) -> Dict:
        """
        Validate fitness function by testing on known good/bad results

        Args:
            test_results: List of backtest results for validation

        Returns:
            Validation report
        """
        min_validation_samples = self.requirements.get('min_validation_samples')
        if len(test_results) < min_validation_samples:
            return {'error': f'Need at least {min_validation_samples} samples for validation'}

        fitness_scores = []
        performance_metrics = []

        good_return_threshold = self.requirements.get('good_return_threshold')
        bad_return_threshold = self.penalties.get('bad_return_threshold')
        good_drawdown_threshold = self.penalties.get('good_drawdown_threshold')
        bad_drawdown_threshold = self.penalties.get('bad_drawdown_threshold')

        for result in test_results:
            fitness = self.calculate_fitness(result['backtest'])
            fitness_scores.append(fitness)

            # Performance classification
            total_return = result['backtest'].get('total_return', self.zero_value)
            max_drawdown = result['backtest'].get('max_drawdown', self.unity_value)

            if total_return > good_return_threshold and max_drawdown < good_drawdown_threshold:
                performance_metrics.append(PerformanceClassification.GOOD.value)
            elif total_return < bad_return_threshold or max_drawdown > bad_drawdown_threshold:
                performance_metrics.append(PerformanceClassification.BAD.value)
            else:
                performance_metrics.append(PerformanceClassification.AVERAGE.value)

        # Check if fitness function ranks correctly
        fitness_ranking = np.argsort(fitness_scores * -1)  # Sort descending (best to worst)

        validation_score = self._calculate_validation_score(fitness_ranking, performance_metrics)

        return {
            'validation_score': validation_score,
            'fitness_scores': fitness_scores,
            'performance_classifications': performance_metrics,
            'ranking_accuracy': self._check_ranking_accuracy(fitness_ranking, performance_metrics),
            'recommendation': self._get_validation_recommendation(validation_score)
        }

    def _calculate_validation_score(self, fitness_ranking: np.ndarray, performance_metrics: List[str]) -> float:
        """Calculate how well fitness function ranks known good/bad results"""

        good_indices = [i for i, perf in enumerate(performance_metrics) if perf == PerformanceClassification.GOOD.value]
        bad_indices = [i for i, perf in enumerate(performance_metrics) if perf == PerformanceClassification.BAD.value]

        if not good_indices or not bad_indices:
            return self.zero_value  # Can't validate without both good and bad examples

        # Create rank lookup dictionary once
        rank_positions = {idx: rank for rank, idx in enumerate(fitness_ranking)}

        # Count how many good results are ranked higher than bad results
        correct_rankings = self.zero_value
        total_comparisons = self.zero_value

        for good_idx in good_indices:
            for bad_idx in bad_indices:
                good_rank = rank_positions[good_idx]
                bad_rank = rank_positions[bad_idx]

                if good_rank < bad_rank:  # Good result ranked higher (lower rank number)
                    correct_rankings += self.unity_value
                total_comparisons += self.unity_value

        return correct_rankings / total_comparisons if total_comparisons > self.zero_value else self.zero_value

    def _check_ranking_accuracy(self, fitness_ranking: np.ndarray, performance_metrics: List[str]) -> Dict:
        """Check ranking accuracy for each performance class"""

        # Create rank lookup dictionary once
        rank_positions = {idx: rank for rank, idx in enumerate(fitness_ranking)}

        good_ranks = []
        average_ranks = []
        bad_ranks = []

        for i, perf in enumerate(performance_metrics):
            rank = rank_positions[i]

            if perf == PerformanceClassification.GOOD.value:
                good_ranks.append(rank)
            elif perf == PerformanceClassification.AVERAGE.value:
                average_ranks.append(rank)
            elif perf == PerformanceClassification.BAD.value:
                bad_ranks.append(rank)

        return {
            'good_average_rank': np.mean(good_ranks) if good_ranks else None,
            'average_average_rank': np.mean(average_ranks) if average_ranks else None,
            'bad_average_rank': np.mean(bad_ranks) if bad_ranks else None,
            'good_count': len(good_ranks),
            'average_count': len(average_ranks),
            'bad_count': len(bad_ranks)
        }

    def _get_validation_recommendation(self, validation_score: float) -> str:
        """Get recommendation based on validation score"""

        excellent_threshold = self.fitness_config.get('excellent_validation_threshold')
        good_threshold = self.fitness_config.get('good_validation_threshold')

        if validation_score >= excellent_threshold:
            return 'EXCELLENT - Fitness function accurately ranks strategies'
        elif validation_score >= good_threshold:
            return 'GOOD - Fitness function shows reasonable ranking accuracy'
        else:
            return 'POOR - Consider adjusting fitness function weights and penalties'