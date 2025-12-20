from typing import Dict


class OptimizationEvaluator:
    def __init__(self, fitness_calculator, robustness_analyzer):
        self.fitness_calculator = fitness_calculator
        self.robustness_analyzer = robustness_analyzer

    def evaluate(self, strategy) -> Dict:
        """Evaluate already-created strategy"""
        # Strategy is pre-configured, pre-wired
        result = strategy.run()
        fitness = self.fitness_calculator.calculate_fitness(result)
        robustness = self.robustness_analyzer.analyze(result)

        return {'fitness': fitness, 'robustness': robustness, 'metrics': result}