# src/hybrid/optimization/__init__.py
"""
Optimization package for trading strategy parameter tuning

Public API: Core interfaces and types only.
Implementations are imported directly when needed.
"""

# Core interfaces and types - no heavy dependencies
from .optimization_interface import IOptimizer, IOptimizerBase
from .optimizer_type import OptimizerType

# Public API
__all__ = [
    'IOptimizer',
    'IOptimizerBase',
    'OptimizerType',
]

# Users import implementations directly when needed:
#   from src.hybrid.optimization.implementation.simple_optimizer import SimpleRandomOptimizer
#   from src.hybrid.optimization.implementation.cached_optimizer import CachedRandomOptimizer
#   from src.hybrid.optimization.implementation.bayesian_optimizer import BayesianOptimizer
#   from src.hybrid.optimization.factory import OptimizerFactory
#   from src.hybrid.optimization.fitness import FitnessCalculator
#   from src.hybrid.optimization.robustness import RobustnessAnalyzer