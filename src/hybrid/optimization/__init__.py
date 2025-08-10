# __init__.py
# JAVA EQUIVALENT: Public API exports (like public class declarations in a package)

"""
Optimization package for trading strategy parameter tuning

JAVA EQUIVALENT:
package com.trading.optimization;

// Public API exports
public class OptimizerFactory { ... }
public enum OptimizationType { ... }
public interface IOptimizer { ... }
public class SimpleRandomOptimizer implements IOptimizer { ... }
public class CachedRandomOptimizer implements IOptimizer { ... }
public class BayesianOptimizer implements IOptimizer { ... }

// Static utility methods
public static OptimizationResult runOptimization(OptimizationType type, ...);
"""

# Core interfaces and types
from .optimization_interface import IOptimizer, IOptimizerBase
from .optimization_types import OptimizationType

# Concrete implementations
from .simple_optimizer import SimpleRandomOptimizer
from .cached_optimizer import CachedRandomOptimizer
from .bayesian_optimizer import BayesianOptimizer

# Factory and main functions
from .factory import (
    OptimizerFactory,
    run_optimization,
    run_fast_optimization,  # Legacy
    run_optimized_fast_optimization,  # Legacy
    run_bayesian_optimization  # Legacy
)

# Import from other modules for backward compatibility
# JAVA EQUIVALENT: import com.trading.optimization.fitness.*;
try:
    from .fitness import FitnessCalculator, PerformanceClassification
    from .robustness import RobustnessAnalyzer

    FITNESS_AVAILABLE = True
except ImportError:
    FITNESS_AVAILABLE = False

# Public API - what other modules can import
# JAVA EQUIVALENT: public class declarations
__all__ = [
    # Core interfaces and types
    'IOptimizer',
    'IOptimizerBase',
    'OptimizationType',

    # Concrete implementations
    'SimpleRandomOptimizer',
    'CachedRandomOptimizer',
    'BayesianOptimizer',

    # Factory and main entry points
    'OptimizerFactory',
    'run_optimization',  # Main factory-based function

    # Legacy functions for backward compatibility
    'run_fast_optimization',
    'run_optimized_fast_optimization',
    'run_bayesian_optimization'
]

# Add fitness classes if available
if FITNESS_AVAILABLE:
    __all__.extend([
        'FitnessCalculator',
        'PerformanceClassification',
        'RobustnessAnalyzer'
    ])