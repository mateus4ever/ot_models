# interfaces.py
# Python equivalent of: public interface IOptimizer { ... }
# Using Protocol instead of ABC because it's closer to Java interfaces

from typing import Protocol, Dict
from src.hybrid.optimization.optimizer_type import OptimizerType
from src.hybrid.config.unified_config import UnifiedConfig

class IOptimizer(Protocol):
    """

    === OPTIMIZATION DESIGN NOTES ===
Date: Saturday morning session

1. RESULT VISUALIZATION
   - Plateau detection (robust) vs Peak detection (overfitted)
   - Heatmap/3D surface for 2-3 parameters
   - For 10+ parameters: sensitivity analysis, clustering, correlation matrix

2. SIGNAL COMBINATION OPTIMIZATION
   - Not just parameter values, but which signals to combine
   - Combinatorial search across signal sets
   - Each combination tested with parameter ranges

3. REFRESH MECHANISM
   - Time-based: scheduled re-optimization (monthly)
   - Performance-based: trigger when live deviates X% from backtest
   - Regime-based: trigger when market conditions change (volatility, trend)

4. ARCHITECTURE REQUIREMENTS
   - Strategy must expose optimizable parameters
   - Strategy must accept different signal combinations
   - Results must be storable/comparable
   - Optimizer lives outside strategy (separate orchestrator)


    """

    def run_optimization(self, data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
        """
        Java equivalent: public OptimizationResult runOptimization(String dataPath, int combinations)
        """
        ...

    def get_optimization_type(self) -> OptimizerType:
        """
        Java equivalent: public OptimizationType getOptimizationType()
        """
        ...

    def get_description(self) -> str:
        """
        Java equivalent: public String getDescription()
        """
        ...


class IOptimizerBase:

    def __init__(self, config: UnifiedConfig):
        """Base class - just stores config"""
        self.config = config