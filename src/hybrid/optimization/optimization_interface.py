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
    """
    Python equivalent of Java abstract class:

    public abstract class OptimizerBase implements IOptimizer {
        protected UnifiedConfig config;
        protected double zeroValue;
        // ... common implementation
        public abstract OptimizationType getOptimizationType();
    }

    This provides shared implementation that all optimizers can inherit
    """

    def __init__(self, config: UnifiedConfig):
        """Java equivalent: protected OptimizerBase(UnifiedConfig config)"""
        self.config = config
        self._initialize_common_config()

    def _initialize_common_config(self):
        """Java equivalent: protected void initializeCommonConfig()"""
        constants = self.config.get_section('mathematical_operations', {})
        self.zero_value = constants.get('zero')
        self.unity_value = constants.get('unity')

        fitness_config = self.config.get_section('optimization', {}).get('fitness', {})
        self.severe_penalty = fitness_config.get('penalties', {}).get('severe_penalty_value')

        # Get fitness calculation requirements from config
        requirements_config = fitness_config.get('requirements', {})
        self.min_trades_required = requirements_config.get('min_trades')
        self.max_drawdown_limit = requirements_config.get('max_drawdown_limit')

        # Get fitness calculation weights from config
        weights_config = fitness_config.get('weights', {})
        self.return_component_weight = weights_config.get('return_component')
        self.sharpe_component_weight = weights_config.get('sharpe_component')
        self.drawdown_component_weight = weights_config.get('drawdown_component')

    def calculate_fitness(self, backtest_results: Dict) -> float:
        """
        Java equivalent: protected double calculateFitness(BacktestResults results)
        Common fitness calculation shared by all optimizers
        ZERO HARDCODED VALUES - ALL THRESHOLDS CONFIGURABLE
        """
        total_return = backtest_results.get('total_return', self.zero_value)
        sharpe_ratio = backtest_results.get('sharpe_ratio', self.zero_value)
        num_trades = backtest_results.get('num_trades', self.zero_value)
        max_drawdown = backtest_results.get('max_drawdown', self.zero_value)

        # Check minimum trades requirement from config
        if num_trades < self.min_trades_required:
            return self.severe_penalty

        # Check maximum drawdown limit from config
        if max_drawdown > self.max_drawdown_limit:
            return self.severe_penalty

        # Calculate fitness using configurable weights
        fitness_score = (
                total_return * self.return_component_weight +
                sharpe_ratio * self.sharpe_component_weight -
                max_drawdown * self.drawdown_component_weight
        )

        return fitness_score