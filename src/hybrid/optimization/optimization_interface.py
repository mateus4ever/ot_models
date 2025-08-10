# interfaces.py
# Python equivalent of: public interface IOptimizer { ... }
# Using Protocol instead of ABC because it's closer to Java interfaces

from typing import Protocol, Dict
from .optimization_types import OptimizationType
from src.hybrid.config.unified_config import UnifiedConfig


class IOptimizer(Protocol):
    """
    Python equivalent of Java interface:

    public interface IOptimizer {
        public OptimizationResult runOptimization(String dataPath, int combinations);
        public OptimizationType getOptimizationType();
        public String getDescription();
    }

    In Python, Protocol provides structural typing (duck typing with type safety)
    Similar to Java interface but without 'implements' keyword
    """

    def run_optimization(self, data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
        """
        Java equivalent: public OptimizationResult runOptimization(String dataPath, int combinations)
        """
        ...

    def get_optimization_type(self) -> OptimizationType:
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

    def calculate_fitness(self, backtest_results: Dict) -> float:
        """
        Java equivalent: protected double calculateFitness(BacktestResults results)
        Common fitness calculation shared by all optimizers
        """
        total_return = backtest_results.get('total_return', self.zero_value)
        sharpe_ratio = backtest_results.get('sharpe_ratio', self.zero_value)
        num_trades = backtest_results.get('num_trades', self.zero_value)

        if num_trades < 10:
            return self.severe_penalty

        return total_return * 100 + sharpe_ratio * 10