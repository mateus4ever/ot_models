# factory.py
# JAVA EQUIVALENT: public class OptimizerFactory { ... }

from typing import Dict, List
from .optimization_types import OptimizationType
from .optimization_interface import IOptimizer
from .cached_optimizer import CachedRandomOptimizer
from .bayesian_optimizer import BayesianOptimizer, SKOPT_AVAILABLE
from src.hybrid.config.unified_config import UnifiedConfig


class OptimizerFactory:
    """
    Factory class for creating optimizer instances

    JAVA EQUIVALENT:
    public class OptimizerFactory {
        public static IOptimizer createOptimizer(OptimizationType type, UnifiedConfig config) throws IllegalArgumentException {
            switch (type) {
                case SIMPLE_RANDOM:
                    return new SimpleRandomOptimizer(config);
                case CACHED_RANDOM:
                    return new CachedRandomOptimizer(config);
                case BAYESIAN:
                    if (!SKOPT_AVAILABLE) {
                        throw new RuntimeException("scikit-optimize is required for Bayesian optimization");
                    }
                    return new BayesianOptimizer(config);
                default:
                    throw new IllegalArgumentException("Unsupported optimizer type: " + type);
            }
        }

        public static List<OptimizationType> getAvailableOptimizers() { ... }
        public static Map<OptimizationType, String> getOptimizerDescriptions() { ... }
    }
    """

    @staticmethod
    def create_optimizer(optimizer_type: OptimizationType, config: UnifiedConfig) -> IOptimizer:
        """
        Factory method to create optimizer instances

        JAVA EQUIVALENT:
        public static IOptimizer createOptimizer(OptimizationType optimizerType, UnifiedConfig config)
                throws IllegalArgumentException {

            switch (optimizerType) {
                case SIMPLE_RANDOM:
                    return new SimpleRandomOptimizer(config);
                case CACHED_RANDOM:
                    return new CachedRandomOptimizer(config);
                case BAYESIAN:
                    if (!SKOPT_AVAILABLE) {
                        throw new RuntimeException("scikit-optimize is required for Bayesian optimization");
                    }
                    return new BayesianOptimizer(config);
                default:
                    throw new IllegalArgumentException("Unsupported optimizer type: " + optimizerType);
            }
        }

        Args:
            optimizer_type: Type of optimizer to create
            config: Configuration object

        Returns:
            IOptimizer: Instance of the requested optimizer

        Raises:
            ValueError: If optimizer type is not supported
            ImportError: If required dependencies are missing
        """
        if optimizer_type == OptimizationType.SIMPLE_RANDOM:
            # Import here to avoid circular dependency
            from .simple_optimizer import SimpleRandomOptimizer
            return SimpleRandomOptimizer(config)
        elif optimizer_type == OptimizationType.CACHED_RANDOM:
            return CachedRandomOptimizer(config)
        elif optimizer_type == OptimizationType.BAYESIAN:
            if not SKOPT_AVAILABLE:
                raise ImportError(
                    "scikit-optimize is required for Bayesian optimization. Install with: pip install scikit-optimize")
            return BayesianOptimizer(config)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    @staticmethod
    def get_available_optimizers() -> List[OptimizationType]:
        """
        Get list of available optimizer types

        JAVA EQUIVALENT:
        public static List<OptimizationType> getAvailableOptimizers() {
            List<OptimizationType> optimizers = new ArrayList<>();
            optimizers.add(OptimizationType.SIMPLE_RANDOM);
            optimizers.add(OptimizationType.CACHED_RANDOM);

            if (SKOPT_AVAILABLE) {
                optimizers.add(OptimizationType.BAYESIAN);
            }

            return optimizers;
        }
        """
        optimizers = [OptimizationType.SIMPLE_RANDOM, OptimizationType.CACHED_RANDOM]
        if SKOPT_AVAILABLE:
            optimizers.append(OptimizationType.BAYESIAN)
        return optimizers

    @staticmethod
    def get_optimizer_descriptions() -> Dict[OptimizationType, str]:
        """
        Get descriptions of all optimizer types

        JAVA EQUIVALENT:
        public static Map<OptimizationType, String> getOptimizerDescriptions() {
            Map<OptimizationType, String> descriptions = new HashMap<>();
            UnifiedConfig config = new UnifiedConfig(); // Temporary config for descriptions

            for (OptimizationType optType : getAvailableOptimizers()) {
                try {
                    IOptimizer optimizer = createOptimizer(optType, config);
                    descriptions.put(optType, optimizer.getDescription());
                } catch (Exception e) {
                    descriptions.put(optType, "Description unavailable");
                }
            }

            return descriptions;
        }
        """
        config = UnifiedConfig()  # Temporary config for descriptions
        descriptions = {}

        for opt_type in OptimizerFactory.get_available_optimizers():
            try:
                optimizer = OptimizerFactory.create_optimizer(opt_type, config)
                descriptions[opt_type] = optimizer.get_description()
            except Exception:
                descriptions[opt_type] = "Description unavailable"

        return descriptions


# Main optimization function using factory pattern
# JAVA EQUIVALENT: public static OptimizationResult runOptimization(OptimizationType type, String dataPath, Integer nCombinations)
def run_optimization(optimizer_type: OptimizationType = OptimizationType.CACHED_RANDOM,
                     data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
    """
    Main optimization function using factory pattern

    JAVA EQUIVALENT:
    public static OptimizationResult runOptimization(
            OptimizationType optimizerType,
            String dataPath,
            Integer nCombinations,
            Map<String, Object> kwargs) {

        UnifiedConfig config = new UnifiedConfig();
        IOptimizer optimizer = OptimizerFactory.createOptimizer(optimizerType, config);
        return optimizer.runOptimization(dataPath, nCombinations, kwargs);
    }

    Args:
        optimizer_type: Type of optimizer to use
        data_path: Path to data file
        n_combinations: Number of parameter combinations to test
        **kwargs: Additional arguments passed to optimizer

    Returns:
        Dict: Optimization results
    """
    config = UnifiedConfig()
    optimizer = OptimizerFactory.create_optimizer(optimizer_type, config)
    return optimizer.run_optimization(data_path=data_path, n_combinations=n_combinations, **kwargs)


# Legacy function names for backward compatibility
# JAVA EQUIVALENT: @Deprecated public static OptimizationResult runFastOptimization(...)
def run_fast_optimization(data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
    """
    Legacy function - use simple random optimizer

    JAVA EQUIVALENT:
    @Deprecated
    public static OptimizationResult runFastOptimization(String dataPath, Integer nCombinations, Map<String, Object> kwargs) {
        return runOptimization(OptimizationType.SIMPLE_RANDOM, dataPath, nCombinations, kwargs);
    }
    """
    return run_optimization(OptimizationType.SIMPLE_RANDOM, data_path, n_combinations, **kwargs)


def run_optimized_fast_optimization(data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
    """
    Legacy function - use cached random optimizer

    JAVA EQUIVALENT:
    @Deprecated
    public static OptimizationResult runOptimizedFastOptimization(String dataPath, Integer nCombinations, Map<String, Object> kwargs) {
        return runOptimization(OptimizationType.CACHED_RANDOM, dataPath, nCombinations, kwargs);
    }
    """
    return run_optimization(OptimizationType.CACHED_RANDOM, data_path, n_combinations, **kwargs)


def run_bayesian_optimization(data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
    """
    Legacy function - use Bayesian optimizer

    JAVA EQUIVALENT:
    @Deprecated
    public static OptimizationResult runBayesianOptimization(String dataPath, Integer nCombinations, Map<String, Object> kwargs) {
        return runOptimization(OptimizationType.BAYESIAN, dataPath, nCombinations, kwargs);
    }
    """
    return run_optimization(OptimizationType.BAYESIAN, data_path, n_combinations, **kwargs)