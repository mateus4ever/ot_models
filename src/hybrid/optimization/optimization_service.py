from typing import Dict, Callable

from src.hybrid.optimization import OptimizerType
from src.hybrid.optimization.optimization_coordinator import OptimizationCoordinator
from src.hybrid.config.unified_config import UnifiedConfig


class OptimizationService:
    """
    Entry point for optimization.

    Purpose:
        Facade that hides orchestration complexity from callers.
        Currently wraps OptimizationCoordinator (synchronous).
        Later: async job management, cloud execution, checkpointing.

    Why separate from OptimizationCoordinator:
        - Service = stable external API
        - Coordinator = internal orchestration (can change)
        - Allows adding async/cloud without changing callers
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.coordinator = OptimizationCoordinator(config)

    def run_optimization(self,
                         strategy_factory: Callable,
                         optimizer_type: OptimizerType,
                         n_combinations: int = None,
                         n_workers: int = None) -> Dict:
        """
        Run optimization (synchronous).

        Args:
            strategy_factory: Function that creates strategy from params
            optimizer_type: Type of optimizer to use
            n_combinations: Number of parameter combinations
            n_workers: Number of parallel workers

        Returns:
            Optimization results
        """
        return self.coordinator.optimize(
            strategy_factory=strategy_factory,
            optimizer_type=optimizer_type,
            n_combinations=n_combinations,
            n_workers=n_workers
        )