# src/hybrid/optimization/optimization_orchestrator.py
"""
Optimization Orchestrator - Coordinates parameter search and evaluation
Manages worker pool, distributes work, aggregates results
"""

import logging
from typing import Dict, List, Callable
from multiprocessing import Pool
from datetime import datetime
from src.hybrid.optimization.optimizer_factory import OptimizerFactory
from src.hybrid.optimization import OptimizerType
from src.hybrid.config.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class OptimizationCoordinator:
    """
    Coordinates optimization process

    Responsibilities:
    - Generate parameter combinations (via optimizer)
    - Distribute work to worker pool
    - Collect and aggregate results
    - Optional checkpointing
    """

    def __init__(self, config: UnifiedConfig):
        """
        Initialize orchestrator

        Args:
            config: Base configuration
        """
        self.config = config
        self.all_evaluations = []

        # Load checkpoint config
        checkpoint_config = config.get_section('optimization', {}).get('checkpointing', {})
        self.checkpoint_interval = checkpoint_config.get('checkpoint_interval', 50)
        self.checkpoint_time_interval = checkpoint_config.get('checkpoint_time_interval', 300)
        self.last_checkpoint_time = datetime.now().timestamp()

    def optimize(self,
                 strategy_factory: Callable,
                 optimizer_type: OptimizerType,
                 n_combinations: int = None,
                 n_workers: int = None) -> Dict:
        """
        Run optimization

        Args:
            strategy_factory: Function that creates strategy with params
                             Signature: (params: Dict) -> Strategy
            optimizer_type: Type of optimizer to use
            n_combinations: Number of parameter combinations to test
            n_workers: Number of parallel workers

        Returns:
            Optimization results with best parameters
        """
        logger.info(f"Starting optimization with {optimizer_type}")
        start_time = datetime.now()

        # Get defaults from config if not provided
        if n_combinations is None:
            n_combinations = self.config.get_section('optimization', {}).get('defaults', {}).get('n_combinations', 100)
        if n_workers is None:
            n_workers = self.config.get_section('optimization', {}).get('defaults', {}).get('n_workers', 16)

        logger.info(f"Combinations: {n_combinations}, Workers: {n_workers}")

        # Create optimizer to generate parameter combinations
        # Note: We need a temporary strategy just for get_optimizable_parameters
        # This is a bit awkward - might need refactoring
        temp_strategy = strategy_factory({})  # Create with empty params to get structure
        optimizer = OptimizerFactory.create_optimizer(optimizer_type, self.config, temp_strategy)

        # Generate parameter combinations
        logger.info("Generating parameter combinations...")
        param_combinations = optimizer.generate_random_parameters(n_combinations)
        logger.info(f"Generated {len(param_combinations)} parameter combinations")

        # Distribute work to workers
        logger.info(f"Distributing work to {n_workers} workers...")
        results = self._distribute_work(strategy_factory, param_combinations, n_workers)

        # Aggregate results
        duration = (datetime.now() - start_time).total_seconds()
        aggregated = self._aggregate_results(results, optimizer_type, duration)

        logger.info(f"Optimization completed in {duration:.2f}s")
        logger.info(f"Valid results: {aggregated['valid_results']}/{aggregated['total_combinations']}")

        return aggregated

    def _distribute_work(self,
                         strategy_factory: Callable,
                         param_combinations: List[Dict],
                         n_workers: int) -> List[Dict]:
        """Distribute parameter evaluation across worker pool"""

        # Create fitness calculator (can be shared - thread-safe)
        from src.hybrid.optimization.fitness import FitnessCalculator
        fitness_calculator = FitnessCalculator(self.config)

        def worker_function(params):
            """Worker that evaluates single parameter set"""
            try:
                # Create strategy with params (factory handles all wiring)
                strategy = strategy_factory(params)

                # Run strategy
                metrics = strategy.run()

                # Calculate fitness
                fitness = fitness_calculator.calculate_fitness(metrics)

                # TODO: Calculate robustness

                return {
                    'params': params,
                    'fitness': fitness,
                    'metrics': metrics,
                    'success': True
                }

            except Exception as e:
                logger.error(f"Worker failed for params {params}: {e}")
                return {
                    'params': params,
                    'fitness': fitness_calculator.severe_penalty,
                    'metrics': {},
                    'success': False,
                    'error': str(e)
                }

        # Use multiprocessing pool
        with Pool(n_workers) as pool:
            results = pool.map(worker_function, param_combinations)

        return results

    def _aggregate_results(self, results: List[Dict], optimizer_type: OptimizerType, duration: float) -> Dict:
        """Aggregate evaluation results"""

        # Filter and sort valid results by fitness
        valid_results = [r for r in results if r['success']]
        valid_results.sort(key=lambda x: x['fitness'], reverse=True)  # Higher fitness = better

        return {
            'optimizer_type': optimizer_type.value,
            'total_combinations': len(results),
            'valid_results': len(valid_results),
            'failed_results': len(results) - len(valid_results),
            'all_results': valid_results,
            'best_result': valid_results[0] if valid_results else None,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }

    def collect_result(self, result: Dict):
        """
        Collect single evaluation result (for incremental collection)

        Args:
            result: Evaluation result
        """
        self.all_evaluations.append(result)

        # Checkpoint by count
        if len(self.all_evaluations) % self.checkpoint_interval == 0:
            self._save_checkpoint()

        # Checkpoint by time
        current_time = datetime.now().timestamp()
        if current_time - self.last_checkpoint_time > self.checkpoint_time_interval:
            self._save_checkpoint()
            self.last_checkpoint_time = current_time

    def _save_checkpoint(self):
        """Save checkpoint to disk"""
        # TODO: Implement checkpointing
        logger.info(f"Checkpoint: {len(self.all_evaluations)} evaluations completed")