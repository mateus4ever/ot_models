# simple_optimizer.py
"""
Simple Random Optimizer - Educational baseline implementation

Generates random parameter combinations and evaluates them through backtesting.
No caching - loads full data for each iteration.
Good for understanding optimization basics and simple parameter searches.
"""

import random
from typing import Dict, List
from src.hybrid.optimization.optimization_interface import IOptimizerBase
from src.hybrid.optimization.optimizer_type import OptimizerType
from src.hybrid.config.unified_config import UnifiedConfig


class SimpleRandomOptimizer(IOptimizerBase):
    """
    Simple random parameter search optimizer

    Generates random parameter combinations within configured ranges and evaluates
    each through full backtest execution. No result caching or intelligent search -
    purely random exploration of parameter space.

    Use cases:
    - Educational baseline for understanding optimization
    - Quick parameter space exploration
    - Comparison benchmark for more sophisticated optimizers
    """

    def __init__(self, config: UnifiedConfig, strategy):
        """
        Initialize optimizer with configuration and strategy

        Args:
            config: System configuration
            strategy: Strategy instance that defines optimizable parameters
        """
        super().__init__(config)
        self.strategy = strategy

        # Get optimizable parameters from strategy
        # These come from active signals, position sizers, and risk managers
        self.param_ranges = strategy.get_optimizable_parameters()

        if not self.param_ranges:
            raise ValueError("No optimizable parameters found in strategy configuration")

    def get_optimization_type(self) -> OptimizerType:
        """Return optimizer type identifier"""
        return OptimizerType.SIMPLE_RANDOM

    def get_description(self) -> str:
        """Return human-readable optimizer description"""
        return "Simple random parameter search with full data loading per iteration"

    def generate_random_parameters(self, n_combinations: int) -> List[Dict]:
        """
        Generate random parameter combinations within configured ranges

        Creates n_combinations parameter sets where each parameter is randomly
        sampled from its configured min/max range. Parameters are determined
        dynamically from strategy configuration.

        Args:
            n_combinations: Number of random parameter sets to generate

        Returns:
            List of parameter dictionaries, each containing all optimizable parameters

        Example:
            If param_ranges = {
                'sma_fast_period': {'min': 10, 'max': 50},
                'sma_slow_period': {'min': 50, 'max': 200}
            }

            Returns: [
                {'sma_fast_period': 23, 'sma_slow_period': 127},
                {'sma_fast_period': 41, 'sma_slow_period': 89},
                ...
            ]
        """
        combinations = []

        for _ in range(n_combinations):
            combo = {}

            # Generate random value for each parameter within its range
            for param_name, param_def in self.param_ranges.items():
                min_val = param_def['min']
                max_val = param_def['max']

                # Generate random value within range
                random_value = random.uniform(min_val, max_val)
                combo[param_name] = random_value

            combinations.append(combo)

        return combinations

    def create_test_config(self, params: Dict) -> UnifiedConfig:
        """
        Create configuration with specific parameter values for testing

        Takes a parameter dictionary and creates a new config with those values
        applied to the appropriate sections (signals, risk management, position sizing).

        Args:
            params: Parameter dictionary with values to test

        Returns:
            New UnifiedConfig instance with parameters applied
        """
        # Create new config based on current config
        new_config = UnifiedConfig(self.config.config_path)
        new_config.config = self.config.config.copy()

        # Apply parameters to appropriate config sections
        # Parameters are prefixed by their source (signal name, 'sizer_', 'risk_')
        updates = self._build_config_updates(params)

        # Disable verbose output during optimization
        if 'general' not in updates:
            updates['general'] = {}
        updates['general']['verbose'] = False

        # Disable debug output during optimization
        if 'debug_configuration' not in updates:
            updates['debug_configuration'] = {}
        updates['debug_configuration'].update({
            'trade_debug_count': 0,
            'enable_fee_debug': False,
            'enable_trade_debug': False
        })

        new_config.update_config(updates)
        return new_config

    def _build_config_updates(self, params: Dict) -> Dict:
        """
        Build config update dictionary from parameters

        Distributes parameters to their appropriate config sections based on
        parameter name prefixes (signal names, 'sizer_', 'risk_').

        Args:
            params: Flat parameter dictionary

        Returns:
            Nested config update dictionary
        """
        updates = {}

        for param_name, param_value in params.items():
            # Determine which config section this parameter belongs to
            if param_name.startswith('sizer_'):
                # Position sizing parameter
                if 'money_management' not in updates:
                    updates['money_management'] = {}
                if 'position_sizers' not in updates['money_management']:
                    updates['money_management']['position_sizers'] = {}

                # Remove prefix and apply to active sizer
                clean_name = param_name.replace('sizer_', '')
                active_sizer = self.config.get_section('money_management', {}).get('position_sizing')
                if active_sizer:
                    if active_sizer not in updates['money_management']['position_sizers']:
                        updates['money_management']['position_sizers'][active_sizer] = {}
                    updates['money_management']['position_sizers'][active_sizer][clean_name] = param_value

            elif param_name.startswith('risk_'):
                # Risk management parameter
                if 'money_management' not in updates:
                    updates['money_management'] = {}
                if 'risk_managers' not in updates['money_management']:
                    updates['money_management']['risk_managers'] = {}

                # Remove prefix and apply to active risk manager
                clean_name = param_name.replace('risk_', '')
                active_risk = self.config.get_section('money_management', {}).get('risk_management')
                if active_risk:
                    if active_risk not in updates['money_management']['risk_managers']:
                        updates['money_management']['risk_managers'][active_risk] = {}
                    updates['money_management']['risk_managers'][active_risk][clean_name] = param_value

            else:
                # Signal parameter (prefixed with signal name)
                # Find which signal this parameter belongs to
                for signal_name in self.param_ranges.keys():
                    if param_name.startswith(f"{signal_name}_"):
                        if 'signals' not in updates:
                            updates['signals'] = {}

                        # Remove signal name prefix
                        clean_name = param_name.replace(f"{signal_name}_", '')

                        # Find signal in config and update
                        # This requires navigating the signals config structure
                        if signal_name not in updates['signals']:
                            updates['signals'][signal_name] = {}
                        updates['signals'][signal_name][clean_name] = param_value
                        break

        return updates

    def run_optimization(self, data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
        """
        Execute random parameter search optimization

        Generates random parameter combinations and evaluates each through
        full backtest execution. No caching - each iteration loads and processes
        complete dataset.

        Args:
            data_path: Path to market data for backtesting
            n_combinations: Number of random combinations to test (default from config)
            **kwargs: Additional arguments (unused, for interface compatibility)

        Returns:
            Dictionary containing:
                - optimizer_type: Type of optimizer used
                - total_combinations: Total combinations tested
                - valid_results: Number of successful backtests
                - best_result: Best performing parameter set
                - all_results: All results sorted by fitness
        """
        # Get number of combinations from config if not provided
        if n_combinations is None:
            opt_config = self.config.get_section('optimization', {})
            defaults = opt_config.get('defaults', {})
            n_combinations = defaults.get('n_combinations', 10)

        print(f"Running {self.get_description()}")
        print(f"Testing {n_combinations} random parameter combinations")
        print(f"Optimizing {len(self.param_ranges)} parameters: {list(self.param_ranges.keys())}")

        # Generate random parameter combinations
        param_combinations = self.generate_random_parameters(n_combinations)
        results = []

        # Evaluate each combination
        for i, params in enumerate(param_combinations):
            print(f"\nTesting combination {i + 1}/{n_combinations}...")
            print(f"Parameters: {params}")

            try:
                # Create config with these parameters
                test_config = self.create_test_config(params)

                # Run backtest (commented out - needs backtest integration)
                # backtest_results = run_hybrid_strategy_backtest(
                #     data_path=data_path,
                #     config=test_config,
                #     save_results_flag=False
                # )
                backtest_results = None

                if backtest_results and 'backtest' in backtest_results:
                    # Calculate fitness from backtest results
                    fitness = self.calculate_fitness(backtest_results['backtest'])
                    results.append({
                        'params': params,
                        'fitness': fitness,
                        'return': backtest_results['backtest'].get('total_return', 0),
                        'sharpe': backtest_results['backtest'].get('sharpe_ratio', 0),
                        'trades': backtest_results['backtest'].get('num_trades', 0)
                    })
                else:
                    # Backtest failed - apply severe penalty
                    results.append({
                        'params': params,
                        'fitness': self.severe_penalty,
                        'return': 0,
                        'sharpe': 0,
                        'trades': 0
                    })

            except Exception as e:
                print(f"Error in combination {i + 1}: {e}")
                results.append({
                    'params': params,
                    'fitness': self.severe_penalty,
                    'return': 0,
                    'sharpe': 0,
                    'trades': 0
                })

        # Filter valid results and sort by fitness
        valid_results = [r for r in results if r['fitness'] != self.severe_penalty]
        valid_results.sort(key=lambda x: x['fitness'], reverse=True)

        return {
            'optimizer_type': self.get_optimization_type().value,
            'total_combinations': len(results),
            'valid_results': len(valid_results),
            'best_result': valid_results[0] if valid_results else None,
            'all_results': valid_results
        }