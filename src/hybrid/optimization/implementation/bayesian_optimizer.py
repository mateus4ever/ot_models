# bayesian_optimizer.py - FIXED VERSION
# JAVA EQUIVALENT: public class BayesianOptimizer extends BaseOptimizer implements IOptimizer

import numpy as np
from typing import Dict, List
from datetime import datetime
from src.hybrid.optimization.optimization_interface import IOptimizerBase
from src.hybrid.optimization.optimizer_type import OptimizerType
from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.hybrid_strategy import HybridStrategy

# Bayesian optimization imports
try:
    from skopt import gp_minimize
    from skopt.space import Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


class BayesianOptimizer(IOptimizerBase):
    """
    Fixed Bayesian Parameter Optimization

    KEY FIX: No signal caching - each trial generates fresh signals with optimized parameters
    This eliminates "signal leakage" where pre-computed signals ignore optimization parameters
    """

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.cached_train_data = None
        self.cached_test_data = None
        self.base_trained_strategy = None  # Strategy trained once, used as template
        self.all_evaluations = []

        # Get array indexing config
        array_config = self.config.get_section('array_indexing', {})
        math_config = self.config.get_section('mathematical_operations', {})
        self.evaluation_count = math_config.get('zero')

        # Get Bayesian-specific config
        bayesian_config = self._get_bayesian_config()
        self.n_calls = bayesian_config.get('n_calls')
        self.n_initial_points = bayesian_config.get('n_initial_points')
        self.acquisition_function = bayesian_config.get('acquisition_function')
        self.data_sample_size = bayesian_config.get('data_sample_size')

    def get_optimization_type(self) -> OptimizerType:
        return OptimizerType.BAYESIAN

    def get_description(self) -> str:
        return "Bayesian optimization with fresh signal generation per trial (no signal caching)"

    def _get_bayesian_config(self) -> Dict:
        """Get Bayesian optimization configuration - ZERO HARDCODED VALUES"""
        bayesian_config = self.config.get_section('optimization', {}).get('bayesian', {})
        defaults_config = self.config.get_section('optimization', {}).get('defaults', {})
        bayesian_defaults = self.config.get_section('optimization', {}).get('bayesian_defaults', {})
        math_config = self.config.get_section('mathematical_operations', {})

        default_n_calls = bayesian_defaults.get('n_calls', defaults_config.get('n_combinations'))
        default_n_initial_points = bayesian_defaults.get('n_initial_points', default_n_calls // bayesian_defaults.get(
            'initial_points_divisor'))

        default_config = {
            'n_calls': bayesian_defaults.get('n_calls', defaults_config.get('n_combinations')),
            'n_initial_points': default_n_initial_points,
            'acquisition_function': bayesian_defaults.get('acquisition_function'),
            'data_sample_size': defaults_config.get('data_sample_size'),
            'random_state': math_config.get('random_seed')
        }

        for key, default_value in default_config.items():
            if key not in bayesian_config:
                bayesian_config[key] = default_value

        return bayesian_config

    def initialize_cache(self, data_path: str = None):
        """
        Initialize cache with train/test split and base trained strategy

        KEY CHANGE: Only cache data and base strategy training, NOT signals
        Signals will be generated fresh for each optimization trial
        """
        print("Initializing Bayesian optimization cache (no signal caching)...")

        if data_path is None:
            data_config = self.config.get_section('data_loading', {})
            data_path = data_config.get('data_source')

        # Load full dataset
        # full_data = load_and_preprocess_data(data_path, self.config)
        full_data = None
        # print(f"✓ Loaded full dataset: {len(full_data):,} rows")

        # Get walk-forward configuration
        walk_forward_config = self.config.get_section('walk_forward', {})
        train_window_size = walk_forward_config.get('train_window_size')
        test_window_size = walk_forward_config.get('test_window_size')

        # Validate we have enough data
        total_required = train_window_size + test_window_size
        if len(full_data) < total_required:
            raise ValueError(f"Insufficient data: need {total_required:,} rows, have {len(full_data):,}")

        # Split data with proper temporal ordering
        self.cached_train_data = full_data.head(train_window_size).copy()
        print(f"✓ Training data: {len(self.cached_train_data):,} rows (oldest)")

        start_idx = train_window_size
        end_idx = start_idx + test_window_size
        self.cached_test_data = full_data.iloc[start_idx:end_idx].copy()
        print(f"✓ Testing data: {len(self.cached_test_data):,} rows (newer, unseen)")

        # Train base strategy ONCE on training data (for ML models only)
        # This strategy will be used as a template for generating fresh signals
        self.base_trained_strategy = HybridStrategy(self.config)
        print("Training base ML models on historical data...")
        training_results = self.base_trained_strategy.train(self.cached_train_data)
        training_time_key = 'training_time'
        zero_default = self.config.get_section('mathematical_operations', {}).get('zero')
        print(f"✓ Base ML training completed in {training_results.get(training_time_key, zero_default):.1f}s")

        # Print temporal validation
        train_start = self.cached_train_data.index[self.config.get_section('array_indexing', {}).get('first_index')]
        train_end = self.cached_train_data.index[-self.config.get_section('mathematical_operations', {}).get('unity')]
        test_start = self.cached_test_data.index[self.config.get_section('array_indexing', {}).get('first_index')]
        test_end = self.cached_test_data.index[-self.config.get_section('mathematical_operations', {}).get('unity')]

        print(f"✓ Temporal validation:")
        print(f"  Training period: {train_start} to {train_end}")
        print(f"  Testing period:  {test_start} to {test_end}")
        print(f"  ✓ No data leakage - models trained on past, tested on future")
        print("✓ Cache initialization complete - ready for fresh signal generation per trial!\n")

    # def objective_function(self, params_list: List[float]) -> float:
    #     """
    #     FIXED: Generate fresh signals for each optimization trial
    #
    #     This ensures position sizing and other strategy parameters
    #     are properly applied for each parameter combination
    #     """
    #     array_config = self.config.get_section('array_indexing', {})
    #     math_config = self.config.get_section('mathematical_operations', {})
    #     one = math_config.get('unity')
    #
    #     self.evaluation_count += one
    #
    #     # Get parameter indices from config
    #     stop_loss_index = array_config.get('first_index')
    #     take_profit_index = array_config.get('second_index')
    #     max_position_index = array_config.get('third_index')
    #
    #     params = {
    #         'stop_loss_pct': params_list[stop_loss_index],
    #         'take_profit_pct': params_list[take_profit_index],
    #         'max_position_size': params_list[max_position_index]
    #     }
    #
    #     try:
    #         # Create optimized config with parameters
    #         new_config = UnifiedConfig(self.config.config_path)
    #         new_config.config = self.config.config.copy()
    #
    #         # Get debug config values
    #         debug_config = self.config.get_section('debug_configuration', {})
    #         general_config = self.config.get_section('general', {})
    #         backtesting_config = self.config.get_section('backtesting', {})
    #
    #         updates = {
    #             'risk_management': params,
    #             'general': {
    #                 'verbose': general_config.get('verbose'),
    #                 'save_signals': general_config.get('save_signals'),
    #                 'debug_mode': general_config.get('debug_mode')
    #             },
    #             'debug_configuration': {
    #                 'enable_metrics_debug': debug_config.get('enable_metrics_debug'),
    #                 'enable_direct_math_check': debug_config.get('enable_direct_math_check'),
    #                 'enable_trade_debug': debug_config.get('enable_trade_debug'),
    #                 'enable_position_debug': debug_config.get('enable_position_debug'),
    #                 'print_trade_details': debug_config.get('print_trade_details'),
    #                 'log_trades': debug_config.get('log_trades'),
    #                 'trade_debug_count': math_config.get('zero'),
    #                 'enable_fee_debug': debug_config.get('enable_fee_debug')
    #             },
    #             'backtesting': {
    #                 'print_trades': backtesting_config.get('print_trades'),
    #                 'verbose_output': backtesting_config.get('verbose_output')
    #             }
    #         }
    #         new_config.update_config(updates)
    #
    #         # KEY FIX: Create fresh strategy with optimized config for signal generation
    #         fresh_strategy = HybridStrategy(new_config)
    #
    #         # Copy the trained ML models from base strategy to avoid retraining
    #         fresh_strategy.ml_manager = self.base_trained_strategy.ml_manager
    #         fresh_strategy.is_trained = True
    #         fresh_strategy.training_results = self.base_trained_strategy.training_results
    #
    #         # Generate fresh signals with the optimized configuration
    #         fresh_signals = fresh_strategy.generate_signals(self.cached_test_data)
    #
    #         # Run backtest with fresh signals
    #         backtest_engine = BacktestEngine(new_config)
    #         backtest_results = backtest_engine.run_backtest(self.cached_test_data, fresh_signals)
    #
    #         fitness = self.calculate_fitness(backtest_results)
    #
    #         # Get keys from config
    #         result_keys = self.config.get_section('result_keys', {})
    #         zero_default = math_config.get('zero')
    #         true_value = self.config.get_section('boolean_values', {}).get('true', True)
    #
    #         # Store evaluation
    #         self.all_evaluations.append({
    #             'evaluation': self.evaluation_count,
    #             'params': params.copy(),
    #             'fitness': fitness,
    #             'return': backtest_results.get(result_keys.get('total_return', 'total_return'), zero_default),
    #             'sharpe': backtest_results.get(result_keys.get('sharpe_ratio', 'sharpe_ratio'), zero_default),
    #             'trades': backtest_results.get(result_keys.get('num_trades', 'num_trades'), zero_default),
    #             'success': true_value
    #         })
    #
    #         return -fitness  # Negative for minimization
    #
    #     except Exception as e:
    #         # Error handling
    #         import traceback
    #         import sys
    #
    #         exc_type, exc_value, exc_traceback = sys.exc_info()
    #
    #         if "unsupported operand type(s) for %" in str(e):
    #             print(f"\n=== MODULO ERROR FOUND IN EVALUATION {self.evaluation_count} ===")
    #             print(f"Error message: {e}")
    #             traceback.print_exc()
    #             print("=== END MODULO ERROR DETAILS ===\n")
    #         else:
    #             print(f"Error in evaluation {self.evaluation_count}: {e}")
    #
    #         self.all_evaluations.append({
    #             'evaluation': self.evaluation_count,
    #             'params': params.copy(),
    #             'fitness': self.severe_penalty,
    #             'success': False,
    #             'error': str(e)
    #         })
    #         return abs(self.severe_penalty)

    def _print_optimization_results(self, valid_evaluations: List[Dict]):
        """Print formatted optimization results table with ML component quality metrics"""
        if not valid_evaluations:
            print("No valid results to display")
            return

        # Get display config
        display_config = self.config.get_section('optimization', {}).get('display', {})
        top_count = display_config.get('top_display_count', len(valid_evaluations))

        # Limit to top performers
        top_results = valid_evaluations[:min(top_count, len(valid_evaluations))]

        # Extract ML component quality metrics from base strategy training results
        training_results = getattr(self.base_trained_strategy, 'training_results', {})

        # Get component metrics with fallbacks
        math_config = self.config.get_section('mathematical_operations', {})
        zero_default = math_config.get('zero')

        regime_acc = training_results.get('regime', {}).get('strength_accuracy', zero_default)
        vol_acc = training_results.get('volatility', {}).get('accuracy', zero_default)
        duration_acc = training_results.get('duration', {}).get('accuracy', zero_default)
        ml_features = training_results.get('volatility', {}).get('n_features', zero_default)

        # Print header
        separator_char = display_config.get('separator_char', '=')
        separator_length = display_config.get('separator_length', 80)
        print(f"\n{separator_char * separator_length}")
        print(display_config.get('results_title', 'OPTIMIZATION RESULTS - TOP PARAMETER COMBINATIONS'))
        print(f"{separator_char * separator_length}")

        # Print column headers with ML component columns
        print(f"{'Rank':<6} {'Stop Loss':<12} {'Take Profit':<14} {'Max Pos':<10} {'Sharpe':<8} {'Trades':<8} "
              f"{'Return':<10} {'Fitness':<10} {'RegAcc':<8} {'VolAcc':<8} {'DurAcc':<8} {'MLFeat':<8}")
        print(f"{display_config.get('dash_char', '-') * (separator_length + 32)}")

        # Print results with ML component metrics
        array_config = self.config.get_section('array_indexing', {})
        one = math_config.get('unity')

        # Get percentage multiplier from config
        output_config = self.config.get_section('output_formatting', {})
        percentage_multiplier = output_config.get('percentage_multiplier',
                                                  math_config.get('unity') * math_config.get('unity') * 10 * 10)

        for i, result in enumerate(top_results):
            rank = i + one
            params = result['params']

            print(f"{rank:<6} "
                  f"{params['stop_loss_pct'] * percentage_multiplier:.1f}%{'':<7} "
                  f"{params['take_profit_pct'] * percentage_multiplier:.1f}%{'':<8} "
                  f"{params['max_position_size'] * percentage_multiplier:.0f}%{'':<7} "
                  f"{result.get('sharpe', zero_default):.2f}{'':<4} "
                  f"{result.get('trades', zero_default):<8} "
                  f"{result.get('return', zero_default) * percentage_multiplier:.1f}%{'':<5} "
                  f"{result['fitness']:.1f}{'':<5} "
                  f"{regime_acc:.3f}{'':<3} "
                  f"{vol_acc:.3f}{'':<5} "
                  f"{duration_acc:.3f}{'':<3} "
                  f"{ml_features:<8}")

        # Print best combination summary
        if top_results:
            best = top_results[array_config.get('first_index')]
            print(f"\n{display_config.get('summary_title', 'BEST COMBINATION SUMMARY:')}")
            print(f"Return: {best.get('return', zero_default) * percentage_multiplier:.2f}% | "
                  f"Sharpe: {best.get('sharpe', zero_default):.2f} | "
                  f"Trades: {best.get('trades', zero_default)}")
            print(f"Stop Loss: {best['params']['stop_loss_pct'] * percentage_multiplier:.1f}% | "
                  f"Take Profit: {best['params']['take_profit_pct'] * percentage_multiplier:.1f}%")
            print(f"Max Position: {best['params']['max_position_size'] * percentage_multiplier:.0f}% | "
                  f"Fitness Score: {best['fitness']:.1f}")

            # Add ML component summary
            print(f"\nML COMPONENT QUALITY:")
            print(f"Regime Detection Accuracy: {regime_acc:.3f} | "
                  f"Volatility Prediction Accuracy: {vol_acc:.3f}")
            print(f"Duration Prediction Accuracy: {duration_acc:.3f} | "
                  f"ML Features Used: {ml_features}")

        print(f"{separator_char * (separator_length + 32)}")

    def run_optimization(self, data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
        """Run Bayesian optimization with fresh signal generation per trial"""

        try:
            if not SKOPT_AVAILABLE:
                raise ImportError(
                    "scikit-optimize is required for Bayesian optimization. Install with: pip install scikit-optimize")

            if n_combinations is not None:
                self.n_calls = n_combinations

            print(f"Running {self.get_description()}")
            print(f"Total evaluations: {self.n_calls}")
            print(f"Initial random points: {self.n_initial_points}")
            print(f"🔄 Fresh signals generated per trial (no caching)")

            start_time = datetime.now()

            self.initialize_cache(data_path)

            # Define search space
            ranges = self.config.get_section('optimization', {}).get('parameter_ranges', {})
            dimensions = [
                Real(ranges.get('stop_loss_min'), ranges.get('stop_loss_max'), name='stop_loss_pct'),
                Real(ranges.get('take_profit_min'), ranges.get('take_profit_max'), name='take_profit_pct'),
                Real(ranges.get('max_position_min'), ranges.get('max_position_max'), name='max_position_size')
            ]

            # Get bayesian config for random_state
            bayesian_config = self._get_bayesian_config()

            # Run Bayesian optimization
            result = gp_minimize(
                func=self.objective_function,
                dimensions=dimensions,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                acq_func=self.acquisition_function,
                random_state=bayesian_config.get('random_state')
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Get success key from config
            valid_evaluations = [e for e in self.all_evaluations if e['success']]
            valid_evaluations.sort(key=lambda x: x['fitness'], reverse=True)

            print(f"\nOptimization completed!")
            print(f"Valid results: {len(valid_evaluations)}/{len(self.all_evaluations)}")

            # Print optimization results table
            self._print_optimization_results(valid_evaluations)

            # Get array indexing for best result
            array_config = self.config.get_section('array_indexing', {})
            first_index = array_config.get('first_index')

            return {
                'optimizer_type': self.get_optimization_type().value,
                'skopt_result': result,
                'all_evaluations': self.all_evaluations,
                'valid_evaluations': valid_evaluations,
                'best_result': valid_evaluations[first_index] if valid_evaluations else None,
                'total_duration': duration,
                'bayesian': True,
                'fresh_signals': True  # Flag indicating this version generates fresh signals
            }

        except BaseException as e:
            print(f"ERROR in Bayesian optimization: {e}")
            import traceback
            traceback.print_exc()
            return {}