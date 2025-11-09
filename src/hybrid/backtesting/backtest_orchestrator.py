import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

# Import and setup Windows compatibility FIRST
# Core imports
import pandas as pd

from src.hybrid.backtesting.backtest_result import BacktestResult
# Configuration and data
from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data.data_manager import DataManager
from src.hybrid.money_management import MoneyManager
from src.hybrid.optimization import OptimizerType, OptimizerFactory
from src.hybrid.results import Result
from src.hybrid.signals import SignalFactory
from src.hybrid.strategies import StrategyFactory
from src.hybrid.strategies import StrategyInterface


logger = logging.getLogger(__name__)

class DataLoadingError(Exception):
    """Raised when DataManager fails to load market data"""
    def __init__(self, message: str, source: str = None):
        super().__init__(message)
        self.source = source

class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid"""
    def __init__(self, message: str, config_section: str = None):
        super().__init__(message)
        self.config_section = config_section

class BacktestOrchestrator:
    """
    Main backtesting orchestrator - chooses appropriate backtesting method
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """
    def __init__(self, config: UnifiedConfig, project_root: Path = None):
        self.config = config
        self.project_root = project_root or Path.cwd()

        file_ops = config.get_section('file_operations', {})
        self.results_dir = file_ops.get('results_dir')
        self.timestamp_format = file_ops.get('timestamp_format')
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache orchestrator configuration values with validation"""

        # Validate sections exist
        general_config = self.config.get_section('general')
        if not general_config:
            raise ConfigurationError("Missing required section: 'general'")

        backtest_config = self.config.get_section('backtesting')
        if not backtest_config:
            raise ConfigurationError("Missing required section: 'backtesting'")

        math_config = self.config.get_section('mathematical_operations')
        if not math_config:
            raise ConfigurationError("Missing required section: 'mathematical_operations'")

        # Validate and cache with type checking
        if 'verbose' not in general_config:
            raise ConfigurationError("Missing required key: 'general.verbose'")
        self.verbose = bool(general_config['verbose'])

        self.backtesting_method = backtest_config.get('method', 'walk_forward')
        if self.backtesting_method not in ['walk_forward', 'simple']:
            raise ConfigurationError(f"Invalid backtesting method: {self.backtesting_method}")

        if 'unity' not in math_config:
            raise ConfigurationError("Missing required key: 'mathematical_operations.unity'")
        self.unity_value = int(math_config['unity'])
        if self.unity_value != 1:
            raise ConfigurationError(f"Invalid unity value: {self.unity_value}, expected 1")

    def _verify_data_loaded(self, data_manager: DataManager,
                            requested_markets: List[str] = None) -> List[str]:
        """Verify DataManager loaded data successfully"""
        loaded_markets = data_manager.get_available_markets()

        if not loaded_markets:
            raise DataLoadingError("DataManager failed to load any markets")

        if requested_markets:
            missing = [m for m in requested_markets if m not in loaded_markets]
            if missing:
                raise DataLoadingError(f"Required markets not loaded: {missing}")

        logger.info(f"Verified {len(loaded_markets)} markets loaded")
        return loaded_markets

    # Updated run_multi_strategy_backtest method with refactored Step 2
    def run_multi_strategy_backtest(self,
                                    strategies: List[Union[StrategyInterface, str]],
                                    markets: List[str] = None,
                                    execution_mode: str = "serial") -> Dict:
        """
        Run backtest with multiple strategies across multiple markets

        Args:
            strategies: List of strategy instances or strategy names
            markets: List of market identifiers (e.g., ['EURUSD', 'GBPUSD'])
            execution_mode: 'serial' or 'parallel'

        Returns:
            Comprehensive backtest results for all strategies
        """
        logger = logging.getLogger(__name__)
        start_time = datetime.now()

        logger.info("Starting multi-strategy backtest orchestration")
        logger.info(f"Strategies: {len(strategies)}, Markets: {markets or 'Auto-discover'}, Mode: {execution_mode}")

        try:
            # 1. Initialize managers
            data_manager = DataManager(self.config,project_root=self.project_root)
            money_manager = MoneyManager(self.config)
            logger.debug("Managers initialized successfully")

            # 2. Load market data using DataManager
            data_manager.load_market_data()
            loaded_markets = self._verify_data_loaded(data_manager, markets)
            logger.info(f"Market data loaded successfully for {len(loaded_markets)} markets")

            # 3. Initialize strategies with components and dependency injection
            # todo: optimizer_type must be derived from config.
            strategy_instances = self._initialize_strategies(strategies, data_manager, money_manager,
                                                             OptimizerType.SIMPLE_RANDOM)
            logger.info(f"Initialized {len(strategy_instances)} strategies with components")

            # 4. Each strategy runs its own backtest
            if execution_mode.lower() == "parallel":
                results = self._execute_strategies_parallel(strategy_instances, loaded_markets)
            else:
                results = self._execute_strategies_serial(strategy_instances, loaded_markets)

            # 5. Aggregate and analyze results
            aggregated_results = self._aggregate_strategy_results(results, start_time)

            logger.info("Multi-strategy backtest completed successfully")
            return aggregated_results

        except Exception as e:
            logger.debug(f"Exception caught in orchestrator: {type(e).__name__}: {e}")
            logger.error(f"Multi-strategy backtest failed: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'method': 'multi_strategy_backtest'
            }

    def _initialize_strategies(self, strategies: List[Union[StrategyInterface, str]],
                               data_manager: DataManager, money_manager: MoneyManager,
                               optimizer_type: OptimizerType = None) -> List[StrategyInterface]:
        """Initialize strategy instances with dependency injection"""
        strategy_factory = StrategyFactory()
        signal_factory = SignalFactory(self.config)
        strategy_instances = []

        # Get signals from config
        signals_config = self.config.get_section('strategy', {}).get('signals', [])

        # Determine optimizer type (parameter overrides config)
        if optimizer_type is None:
            opt_config = self.config.get_section('optimization', {})
            optimizer_type_str = opt_config.get('default_type', 'CACHED_RANDOM')
            optimizer_type = OptimizerType[optimizer_type_str]

        for strategy in strategies:
            if isinstance(strategy, str):
                created_strategy = strategy_factory.create_strategy(strategy, self.config)
            else:
                created_strategy = strategy

            # Inject managers
            created_strategy.setMoneyManager(money_manager)
            created_strategy.setDataManager(data_manager)

            for signal_name in signals_config:
                signal = signal_factory.create_signal(signal_name)
                created_strategy.addSignal(signal)

            # Add components
            created_strategy.addPredictor("placeholder_predictor")
            created_strategy.addOptimizer(OptimizerFactory.create_optimizer(optimizer_type, self.config))
            created_strategy.addRunner("placeholder_runner")
            created_strategy.addMetric("placeholder_metric")

            strategy_instances.append(created_strategy)

        return strategy_instances

    def _execute_strategies_serial(self, strategy_instances: List[StrategyInterface], market_data: Dict) -> List[
        Result]:
        """Execute strategies in serial mode"""
        results = []

        for strategy in strategy_instances:
            backtest_results = strategy.run(market_data)
            result = Result(strategy.name, backtest_results)
            results.append(result)
        return results

    def _execute_strategies_parallel(self, strategy_instances: List[StrategyInterface], market_data: Dict) -> List[
        Result]:
        """Execute strategies in parallel mode"""
        results = []

        # TODO: Implement parallel execution
        for strategy in strategy_instances:
            result = Result(strategy.name, "placeholder_data")
            results.append(result)

        return results

    def _aggregate_strategy_results(self, strategy_results: List[Result], start_time: datetime) -> Dict:
        """Aggregate results from multiple strategies"""

        # TODO: Implement result aggregation logic
        return {
            'method': self.backtesting_method,  # Add this line
            'total_strategies': len(strategy_results),
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'results': strategy_results
        }

    def _run_walkforward_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Walk-Forward Optimization Backtest

        PURPOSE:
        Walk-forward optimization tests strategy robustness by continuously re-optimizing
        parameters as new data becomes available, simulating real-world parameter tuning.

        METHODOLOGY:
        1. Split data into rolling windows: [Train1|Test1] [Train2|Test2] [Train3|Test3]
        2. Optimize parameters on Train1 → test on Test1
        3. Re-optimize on Train1+Train2 → test on Test2 (parameters may change)
        4. Re-optimize on Train1+Train2+Train3 → test on Test3
        5. Aggregate results across all test periods

        PREVENTS OVERFITTING:
        - Parameters are re-optimized periodically (not fixed forever)
        - Tests if strategy adapts well to changing market conditions
        - Simulates realistic parameter maintenance schedule

        USE CASE:
        Essential for validating that optimized parameters remain profitable over time,
        not just curve-fit to one specific market regime.

        TEMPORAL ISOLATION:
        Uses TemporalDataGuard to ensure training only uses past data (no look-ahead bias).
        Uses WalkForwardRetrainingStrategy to decide WHEN to re-optimize.

        STATUS: Currently being refactored to work with new BacktestEngine architecture.
        """
        raise NotImplementedError(
            "Walk-forward optimization temporarily disabled during architecture refactor.\n"
            "Will be reimplemented using:\n"
            "  - TemporalDataGuard (prevents look-ahead bias)\n"
            "  - WalkForwardRetrainingStrategy (retraining decisions)\n"
            "  - BacktestEngine (executes each window)\n"
            "  - BacktestResult (aggregates results)\n"
            "See temporal_data_guard.py and walk_forward_retraining_strategy.py for components."
        )

    def _create_config_summary(self) -> Dict:
        """Create configuration summary for results"""
        regime_config = self.config.get_section('regime_detection', {})
        risk_config = self.config.get_section('risk_management', {})
        vol_config = self.config.get_section('volatility_prediction', {})
        duration_config = self.config.get_section('trend_duration_prediction', {})
        backtest_config = self.config.get_section('backtesting', {})

        return {
            'strategy_type': 'Hybrid ML-Technical Walk-Forward',
            'temporal_method': 'strict_isolation',
            'ml_components': {
                'regime': regime_config.get('method'),
                'volatility': vol_config.get('use_volatility_ml'),
                'duration': duration_config.get('enabled')
            },
            'backtesting': {
                'method': backtest_config.get('method'),
                'pretrain_rows': backtest_config.get('pretrain_rows'),
                'retrain_frequency': backtest_config.get('retrain_frequency')
            },
            'risk_management': {
                'stop_loss_pct': risk_config.get('stop_loss_pct'),
                'take_profit_pct': risk_config.get('take_profit_pct'),
                'max_position_size': risk_config.get('max_position_size')
            }
        }

    def _save_results(self, results: Dict):
        """Save results using BacktestResult"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Convert dict results to BacktestResult if needed
            if isinstance(results, dict) and 'results' in results:
                # Multi-strategy results
                for strategy_result in results['results']:
                    if isinstance(strategy_result, BacktestResult):
                        output_dir = Path(self.results_dir) / f"backtest_{timestamp}"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        strategy_result.save_with_config(str(output_dir))
                        logger.info(f"Results saved to: {output_dir}")

            elif isinstance(results, BacktestResult):
                # Single result
                output_dir = Path(self.results_dir) / f"backtest_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)
                results.save_with_config(str(output_dir))
                logger.info(f"Results saved to: {output_dir}")

            else:
                logger.warning(f"Unknown result type: {type(results)}")

        except Exception as e:
            logger.error(f"Could not save results: {e}")

    def _print_final_summary(self, start_time: datetime, results: Dict):
        """Print final backtest summary"""
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Method: {results.get('method', 'unknown')}")
        logger.info(f"Total time: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)")

        # Print data statistics if available
        data_info = results.get('data_info', {})
        total_records = data_info.get('total_records', 0)
        if total_records > 0:
            logger.info(f"Data records: {total_records:,}")
            logger.info(f"Records/second: {total_records / total_duration:,.0f}")

        # Print strategy results summary
        if 'results' in results:
            num_strategies = len(results['results'])
            logger.info(f"Strategies tested: {num_strategies}")

        logger.info("=" * 80)

    def _handle_error(self, error: Exception, start_time: datetime):
        """Handle errors during backtesting"""
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        logger.error(f"Backtest failed: {error}")
        logger.error(f"Failed after {total_duration:.1f} seconds")
        logger.error(traceback.format_exc())