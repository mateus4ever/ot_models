# cached_optimizer.py
# JAVA EQUIVALENT: public class CachedRandomOptimizer extends BaseOptimizer implements IOptimizer

import random
from typing import Dict, List
from src.hybrid.optimization.optimization_interface import IOptimizerBase
from src.hybrid.optimization.optimizer_type import OptimizerType
from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.hybrid_strategy import HybridStrategy
from src.hybrid.backtesting import BacktestEngine


class CachedRandomOptimizer(IOptimizerBase):
    """
    Cached random parameter optimization
    Loads data and trains ML once, then tests multiple parameter combinations

    JAVA EQUIVALENT:
    public class CachedRandomOptimizer extends BaseOptimizer implements IOptimizer {
        private DataFrame cachedData;
        private HybridStrategy cachedStrategy;
        private DataFrame cachedSignals;

        public CachedRandomOptimizer(UnifiedConfig config) {
            super(config);
        }

        @Override
        public OptimizationType getOptimizationType() {
            return OptimizationType.CACHED_RANDOM;
        }
    }
    """

    def __init__(self, config: UnifiedConfig):
        """
        JAVA EQUIVALENT:
        public CachedRandomOptimizer(UnifiedConfig config) {
            super(config);
            this.cachedData = null;
            this.cachedStrategy = null;
            this.cachedSignals = null;
        }
        """
        super().__init__(config)
        self.cached_data = None
        self.cached_strategy = None
        self.cached_signals = None

    def get_optimization_type(self) -> OptimizerType:
        """
        JAVA EQUIVALENT:
        @Override
        public OptimizationType getOptimizationType() {
            return OptimizationType.CACHED_RANDOM;
        }
        """
        return OptimizerType.CACHED_RANDOM

    def get_description(self) -> str:
        """
        JAVA EQUIVALENT:
        @Override
        public String getDescription() {
            return "Cached random parameter search with optimized data/ML reuse";
        }
        """
        return "Cached random parameter search with optimized data/ML reuse"

    def initialize_cache(self, data_path: str = None):
        """
        Initialize optimization cache with data and ML training

        JAVA EQUIVALENT:
        private void initializeCache(String dataPath) {
            System.out.println("Initializing optimization cache...");

            if (dataPath == null) {
                Map<String, Object> dataConfig = config.getSection("data_loading");
                dataPath = (String) dataConfig.getOrDefault("data_source", "data/eurusd");
            }

            this.cachedData = loadAndPreprocessData(dataPath, config);
            // ...
        }
        """
        print("Initializing optimization cache...")

        if data_path is None:
            data_config = self.config.get_section('data_loading', {})
            data_path = data_config.get('data_source', 'data/eurusd')

        # self.cached_data = load_and_preprocess_data(data_path, self.config)
        print(f"✓ Data loaded: {len(self.cached_data)} records")


        self.cached_strategy = HybridStrategy(self.config)
        training_results = self.cached_strategy.train(self.cached_data)
        print(f"✓ ML training completed in {training_results.get('training_time', 0):.1f}s")

        self.cached_signals = self.cached_strategy.generate_signals(self.cached_data)
        print(f"✓ Signals generated: {len(self.cached_signals)} records")
        print("Cache initialization complete!\n")

    def create_optimized_config(self, params: Dict) -> UnifiedConfig:
        """
        Create optimized config for backtesting

        JAVA EQUIVALENT:
        private UnifiedConfig createOptimizedConfig(Map<String, Double> params) {
            UnifiedConfig newConfig = new UnifiedConfig(config.getConfigPath());
            newConfig.setConfig(config.getConfig().copy());

            Map<String, Object> updates = new HashMap<>();
            // ... build updates map
            newConfig.updateConfig(updates);
            return newConfig;
        }
        """
        new_config = UnifiedConfig(self.config.config_path)
        new_config.config = self.config.config.copy()

        updates = {
            'risk_management': {
                'stop_loss_pct': params['stop_loss_pct'],
                'take_profit_pct': params['take_profit_pct'],
                'max_position_size': params['max_position_size']
            },
            'general': {'verbose': False, 'save_signals': False, 'debug_mode': False},
            'debug_configuration': {
                'trade_debug_count': 0,
                'enable_fee_debug': False,
                'enable_trade_debug': False,
                'enable_metrics_debug': False,
                'enable_direct_math_check': False
            }
        }
        new_config.update_config(updates)
        return new_config

    def run_single_backtest(self, params: Dict) -> Dict:
        """
        Run single backtest with cached data

        JAVA EQUIVALENT:
        private BacktestResult runSingleBacktest(Map<String, Double> params) {
            try {
                UnifiedConfig testConfig = createOptimizedConfig(params);
                BacktestEngine backtestEngine = new BacktestEngine(testConfig);
                Map<String, Object> backtestResults = backtestEngine.runBacktest(cachedData, cachedSignals);

                return new BacktestResult(backtestResults, true);
            } catch (Exception e) {
                return new BacktestResult(new HashMap<>(), false, e.getMessage());
            }
        }
        """
        try:
            test_config = self.create_optimized_config(params)
            backtest_engine = BacktestEngine(test_config)
            backtest_results = backtest_engine.run_backtest(self.cached_data, self.cached_signals)

            return {
                'backtest': backtest_results,
                'success': True
            }
        except Exception as e:
            return {
                'backtest': {},
                'success': False,
                'error': str(e)
            }

    def generate_random_parameters(self, n_combinations: int) -> List[Dict]:
        """
        Generate random parameter combinations

        JAVA EQUIVALENT:
        private List<Map<String, Double>> generateRandomParameters(int nCombinations) {
            Map<String, Object> ranges = config.getSection("optimization").get("parameter_ranges");
            List<Map<String, Double>> combinations = new ArrayList<>();

            for (int i = 0; i < nCombinations; i++) {
                Map<String, Double> combo = new HashMap<>();
                combo.put("stop_loss_pct", Random.uniform(ranges.get("stop_loss_min"), ranges.get("stop_loss_max")));
                // ...
                combinations.add(combo);
            }
            return combinations;
        }
        """
        ranges = self.config.get_section('optimization', {}).get('parameter_ranges', {})
        combinations = []

        for _ in range(n_combinations):
            combo = {
                'stop_loss_pct': random.uniform(ranges.get('stop_loss_min'), ranges.get('stop_loss_max')),
                'take_profit_pct': random.uniform(ranges.get('take_profit_min'), ranges.get('take_profit_max')),
                'max_position_size': random.uniform(ranges.get('max_position_min'), ranges.get('max_position_max'))
            }
            combinations.append(combo)

        return combinations

    def run_optimization(self, data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
        """
        Run cached random optimization

        JAVA EQUIVALENT:
        @Override
        public OptimizationResult runOptimization(String dataPath, Integer nCombinations, Map<String, Object> kwargs) {
            if (nCombinations == null) {
                nCombinations = config.getSection("optimization").get("defaults").get("n_combinations", 10);
            }

            System.out.println("Running " + getDescription());
            System.out.println("Combinations: " + nCombinations);

            initializeCache(dataPath);
            List<Map<String, Double>> paramCombinations = generateRandomParameters(nCombinations);
            List<OptimizationResult> results = new ArrayList<>();

            // ... rest of optimization logic
        }
        """
        if n_combinations is None:
            n_combinations = self.config.get_section('optimization', {}).get('defaults', {}).get('n_combinations', 10)

        print(f"Running {self.get_description()}")
        print(f"Combinations: {n_combinations}")

        self.initialize_cache(data_path)
        param_combinations = self.generate_random_parameters(n_combinations)
        results = []

        for i, params in enumerate(param_combinations):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Progress: {i + 1}/{n_combinations} combinations tested...")

            backtest_result = self.run_single_backtest(params)

            if backtest_result['success'] and backtest_result['backtest']:
                fitness = self.calculate_fitness(backtest_result['backtest'])
                results.append({
                    'params': params,
                    'fitness': fitness,
                    'return': backtest_result['backtest'].get('total_return', 0),
                    'sharpe': backtest_result['backtest'].get('sharpe_ratio', 0),
                    'trades': backtest_result['backtest'].get('num_trades', 0)
                })
            else:
                results.append({
                    'params': params,
                    'fitness': self.severe_penalty,
                    'return': 0, 'sharpe': 0, 'trades': 0,
                    'error': backtest_result.get('error', 'Unknown error')
                })

        valid_results = [r for r in results if r['fitness'] != self.severe_penalty]
        valid_results.sort(key=lambda x: x['fitness'], reverse=True)

        return {
            'optimizer_type': self.get_optimization_type().value,
            'total_combinations': len(results),
            'valid_results': len(valid_results),
            'best_result': valid_results[0] if valid_results else None,
            'all_results': valid_results,
            'cache_used': True
        }