# simple_optimizer.py
# JAVA EQUIVALENT: public class SimpleRandomOptimizer extends BaseOptimizer implements IOptimizer

import random
from typing import Dict, List
from .optimization_interface import IOptimizerBase
from .optimization_types import OptimizationType
from src.hybrid.config.unified_config import UnifiedConfig


class SimpleRandomOptimizer(IOptimizerBase):
    """
    Simple random parameter optimization
    Original implementation - good for educational purposes and simple testing

    JAVA EQUIVALENT:
    public class SimpleRandomOptimizer extends BaseOptimizer implements IOptimizer {

        public SimpleRandomOptimizer(UnifiedConfig config) {
            super(config);
        }

        @Override
        public OptimizationType getOptimizationType() {
            return OptimizationType.SIMPLE_RANDOM;
        }

        @Override
        public String getDescription() {
            return "Simple random parameter search with full data loading per iteration";
        }

        // ... implementation methods
    }
    """

    def get_optimization_type(self) -> OptimizationType:
        """
        JAVA EQUIVALENT:
        @Override
        public OptimizationType getOptimizationType() {
            return OptimizationType.SIMPLE_RANDOM;
        }
        """
        return OptimizationType.SIMPLE_RANDOM

    def get_description(self) -> str:
        """
        JAVA EQUIVALENT:
        @Override
        public String getDescription() {
            return "Simple random parameter search with full data loading per iteration";
        }
        """
        return "Simple random parameter search with full data loading per iteration"

    def generate_random_parameters(self, n_combinations: int) -> List[Dict]:
        """
        Generate random parameter combinations

        JAVA EQUIVALENT:
        private List<Map<String, Double>> generateRandomParameters(int nCombinations) {
            Map<String, Object> ranges = config.getSection("optimization").get("parameter_ranges");
            List<Map<String, Double>> combinations = new ArrayList<>();

            for (int i = 0; i < nCombinations; i++) {
                Map<String, Double> combo = new HashMap<>();
                combo.put("stop_loss_pct",
                    ThreadLocalRandom.current().nextDouble(
                        (Double) ranges.get("stop_loss_min"),
                        (Double) ranges.get("stop_loss_max")
                    )
                );
                combo.put("take_profit_pct",
                    ThreadLocalRandom.current().nextDouble(
                        (Double) ranges.get("take_profit_min"),
                        (Double) ranges.get("take_profit_max")
                    )
                );
                combo.put("max_position_size",
                    ThreadLocalRandom.current().nextDouble(
                        (Double) ranges.get("max_position_min"),
                        (Double) ranges.get("max_position_max")
                    )
                );
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

    def create_test_config(self, params: Dict) -> UnifiedConfig:
        """
        Create test configuration with parameters

        JAVA EQUIVALENT:
        private UnifiedConfig createTestConfig(Map<String, Double> params) {
            UnifiedConfig newConfig = new UnifiedConfig(config.getConfigPath());
            newConfig.setConfig(config.getConfig().copy());

            Map<String, Object> updates = new HashMap<>();
            Map<String, Object> riskManagement = new HashMap<>();
            riskManagement.put("stop_loss_pct", params.get("stop_loss_pct"));
            riskManagement.put("take_profit_pct", params.get("take_profit_pct"));
            riskManagement.put("max_position_size", params.get("max_position_size"));
            updates.put("risk_management", riskManagement);

            Map<String, Object> general = new HashMap<>();
            general.put("verbose", false);
            updates.put("general", general);

            Map<String, Object> debugConfig = new HashMap<>();
            debugConfig.put("trade_debug_count", 0);
            debugConfig.put("enable_fee_debug", false);
            debugConfig.put("enable_trade_debug", false);
            updates.put("debug_configuration", debugConfig);

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
            'general': {'verbose': False},
            'debug_configuration': {
                'trade_debug_count': 0,
                'enable_fee_debug': False,
                'enable_trade_debug': False
            }
        }
        new_config.update_config(updates)
        return new_config

    def run_optimization(self, data_path: str = None, n_combinations: int = None, **kwargs) -> Dict:
        """
        Run simple random optimization

        JAVA EQUIVALENT:
        @Override
        public OptimizationResult runOptimization(String dataPath, Integer nCombinations, Map<String, Object> kwargs) {
            // Import here to avoid circular dependency - equivalent to dynamic import
            HybridStrategyBacktest backtest = new HybridStrategyBacktest();

            if (nCombinations == null) {
                nCombinations = config.getSection("optimization")
                    .get("defaults")
                    .getOrDefault("n_combinations", 10);
            }

            System.out.println("Running " + getDescription());
            System.out.println("Combinations: " + nCombinations);

            List<Map<String, Double>> paramCombinations = generateRandomParameters(nCombinations);
            List<OptimizationResult> results = new ArrayList<>();

            for (int i = 0; i < paramCombinations.size(); i++) {
                Map<String, Double> params = paramCombinations.get(i);
                System.out.println("Testing combination " + (i+1) + "/" + nCombinations + "...");

                try {
                    UnifiedConfig testConfig = createTestConfig(params);
                    BacktestResult backtestResults = backtest.runHybridStrategyBacktest(
                        dataPath, testConfig, false
                    );

                    if (backtestResults != null && backtestResults.containsKey("backtest")) {
                        double fitness = calculateFitness(backtestResults.get("backtest"));
                        OptimizationResult result = new OptimizationResult(
                            params, fitness,
                            backtestResults.get("backtest").get("total_return"),
                            backtestResults.get("backtest").get("sharpe_ratio"),
                            backtestResults.get("backtest").get("num_trades")
                        );
                        results.add(result);
                    } else {
                        results.add(new OptimizationResult(params, severePenalty, 0, 0, 0));
                    }

                } catch (Exception e) {
                    System.out.println("Error in combination " + (i+1) + ": " + e.getMessage());
                    results.add(new OptimizationResult(params, severePenalty, 0, 0, 0));
                }
            }

            List<OptimizationResult> validResults = results.stream()
                .filter(r -> r.getFitness() != severePenalty)
                .sorted(Comparator.comparing(OptimizationResult::getFitness).reversed())
                .collect(Collectors.toList());

            return new OptimizationSummary(
                getOptimizationType().getValue(),
                results.size(),
                validResults.size(),
                validResults.isEmpty() ? null : validResults.get(0),
                validResults
            );
        }
        """
        # Import here to avoid circular dependency
        from src.hybrid.backtest import run_hybrid_strategy_backtest

        if n_combinations is None:
            n_combinations = self.config.get_section('optimization', {}).get('defaults', {}).get('n_combinations', 10)

        print(f"Running {self.get_description()}")
        print(f"Combinations: {n_combinations}")

        param_combinations = self.generate_random_parameters(n_combinations)
        results = []

        for i, params in enumerate(param_combinations):
            print(f"Testing combination {i + 1}/{n_combinations}...")

            try:
                test_config = self.create_test_config(params)
                backtest_results = run_hybrid_strategy_backtest(
                    data_path=data_path,
                    config=test_config,
                    save_results_flag=False
                )

                if backtest_results and 'backtest' in backtest_results:
                    fitness = self.calculate_fitness(backtest_results['backtest'])
                    results.append({
                        'params': params,
                        'fitness': fitness,
                        'return': backtest_results['backtest'].get('total_return', 0),
                        'sharpe': backtest_results['backtest'].get('sharpe_ratio', 0),
                        'trades': backtest_results['backtest'].get('num_trades', 0)
                    })
                else:
                    results.append({
                        'params': params,
                        'fitness': self.severe_penalty,
                        'return': 0, 'sharpe': 0, 'trades': 0
                    })

            except Exception as e:
                print(f"Error in combination {i + 1}: {e}")
                results.append({
                    'params': params,
                    'fitness': self.severe_penalty,
                    'return': 0, 'sharpe': 0, 'trades': 0
                })

        valid_results = [r for r in results if r['fitness'] != self.severe_penalty]
        valid_results.sort(key=lambda x: x['fitness'], reverse=True)

        return {
            'optimizer_type': self.get_optimization_type().value,
            'total_combinations': len(results),
            'valid_results': len(valid_results),
            'best_result': valid_results[0] if valid_results else None,
            'all_results': valid_results
        }