# tests/hybrid/test_strategy_factory.py
"""
pytest-bdd test runner for StrategyFactory creation and error handling
Tests factory's core responsibility: create strategies successfully and handle errors properly
ZERO MOCKS - Real strategy factory with actual strategy creation
"""
import logging
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.optimization import OptimizerType
from src.hybrid.optimization.optimization_coordinator import OptimizationCoordinator
# Import the system under test
from src.hybrid.strategies.strategy_factory import StrategyFactoryCallable

# Only when you need project root path:

# Load scenarios from the strategy_factory.feature
scenarios('optimization_coordinator.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# Test fixtures and shared state
@pytest.fixture
def test_context(request):
    """A per-scenario context dict with scenario name pre-attached."""
    ctx = {}
    scenario = getattr(request.node._obj, "__scenario__", None)
    if scenario:
        ctx["scenario_name"] = scenario.name
    else:
        ctx["scenario_name"] = request.node.name
    return ctx


# =============================================================================
# GIVEN steps - Setup
# =============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    unified_config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['unified_config'] = unified_config

@given('I create an OptimizationCoordinator')
def step_create_optimization_coordinator(test_context):
    """Create OptimizationCoordinator instance"""
    config = test_context['unified_config']
    coordinator = OptimizationCoordinator(config)
    test_context['coordinator'] = coordinator


@given(parsers.parse('data source is set to {data_path}'))
def step_set_data_source(test_context, data_path):
    """Set data source path in config"""

    # Set project root for resolving relative paths
    test_root = Path(__file__).parent.parent.parent
    test_context['test_root'] = test_root


@given(parsers.parse('initial capital is set to {capital}'))
def step_set_initial_capital(test_context, capital):
    """Set initial capital for optimization"""
    test_context['initial_capital'] = float(capital)

@given('a strategy factory function')
def step_create_strategy_factory_function(test_context):
    config = test_context['unified_config']
    initial_capital = test_context['initial_capital']
    test_root = test_context['test_root']

    factory = StrategyFactoryCallable(config, 'base', initial_capital,test_root)
    test_context['strategy_factory'] = factory

# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when(parsers.parse('I run optimization with {optimizer_type} optimizer'))
def step_run_optimization_with_params(test_context, optimizer_type, datatable):
    """Run optimization with parameters from table"""

    coordinator = test_context['coordinator']
    strategy_factory = test_context['strategy_factory']
    optimizer_enum = OptimizerType[optimizer_type]

    # Extract parameters from table
    params = {}
    for row in datatable[1:]:  # Skip header
        param_name = row[0]
        param_value = int(row[1])
        params[param_name] = param_value

    # Run optimization
    results = coordinator.optimize(
        strategy_factory=strategy_factory,
        optimizer_type=optimizer_enum,
        n_combinations=params.get('n_combinations'),
        n_workers=params.get('n_workers')
    )

    test_context['optimization_results'] = results
    test_context['n_combinations'] = params.get('n_combinations')

@when(parsers.parse('n_combinations is {n_combinations:d}'))
def step_set_n_combinations(test_context, n_combinations):
    """Set number of combinations"""
    test_context['n_combinations'] = n_combinations


@when(parsers.parse('n_workers is {n_workers}'))
def step_set_n_workers(test_context, n_workers):
    """Set number of workers"""
    test_context['n_workers'] = int(n_workers)

    # Now run the optimization with all parameters
    coordinator = test_context['coordinator']
    strategy_factory = test_context['strategy_factory']
    optimizer_type = test_context['optimizer_type']
    n_combinations = test_context['n_combinations']

    results = coordinator.optimize(
        strategy_factory=strategy_factory,
        optimizer_type=optimizer_type,
        n_combinations=n_combinations,
        n_workers=n_workers
    )

    test_context['optimization_results'] = results



# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('coordinator should be initialized')
def step_coordinator_initialized(test_context):
    """Verify coordinator is initialized"""
    coordinator = test_context['coordinator']
    assert coordinator is not None, "Coordinator should be created"
    assert hasattr(coordinator, 'config'), "Coordinator should have config"
    assert hasattr(coordinator, 'all_evaluations'), "Coordinator should have evaluations list"
    assert len(coordinator.all_evaluations) == 0, "Evaluations list should be empty"

@then('checkpoint settings should be loaded from config')
def step_checkpoint_settings_loaded(test_context):
    """Verify checkpoint settings loaded from config"""
    coordinator = test_context['coordinator']

    assert hasattr(coordinator, 'checkpoint_interval'), "Should have checkpoint_interval"
    assert hasattr(coordinator, 'checkpoint_time_interval'), "Should have checkpoint_time_interval"
    assert isinstance(coordinator.checkpoint_interval, int), "checkpoint_interval should be int"
    assert isinstance(coordinator.checkpoint_time_interval, int), "checkpoint_time_interval should be int"


@then('parameter combinations should be generated')
def step_verify_parameters_generated(test_context):
    """Verify parameter combinations were generated"""
    results = test_context['optimization_results']
    assert 'total_combinations' in results, "Results should include total_combinations"
    assert results['total_combinations'] > 0, "Should have generated combinations"


@then('work should be distributed to workers')
def step_verify_work_distributed(test_context):
    """Verify work was distributed (all combinations processed)"""
    results = test_context['optimization_results']
    n_combinations = test_context['n_combinations']

    assert results['total_combinations'] == n_combinations, \
        f"Should have processed {n_combinations} combinations"


@then('results should be aggregated')
def step_verify_results_aggregated(test_context):
    """Verify results were aggregated"""
    results = test_context['optimization_results']

    assert 'valid_results' in results, "Results should include valid_results count"
    assert 'failed_results' in results, "Results should include failed_results count"
    assert 'all_results' in results, "Results should include all_results list"
    assert 'duration_seconds' in results, "Results should include duration"


@then('best result should be identified')
def step_verify_best_result(test_context):
    """Verify best result was identified and is actually the best"""
    results = test_context['optimization_results']

    assert 'best_result' in results, "Results should include best_result"

    if results['valid_results'] > 0:

        best = results['best_result']
        assert best is not None, "Should have best result when valid results exist"
        assert 'params' in best, "Best result should have params"
        assert 'fitness' in best, "Best result should have fitness"
        assert 'metrics' in best, "Best result should have metrics"

        # Verify it's actually the best
        all_results = results['all_results']
        best_fitness = best['fitness']

        # Check that no other result has better fitness
        for result in all_results:
            if result['success']:
                assert result['fitness'] <= best_fitness, \
                    f"Found result with better fitness ({result['fitness']}) than best ({best_fitness})"

        # Verify parameters are different from defaults
        assert best['params'], "Best result should have non-empty params"

        # Verify metrics are realistic
        metrics = best['metrics']
        assert 'total_trades' in metrics, "Should have total_trades"
        assert 'total_pnl' in metrics, "Should have total_pnl"
        assert 'win_rate' in metrics, "Should have win_rate"

        # Verify diversity - not all results identical
        unique_trade_counts = set()
        unique_pnls = set()
        for result in all_results:
            if result['success']:
                unique_trade_counts.add(result['metrics']['total_trades'])
                unique_pnls.add(float(result['metrics']['total_pnl']))

        assert len(unique_trade_counts) > 1, \
            f"All results have identical trade counts: {unique_trade_counts} - parameters not being applied!"
        assert len(unique_pnls) > 1, \
            f"All results have identical P&L: {unique_pnls} - parameters not being applied!"