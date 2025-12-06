# tests/hybrid/test_strategy_factory.py
"""
pytest-bdd test runner for StrategyFactory creation and error handling
Tests factory's core responsibility: create strategies successfully and handle errors properly
ZERO MOCKS - Real strategy factory with actual strategy creation
"""
from pathlib import Path

import pytest
import logging
# Only when you need project root path:

from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data import DataManager
from src.hybrid.money_management import MoneyManager
from src.hybrid.optimization import OptimizerType
from src.hybrid.optimization.optimization_coordinator import OptimizationCoordinator
from src.hybrid.optimization.optimizer_factory import OptimizerFactory
from src.hybrid.positions.position_orchestrator import PositionOrchestrator
from src.hybrid.positions.trade_history import TradeHistory
from src.hybrid.signals import SignalFactory
# Import the system under test
from src.hybrid.strategies.strategy_factory import StrategyFactory

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


@given(parsers.parse('data_management config points to {data_path}'))
def set_data_management_source(test_context, data_path):
    """Set data source in data_management config"""
    config = test_context['config']


    update_payload = {
        'data_loading': {
            'directory_path': data_path
        }
    }

    config.update_config(update_payload)

@given('a strategy instance')
def step_create_strategy(test_context):
    """Create fully-wired strategy"""
    config = test_context['unified_config']

    # Create dependencies
    data_manager = DataManager(config)
    money_manager = MoneyManager(config)
    position_orchestrator = PositionOrchestrator(config)

    # Create strategy
    strategy_factory = StrategyFactory()
    strategy = strategy_factory.create_strategy('base', config)

    # Create and wire signals
    signal_factory = SignalFactory(config)
    entry_signal = signal_factory.create_signal(config.get_section('strategy', {}).get('entry_signal'), config)
    exit_signal = signal_factory.create_signal(config.get_section('strategy', {}).get('exit_signal'), config)

    strategy.add_entry_signal(entry_signal)
    strategy.add_exit_signal(exit_signal)

    # Wire other dependencies
    strategy.set_data_manager(data_manager)
    strategy.set_money_manager(money_manager)
    strategy.set_position_orchestrator(position_orchestrator)

    test_context['strategy'] = strategy
    #todo: the optimize parameters must be obtained from the strategy.

# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when(parsers.parse('I run optimization with {optimizer_type} optimizer'))
def step_run_optimization(test_context, optimizer_type):
    """Run optimization with specified optimizer type"""
    from src.hybrid.optimization import OptimizerType

    coordinator = test_context['coordinator']
    strategy_factory = test_context['strategy_factory']

    # Get optimizer type enum
    optimizer_enum = OptimizerType[optimizer_type]

    # Store for later steps
    test_context['optimizer_type'] = optimizer_enum
    test_context['optimization_started'] = True


@when(parsers.parse('n_combinations is {n_combinations:d}'))
def step_set_n_combinations(test_context, n_combinations):
    """Set number of combinations"""
    test_context['n_combinations'] = n_combinations


@when(parsers.parse('n_workers is {n_workers:d}'))
def step_set_n_workers(test_context, n_workers):
    """Set number of workers"""
    test_context['n_workers'] = n_workers

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
    """Verify best result was identified"""
    results = test_context['optimization_results']

    assert 'best_result' in results, "Results should include best_result"

    if results['valid_results'] > 0:
        best = results['best_result']
        assert best is not None, "Should have best result when valid results exist"
        assert 'params' in best, "Best result should have params"
        assert 'fitness' in best, "Best result should have fitness"
        assert 'metrics' in best, "Best result should have metrics"