# tests/hybrid/test_strategy_factory.py

from pathlib import Path

import pytest
import logging
# Only when you need project root path:

from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.optimization import OptimizerType
from src.hybrid.optimization.optimization_service import OptimizationService
from src.hybrid.optimization.optimizer_factory import OptimizerFactory
# Import the system under test
from src.hybrid.strategies.strategy_factory import StrategyFactory, StrategyFactoryCallable

# Load scenarios from the strategy_factory.feature
scenarios('optimization_service.feature')

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
    test_root = Path(__file__).parent.parent.parent

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    unified_config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['unified_config'] = unified_config
    test_context['test_root'] = test_root

@given('an OptimizationService')
def given_optimization_service(test_context):
    config = test_context['unified_config']
    test_context['service'] = OptimizationService(config)


@given('a strategy factory')
def given_strategy_factory(test_context):
    config = test_context['unified_config']
    test_root = Path(__file__).parent.parent.parent

    factory = StrategyFactoryCallable(
        config, 'base', 10, test_root)
    test_context['strategy_factory'] = factory

# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when(parsers.parse('I run optimization with optimizer type {optimizer_type}'))
def when_run_optimization(test_context, optimizer_type):
    service = test_context['service']
    strategy_factory = test_context['strategy_factory']

    opt_type = OptimizerType[optimizer_type]

    test_context['results'] = service.run_optimization(
        strategy_factory=strategy_factory,
        optimizer_type=opt_type,
        n_combinations=5,
        n_workers=2
    )

# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('optimization results should be returned')
def then_results_returned(test_context):
    results = test_context['results']
    assert results is not None
    assert isinstance(results, dict)


@then('results should contain best_result')
def then_results_contain_best(test_context):
    results = test_context['results']
    assert 'best_result' in results


@then('results should contain all_results')
def then_results_contain_all(test_context):
    results = test_context['results']
    assert 'all_results' in results