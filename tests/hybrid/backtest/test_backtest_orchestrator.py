# tests/hybrid/test_backtest.py
"""
pytest-bdd test runner for BacktestOrchestrator orchestration testing
Tests backtest.py orchestration behavior, not implementation details
ZERO MOCKS - Real orchestration testing with actual dependencies
"""

import logging
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.backtesting.backtest_orchestrator import BacktestOrchestrator
from src.hybrid.config.unified_config import UnifiedConfig

# Load scenarios from the backtest.feature
scenarios('backtest_orchestrator.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# Test fixtures and shared state
@pytest.fixture
def test_context(request):
    """
    A per-scenario context dict with scenario name pre-attached.
    """
    ctx = {}
    # pytest node is the test function generated for the scenario
    # ._obj is the underlying function object
    # __scenario__ is attached by pytest-bdd
    scenario = getattr(request.node._obj, "__scenario__", None)
    if scenario:
        ctx["scenario_name"] = scenario.name
    else:
        ctx["scenario_name"] = request.node.name  # fallback
    return ctx

# =============================================================================
# GIVEN steps - Setup
# =============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    root_path = Path(__file__).parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['config'] = config


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


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('BacktestOrchestrator is initialized')
def initialize_orchestrator(test_context):
    """Initialize BacktestOrchestrator with loaded config"""

    config = test_context['config']

    root_path = Path(__file__).parent.parent.parent
    orchestrator = BacktestOrchestrator(config,project_root=root_path)
    #TODO: this should be as parameter in feature file
    orchestrator.position_orchestrator.set_initial_capital(100000)

    test_context['orchestrator'] = orchestrator

@when('multi-strategy backtest is executed')
def execute_multi_strategy_backtest(test_context):
    """Execute multi-strategy backtest"""
    orchestrator = test_context['orchestrator']

    try:
        # Run with default strategy from config
        results = orchestrator.run_multi_strategy_backtest(
            strategies=['base'],  # Or get from config
            markets=None,  # Use all loaded markets
            execution_mode='serial'
        )

        test_context['backtest_results'] = results
        test_context['backtest_error'] = None

    except Exception as e:
        test_context['backtest_results'] = None
        test_context['backtest_error'] = e


# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('orchestrator should be ready')
def verify_orchestrator_ready(test_context):
    """Verify orchestrator initialized successfully"""
    orchestrator = test_context.get('orchestrator')

    assert orchestrator is not None, "Orchestrator was not created"
    assert hasattr(orchestrator, 'config'), "Orchestrator missing config"


@then('backtest results should be returned')
def verify_results_returned(test_context):
    """Verify backtest returned results"""
    error = test_context.get('backtest_error')
    assert error is None, f"Backtest failed with error: {error}"

    results = test_context.get('backtest_results')
    assert results is not None, "No results returned"
    assert isinstance(results, dict), "Results should be a dictionary"


@then('results should contain performance metrics')
def verify_performance_metrics(test_context):
    """Verify results contain expected metrics"""
    results = test_context['backtest_results']

    # Just verify basic structure without overtesting
    assert 'method' in results, "Results should contain backtesting method"
    assert 'results' in results, "Results should contain results list"
    assert 'execution_time' in results, "Results should contain execution time"
    # Don't test internal Result object structure - that can change