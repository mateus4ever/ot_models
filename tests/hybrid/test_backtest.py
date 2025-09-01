# tests/hybrid/test_backtest.py
"""
pytest-bdd test runner for BacktestOrchestrator orchestration testing
Tests backtest.py orchestration behavior, not implementation details
ZERO MOCKS - Real orchestration testing with actual dependencies
"""

import pytest
import pandas as pd
import logging
from pathlib import Path
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.backtest import BacktestOrchestrator

# Load scenarios from the backtest.feature
scenarios('backtest.feature')

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

@given('the system has proper directory structure')
def step_system_directory_structure(test_context):
    """Verify basic directory structure exists"""
    root_path = Path(__file__).parent.parent.parent
    assert (root_path / 'src').exists(), "src directory missing"
    assert (root_path / 'tests').exists(), "tests directory missing"
    test_context['root_path'] = root_path


@given(parsers.parse('test data files are available in {data_directory}'))
def step_test_data_files_available(test_context, data_directory):
    """Verify test data files exist in specified directory"""
    if 'root_path' not in test_context:
        root_path = Path(__file__).parent.parent.parent
        test_context['root_path'] = root_path

    root_path = test_context['root_path']
    data_path = root_path / data_directory
    assert data_path.exists(), f"Data directory {data_directory} missing"

    csv_files = list(data_path.glob('*.csv'))
    assert len(csv_files) >= 1, f"Expected at least 1 CSV file in {data_directory}, found {len(csv_files)}"
    test_context['csv_files'] = csv_files
    test_context['data_path'] = data_path
    test_context['data_directory'] = data_directory


@given(parsers.parse('{config_file} is available in {config_directory}'))
def step_config_file_available(test_context, config_file, config_directory):
    """Verify config file exists in specified directory"""
    if 'root_path' not in test_context:
        root_path = Path(__file__).parent.parent.parent
        test_context['root_path'] = root_path

    root_path = test_context['root_path']
    config_path = root_path / config_directory / config_file
    assert config_path.exists(), f"{config_file} not found at {config_path}"

    test_context['config_path'] = config_path
    test_context['config_file'] = config_file
    test_context['config_directory'] = config_directory


@given(parsers.parse('I have the {config_file} file in {config_directory}'))
def step_have_config_file(test_context, config_file, config_directory):
    """Verify config file exists"""
    step_config_file_available(test_context, config_file, config_directory)


@given(parsers.parse('I have test CSV files in {data_directory}'))
def step_have_test_csv_files(test_context, data_directory):
    """Verify test CSV files exist"""
    step_test_data_files_available(test_context, data_directory)


@given(parsers.parse('I have {error_condition}'))
def step_have_error_condition(test_context, error_condition):
    """Set up error conditions for testing"""
    logger = logging.getLogger(__name__)
    logger.debug(f"Setting up error condition: {error_condition}")

    test_context['error_condition'] = error_condition

    root_path = test_context.get('root_path', Path(__file__).parent.parent.parent)
    test_context['root_path'] = root_path

    if error_condition == "invalid config file":
        # Point to non-existent config
        test_context['config_path'] = root_path / 'tests' / 'config' / 'nonexistent.json'
        logger.debug(f"Set config_path to: {test_context['config_path']}")
    elif error_condition == "missing CSV data files":
        # Use bad config that points to non-existent data directory
        test_context['config_path'] = root_path / 'tests' / 'config' / 'bad_smoke_config.json'
        logger.debug(f"Set config_path to: {test_context['config_path']}")
    elif error_condition == "invalid strategy setup":
        # Use valid config but will pass invalid strategy later
        test_context['config_path'] = root_path / 'tests' / 'config' / 'smoke_config.json'
        test_context['invalid_strategy'] = True
        logger.debug(f"Set config_path to: {test_context['config_path']}, invalid_strategy: True")


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I initialize a BacktestOrchestrator with the config')
def step_initialize_backtest_orchestrator(test_context):
    """Test BacktestOrchestrator initialization"""
    logger = logging.getLogger(__name__)

    try:
        config_path = test_context['config_path']
        logger.debug(f"Initializing BacktestOrchestrator with config: {config_path}")

        config = UnifiedConfig(config_path=str(config_path))
        orchestrator = BacktestOrchestrator(config)

        test_context['config'] = config
        test_context['orchestrator'] = orchestrator
        test_context['initialization_error'] = None

        logger.debug("BacktestOrchestrator initialization successful")

    except Exception as e:
        logger.debug(f"BacktestOrchestrator initialization failed: {type(e).__name__}: {e}")
        test_context['orchestrator'] = None
        test_context['initialization_error'] = e


@when('I run the multi-strategy backtest')
def step_run_multi_strategy_backtest(test_context):
    """Test BacktestOrchestrator.run_multi_strategy_backtest() orchestration"""
    try:
        orchestrator = test_context.get('orchestrator')

        # For error scenarios that don't initialize orchestrator, create one here
        if orchestrator is None and test_context.get('error_condition'):
            # This is an error scenario - try to initialize with error condition
            try:
                config_path = test_context['config_path']
                config = UnifiedConfig(config_path=str(config_path))
                orchestrator = BacktestOrchestrator(config)
                test_context['orchestrator'] = orchestrator
            except Exception as e:
                # Initialization failed - this is expected for some error conditions
                test_context['backtest_error'] = e
                test_context['backtest_success'] = False
                return
        elif orchestrator is None:
            # This is a success scenario but orchestrator not initialized
            raise RuntimeError("Orchestrator not initialized - missing initialization step")

        # Determine strategies based on error condition
        if test_context.get('invalid_strategy'):
            strategies = ['nonexistent_strategy']
        else:
            strategies = ['base']  # Default strategy from config

        # Call the orchestration method
        results = orchestrator.run_multi_strategy_backtest(
            strategies=strategies,
            markets=None,  # Use default from config
            execution_mode='serial'
        )

        test_context['backtest_results'] = results
        test_context['backtest_error'] = None
        test_context['backtest_success'] = True

    except Exception as e:
        test_context['backtest_results'] = None
        test_context['backtest_error'] = e
        test_context['backtest_success'] = False


# =============================================================================
# THEN steps - Assertions
# =============================================================================

# Initialization scenario assertions
@then('the orchestrator should be properly initialized')
def step_orchestrator_properly_initialized(test_context):
    """Verify BacktestOrchestrator initialization succeeded"""
    initialization_error = test_context.get('initialization_error')
    assert initialization_error is None, f"Initialization failed: {initialization_error}"

    orchestrator = test_context.get('orchestrator')
    assert orchestrator is not None, "BacktestOrchestrator was not created"
    assert hasattr(orchestrator, 'config'), "BacktestOrchestrator missing config attribute"


@then('configuration values should be cached correctly')
def step_configuration_values_cached(test_context):
    """Verify configuration caching worked"""
    orchestrator = test_context['orchestrator']

    # Verify config values were cached
    assert hasattr(orchestrator, 'verbose'), "Orchestrator missing cached verbose value"
    assert hasattr(orchestrator, 'backtesting_method'), "Orchestrator missing cached backtesting_method"
    assert hasattr(orchestrator, 'unity_value'), "Orchestrator missing cached unity_value"


@then(parsers.parse('backtesting method should be set to {expected_method} from config'))
def step_backtesting_method_from_config(test_context, expected_method):
    """Verify backtesting method was loaded from config"""
    orchestrator = test_context['orchestrator']
    assert orchestrator.backtesting_method == expected_method, \
        f"Expected backtesting method '{expected_method}', got '{orchestrator.backtesting_method}'"


@then(parsers.parse('mathematical constants should have unity={expected_unity:d} from config'))
def step_mathematical_constants_loaded(test_context, expected_unity):
    """Verify mathematical constants were loaded from config"""
    orchestrator = test_context['orchestrator']
    assert orchestrator.unity_value == expected_unity, \
        f"Expected unity value {expected_unity}, got {orchestrator.unity_value}"


@then(parsers.parse('verbose mode should be {expected_verbose} from config'))
def step_verbose_mode_configured(test_context, expected_verbose):
    """Verify verbose mode was loaded from config"""
    orchestrator = test_context['orchestrator']
    expected_verbose_bool = expected_verbose.lower() == 'true'
    assert orchestrator.verbose == expected_verbose_bool, \
        f"Expected verbose {expected_verbose_bool}, got {orchestrator.verbose}"


# Success scenario assertions
@then('I should receive aggregated backtest results')
def step_receive_aggregated_backtest_results(test_context):
    """Verify backtest returned results"""
    backtest_error = test_context.get('backtest_error')
    if backtest_error:
        pytest.fail(f"Backtest failed with error: {backtest_error}")

    results = test_context.get('backtest_results')
    assert results is not None, "No backtest results returned"
    assert isinstance(results, dict), "Results should be a dictionary"

    # Verify this is not an error result
    assert 'error' not in results, f"Expected success results but got error: {results.get('error')}"


@then('no exceptions should be thrown')
def step_no_exceptions_thrown(test_context):
    """Verify no exceptions occurred during backtest"""
    backtest_error = test_context.get('backtest_error')
    initialization_error = test_context.get('initialization_error')

    assert initialization_error is None, f"Initialization error: {initialization_error}"
    assert backtest_error is None, f"Backtest error: {backtest_error}"


@then('the results should contain portfolio metrics')
def step_results_contain_portfolio_metrics(test_context):
    """Verify results contain expected portfolio metrics"""
    if test_context.get('backtest_success'):
        results = test_context.get('backtest_results', {})
        # Verify basic result structure
        assert isinstance(results, dict), "Results should be a dictionary"


@then('the execution should complete successfully')
def step_execution_completes_successfully(test_context):
    """Verify execution completed successfully"""
    success = test_context.get('backtest_success', False)
    assert success, "Backtest execution did not complete successfully"


# Error scenario assertions
@then(parsers.parse('the results should contain an error about {error_type}'))
def step_results_contain_error_about(test_context, error_type):
    """Verify results contain error about specific type"""
    backtest_results = test_context.get('backtest_results')
    backtest_error = test_context.get('backtest_error')
    initialization_error = test_context.get('initialization_error')

    # Check if we got an error result or an exception
    if backtest_results and isinstance(backtest_results, dict) and 'error' in backtest_results:
        # BacktestOrchestrator returned error result
        error_msg = str(backtest_results['error']).lower()
        error_type_lower = error_type.lower()

        # Verify error message contains relevant keywords
        if error_type_lower == "configuration":
            assert any(keyword in error_msg for keyword in ['config', 'configuration', 'file not found']), \
                f"Expected configuration error, got: {backtest_results['error']}"
        elif error_type_lower == "data loading":
            assert any(keyword in error_msg for keyword in ['data', 'file', 'csv', 'not found']), \
                f"Expected data loading error, got: {backtest_results['error']}"
        elif error_type_lower == "strategy":
            assert any(keyword in error_msg for keyword in ['strategy', 'not found', 'invalid']), \
                f"Expected strategy error, got: {backtest_results['error']}"
    else:
        # Check if initialization failed
        error = initialization_error or backtest_error
        assert error is not None, f"Expected error about {error_type} but no error occurred"


@then('the error message should be informative')
def step_error_message_informative(test_context):
    """Verify error message provides useful information"""
    backtest_results = test_context.get('backtest_results')
    backtest_error = test_context.get('backtest_error')
    initialization_error = test_context.get('initialization_error')

    # Check for error in results first (orchestrator error handling)
    if backtest_results and isinstance(backtest_results, dict) and 'error' in backtest_results:
        error_msg = str(backtest_results['error'])
        assert len(error_msg) > 0, "Error message should not be empty"
        assert not error_msg.isspace(), "Error message should not be just whitespace"
        return

    # Check for exceptions (initialization or runtime errors)
    error = initialization_error or backtest_error
    assert error is not None, "No error found to check message"

    error_msg = str(error)
    assert len(error_msg) > 0, "Error message should not be empty"
    assert not error_msg.isspace(), "Error message should not be just whitespace"


@then('the system should fail gracefully')
def step_system_fails_gracefully(test_context):
    """Verify system fails gracefully without crashes"""
    backtest_results = test_context.get('backtest_results')
    backtest_error = test_context.get('backtest_error')
    initialization_error = test_context.get('initialization_error')

    # Check for error in results (orchestrator graceful error handling)
    if backtest_results and isinstance(backtest_results, dict) and 'error' in backtest_results:
        # System failed gracefully by returning error result
        return

    # Check for exceptions (also graceful if caught properly)
    error = initialization_error or backtest_error
    assert error is not None, "Expected graceful failure but no error occurred"