# tests/hybrid/test_strategy_interface.py
"""
pytest-bdd test runner for StrategyInterface protocol compliance
Tests interface signature compliance - no implementation execution
"""

import logging
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data import DataManager
from src.hybrid.money_management import MoneyManager
from src.hybrid.positions.position_orchestrator import PositionOrchestrator
from src.hybrid.strategies import StrategyFactory

# Add project root to Python path

# Load scenarios from the strategy_interface.feature
scenarios('strategy_interface.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


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

def _create_strategy_dependencies(config):
    """Helper to create strategy dependencies - reduces boilerplate"""
    data_manager = DataManager(config)
    money_manager = MoneyManager(config)
    position_orchestrator = PositionOrchestrator(config)
    return data_manager, money_manager, position_orchestrator



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


@given(parsers.parse('a strategy of type "{strategy_type}"'))
def step_create_strategy_by_type(test_context, strategy_type):
    """Create strategy instance by type name"""
    config = test_context['unified_config']
    dm, mm, po = _create_strategy_dependencies(config)

    # Create strategy using factory
    strategy_factory = StrategyFactory()
    strategy = strategy_factory.create_strategy_shared(strategy_type, config, dm, mm, po)

    test_context['strategy'] = strategy
    test_context['strategy_type'] = strategy_type


@then(parsers.parse('the strategy should have method "{method_name}"'))
def step_verify_strategy_has_method(test_context, method_name):
    """Verify strategy has specified method"""
    strategy = test_context['strategy']
    strategy_type = test_context['strategy_type']

    assert hasattr(strategy, method_name), \
        f"Strategy {strategy_type} missing method: {method_name}"

    # Verify it's actually callable
    method = getattr(strategy, method_name)
    assert callable(method), \
        f"Strategy {strategy_type} has {method_name} but it's not callable"


# =============================================================================
# WHEN steps - Actions
# =============================================================================
@when('I create mock dependencies')
def step_create_mock_dependencies(test_context):
    """Create simple mock objects for dependency injection"""
    test_context['mock_money_manager'] = object()
    test_context['mock_data_manager'] = object()
    test_context['mock_position_orchestrator'] = object()
    test_context['injection_error'] = None


@when('I inject MoneyManager into the strategy')
def step_inject_money_manager(test_context):
    """Inject MoneyManager dependency"""
    strategy = test_context['strategy']
    mock_mm = test_context['mock_money_manager']

    try:
        strategy.set_money_manager(mock_mm)
    except Exception as e:
        test_context['injection_error'] = e


@when('I inject DataManager into the strategy')
def step_inject_data_manager(test_context):
    """Inject DataManager dependency"""
    strategy = test_context['strategy']
    mock_dm = test_context['mock_data_manager']

    try:
        strategy.set_data_manager(mock_dm)
    except Exception as e:
        test_context['injection_error'] = e


@when('I inject PositionOrchestrator into the strategy')
def step_inject_position_orchestrator(test_context):
    """Inject PositionOrchestrator dependency"""
    strategy = test_context['strategy']
    mock_po = test_context['mock_position_orchestrator']

    try:
        strategy.set_position_orchestrator(mock_po)
    except Exception as e:
        test_context['injection_error'] = e

@when('I create mock components')
def step_create_mock_components(test_context):
    """Create simple mock objects for component addition"""
    test_context['mock_entry_signal'] = object()
    test_context['mock_exit_signal'] = object()
    test_context['mock_optimizer'] = object()
    test_context['mock_predictor'] = object()
    test_context['mock_metric'] = object()
    test_context['component_error'] = None


@when('I add entry signal to the strategy')
def step_add_entry_signal(test_context):
    """Add entry signal component"""
    strategy = test_context['strategy']
    mock_signal = test_context['mock_entry_signal']

    try:
        strategy.add_entry_signal(mock_signal)
    except Exception as e:
        test_context['component_error'] = e


@when('I add exit signal to the strategy')
def step_add_exit_signal(test_context):
    """Add exit signal component"""
    strategy = test_context['strategy']
    mock_signal = test_context['mock_exit_signal']

    try:
        strategy.add_exit_signal(mock_signal)
    except Exception as e:
        test_context['component_error'] = e


@when('I add optimizer to the strategy')
def step_add_optimizer(test_context):
    """Add optimizer component"""
    strategy = test_context['strategy']
    mock_optimizer = test_context['mock_optimizer']

    try:
        strategy.add_optimizer(mock_optimizer)
    except Exception as e:
        test_context['component_error'] = e


@when('I add predictor to the strategy')
def step_add_predictor(test_context):
    """Add predictor component"""
    strategy = test_context['strategy']
    mock_predictor = test_context['mock_predictor']

    try:
        strategy.add_predictor(mock_predictor)
    except Exception as e:
        test_context['component_error'] = e


@when('I add metric to the strategy')
def step_add_metric(test_context):
    """Add metric component"""
    strategy = test_context['strategy']
    mock_metric = test_context['mock_metric']

    try:
        strategy.add_metric(mock_metric)
    except Exception as e:
        test_context['component_error'] = e

# =============================================================================
# THEN steps - Assertions
# =============================================================================
@then(parsers.parse('the strategy should have method "{method_name}"'))
def step_verify_strategy_has_method(test_context, method_name):
    """Verify strategy has specified method"""
    strategy = test_context['strategy']
    strategy_type = test_context['strategy_type']

    assert hasattr(strategy, method_name), \
        f"Strategy {strategy_type} missing method: {method_name}"

    method = getattr(strategy, method_name)
    assert callable(method), \
        f"Strategy {strategy_type} has {method_name} but it's not callable"


@then(parsers.parse('it should have attribute "{attribute_name}"'))
def step_verify_strategy_has_attribute(test_context, attribute_name):
    """Verify strategy has specified attribute"""
    strategy = test_context['strategy']
    strategy_type = test_context['strategy_type']

    assert hasattr(strategy, attribute_name), \
        f"Strategy {strategy_type} missing attribute: {attribute_name}"

@then('no injection errors should occur')
def step_no_injection_errors(test_context):
    """Verify no errors occurred during dependency injection"""
    error = test_context.get('injection_error')
    assert error is None, f"Dependency injection failed: {error}"


@then('no component addition errors should occur')
def step_no_component_errors(test_context):
    """Verify no errors occurred during component addition"""
    error = test_context.get('component_error')
    assert error is None, f"Component addition failed: {error}"