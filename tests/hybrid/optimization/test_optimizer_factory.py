# tests/hybrid/test_strategy_factory.py

from pathlib import Path

import pytest
import logging
# Only when you need project root path:

from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.optimization import OptimizerType
from src.hybrid.optimization.optimizer_factory import OptimizerFactory
# Import the system under test
from src.hybrid.strategies.strategy_factory import StrategyFactory

# Load scenarios from the strategy_factory.feature
scenarios('optimizer_factory.feature')

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

@given(parsers.parse('optimizer type "{optimizer_type}"'))
def step_given_optimizer_type(test_context, optimizer_type):
    """Store optimizer type for creation"""
    test_context['optimizer_type'] = optimizer_type

@given(parsers.parse('an invalid optimizer type "{invalid_type}"'))
def step_given_invalid_optimizer_type(test_context, invalid_type):
    """Store invalid optimizer type"""
    test_context['invalid_optimizer_type'] = invalid_type
# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I create an optimizer using the factory')
def step_create_optimizer_using_factory(test_context):
    """Create optimizer instance using factory"""
    config = test_context['unified_config']
    optimizer_type = test_context['optimizer_type']
    optimizer_type_enum = OptimizerType[optimizer_type]

    initial_capital = config.config['testing']['initial_capital']
    test_root = test_context['test_root']

    factory = OptimizerFactory()

    strategy_factory = StrategyFactory()
    strategy = strategy_factory.create_strategy_isolated(
        'base', config, initial_capital,test_root)

    try:
        optimizer = factory.create_optimizer(optimizer_type_enum, config, strategy)
        test_context['optimizer'] = optimizer
        test_context['creation_error'] = None
    except Exception as e:
        test_context['optimizer'] = None
        test_context['creation_error'] = e


@when('I attempt to create an optimizer')
def step_attempt_create_optimizer(test_context):
    """Attempt to create optimizer with invalid type"""
    config = test_context['unified_config']
    invalid_type = test_context['invalid_optimizer_type']
    initial_capital = config.config['testing']['initial_capital']
    test_root = test_context['test_root']

    factory = OptimizerFactory()
    strategy_factory = StrategyFactory()
    strategy = strategy_factory.create_strategy_isolated(
        'base', config, initial_capital,test_root)

    try:
        optimizer = factory.create_optimizer(invalid_type, config, strategy)
        test_context['optimizer'] = optimizer
        test_context['creation_error'] = None
    except Exception as e:
        test_context['optimizer'] = None
        test_context['creation_error'] = e


@when('I request available optimizer types')
def step_request_available_optimizer_types(test_context):
    """Request list of available optimizer types"""
    factory = OptimizerFactory()

    try:
        available_types = factory.get_available_optimizers()
        test_context['available_optimizers'] = available_types
        test_context['list_error'] = None
    except Exception as e:
        test_context['available_optimizers'] = None
        test_context['list_error'] = e


@when(parsers.parse('I try to create an optimizer with {invalid_input}'))
def step_try_create_optimizer_with_invalid_input(test_context, invalid_input):
    """Try to create optimizer with invalid input"""
    config = test_context['unified_config']

    # Convert string representation to actual values
    if invalid_input == 'None':
        optimizer_type = None
    elif invalid_input == '""':
        optimizer_type = ""
    elif invalid_input == '"   "':
        optimizer_type = "   "
    else:
        optimizer_type = invalid_input

    initial_capital = config.config['testing']['initial_capital']
    test_root = test_context['test_root']
    factory = OptimizerFactory()
    strategy_factory = StrategyFactory()
    strategy = strategy_factory.create_strategy_isolated(
        'base', config,initial_capital,test_root)

    try:
        optimizer = factory.create_optimizer(optimizer_type, config, strategy)
        test_context['optimizer'] = optimizer
        test_context['creation_error'] = None
    except Exception as e:
        test_context['optimizer'] = None
        test_context['creation_error'] = e

# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('an optimizer instance should be returned')
def step_optimizer_instance_returned(test_context):
    """Verify optimizer instance was created"""
    error = test_context.get('creation_error')
    assert error is None, f"Optimizer creation failed: {error}"

    optimizer = test_context.get('optimizer')
    assert optimizer is not None, "Optimizer instance should not be None"


@then('the optimizer should implement IOptimizer interface')
def step_optimizer_implements_interface(test_context):
    """Verify optimizer implements IOptimizer interface"""
    optimizer = test_context['optimizer']

    required_methods = ['run_optimization', 'get_optimization_type', 'get_description']

    for method in required_methods:
        assert hasattr(optimizer, method), f"Optimizer missing method: {method}"
        assert callable(getattr(optimizer, method)), f"Optimizer {method} not callable"


@then(parsers.parse('the optimizer type should be "{expected_type}"'))
def step_optimizer_type_matches(test_context, expected_type):
    """Verify optimizer type matches expected"""
    optimizer = test_context['optimizer']
    actual_type = optimizer.get_optimization_type()

    assert actual_type.name == expected_type, \
        f"Expected type {expected_type}, got {actual_type.name}"


@then('a ValueError should be raised')
def step_value_error_raised(test_context):
    """Verify ValueError was raised"""
    error = test_context.get('creation_error')
    assert error is not None, "Expected ValueError but no error occurred"
    assert isinstance(error, ValueError), f"Expected ValueError, got {type(error)}"


@then(parsers.parse('error message should include "{expected_text}"'))
def step_error_message_includes(test_context, expected_text):
    """Verify error message contains expected text"""
    error = test_context.get('creation_error')
    assert error is not None, "Expected error but none occurred"

    error_message = str(error)
    assert expected_text in error_message, \
        f"Expected '{expected_text}' in error message: {error_message}"


@then(parsers.parse('the list should contain "{optimizer_type}"'))
def step_list_contains_optimizer_type(test_context, optimizer_type):
    """Verify list contains expected optimizer type"""
    error = test_context.get('list_error')
    assert error is None, f"Failed to get available optimizers: {error}"

    available = test_context.get('available_optimizers')
    assert available is not None, "Available optimizers list should not be None"

    # Convert string to enum for comparison
    optimizer_type_enum = OptimizerType[optimizer_type]

    assert optimizer_type_enum in available, \
        f"Expected '{optimizer_type_enum}' in available optimizers: {available}"


@then('the error message should be informative')
def step_error_message_informative(test_context):
    """Verify error message is informative"""
    error = test_context.get('creation_error')
    assert error is not None, "Expected error but none occurred"

    error_message = str(error)
    assert len(error_message) > 0, "Error message should not be empty"
    assert not error_message.isspace(), "Error message should not be just whitespace"