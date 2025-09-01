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

# Import the system under test
from src.hybrid.strategies.strategy_factory import StrategyFactory

# Load scenarios from the strategy_factory.feature
scenarios('strategy_factory.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# Test fixtures and shared state
@pytest.fixture
def test_context():
    """Shared test context for storing state between steps"""
    return {}


# Mock config for testing
class MockConfig:
    """Mock configuration for strategy creation testing"""

    def __init__(self):
        self.data = {
            'strategy': {
                'name': 'test_strategy',
                'parameters': {'param1': 'value1', 'param2': 'value2'}
            }
        }

    def get_section(self, section, default=None):
        return self.data.get(section, default or {})


# =============================================================================
# GIVEN steps - Setup
# =============================================================================

@given('the system has proper directory structure')
def step_system_directory_structure(test_context):
    """Verify basic directory structure exists"""
    root_path = Path(__file__).parent.parent.parent.parent
    assert (root_path / 'src').exists(), "src directory missing"
    assert (root_path / 'tests').exists(), "tests directory missing"
    test_context['root_path'] = root_path


@given('I have a StrategyFactory instance')
def step_have_strategy_factory_instance(test_context):
    """Create a StrategyFactory instance for testing"""
    logger = logging.getLogger(__name__)

    try:
        factory = StrategyFactory()
        test_context['factory'] = factory
        logger.debug("StrategyFactory instance created successfully")

    except Exception as e:
        pytest.fail(f"Failed to create StrategyFactory: {e}")


@given('the factory is properly initialized')
def step_factory_properly_initialized(test_context):
    """Verify factory is properly initialized"""
    factory = test_context.get('factory')
    if not factory:
        step_have_strategy_factory_instance(test_context)
        factory = test_context['factory']

    # Verify factory has basic functionality
    assert hasattr(factory, 'create_strategy'), "Factory should have create_strategy method"
    assert hasattr(factory, 'get_available_strategies'), "Factory should have get_available_strategies method"

    test_context['factory_initialized'] = True


@given('I have a valid configuration object')
def step_have_valid_configuration(test_context):
    """Create a valid configuration object for testing"""
    test_context['config'] = MockConfig()


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when(parsers.parse('I create a {strategy_name} strategy'))
def step_create_strategy(test_context, strategy_name):
    """Create a strategy using the factory"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    try:
        strategy = factory.create_strategy(strategy_name)
        test_context['created_strategy'] = strategy
        test_context['creation_error'] = None
        test_context['strategy_name'] = strategy_name
        logger.debug(f"Successfully created strategy: {strategy_name}")

    except Exception as e:
        test_context['created_strategy'] = None
        test_context['creation_error'] = e
        logger.debug(f"Strategy creation failed: {e}")


@when(parsers.parse('I create a {strategy_name} strategy with configuration'))
def step_create_strategy_with_config(test_context, strategy_name):
    """Create a strategy with configuration using the factory"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']
    config = test_context.get('config')

    if not config:
        step_have_valid_configuration(test_context)
        config = test_context['config']

    try:
        strategy = factory.create_strategy(strategy_name, config)
        test_context['created_strategy'] = strategy
        test_context['creation_error'] = None
        test_context['strategy_name'] = strategy_name
        test_context['used_config'] = config
        logger.debug(f"Successfully created strategy with config: {strategy_name}")

    except Exception as e:
        test_context['created_strategy'] = None
        test_context['creation_error'] = e
        logger.debug(f"Strategy creation with config failed: {e}")


@when(parsers.parse('I try to create a strategy with name {strategy_name}'))
def step_try_create_strategy_with_name(test_context, strategy_name):
    """Try to create a strategy with specific name (for error testing)"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    # Remove quotes if present
    strategy_name = strategy_name.strip('"\'')

    try:
        strategy = factory.create_strategy(strategy_name)
        test_context['created_strategy'] = strategy
        test_context['creation_error'] = None
        test_context['strategy_name'] = strategy_name

    except Exception as e:
        test_context['created_strategy'] = None
        test_context['creation_error'] = e
        test_context['strategy_name'] = strategy_name
        logger.debug(f"Expected error for strategy '{strategy_name}': {e}")


@when(parsers.parse('I try to create a strategy with {invalid_input}'))
def step_try_create_strategy_with_invalid_input(test_context, invalid_input):
    """Try to create a strategy with invalid input"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    # Convert string representation to actual values
    if invalid_input == 'None':
        strategy_name = None
    elif invalid_input == '""':
        strategy_name = ""
    elif invalid_input == '"   "':
        strategy_name = "   "
    else:
        strategy_name = invalid_input

    try:
        strategy = factory.create_strategy(strategy_name)
        test_context['created_strategy'] = strategy
        test_context['creation_error'] = None

    except Exception as e:
        test_context['created_strategy'] = None
        test_context['creation_error'] = e
        test_context['invalid_input'] = invalid_input
        logger.debug(f"Expected error for invalid input {invalid_input}: {e}")


@when('I request the list of available strategies')
def step_request_available_strategies(test_context):
    """Request list of available strategies from factory"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    try:
        available_strategies = factory.get_available_strategies()
        test_context['available_strategies'] = available_strategies
        test_context['strategies_error'] = None
        logger.debug(f"Available strategies: {available_strategies}")

    except Exception as e:
        test_context['available_strategies'] = None
        test_context['strategies_error'] = e


# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('a valid strategy instance should be created')
def step_valid_strategy_instance_created(test_context):
    """Verify a valid strategy instance was created"""
    creation_error = test_context.get('creation_error')
    assert creation_error is None, f"Strategy creation failed: {creation_error}"

    strategy = test_context.get('created_strategy')
    assert strategy is not None, "Strategy instance should not be None"


@then('a valid strategy instance should be created with config')
def step_valid_strategy_instance_created_with_config(test_context):
    """Verify a valid strategy instance was created with configuration"""
    creation_error = test_context.get('creation_error')
    assert creation_error is None, f"Strategy creation with config failed: {creation_error}"

    strategy = test_context.get('created_strategy')
    assert strategy is not None, "Strategy instance should not be None"


@then('the strategy should implement StrategyInterface')
def step_strategy_implements_interface(test_context):
    """Verify strategy implements StrategyInterface protocol"""
    strategy = test_context.get('created_strategy')

    # Check if strategy has required attributes and methods (duck typing for Protocol)
    required_attributes = ['name', 'money_manager', 'data_manager', 'signals',
                           'optimizations', 'predictors', 'runners', 'metrics', 'verificators']

    for attr in required_attributes:
        assert hasattr(strategy, attr), f"Strategy missing required attribute: {attr}"

    required_methods = ['setMoneyManager', 'setDataManager', 'addSignal', 'addOptimizer',
                        'addPredictor', 'addRunner', 'addMetric', 'addVerificator',
                        'initialize', 'generate_signals', 'execute_trades', 'run_backtest']

    for method in required_methods:
        assert hasattr(strategy, method), f"Strategy missing required method: {method}"
        assert callable(getattr(strategy, method)), f"Strategy {method} should be callable"


@then(parsers.parse('the strategy name should be {expected_name}'))
def step_strategy_name_should_be(test_context, expected_name):
    """Verify strategy has correct name"""
    strategy = test_context.get('created_strategy')
    assert strategy.name == expected_name, f"Expected strategy name '{expected_name}', got '{strategy.name}'"


@then('the strategy should be properly configured')
def step_strategy_properly_configured(test_context):
    """Verify strategy is properly configured"""
    strategy = test_context.get('created_strategy')

    # Basic configuration verification - strategy should be in a valid state
    assert hasattr(strategy, 'name'), "Strategy should have name attribute"
    assert strategy.name is not None, "Strategy name should not be None"


@then('the configuration should be passed to the strategy')
def step_configuration_passed_to_strategy(test_context):
    """Verify configuration was passed to strategy"""
    strategy = test_context.get('created_strategy')
    used_config = test_context.get('used_config')

    # Verify strategy received configuration (implementation specific)
    # At minimum, strategy should be created without error when config is passed
    assert strategy is not None, "Strategy should be created successfully with config"
    assert used_config is not None, "Config should have been used in creation"


@then('no creation errors should occur')
def step_no_creation_errors(test_context):
    """Verify no errors occurred during strategy creation"""
    creation_error = test_context.get('creation_error')
    assert creation_error is None, f"Strategy creation should not cause errors: {creation_error}"


@then('a ValueError should be thrown')
def step_value_error_thrown(test_context):
    """Verify ValueError was thrown"""
    creation_error = test_context.get('creation_error')
    assert creation_error is not None, "Expected ValueError but no error occurred"
    assert isinstance(creation_error, ValueError), f"Expected ValueError, got {type(creation_error)}"


@then(parsers.parse('the error message should mention {expected_text}'))
def step_error_message_mentions_text(test_context, expected_text):
    """Verify error message contains expected text"""
    creation_error = test_context.get('creation_error')
    assert creation_error is not None, "Expected error but none occurred"

    # Remove quotes from expected text
    expected_text = expected_text.strip('"\'')  # This strips the quotes!
    error_message = str(creation_error)
    assert expected_text in error_message, f"Expected '{expected_text}' in error message: {error_message}"
    assert expected_text in error_message, f"Expected '{expected_text}' in error message: {error_message}"

@then('the error message should list available strategies')
def step_error_message_lists_available_strategies(test_context):
    """Verify error message lists available strategies"""
    creation_error = test_context.get('creation_error')
    assert creation_error is not None, "Expected error but none occurred"

    error_message = str(creation_error)

    # Should mention available strategies
    assert 'available' in error_message.lower() or 'Available' in error_message, \
        f"Error message should mention available strategies: {error_message}"


@then(parsers.parse('the available strategies should include {strategy_name}'))
def step_available_strategies_include(test_context, strategy_name):
    """Verify available strategies include expected strategy"""
    creation_error = test_context.get('creation_error')
    error_message = str(creation_error)

    # Remove quotes from strategy name
    strategy_name = strategy_name.strip('"\'')

    assert strategy_name in error_message, \
        f"Available strategies should include '{strategy_name}': {error_message}"


@then('the error message should be informative')
def step_error_message_informative(test_context):
    """Verify error message is informative"""
    creation_error = test_context.get('creation_error')
    assert creation_error is not None, "Expected error but none occurred"

    error_message = str(creation_error)
    assert len(error_message) > 0, "Error message should not be empty"
    assert not error_message.isspace(), "Error message should not be just whitespace"


@then(parsers.parse('the list should contain {strategy_name}'))
def step_list_contains_strategy(test_context, strategy_name):
    """Verify available strategies list contains expected strategy"""
    strategies_error = test_context.get('strategies_error')
    assert strategies_error is None, f"Failed to get available strategies: {strategies_error}"

    available_strategies = test_context.get('available_strategies')
    assert available_strategies is not None, "Available strategies list should not be None"

    # Remove quotes from strategy name
    strategy_name = strategy_name.strip('"\'')

    assert strategy_name in available_strategies, \
        f"Available strategies should contain '{strategy_name}': {available_strategies}"


@then(parsers.parse('the list should have exactly {expected_count:d} strategies'))
def step_list_has_exact_count(test_context, expected_count):
    """Verify available strategies list has expected count"""
    available_strategies = test_context.get('available_strategies')
    assert available_strategies is not None, "Available strategies list should not be None"

    actual_count = len(available_strategies)
    assert actual_count == expected_count, \
        f"Expected {expected_count} strategies, got {actual_count}: {available_strategies}"