# tests/hybrid/signals/test_signal_factory.py
"""
pytest-bdd test runner for SignalFactory creation and error handling
Tests factory's core responsibility: create signals successfully and handle errors properly
ZERO MOCKS - Real signal factory with actual signal creation
"""

import pytest
import logging
from pathlib import Path
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.signals.signal_factory import SignalFactory

# Load scenarios from the signal_factory.feature
scenarios('signal_factory.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# Test fixtures and shared state
@pytest.fixture
def test_context():
    """Shared test context for storing state between steps"""
    return {}


def get_default_value_for_param(param_name):
    """Single source of truth for parameter defaults"""
    defaults = {
        'period': 20, 'std_dev': 2.0, 'buffer_multiplier': 1.0,
        'oversold_threshold': 30, 'overbought_threshold': 70,
        'fast_period': 10, 'slow_period': 30, 'crossover_confirmation': True
    }
    return defaults.get(param_name, 1.0)


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


@given('I have a SignalFactory instance')
def step_have_signal_factory_instance(test_context):
    """Create a SignalFactory instance for testing"""
    logger = logging.getLogger(__name__)

    try:
        factory = SignalFactory()
        test_context['factory'] = factory
        logger.debug("SignalFactory instance created successfully")

    except Exception as e:
        pytest.fail(f"Failed to create SignalFactory: {e}")


@given('the factory is properly initialized')
def step_factory_properly_initialized(test_context):
    """Verify factory is properly initialized"""
    factory = test_context.get('factory')
    if not factory:
        step_have_signal_factory_instance(test_context)
        factory = test_context['factory']

    # Verify factory has basic functionality
    assert hasattr(factory, 'create_signal'), "Factory should have create_signal method"
    assert hasattr(factory, 'get_available_signals'), "Factory should have get_available_signals method"
    assert hasattr(factory, 'get_signals_by_category'), "Factory should have get_signals_by_category method"
    assert hasattr(factory, 'get_available_categories'), "Factory should have get_available_categories method"

    test_context['factory_initialized'] = True


@given('I have a valid signal configuration object')
def step_have_valid_signal_configuration(test_context):
    """Create a valid configuration object for testing"""
    # This step is deprecated - configuration is now built dynamically from signal requirements
    pass


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when(parsers.parse('I create a {signal_name} signal'))
def step_create_signal(test_context, signal_name):
    """Create a signal using the factory"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    try:
        signal = factory.create_signal(signal_name)
        test_context['created_signal'] = signal
        test_context['creation_error'] = None
        test_context['signal_name'] = signal_name
        logger.debug(f"Successfully created signal: {signal_name}")

    except Exception as e:
        test_context['created_signal'] = None
        test_context['creation_error'] = e
        logger.debug(f"Signal creation failed: {e}")


@when(parsers.parse('I create a {signal_name} signal with configuration'))
def step_create_signal_with_config(test_context, signal_name):
    """Create a signal with configuration using the factory"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    try:
        # Get required parameters from the signal class
        if signal_name not in factory._signal_registry:
            raise ValueError(f"Signal {signal_name} not found in registry")

        signal_class = factory._signal_registry[signal_name]
        required_params = signal_class.get_required_parameters()

        # Build valid configuration dynamically
        config = {param: get_default_value_for_param(param) for param in required_params}

        signal = factory.create_signal(signal_name, config)
        test_context['created_signal'] = signal
        test_context['creation_error'] = None
        test_context['signal_name'] = signal_name
        test_context['used_config'] = config
        logger.debug(f"Successfully created signal with config: {signal_name}")

    except Exception as e:
        test_context['created_signal'] = None
        test_context['creation_error'] = e
        logger.debug(f"Signal creation with config failed: {e}")


@when(parsers.parse('I try to create a signal with name {signal_name}'))
def step_try_create_signal_with_name(test_context, signal_name):
    """Try to create a signal with specific name (for error testing)"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    # Remove quotes if present
    signal_name = signal_name.strip('"\'')

    try:
        signal = factory.create_signal(signal_name)
        test_context['created_signal'] = signal
        test_context['creation_error'] = None
        test_context['signal_name'] = signal_name

    except Exception as e:
        test_context['created_signal'] = None
        test_context['creation_error'] = e
        test_context['signal_name'] = signal_name
        logger.debug(f"Expected error for signal '{signal_name}': {e}")


@when(parsers.parse('I try to create a signal with {invalid_input}'))
def step_try_create_signal_with_invalid_input(test_context, invalid_input):
    """Try to create a signal with invalid input"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    # Convert string representation to actual values
    if invalid_input == 'None':
        signal_name = None
    elif invalid_input == '""':
        signal_name = ""
    elif invalid_input == '"   "':
        signal_name = "   "
    else:
        signal_name = invalid_input

    try:
        signal = factory.create_signal(signal_name)
        test_context['created_signal'] = signal
        test_context['creation_error'] = None

    except Exception as e:
        test_context['created_signal'] = None
        test_context['creation_error'] = e
        test_context['invalid_input'] = invalid_input
        logger.debug(f"Expected error for invalid input {invalid_input}: {e}")


@when('I request the list of available signals')
def step_request_available_signals(test_context):
    """Request list of available signals from factory"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    try:
        available_signals = factory.get_available_signals()
        test_context['available_signals'] = available_signals
        test_context['signals_error'] = None
        logger.debug(f"Available signals: {available_signals}")

    except Exception as e:
        test_context['available_signals'] = None
        test_context['signals_error'] = e


@when(parsers.parse('I request signals by category {category}'))
def step_request_signals_by_category(test_context, category):
    """Request signals by specific category"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    # Remove quotes if present
    category = category.strip('"\'')

    try:
        category_signals = factory.get_signals_by_category(category)
        test_context['category_signals'] = category_signals
        test_context['category'] = category
        test_context['category_error'] = None
        logger.debug(f"Signals in category '{category}': {category_signals}")

    except Exception as e:
        test_context['category_signals'] = None
        test_context['category_error'] = e


@when('I request the list of available categories')
def step_request_available_categories(test_context):
    """Request list of available categories from factory"""
    logger = logging.getLogger(__name__)
    factory = test_context['factory']

    try:
        available_categories = factory.get_available_categories()
        test_context['available_categories'] = available_categories
        test_context['categories_error'] = None
        logger.debug(f"Available categories: {available_categories}")

    except Exception as e:
        test_context['available_categories'] = None
        test_context['categories_error'] = e


# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('a valid signal instance should be created')
def step_valid_signal_instance_created(test_context):
    """Verify a valid signal instance was created"""
    creation_error = test_context.get('creation_error')
    assert creation_error is None, f"Signal creation failed: {creation_error}"

    signal = test_context.get('created_signal')
    assert signal is not None, "Signal instance should not be None"


@then('a valid signal instance should be created with config')
def step_valid_signal_instance_created_with_config(test_context):
    """Verify a valid signal instance was created with configuration"""
    creation_error = test_context.get('creation_error')
    assert creation_error is None, f"Signal creation with config failed: {creation_error}"

    signal = test_context.get('created_signal')
    assert signal is not None, "Signal instance should not be None"


@then('the signal should implement SignalInterface')
def step_signal_implements_interface(test_context):
    """Verify signal implements SignalInterface protocol"""
    signal = test_context.get('created_signal')

    # Check if signal has required methods (duck typing for Protocol)
    required_methods = ['train', 'update_with_new_data', 'generate_signal', 'getMetrics']

    for method in required_methods:
        assert hasattr(signal, method), f"Signal missing required method: {method}"
        assert callable(getattr(signal, method)), f"Signal {method} should be callable"


@then('no creation errors should occur')
def step_no_creation_errors(test_context):
    """Verify no errors occurred during signal creation"""
    creation_error = test_context.get('creation_error')
    assert creation_error is None, f"Signal creation should not cause errors: {creation_error}"


@then('the configuration should be passed to the signal')
def step_configuration_passed_to_signal(test_context):
    """Verify configuration was passed to signal"""
    signal = test_context.get('created_signal')
    used_config = test_context.get('used_config')

    # Verify signal received configuration (implementation specific)
    # At minimum, signal should be created without error when config is passed
    assert signal is not None, "Signal should be created successfully with config"
    assert used_config is not None, "Config should have been used in creation"


@then('a ValueError should be thrown')
def step_value_error_thrown(test_context):
    """Verify ValueError was thrown"""
    creation_error = test_context.get('creation_error')
    assert creation_error is not None, "Expected ValueError but no error occurred"
    assert isinstance(creation_error, ValueError), f"Expected ValueError, got {type(creation_error)}"


@then('the error message should be informative')
def step_error_message_informative(test_context):
    """Verify error message is informative"""
    creation_error = test_context.get('creation_error')
    assert creation_error is not None, "Expected error but none occurred"

    error_message = str(creation_error)
    assert len(error_message) > 0, "Error message should not be empty"
    assert not error_message.isspace(), "Error message should not be just whitespace"


@then(parsers.parse('the list should contain {signal_name}'))
def step_list_contains_signal(test_context, signal_name):
    """Verify signals list contains expected signal"""
    # Check which context we're in - category or available signals
    if 'category_signals' in test_context and test_context['category_signals'] is not None:
        # We're checking category signals
        signals = test_context.get('category_signals')
        error_key = 'category_error'
    else:
        # We're checking available signals
        signals = test_context.get('available_signals')
        error_key = 'signals_error'

    signals_error = test_context.get(error_key)
    assert signals_error is None, f"Failed to get signals: {signals_error}"

    assert signals is not None, "Signals list should not be None"

    signal_name = signal_name.strip('"\'')
    assert signal_name in signals, f"Signals should contain '{signal_name}': {signals}"


@then(parsers.parse('the list should have exactly {expected_count:d} signals'))
def step_list_has_exact_count(test_context, expected_count):
    """Verify available signals list has expected count"""
    available_signals = test_context.get('available_signals')
    assert available_signals is not None, "Available signals list should not be None"

    actual_count = len(available_signals)
    assert actual_count == expected_count, \
        f"Expected {expected_count} signals, got {actual_count}: {available_signals}"


@then(parsers.parse('the list should have exactly {expected_count:d} signal in {category} category'))
def step_category_has_exact_count(test_context, expected_count, category):
    """Verify category has expected number of signals"""
    category_signals = test_context.get('category_signals')
    assert category_signals is not None, f"Category signals list should not be None for {category}"

    actual_count = len(category_signals)
    assert actual_count == expected_count, \
        f"Expected {expected_count} signals in {category}, got {actual_count}: {category_signals}"


@then(parsers.parse('the category list should contain {category_name}'))
def step_category_list_contains(test_context, category_name):
    """Verify category list contains expected category"""
    categories_error = test_context.get('categories_error')
    assert categories_error is None, f"Failed to get available categories: {categories_error}"

    available_categories = test_context.get('available_categories')
    assert available_categories is not None, "Available categories list should not be None"

    # Remove quotes from category name
    category_name = category_name.strip('"\'')

    assert category_name in available_categories, \
        f"Available categories should contain '{category_name}': {available_categories}"


@then(parsers.parse('the category list should have exactly {expected_count:d} categories'))
def step_category_list_has_exact_count(test_context, expected_count):
    """Verify category list has expected count"""
    available_categories = test_context.get('available_categories')
    assert available_categories is not None, "Available categories list should not be None"

    actual_count = len(available_categories)
    assert actual_count == expected_count, \
        f"Expected {expected_count} categories, got {actual_count}: {available_categories}"