# tests/hybrid/signals/test_signal_interface.py
"""
pytest-bdd test runner for SignalInterface protocol compliance
Tests interface contracts: method existence, signatures, dependency management
ZERO BEHAVIORAL TESTING - Only protocol compliance validation
"""

import pytest
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from pytest_bdd import scenarios, given, when, then, parsers
from src.hybrid.signals import SignalInterface

# Load scenarios from the signal_interface.feature
scenarios('signal_interface.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# Test fixtures and shared state
@pytest.fixture
def test_context():
    """Shared test context for storing state between steps"""
    return {}


# Mock signal implementation for protocol testing
class MockSignalImplementation(SignalInterface):
    """Mock signal implementation for testing protocol compliance only"""

    def __init__(self):
        self.training_data = None
        self.historical_data = []
        self.is_trained = False

    def train(self, training_data: pd.DataFrame) -> None:
        """Mock training method"""
        self.training_data = training_data
        self.is_trained = True

    def update_with_new_data(self, data_point: pd.Series) -> None:
        """Mock data update method"""
        self.historical_data.append(data_point)

    def generate_signal(self, current_data: pd.Series) -> str:
        """Mock signal generation method"""
        if not self.is_trained:
            raise ValueError("Signal must be trained first")
        return "HOLD"

    def getMetrics(self) -> dict:
        """Mock metrics method"""
        return {"mock_metric": 1.0}


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


@given('I have a mock signal implementation')
def step_have_mock_signal_implementation(test_context):
    """Create mock signal implementation for testing"""
    logger = logging.getLogger(__name__)

    try:
        mock_signal = MockSignalImplementation()
        test_context['signal'] = mock_signal
        logger.debug("Mock signal implementation created successfully")

    except Exception as e:
        pytest.fail(f"Failed to create mock signal implementation: {e}")


@given('I have mock training data for testing')
def step_have_mock_training_data(test_context):
    """Create mock training data for signal testing"""
    # Create simple mock training data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = np.random.randn(100).cumsum() + 100  # Random walk around 100

    training_data = pd.DataFrame({
        'close': prices,
        'high': prices + np.random.rand(100),
        'low': prices - np.random.rand(100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    test_context['training_data'] = training_data


@given('the signal has been properly initialized')
def step_signal_properly_initialized(test_context):
    """Ensure signal is properly initialized for testing"""
    signal = test_context.get('signal')
    if not signal:
        step_have_mock_signal_implementation(test_context)
        signal = test_context['signal']

    # Initialize with mock training data if not already done
    if not hasattr(signal, 'is_trained') or not signal.is_trained:
        if 'training_data' not in test_context:
            step_have_mock_training_data(test_context)
        signal.train(test_context['training_data'])

    test_context['signal_initialized'] = True


@given('the signal has mock training data configured')
def step_signal_has_training_data_configured(test_context):
    """Ensure signal has training data configured"""
    if 'training_data' not in test_context:
        step_have_mock_training_data(test_context)

    signal = test_context.get('signal')
    if signal:
        signal.train(test_context['training_data'])


@given('I have test parameters of various types')
def step_have_test_parameters(test_context):
    """Create test parameters of various types for validation"""
    test_context['valid_training_data'] = test_context.get('training_data')
    test_context['valid_data_point'] = pd.Series({'close': 100.0, 'volume': 1000})
    test_context['invalid_training_data'] = "not_a_dataframe"
    test_context['invalid_data_point'] = "not_a_series"


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I inspect the signal interface compliance')
def step_inspect_signal_interface_compliance(test_context):
    """Inspect signal for interface compliance"""
    signal = test_context['signal']
    test_context['compliance_results'] = {
        'has_train': hasattr(signal, 'train') and callable(getattr(signal, 'train')),
        'has_update': hasattr(signal, 'update_with_new_data') and callable(getattr(signal, 'update_with_new_data')),
        'has_generate': hasattr(signal, 'generate_signal') and callable(getattr(signal, 'generate_signal')),
        'has_metrics': hasattr(signal, 'getMetrics') and callable(getattr(signal, 'getMetrics'))
    }


@when('I execute the training lifecycle')
def step_execute_training_lifecycle(test_context):
    """Execute signal training lifecycle"""
    signal = test_context['signal']
    training_data = test_context['training_data']

    try:
        # Test training
        signal.train(training_data)
        test_context['training_error'] = None

        # Test data update
        new_data_point = pd.Series({'close': 105.0, 'volume': 1200})
        signal.update_with_new_data(new_data_point)
        test_context['update_error'] = None

    except Exception as e:
        test_context['training_error'] = e
        test_context['update_error'] = e


@when('I call signal generation methods')
def step_call_signal_generation_methods(test_context):
    """Call signal generation methods"""
    signal = test_context['signal']
    current_data = pd.Series({'close': 102.0, 'volume': 1500})

    try:
        signal_result = signal.generate_signal(current_data)
        metrics_result = signal.getMetrics()

        test_context['signal_result'] = signal_result
        test_context['metrics_result'] = metrics_result
        test_context['generation_error'] = None

    except Exception as e:
        test_context['generation_error'] = e


@when('I call signal methods with valid inputs')
def step_call_methods_with_valid_inputs(test_context):
    """Call signal methods with valid input types"""
    signal = test_context['signal']

    try:
        # Test with valid inputs
        train_result = signal.train(test_context['valid_training_data'])
        update_result = signal.update_with_new_data(test_context['valid_data_point'])
        signal_result = signal.generate_signal(test_context['valid_data_point'])
        metrics_result = signal.getMetrics()

        test_context['method_results'] = {
            'train': train_result,
            'update': update_result,
            'signal': signal_result,
            'metrics': metrics_result
        }
        test_context['valid_input_error'] = None

    except Exception as e:
        test_context['valid_input_error'] = e


@when('I call signal methods with invalid parameter types')
def step_call_methods_with_invalid_inputs(test_context):
    """Call signal methods with invalid input types"""
    signal = test_context['signal']

    test_context['invalid_input_errors'] = {}

    # Test invalid training data
    try:
        signal.train(test_context['invalid_training_data'])
    except Exception as e:
        test_context['invalid_input_errors']['train'] = e

    # Test invalid data point
    try:
        signal.update_with_new_data(test_context['invalid_data_point'])
    except Exception as e:
        test_context['invalid_input_errors']['update'] = e


@when('I call methods on untrained signal')
def step_call_methods_on_untrained_signal(test_context):
    """Call methods on signal that hasn't been trained"""
    # Create fresh untrained signal
    untrained_signal = MockSignalImplementation()
    test_context['untrained_signal'] = untrained_signal

    try:
        current_data = pd.Series({'close': 100.0})
        untrained_signal.generate_signal(current_data)
        test_context['untrained_error'] = None
    except Exception as e:
        test_context['untrained_error'] = e


@when('I call methods with missing data')
def step_call_methods_with_missing_data(test_context):
    """Call methods with insufficient or missing data"""
    signal = test_context['signal']

    try:
        # Try with empty DataFrame
        empty_data = pd.DataFrame()
        signal.train(empty_data)
        test_context['missing_data_error'] = None
    except Exception as e:
        test_context['missing_data_error'] = e


@when('I call methods with invalid inputs')
def step_call_methods_with_invalid_inputs_general(test_context):
    """Call methods with various invalid inputs"""
    signal = test_context['signal']

    test_context['invalid_errors'] = {}

    # Test with None inputs
    try:
        signal.generate_signal(None)
    except Exception as e:
        test_context['invalid_errors']['none_input'] = e


@when('I train the signal with data')
def step_train_signal_with_data(test_context):
    """Train signal with data and check state"""
    signal = test_context['signal']
    training_data = test_context['training_data']

    # Record state before training
    test_context['state_before_training'] = {
        'is_trained': getattr(signal, 'is_trained', False),
        'training_data': getattr(signal, 'training_data', None)
    }

    # Train signal
    signal.train(training_data)

    # Record state after training
    test_context['state_after_training'] = {
        'is_trained': getattr(signal, 'is_trained', False),
        'training_data': getattr(signal, 'training_data', None)
    }


@when('I update the signal with new data')
def step_update_signal_with_new_data(test_context):
    """Update signal with new data and check state"""
    signal = test_context['signal']
    new_data = pd.Series({'close': 110.0, 'volume': 2000})

    # Record state before update
    test_context['state_before_update'] = {
        'historical_data_count': len(getattr(signal, 'historical_data', []))
    }

    # Update signal
    signal.update_with_new_data(new_data)

    # Record state after update
    test_context['state_after_update'] = {
        'historical_data_count': len(getattr(signal, 'historical_data', []))
    }


# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('the signal should have all required methods')
def step_signal_has_required_methods(test_context):
    """Verify signal has all required methods"""
    compliance = test_context['compliance_results']
    assert compliance['has_train'], "Signal should have train method"
    assert compliance['has_update'], "Signal should have update_with_new_data method"
    assert compliance['has_generate'], "Signal should have generate_signal method"
    assert compliance['has_metrics'], "Signal should have getMetrics method"


@then('all method signatures should match the protocol')
def step_method_signatures_match_protocol(test_context):
    """Verify method signatures match SignalInterface protocol"""
    signal = test_context['signal']

    # Check method signatures (basic callable verification)
    assert callable(getattr(signal, 'train')), "train method should be callable"
    assert callable(getattr(signal, 'update_with_new_data')), "update_with_new_data method should be callable"
    assert callable(getattr(signal, 'generate_signal')), "generate_signal method should be callable"
    assert callable(getattr(signal, 'getMetrics')), "getMetrics method should be callable"


@then('the signal should be callable and functional')
def step_signal_callable_and_functional(test_context):
    """Verify signal is callable and functional"""
    compliance = test_context['compliance_results']
    assert all(compliance.values()), f"All methods should be callable: {compliance}"


@then('the signal should accept training data without errors')
def step_signal_accepts_training_data(test_context):
    """Verify signal accepts training data"""
    training_error = test_context.get('training_error')
    assert training_error is None, f"Training should not cause errors: {training_error}"


@then('the signal should accept new data updates')
def step_signal_accepts_data_updates(test_context):
    """Verify signal accepts data updates"""
    update_error = test_context.get('update_error')
    assert update_error is None, f"Data updates should not cause errors: {update_error}"


@then('the training methods should be callable')
def step_training_methods_callable(test_context):
    """Verify training methods are callable"""
    signal = test_context['signal']
    assert hasattr(signal, 'train') and callable(signal.train), "train method should be callable"
    assert hasattr(signal, 'update_with_new_data') and callable(
        signal.update_with_new_data), "update method should be callable"


@then('the signal should generate signals with current data')
def step_signal_generates_signals(test_context):
    """Verify signal generates signals"""
    generation_error = test_context.get('generation_error')
    assert generation_error is None, f"Signal generation should not cause errors: {generation_error}"

    signal_result = test_context.get('signal_result')
    assert signal_result is not None, "Signal should return a result"


@then('the signal should provide metrics information')
def step_signal_provides_metrics(test_context):
    """Verify signal provides metrics"""
    metrics_result = test_context.get('metrics_result')
    assert metrics_result is not None, "Signal should provide metrics"
    assert isinstance(metrics_result, dict), "Metrics should be a dictionary"


@then('signal generation should not cause errors')
def step_signal_generation_no_errors(test_context):
    """Verify signal generation doesn't cause errors"""
    generation_error = test_context.get('generation_error')
    assert generation_error is None, f"Signal generation should not cause errors: {generation_error}"


@then('the train method should return None type')
def step_train_returns_none(test_context):
    """Verify train method returns None"""
    method_results = test_context.get('method_results', {})
    train_result = method_results.get('train')
    assert train_result is None, f"train method should return None, got {type(train_result)}"


@then('the update_with_new_data method should return None type')
def step_update_returns_none(test_context):
    """Verify update_with_new_data method returns None"""
    method_results = test_context.get('method_results', {})
    update_result = method_results.get('update')
    assert update_result is None, f"update_with_new_data method should return None, got {type(update_result)}"


@then('the generate_signal method should return string type')
def step_generate_signal_returns_string(test_context):
    """Verify generate_signal method returns string"""
    method_results = test_context.get('method_results', {})
    signal_result = method_results.get('signal')
    assert isinstance(signal_result, str), f"generate_signal method should return string, got {type(signal_result)}"


@then('the getMetrics method should return dict type')
def step_get_metrics_returns_dict(test_context):
    """Verify getMetrics method returns dict"""
    method_results = test_context.get('method_results', {})
    metrics_result = method_results.get('metrics')
    assert isinstance(metrics_result, dict), f"getMetrics method should return dict, got {type(metrics_result)}"


@then('return types should match the protocol specifications')
def step_return_types_match_protocol(test_context):
    """Verify all return types match protocol specifications"""
    method_results = test_context.get('method_results', {})

    # Verify None returns
    assert method_results.get('train') is None, "train should return None"
    assert method_results.get('update') is None, "update_with_new_data should return None"

    # Verify typed returns
    assert isinstance(method_results.get('signal'), str), "generate_signal should return str"
    assert isinstance(method_results.get('metrics'), dict), "getMetrics should return dict"


@then('the methods should accept the parameters without type errors')
def step_methods_accept_valid_parameters(test_context):
    """Verify methods accept valid parameter types"""
    valid_error = test_context.get('valid_input_error')
    assert valid_error is None, f"Methods should accept valid parameters: {valid_error}"


@then('the methods should reject invalid parameters appropriately')
def step_methods_reject_invalid_parameters(test_context):
    """Verify methods reject invalid parameter types"""
    invalid_errors = test_context.get('invalid_input_errors', {})
    assert len(invalid_errors) > 0, "Methods should reject invalid parameters"


@then('proper parameter validation should occur')
def step_proper_parameter_validation(test_context):
    """Verify proper parameter validation occurs"""
    invalid_errors = test_context.get('invalid_input_errors', {})
    # Should have errors for invalid inputs
    assert 'train' in invalid_errors or 'update' in invalid_errors, "Parameter validation should catch invalid inputs"


@then('proper exceptions should be raised for invalid states')
def step_proper_exceptions_invalid_states(test_context):
    """Verify proper exceptions for invalid states"""
    untrained_error = test_context.get('untrained_error')
    assert untrained_error is not None, "Should raise exception when calling untrained signal"
    assert isinstance(untrained_error,
                      (ValueError, RuntimeError)), f"Should raise appropriate exception type: {type(untrained_error)}"


@then('proper exceptions should be raised for insufficient data')
def step_proper_exceptions_insufficient_data(test_context):
    """Verify proper exceptions for insufficient data"""
    missing_data_error = test_context.get('missing_data_error')
    assert missing_data_error is not None, "Should raise exception for insufficient data"


@then('proper exceptions should be raised for invalid inputs')
def step_proper_exceptions_invalid_inputs(test_context):
    """Verify proper exceptions for invalid inputs"""
    invalid_errors = test_context.get('invalid_errors', {})
    assert len(invalid_errors) > 0, "Should raise exceptions for invalid inputs"


@then('all exceptions should be informative and appropriate')
def step_exceptions_informative_and_appropriate(test_context):
    """Verify exceptions are informative and appropriate"""
    # Check that exceptions have meaningful messages
    untrained_error = test_context.get('untrained_error')
    if untrained_error:
        assert len(str(untrained_error)) > 0, "Exception messages should be informative"


@then('the signal should maintain proper training state')
def step_signal_maintains_training_state(test_context):
    """Verify signal maintains proper training state"""
    state_before = test_context['state_before_training']
    state_after = test_context['state_after_training']

    assert not state_before['is_trained'], "Signal should not be trained initially"
    assert state_after['is_trained'], "Signal should be trained after train() call"


@then('data references should be correctly stored')
def step_data_references_stored_correctly(test_context):
    """Verify data references are correctly stored"""
    state_after = test_context['state_after_training']
    assert state_after['training_data'] is not None, "Training data should be stored"


@then('the signal should maintain proper data state')
def step_signal_maintains_data_state(test_context):
    """Verify signal maintains proper data state after updates"""
    state_before = test_context['state_before_update']
    state_after = test_context['state_after_update']

    assert state_after['historical_data_count'] > state_before[
        'historical_data_count'], "Historical data should increase after update"


@then('historical data should be correctly managed')
def step_historical_data_managed_correctly(test_context):
    """Verify historical data is correctly managed"""
    signal = test_context['signal']
    assert hasattr(signal, 'historical_data'), "Signal should maintain historical data"


@then('the signal state should remain consistent throughout')
def step_signal_state_consistent(test_context):
    """Verify signal state remains consistent"""
    # Basic consistency check - signal should remain functional
    signal = test_context['signal']
    assert hasattr(signal, 'is_trained'), "Signal should maintain training state"
    current_data = pd.Series({'close': 100.0})

    try:
        result = signal.generate_signal(current_data)
        assert isinstance(result, str), "Signal should still generate valid signals"
    except Exception as e:
        pytest.fail(f"Signal state should remain consistent: {e}")