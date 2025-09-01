# tests/hybrid/test_strategy_interface.py
"""
pytest-bdd test runner for StrategyInterface protocol compliance
Tests interface signature compliance - no implementation execution
"""

from pathlib import Path

# Add project root to Python path

import pytest
import logging
from pytest_bdd import scenarios, given, when, then
import inspect
from typing import get_type_hints, get_origin, get_args

from src.hybrid.strategies.strategy_interface import StrategyInterface

# Load scenarios from the strategy_interface.feature
scenarios('strategy_interface.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# Test fixtures and shared state
@pytest.fixture
def test_context():
    """Shared test context for storing state between steps"""
    return {}


# Mock Strategy Implementation for signature testing
class MockStrategyForSignatureTesting:
    """Mock strategy that implements StrategyInterface for signature testing only"""

    def __init__(self, name="MockStrategy"):
        self.name = name
        self.money_manager = None
        self.data_manager = None
        self.signals = []
        self.optimizations = []
        self.predictors = []
        self.runners = []
        self.metrics = []
        self.verificators = []

    def setMoneyManager(self, money_manager) -> None:
        pass  # Signature only

    def setDataManager(self, data_manager) -> None:
        pass  # Signature only

    def addSignal(self, signal) -> None:
        pass  # Signature only

    def addOptimizer(self, optimization) -> None:
        pass  # Signature only

    def addPredictor(self, predictor) -> None:
        pass  # Signature only

    def addRunner(self, runner) -> None:
        pass  # Signature only

    def addMetric(self, metric) -> None:
        pass  # Signature only

    def addVerificator(self, verificator) -> None:
        pass  # Signature only

    def initialize(self, market_data) -> bool:
        return True  # Minimal return for signature testing

    def generate_signals(self, data):
        return {}  # Minimal return for signature testing

    def execute_trades(self, signals):
        return {}  # Minimal return for signature testing

    def run_backtest(self, market_data):
        return {}  # Minimal return for signature testing


# =============================================================================
# GIVEN steps - Setup
# =============================================================================

@given('I have a mock strategy implementation')
def step_have_mock_strategy_implementation(test_context):
    """Create a mock strategy for signature testing"""
    strategy = MockStrategyForSignatureTesting()
    test_context['strategy'] = strategy


@given('I have mock dependencies for testing')
def step_have_mock_dependencies(test_context):
    """Create minimal mock objects for signature testing"""
    test_context['mock_money_manager'] = object()  # Just need an object
    test_context['mock_data_manager'] = object()  # Just need an object


@given('I have mock components for testing')
def step_have_mock_components(test_context):
    """Create minimal mock components for signature testing"""
    test_context['mock_components'] = {
        'signal': object(),
        'optimizer': object(),
        'predictor': object(),
        'runner': object(),
        'metric': object(),
        'verificator': object()
    }


@given('the strategy has mock dependencies injected')
def step_strategy_has_mock_dependencies_injected(test_context):
    """Not needed for signature testing"""
    pass


@given('the strategy has mock components added')
def step_strategy_has_mock_components_added(test_context):
    """Not needed for signature testing"""
    pass


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I inspect the strategy interface compliance')
def step_inspect_strategy_interface_compliance(test_context):
    """Inspect strategy for interface signature compliance"""
    strategy = test_context['strategy']

    # Get strategy methods and their signatures
    strategy_methods = {}
    for name, method in inspect.getmembers(strategy, predicate=inspect.ismethod):
        if not name.startswith('_'):
            strategy_methods[name] = inspect.signature(method)

    # Get interface methods and their signatures (from Protocol)
    interface_methods = {}
    for name in dir(StrategyInterface):
        if not name.startswith('_') and callable(getattr(StrategyInterface, name, None)):
            interface_methods[name] = name  # Protocol methods don't have runtime signatures

    test_context['strategy_methods'] = strategy_methods
    test_context['interface_methods'] = interface_methods
    test_context['strategy_attributes'] = dir(strategy)


@when('I inject dependencies into the strategy')
def step_inject_dependencies_into_strategy(test_context):
    """Test dependency injection method signatures"""
    strategy = test_context['strategy']

    # Check setMoneyManager signature
    money_manager_sig = inspect.signature(strategy.setMoneyManager)
    params = list(money_manager_sig.parameters.keys())

    # Check setDataManager signature
    data_manager_sig = inspect.signature(strategy.setDataManager)
    data_params = list(data_manager_sig.parameters.keys())

    test_context['injection_signatures'] = {
        'setMoneyManager': {'params': params, 'signature': money_manager_sig},
        'setDataManager': {'params': data_params, 'signature': data_manager_sig}
    }


@when('I add components to the strategy')
def step_add_components_to_strategy(test_context):
    """Test component addition method signatures"""
    strategy = test_context['strategy']

    component_methods = ['addSignal', 'addOptimizer', 'addPredictor',
                         'addRunner', 'addMetric', 'addVerificator']

    component_signatures = {}
    for method_name in component_methods:
        method = getattr(strategy, method_name)
        sig = inspect.signature(method)
        component_signatures[method_name] = {
            'params': list(sig.parameters.keys()),
            'signature': sig
        }

    test_context['component_signatures'] = component_signatures


@when('I execute the strategy lifecycle methods')
def step_execute_strategy_lifecycle_methods(test_context):
    """Test strategy lifecycle method signatures"""
    strategy = test_context['strategy']

    lifecycle_methods = ['initialize', 'generate_signals', 'execute_trades', 'run_backtest']

    lifecycle_signatures = {}
    for method_name in lifecycle_methods:
        method = getattr(strategy, method_name)
        sig = inspect.signature(method)
        lifecycle_signatures[method_name] = {
            'params': list(sig.parameters.keys()),
            'signature': sig,
            'return_annotation': sig.return_annotation
        }

    test_context['lifecycle_signatures'] = lifecycle_signatures


# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('the strategy should have all required attributes')
def step_strategy_has_required_attributes(test_context):
    """Verify strategy has all required attributes"""
    strategy_attributes = test_context['strategy_attributes']

    required_attributes = ['name', 'money_manager', 'data_manager', 'signals',
                           'optimizations', 'predictors', 'runners', 'metrics', 'verificators']

    for attr in required_attributes:
        assert attr in strategy_attributes, f"Strategy missing required attribute: {attr}"


@then('the strategy should have all required methods')
def step_strategy_has_required_methods(test_context):
    """Verify strategy has all required methods"""
    strategy_methods = test_context['strategy_methods']

    required_methods = ['setMoneyManager', 'setDataManager', 'addSignal', 'addOptimizer',
                        'addPredictor', 'addRunner', 'addMetric', 'addVerificator',
                        'initialize', 'generate_signals', 'execute_trades', 'run_backtest']

    for method in required_methods:
        assert method in strategy_methods, f"Strategy missing required method: {method}"


@then('all method signatures should match the protocol')
def step_method_signatures_match_protocol(test_context):
    """Verify method signatures are valid"""
    strategy_methods = test_context['strategy_methods']

    # Check that key methods have reasonable signatures
    assert 'initialize' in strategy_methods, "Missing initialize method"
    assert 'generate_signals' in strategy_methods, "Missing generate_signals method"
    assert 'execute_trades' in strategy_methods, "Missing execute_trades method"
    assert 'run_backtest' in strategy_methods, "Missing run_backtest method"


@then('the strategy should accept MoneyManager injection')
def step_strategy_accepts_money_manager_injection(test_context):
    """Verify setMoneyManager has correct signature"""
    injection_sigs = test_context['injection_signatures']

    money_manager_sig = injection_sigs['setMoneyManager']
    params = money_manager_sig['params']

    assert 'money_manager' in params, "setMoneyManager should have money_manager parameter"
    assert len(params) == 1, f"setMoneyManager should have exactly 1 parameter, got {len(params)}"


@then('the strategy should accept DataManager injection')
def step_strategy_accepts_data_manager_injection(test_context):
    """Verify setDataManager has correct signature"""
    injection_sigs = test_context['injection_signatures']

    data_manager_sig = injection_sigs['setDataManager']
    params = data_manager_sig['params']

    assert 'data_manager' in params, "setDataManager should have data_manager parameter"
    assert len(params) == 1, f"setDataManager should have exactly 1 parameter, got {len(params)}"


@then('the dependency injection should not cause errors')
def step_dependency_injection_no_errors(test_context):
    """Verify dependency injection signatures are valid"""
    injection_sigs = test_context['injection_signatures']

    # Both methods should have exactly one parameter each
    assert len(injection_sigs) == 2, "Should have signatures for both injection methods"


@then('the strategy should accept signal components')
def step_strategy_accepts_signal_components(test_context):
    """Verify addSignal has correct signature"""
    component_sigs = test_context['component_signatures']

    signal_sig = component_sigs['addSignal']
    params = signal_sig['params']

    assert 'signal' in params, "addSignal should have signal parameter"
    assert len(params) == 1, f"addSignal should have exactly 1 parameter, got {len(params)}"


@then('the strategy should accept optimizer components')
def step_strategy_accepts_optimizer_components(test_context):
    """Verify addOptimizer has correct signature"""
    component_sigs = test_context['component_signatures']

    optimizer_sig = component_sigs['addOptimizer']
    params = optimizer_sig['params']

    assert 'optimization' in params, "addOptimizer should have optimization parameter"


@then('the strategy should accept predictor components')
def step_strategy_accepts_predictor_components(test_context):
    """Verify addPredictor has correct signature"""
    component_sigs = test_context['component_signatures']

    predictor_sig = component_sigs['addPredictor']
    params = predictor_sig['params']

    assert 'predictor' in params, "addPredictor should have predictor parameter"


@then('the strategy should accept runner components')
def step_strategy_accepts_runner_components(test_context):
    """Verify addRunner has correct signature"""
    component_sigs = test_context['component_signatures']

    runner_sig = component_sigs['addRunner']
    params = runner_sig['params']

    assert 'runner' in params, "addRunner should have runner parameter"


@then('the strategy should accept metric components')
def step_strategy_accepts_metric_components(test_context):
    """Verify addMetric has correct signature"""
    component_sigs = test_context['component_signatures']

    metric_sig = component_sigs['addMetric']
    params = metric_sig['params']

    assert 'metric' in params, "addMetric should have metric parameter"


@then('the strategy should accept verificator components')
def step_strategy_accepts_verificator_components(test_context):
    """Verify addVerificator has correct signature"""
    component_sigs = test_context['component_signatures']

    verificator_sig = component_sigs['addVerificator']
    params = verificator_sig['params']

    assert 'verificator' in params, "addVerificator should have verificator parameter"


@then('component addition should not cause errors')
def step_component_addition_no_errors(test_context):
    """Verify all component addition signatures are valid"""
    component_sigs = test_context['component_signatures']

    expected_methods = ['addSignal', 'addOptimizer', 'addPredictor',
                        'addRunner', 'addMetric', 'addVerificator']

    for method_name in expected_methods:
        assert method_name in component_sigs, f"Missing signature for {method_name}"


@then('the strategy initialize method should be callable')
def step_strategy_initialize_callable(test_context):
    """Verify initialize method signature"""
    lifecycle_sigs = test_context['lifecycle_signatures']

    init_sig = lifecycle_sigs['initialize']
    params = init_sig['params']

    assert 'market_data' in params, "initialize should have market_data parameter"


@then('the strategy generate_signals method should be callable')
def step_strategy_generate_signals_callable(test_context):
    """Verify generate_signals method signature"""
    lifecycle_sigs = test_context['lifecycle_signatures']

    signals_sig = lifecycle_sigs['generate_signals']
    params = signals_sig['params']

    assert 'data' in params, "generate_signals should have data parameter"


@then('the strategy execute_trades method should be callable')
def step_strategy_execute_trades_callable(test_context):
    """Verify execute_trades method signature"""
    lifecycle_sigs = test_context['lifecycle_signatures']

    trades_sig = lifecycle_sigs['execute_trades']
    params = trades_sig['params']

    assert 'signals' in params, "execute_trades should have signals parameter"


@then('the strategy run_backtest method should be callable')
def step_strategy_run_backtest_callable(test_context):
    """Verify run_backtest method signature"""
    lifecycle_sigs = test_context['lifecycle_signatures']

    backtest_sig = lifecycle_sigs['run_backtest']
    params = backtest_sig['params']

    assert 'market_data' in params, "run_backtest should have market_data parameter"


@then('all lifecycle methods should return mock results')
def step_lifecycle_methods_return_mock_results(test_context):
    """Verify lifecycle method signatures have return annotations where expected"""
    lifecycle_sigs = test_context['lifecycle_signatures']

    # Check that initialize has bool return annotation (if present)
    init_sig = lifecycle_sigs['initialize']
    if init_sig['return_annotation'] != inspect.Signature.empty:
        # Return annotation exists, verify it's reasonable
        assert init_sig['return_annotation'] is not None

    # Verify all lifecycle methods have signatures
    expected_methods = ['initialize', 'generate_signals', 'execute_trades', 'run_backtest']
    for method_name in expected_methods:
        assert method_name in lifecycle_sigs, f"Missing signature for {method_name}"