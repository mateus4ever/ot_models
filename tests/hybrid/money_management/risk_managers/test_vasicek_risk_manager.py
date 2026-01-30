# tests/bdd/step_defs/test_atr_based_risk_manager_steps.py
import logging
from pathlib import Path

import pandas as pd
import pytest
from pytest_bdd import scenarios, given, parsers, when, then

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.money_management.risk_managers.vasicek_risk_manager import VasicekRiskManager
from src.hybrid.positions.types import TradingSignal, PortfolioState
from src.hybrid.products.product_types import PositionDirection

# Load all scenarios from feature file
scenarios('vasicek_risk_manager.feature')


# ============================================================================
# FIXTURES
# ============================================================================

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


# ============================================================================
# GET STEPS - Actions
# ============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    root_path = Path(__file__).parent.parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['unified_config'] = config
    test_context['root_path'] = root_path


@given(parsers.parse('Vasicek parameters are set with theta={theta:f}, sigma={sigma:f}'))
def set_vasicek_parameters(test_context, theta, sigma):
    """Set theta and sigma in test context"""
    test_context['theta'] = theta
    test_context['sigma'] = sigma


@given(parsers.parse(
    'signal is {direction} with entry price {entry_price}, strength {strength} and entry Z-score {entry_z}'))
def create_signal_with_z_score(test_context, direction, entry_price, strength, entry_z):
    """Create trading signal with Z-score metadata"""
    direction_map = {
        'LONG': PositionDirection.LONG,
        'SHORT': PositionDirection.SHORT
    }

    signal = TradingSignal(
        symbol='TEST_SPREAD',
        direction=direction_map[direction],
        signal_strength=float(strength),
        entry_price=float(entry_price),
        timestamp=pd.Timestamp.now(),
        metadata={'entry_z_score': float(entry_z)}
    )
    test_context['signal'] = signal



@given(parsers.parse('stop loss Z-score threshold is {stop_z_threshold}, max daily loss is {max_daily_loss}, max drawdown is {max_drawdown}'))
def set_stop_z_threshold(test_context, stop_z_threshold, max_daily_loss, max_drawdown):
    """Override stop loss Z-score threshold in config"""
    unified_config = test_context['unified_config']

    updates = {
        'money_management': {
            'risk_managers': {
                'vasicek': {
                    'parameters': {
                        'stop_loss_z_score': float(stop_z_threshold),
                        'max_daily_loss': float(max_daily_loss),
                        'max_drawdown': float(max_drawdown)
                    }
                }
            }
        }
    }

    unified_config.update_config(updates)
    test_context['unified_config'] = unified_config


@given(parsers.parse('parameters are set with theta={theta:f}, sigma={sigma:f}, kappa={kappa:f}'))
def set_parameters(test_context, theta, sigma, kappa):
    test_context['theta'] = theta
    test_context['sigma'] = sigma
    test_context['kappa'] = kappa

@given('Vasicek risk manager is initialized')
def initialize_vasicek_risk_manager(test_context):
    """Create and initialize Vasicek risk manager"""
    config = test_context['unified_config']

    # Create risk manager
    risk_manager = VasicekRiskManager(config)

    # Set Vasicek parameters (theta, sigma from earlier step)
    if 'theta' in test_context and 'sigma' in test_context:
        kappa = test_context.get('kappa')  # ← Could be None
        risk_manager.set_vasicek_parameters(
            theta=test_context['theta'],  # ← Must exist (already checked)
            sigma=test_context['sigma'],  # ← Must exist (already checked)
            kappa=kappa  # ← Might be None (optional)
        )

    test_context['risk_manager'] = risk_manager

@given('parameters are NOT set')
def clear_parameters(test_context):
    """Ensure no Vasicek parameters are in test context"""
    # Remove any leftover parameters from previous tests
    test_context.pop('theta', None)
    test_context.pop('sigma', None)
    test_context.pop('kappa', None)


@given('config file is missing vasicek section')
def config_missing_vasicek(test_context):
    """Create config without vasicek section"""
    unified_config = test_context.get('unified_config')
    if unified_config is None:
        # Load base config first
        root_path = Path(__file__).parent.parent.parent.parent.parent
        config_path = root_path / 'config'
        unified_config = UnifiedConfig(config_path=str(config_path), environment="test")

    # Remove vasicek section
    updates = {
        'money_management': {
            'risk_managers': {
                'vasicek': None
            }
        }
    }
    unified_config.update_config(updates)
    test_context['unified_config'] = unified_config

@given(parsers.parse('signal without Z-score is {direction} with entry price {entry_price}, strength {strength}'))
def create_signal_without_z_score(test_context, direction, entry_price, strength):
    """Create trading signal without Z-score metadata"""
    direction_map = {
        'LONG': PositionDirection.LONG,
        'SHORT': PositionDirection.SHORT
    }

    signal = TradingSignal(
        symbol='TEST_SPREAD',
        direction=direction_map[direction],
        signal_strength=float(strength),
        entry_price=float(entry_price),
        timestamp=pd.Timestamp.now(),
        metadata={}
    )
    test_context['signal'] = signal


# ============================================================================
# GIVEN STEPS - Portfolio conditions
# ============================================================================

@given(parsers.parse('portfolio with total equity {total_equity}'))
def set_portfolio_equity(test_context, total_equity):
    """Set portfolio total equity"""
    test_context['total_equity'] = float(total_equity)


@given(parsers.parse('portfolio daily P&L is {daily_pnl}'))
def set_daily_pnl(test_context, daily_pnl):
    """Set portfolio daily P&L"""
    test_context['daily_pnl'] = float(daily_pnl)


@given(parsers.parse('portfolio peak equity {peak_equity}'))
def set_peak_equity(test_context, peak_equity):
    """Set portfolio peak equity"""
    test_context['peak_equity'] = float(peak_equity)


@given(parsers.parse('max daily loss threshold is {max_daily_loss}'))
def set_max_daily_loss(test_context, max_daily_loss):
    """Set max daily loss threshold"""
    test_context['max_daily_loss'] = float(max_daily_loss)


@given(parsers.parse('max drawdown threshold is {max_drawdown}'))
def set_max_drawdown(test_context, max_drawdown):
    """Set max drawdown threshold"""
    test_context['max_drawdown'] = float(max_drawdown)


# ============================================================================
# WHEN STEPS - Actions
# ============================================================================

@when('stop loss is calculated')
def calculate_stop_loss(test_context):
    """Calculate stop loss using risk manager"""
    risk_manager = test_context['risk_manager']
    signal = test_context['signal']

    try:
        stop_loss = risk_manager.calculate_stop_loss(signal, None)
        test_context['stop_loss'] = stop_loss
        test_context['error'] = None
    except Exception as e:
        test_context['stop_loss'] = None
        test_context['error'] = e


@when('risk reduction check is performed')
def perform_risk_reduction_check(test_context):
    """Check if risk reduction should be triggered"""
    risk_manager = test_context['risk_manager']

    # Create PortfolioState from test context
    portfolio = PortfolioState(
        total_equity=test_context['total_equity'],
        daily_pnl=test_context['daily_pnl'],
        peak_equity=test_context['peak_equity'],
        available_cash=test_context['total_equity'],
        positions={}  # Empty dict, no open positions for test
    )
    # Override thresholds from scenario
    risk_manager.max_daily_loss = test_context['max_daily_loss']
    risk_manager.max_drawdown = test_context['max_drawdown']

    result = risk_manager.should_reduce_risk(portfolio)

    test_context['risk_reduction_result'] = result


@when('Vasicek risk manager initialization is attempted')
def attempt_initialization(test_context):
    """Attempt to initialize risk manager, capture error"""
    config = test_context['unified_config']

    try:
        risk_manager = VasicekRiskManager(config)
        test_context['risk_manager'] = risk_manager
        test_context['init_error'] = None
    except Exception as e:
        test_context['risk_manager'] = None
        test_context['init_error'] = e

# ============================================================================
# THEN STEPS - Assertions
# ============================================================================

@then(parsers.parse('stop loss should be approximately {expected_stop_loss:f} within {tolerance:f}'))
def assert_stop_loss_approximately(test_context, expected_stop_loss, tolerance):
    """Assert stop loss is within tolerance of expected value"""
    actual_stop_loss = test_context['stop_loss']

    assert abs(actual_stop_loss - expected_stop_loss) < tolerance, \
        f"Stop loss {actual_stop_loss:.6f} not within {tolerance} of expected {expected_stop_loss:.6f}"

@then(parsers.parse('risk manager should have theta={expected_theta}'))
def verify_theta(test_context, expected_theta):
    """Verify risk manager has correct theta"""
    actual_theta = test_context['risk_manager'].theta
    assert actual_theta == float(expected_theta), \
        f"Expected theta={expected_theta}, got {actual_theta}"

@then(parsers.parse('risk manager should have sigma={expected_sigma}'))
def verify_sigma(test_context, expected_sigma):
    """Verify risk manager has correct sigma"""
    actual_sigma = test_context['risk_manager'].sigma
    assert actual_sigma == float(expected_sigma), \
        f"Expected sigma={expected_sigma}, got {actual_sigma}"


@then(parsers.parse('risk manager should have kappa={expected_kappa:f}'))
def verify_kappa(test_context, expected_kappa):
    """Verify risk manager has correct kappa"""
    actual_kappa = test_context['risk_manager'].kappa
    assert actual_kappa == float(expected_kappa), \
        f"Expected kappa={expected_kappa}, got {actual_kappa}"

@then(parsers.parse('stop loss should fallback to entry price {expected_price:f}'))
def assert_stop_loss_fallback(test_context, expected_price):
    """Assert stop loss fell back to entry price"""
    actual_stop_loss = test_context['stop_loss']
    assert actual_stop_loss == expected_price, \
        f"Expected fallback to {expected_price}, got {actual_stop_loss}"

@then('calculation should raise error about missing parameters')
def assert_calculation_raises(test_context):
    """Assert that calculation raised an error"""
    error = test_context.get('error')
    assert error is not None, "Expected exception but calculation succeeded"
    assert 'parameter' in str(error).lower(), \
        f"Expected error about parameters, got: {error}"


@then(parsers.parse('risk reduction should be {expected_result}'))
def assert_risk_reduction_result(test_context, expected_result):
    """Assert risk reduction trigger result"""
    result = test_context['risk_reduction_result']

    if expected_result == 'triggered':
        assert result is True, f"Expected risk reduction triggered, but got {result}"
    else:
        assert result is False, f"Expected risk reduction not triggered, but got {result}"

@then('calculation should raise error about invalid sigma')
def assert_invalid_sigma_error(test_context):
    """Assert that calculation raised an error about invalid sigma"""
    error = test_context.get('error')
    assert error is not None, "Expected exception but calculation succeeded"
    assert 'sigma' in str(error).lower(), \
        f"Expected error about sigma, got: {error}"

@then('initialization should raise error')
def assert_init_error(test_context):
    """Assert any error was raised during initialization"""
    error = test_context.get('init_error')
    assert error is not None, "Expected error but initialization succeeded"

@then(parsers.parse('error message should contain "{text}"'))
def assert_error_contains(test_context, text):
    """Assert error message contains text"""
    error = test_context.get('init_error')
    assert text.lower() in str(error).lower(), \
        f"Expected '{text}' in error message, got: {error}"

@then('calculation should raise error about missing Z-score')
def assert_missing_z_score_error(test_context):
    """Assert that calculation raised an error about missing Z-score"""
    error = test_context.get('error')
    assert error is not None, "Expected exception but calculation succeeded"
    assert 'z_score' in str(error).lower() or 'z-score' in str(error).lower(), \
        f"Expected error about Z-score, got: {error}"