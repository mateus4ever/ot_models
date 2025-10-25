# tests/bdd/step_defs/test_atr_based_risk_manager_steps.py
from pathlib import Path

import pandas as pd
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data.data_manager import DataManager
from src.hybrid.money_management import PositionDirection, TradingSignal, PortfolioState
from src.hybrid.money_management.risk_managers import ATRBasedRiskManager

# Load all scenarios from feature file
scenarios('atr_based_risk_manager.feature')


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
# BACKGROUND STEPS - Configuration Loading
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

# ============================================================================
# GIVEN STEPS - Setup Test Conditions
# ============================================================================

@given(parsers.parse('market data has ATR of {atr_value}'))
def mock_market_data_with_atr(test_context, atr_value):
    """Create mock market data that results in specified ATR"""
    atr = float(atr_value)
    test_context['expected_atr'] = atr

    # Create simple market data that will produce this ATR
    # For testing stop loss calculation, we just need the ATR result
    # We'll mock the _calculate_atr method instead of building complex data
    test_context['mock_atr'] = atr


@given(parsers.parse('signal is {direction} with entry price {entry_price} and strength {strength}'))
def create_trading_signal(test_context, direction, entry_price, strength):
    """Create trading signal with specified direction, entry price, and strength"""
    signal_direction = PositionDirection.LONG if direction == 'LONG' else PositionDirection.SHORT

    signal = TradingSignal(
        symbol='TEST',
        direction=signal_direction,
        entry_price=float(entry_price),
        strength=float(strength),
        timestamp=pd.Timestamp.now()
    )

    test_context['signal'] = signal


@given(parsers.parse('ATR multiplier is {multiplier}'))
def set_atr_multiplier(test_context, multiplier):
    """Override ATR multiplier in configuration"""
    config = test_context['unified_config']

    # Get the money_management section
    mm_config = config.config['money_management']

    # Override the ATR multiplier
    mm_config['risk_managers']['atr_based']['stop_loss_atr_multiplier'] = float(multiplier)

    test_context['atr_multiplier'] = float(multiplier)


@given(parsers.parse('market data is loaded from {data_file}'))
def load_market_data(test_context, data_file):
    """Load market data using DataManager"""

    config = test_context['unified_config']

    data_manager = DataManager(config)

    root_path = Path(__file__).parent.parent.parent.parent.parent
    data_file_path = root_path / data_file

    success = data_manager.load_market_data(str(data_file_path))  # Now uses explicit path

    if not success:
        raise ValueError(f"Failed to load market data from {data_file}")

    available_markets = data_manager.get_available_markets()

    if not available_markets:
        raise ValueError(f"No markets loaded from {data_file}")

    market_id = available_markets[0]

    test_context['data_manager'] = data_manager
    test_context['market_id'] = market_id


@given(parsers.parse('market data with {data_condition}'))
def prepare_market_data_condition(test_context, data_condition):
    """Prepare market data slice based on condition using temporal pointer"""
    data_manager = test_context['data_manager']

    if 'periods' in data_condition:
        periods = int(data_condition.split()[0])

        # DataManager already has data loaded, now set temporal pointer
        market_id = test_context['market_id']
        data_manager.set_active_market(market_id)

        # Use the internal active market data to initialize pointer
        data_manager.initialize_temporal_pointer(
            data_manager._active_market_data,
            periods
        )

        past_data = data_manager.get_past_data()
        test_context['market_data'] = past_data[market_id]

@given(parsers.parse('ATR period is {atr_period}'))
def set_atr_period(test_context, atr_period):
    """Set ATR period for calculation"""
    test_context['atr_period'] = int(atr_period)


@given(parsers.parse('portfolio with total equity {total_equity}'))
def create_portfolio_with_equity(test_context, total_equity):
    """Create portfolio with specified total equity"""
    equity = float(total_equity)

    portfolio = PortfolioState(
        total_equity=equity,
        available_cash=equity,
        positions={}
    )

    test_context['portfolio'] = portfolio

@given(parsers.parse('portfolio daily P&L is {daily_pnl}'))
def set_portfolio_daily_pnl(test_context, daily_pnl):
    """Set portfolio daily P&L"""
    portfolio = test_context['portfolio']
    portfolio.daily_pnl = float(daily_pnl)


@given(parsers.parse('portfolio peak equity {peak_equity}'))
def set_portfolio_peak_equity(test_context, peak_equity):
    """Set portfolio peak equity"""
    portfolio = test_context['portfolio']
    portfolio.peak_equity = float(peak_equity)

@given(parsers.parse('max daily loss threshold is {max_daily_loss}'))
def set_max_daily_loss_threshold(test_context, max_daily_loss):
    """Override max daily loss threshold in config"""
    config = test_context['unified_config']
    risk_config = config.get_section('money_management')['risk_managers']['atr_based']
    risk_config['max_daily_loss'] = float(max_daily_loss)


@given(parsers.parse('max drawdown threshold is {max_drawdown}'))
def set_max_drawdown_threshold(test_context, max_drawdown):
    """Override max drawdown threshold in config"""
    config = test_context['unified_config']
    risk_config = config.get_section('money_management')['risk_managers']['atr_based']
    risk_config['max_drawdown'] = float(max_drawdown)

# ============================================================================
# WHEN STEPS - Actions
# ============================================================================
@when('stop loss is calculated')
def calculate_stop_loss(test_context):
    """Calculate stop loss using ATRBasedRiskManager"""
    config = test_context['unified_config']
    signal = test_context['signal']
    mock_atr = test_context['mock_atr']

    # Create risk manager
    risk_manager = ATRBasedRiskManager(config)

    # Create empty market data (mocked ATR won't use it)
    market_data = pd.DataFrame()

    # Mock the ATR calculation to return our test value
    risk_manager._calculate_atr = lambda data: mock_atr

    # Calculate stop loss
    stop_loss = risk_manager.calculate_stop_loss(signal, market_data)

    test_context['calculated_stop_loss'] = stop_loss


@when('ATR is calculated')
def calculate_atr(test_context):
    """Calculate ATR using ATRBasedRiskManager"""
    config = test_context['unified_config']
    market_data = test_context['market_data']

    # Create risk manager
    risk_manager = ATRBasedRiskManager(config)

    # Calculate ATR
    atr = risk_manager._calculate_atr(market_data)
    #open issues: this is using SMA instead of EMA
    #predefined ATR is not used properly, the value on row 2 is lower than on 1 and 3
    test_context['calculated_atr'] = atr


@when('risk reduction check is performed')
def perform_risk_reduction_check(test_context):
    """Check if risk reduction is needed"""
    config = test_context['unified_config']
    portfolio = test_context['portfolio']

    risk_manager = ATRBasedRiskManager(config)
    should_reduce = risk_manager.should_reduce_risk(portfolio)

    test_context['risk_reduction_result'] = should_reduce

# ============================================================================
# THEN STEPS - Assertions
# ============================================================================

@then(parsers.parse('stop loss should be {expected_stop_loss}'))
def verify_stop_loss(test_context, expected_stop_loss):
    """Verify calculated stop loss matches expected value"""
    config = test_context['unified_config']
    risk_config = config.get_section('money_management')['risk_managers']['atr_based']
    tolerance = risk_config['calculation_tolerance']

    calculated = test_context['calculated_stop_loss']
    expected = float(expected_stop_loss)

    assert abs(calculated - expected) < tolerance, \
        f"Stop loss {calculated:.2f} does not match expected {expected:.2f} (tolerance: {tolerance})"


@then(parsers.parse('ATR should be {expected_atr}'))
def verify_atr_value(test_context, expected_atr):
    """Verify calculated ATR matches expected value"""
    config = test_context['unified_config']
    risk_config = config.get_section('money_management')['risk_managers']['atr_based']
    tolerance = risk_config['calculation_tolerance']

    calculated = test_context['calculated_atr']
    expected = float(expected_atr)

    assert abs(calculated - expected) < tolerance, \
        f"ATR {calculated:.6f} does not match expected {expected:.6f} (tolerance: {tolerance})"


@then(parsers.parse('risk reduction should be {expected_result}'))
def verify_risk_reduction_result(test_context, expected_result):
    """Verify risk reduction result matches expected"""
    actual = test_context['risk_reduction_result']
    expected = (expected_result == 'triggered')

    assert actual == expected, \
        f"Expected risk reduction to be {expected}, got {actual}"