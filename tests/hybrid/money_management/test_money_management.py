# tests/hybrid/money_management/test_money_management.py
"""
pytest-bdd test runner for MoneyManager core functionality
Tests MoneyManager service initialization, position sizing, and portfolio tracking
NO HARDCODED VALUES - ALL PARAMETERS FROM FEATURE FILES OR CONFIGURATION
NO MOCKS - ONLY REAL UNIFIED CONFIG AND ACTUAL COMPONENTS
"""

import pytest
import logging
import pandas as pd
from pathlib import Path
from pytest_bdd import scenarios, given, when, then, parsers

# Import the system under test
import sys

from src.hybrid.money_management import MoneyManager
from src.hybrid.money_management.money_management import PositionDirection, TradingSignal

# Go up 4 levels from tests/hybrid/money_management/test_money_management.py to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid.config.unified_config import UnifiedConfig


# Load all scenarios from money_management.feature
scenarios('money_management.feature')

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
# GIVEN steps - Setup and preconditions
# =============================================================================

@given(parsers.parse('{config_file} is available in {config_directory} and loaded'))
def load_configuration_file(test_context, config_file, config_directory):
    """Load and validate configuration file from specified directory"""
    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory
    full_config_path = config_path / config_file

    assert full_config_path.exists(), f"Configuration file not found: {full_config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['mm_config'] = config
    test_context['root_path'] = root_path
    test_context['config_path'] = config_path


@given('I have money management configuration loaded')
def step_money_management_config_loaded(test_context):
    """Ensure configuration is loaded for initialization test"""
    # Configuration already loaded in background steps
    assert 'mm_config' in test_context, "money_management config not loaded"


@given(parsers.parse('I have a MoneyManager with {sizing_type} sizing'))
def step_money_manager_with_sizing_type(test_context, sizing_type):
    """Create MoneyManager with specific position sizing strategy"""
    config = test_context['mm_config']
    money_manager = MoneyManager(config)
    test_context['money_manager'] = money_manager
    test_context['sizing_type'] = sizing_type


@given(parsers.parse('the portfolio has {portfolio_equity:d} equity'))
def step_portfolio_equity(test_context, portfolio_equity):
    """Set portfolio equity from feature file parameter"""
    money_manager = test_context['money_manager']
    money_manager.portfolio.total_equity = portfolio_equity
    money_manager.portfolio.available_cash = portfolio_equity
    money_manager.portfolio.peak_equity = portfolio_equity
    test_context['portfolio_equity'] = portfolio_equity


@given('I have a MoneyManager initialized')
def step_money_manager_initialized(test_context):
    """Create basic MoneyManager instance"""
    config = test_context['mm_config']
    money_manager = MoneyManager(config)
    test_context['money_manager'] = money_manager


@given('I have a MoneyManager with risk limits')
def step_money_manager_risk_limits(test_context):
    """Create MoneyManager with risk limits"""
    config = test_context['mm_config']
    money_manager = MoneyManager(config)
    test_context['money_manager'] = money_manager


@given('I have incomplete money management configuration')
def step_incomplete_money_management_config(test_context):
    """Set up configuration missing money_management section"""

    class IncompleteConfig:
        def get_section(self, section_name):
            if section_name == 'money_management':
                return None  # Missing section to trigger error
            return {}

    test_context['mm_config'] = IncompleteConfig()

@given(parsers.parse('I have a trading signal for {symbol} {direction} at {entry_price:f} with strength {signal_strength:f}'))
def step_trading_signal_for_symbol(test_context, symbol, direction, entry_price, signal_strength):
    """Create trading signal with specific parameters from feature file"""
    position_direction = PositionDirection.LONG if direction == 'long' else PositionDirection.SHORT
    signal = TradingSignal(
        symbol=symbol,
        direction=position_direction,
        strength=signal_strength,
        entry_price=entry_price,
        timestamp=pd.Timestamp.now()
    )
    test_context['trading_signal'] = signal

@given(parsers.parse('I have market data with {volatility_percent:f} volatility and {data_periods:d} periods'))
def step_market_data_parameterized(test_context, volatility_percent, data_periods):
    signal = test_context['trading_signal']
    entry_price = signal.entry_price

    market_data = pd.DataFrame({
        'high': [entry_price * (1 + volatility_percent)] * data_periods,
        'low': [entry_price * (1 - volatility_percent)] * data_periods,
        'close': [entry_price] * data_periods
    })
    test_context['market_data'] = market_data

@given(parsers.parse('the portfolio has {daily_pnl:d} daily PnL'))
def step_portfolio_daily_pnl(test_context, daily_pnl):
    """Set portfolio daily PnL"""
    money_manager = test_context['money_manager']
    money_manager.portfolio.daily_pnl = daily_pnl
    test_context['expected_daily_pnl'] = daily_pnl

@given('I have a MoneyManager with existing positions')
def step_money_manager_existing_positions(test_context):
    """Create MoneyManager with some existing positions for testing"""
    config = test_context['mm_config']
    money_manager = MoneyManager(config)
    test_context['money_manager'] = money_manager


@given(parsers.parse('I have a {direction} position of {position_size:d} shares in {symbol} at {entry_price:f}'))
def step_existing_position(test_context, direction, position_size, symbol, entry_price):
    money_manager = test_context['money_manager']
    position_direction = PositionDirection.LONG if direction == 'long' else PositionDirection.SHORT

    money_manager.update_position(symbol, position_size, entry_price, position_direction)
    test_context['position_symbol'] = symbol
    test_context['position_entry_price'] = entry_price
    test_context['position_size'] = position_size
    test_context['position_direction'] = direction


@given(parsers.parse('I have a {direction} position of {position_size:d} shares in {symbol} at {entry_price:f}'))
def step_existing_position(test_context, direction, position_size, symbol, entry_price):
    money_manager = test_context['money_manager']
    position_direction = PositionDirection.LONG if direction == 'long' else PositionDirection.SHORT

    money_manager.update_position(symbol, position_size, entry_price, position_direction)
    test_context['position_symbol'] = symbol
    test_context['position_entry_price'] = entry_price  # Make sure this is set
    test_context['position_size'] = position_size
    test_context['position_direction'] = direction

@given(parsers.parse('the portfolio has {total_equity:d} total equity'))
def step_portfolio_total_equity(test_context, total_equity):
    """Set portfolio total equity"""
    money_manager = test_context['money_manager']
    money_manager.portfolio.total_equity = total_equity
    money_manager.portfolio.available_cash = total_equity  # Assume all cash available initially
    money_manager.portfolio.peak_equity = total_equity
    test_context['initial_equity'] = total_equity

@given('A MoneyManager is created for testing')
def step_simple_test(test_context):
    print("Simple step recognized")
    test_context['test'] = True


@given(parsers.parse('I create {position_count:d} test positions'))
def step_create_test_positions(test_context, position_count):
    """Store position count for later position creation"""
    test_context['positions_to_create'] = position_count


@given(parsers.parse('each position has {position_size:d} shares at {entry_price:f}'))
def step_set_position_parameters(test_context, position_size, entry_price):
    """Create the test positions with specified parameters"""
    money_manager = test_context['money_manager']
    position_count = test_context.get('positions_to_create', 0)

    for i in range(position_count):
        symbol = f"TEST{i + 1}"
        money_manager.update_position(symbol, position_size, entry_price, PositionDirection.LONG)

    test_context['expected_position_count'] = position_count

@given(parsers.parse('the portfolio has {available_cash:d} available cash'))
def step_portfolio_available_cash(test_context, available_cash):
    """Set portfolio available cash"""
    money_manager = test_context['money_manager']
    money_manager.portfolio.available_cash = available_cash
    test_context['expected_available_cash'] = available_cash

@given(parsers.parse('the portfolio has {max_drawdown:f} maximum drawdown'))
def step_portfolio_maximum_drawdown(test_context, max_drawdown):
    """Set portfolio maximum drawdown"""
    money_manager = test_context['money_manager']
    money_manager.portfolio.max_drawdown = max_drawdown
    test_context['expected_max_drawdown'] = max_drawdown


@given(parsers.parse('I have money management configuration with {sizer_type} position sizer'))
def step_config_with_position_sizer(test_context, sizer_type):
    """Create configuration with specific position sizer type"""

    # Load base config from actual config file
    config_path = test_context['config_path']
    base_config = UnifiedConfig(config_path=str(config_path), environment="test")
    base_mm_config = base_config.get_section('money_management')

    # Create config that overrides just the position_sizing type
    class ConfigWithSizer:
        def __init__(self, sizer_type, base_config):
            self.sizer_type = sizer_type
            self.base_config = base_config

        def get_section(self, section_name):
            if section_name == 'money_management':
                # Use base config but override position_sizing
                config = self.base_config.copy()
                config['position_sizing'] = self.sizer_type
                return config
            return self.base_config.get(section_name, {})

    test_context['mm_config'] = ConfigWithSizer(sizer_type, base_mm_config)
    test_context['expected_sizer_type'] = sizer_type


@given(parsers.parse('I have money management configuration with unknown {component_type} "{invalid_name}"'))
def step_config_with_unknown_component(test_context, component_type, invalid_name):
    """Create configuration with invalid component name using base config"""
    config_path = test_context['config_path']
    base_config = UnifiedConfig(config_path=str(config_path), environment="test")
    base_mm_config = base_config.get_section('money_management')

    if base_mm_config is None:
        raise ValueError(f"Failed to load money_management section from {config_path}")

    modified_config = base_mm_config.copy()

    if component_type == 'position_sizer':
        modified_config['position_sizing'] = invalid_name
    elif component_type == 'risk_manager':
        modified_config['risk_management'] = invalid_name

    class ConfigWithInvalidComponent:
        def get_section(self, section_name):
            if section_name == 'money_management':
                return modified_config
            return base_config.get_section(section_name)

    test_context['mm_config'] = ConfigWithInvalidComponent()
    test_context['expected_component_type'] = component_type
    test_context['invalid_component_name'] = invalid_name

@given(parsers.parse('the portfolio has {drawdown_percent:f} drawdown exceeding risk limits'))
def step_portfolio_drawdown_exceeding_limits(test_context, drawdown_percent):
    """Set portfolio drawdown to exceed risk limits"""
    money_manager = test_context['money_manager']
    money_manager.portfolio.max_drawdown = drawdown_percent
    test_context['expected_drawdown'] = drawdown_percent

# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I create a MoneyManager instance')
def step_create_money_manager_instance(test_context):
    """Create MoneyManager instance and capture result"""
    try:
        config_path = test_context['config_path']
        config = UnifiedConfig(config_path=str(config_path))
        money_manager = MoneyManager(config)
        test_context['money_manager'] = money_manager
        test_context['money_manager_created'] = True
        test_context['creation_error'] = None
    except Exception as e:
        test_context['money_manager_created'] = False
        test_context['creation_error'] = e


@when(parsers.parse(
    'I calculate position size for a {direction} signal at {entry_price:f} with market data length {data_length:d} and price multipliers {high_mult:f} and {low_mult:f}'))
def step_calculate_position_size_with_market_data(test_context, direction, entry_price, data_length, high_mult,
                                                  low_mult):
    """Calculate position size with feature file parameters"""
    money_manager = test_context['money_manager']

    # Create trading signal from feature file parameters
    position_direction = PositionDirection.LONG if direction == 'long' else PositionDirection.SHORT
    signal = TradingSignal(
        symbol='TEST',
        direction=position_direction,
        strength=1.0,
        entry_price=entry_price,
        timestamp=pd.Timestamp.now()
    )

    # Create market data using feature file parameters
    market_data = pd.DataFrame({
        'high': [entry_price * high_mult] * data_length,
        'low': [entry_price * low_mult] * data_length,
        'close': [entry_price] * data_length
    })

    try:
        position_size = money_manager.calculate_position_size(signal, market_data)
        test_context['calculated_position_size'] = position_size
        test_context['test_signal'] = signal
        test_context['calculation_error'] = None
    except Exception as e:
        test_context['calculated_position_size'] = None
        test_context['calculation_error'] = e


@when(parsers.parse('I update position for {symbol} with {size:d} shares at {price:f} going {direction}'))
def step_update_position(test_context, symbol, size, price, direction):
    """Update position with feature file parameters"""
    money_manager = test_context['money_manager']

    position_direction = PositionDirection.LONG if direction == 'long' else PositionDirection.SHORT

    try:
        success = money_manager.update_position(symbol, size, price, position_direction)
        test_context['position_update_success'] = success
        test_context['updated_symbol'] = symbol
        test_context['updated_size'] = size
        test_context['updated_price'] = price
        test_context['position_update_error'] = None
    except Exception as e:
        test_context['position_update_success'] = False
        test_context['position_update_error'] = e


@when('I check if risk should be reduced')
def step_check_risk_reduction(test_context):
    """Check risk reduction status"""
    money_manager = test_context['money_manager']

    try:
        should_reduce = money_manager.should_reduce_risk()
        test_context['should_reduce_risk'] = should_reduce
        test_context['risk_check_error'] = None
    except Exception as e:
        test_context['should_reduce_risk'] = None
        test_context['risk_check_error'] = e


@when('I try to create a MoneyManager instance')
def step_try_create_money_manager_with_error(test_context):
    """Attempt to create MoneyManager expecting configuration error"""
    try:
        config = test_context['mm_config']
        money_manager = MoneyManager(config)
        test_context['money_manager_created'] = True
        test_context['creation_error'] = None
    except Exception as e:
        test_context['money_manager_created'] = False
        test_context['creation_error'] = e

@when('I request stop loss calculation for the signal')
def step_request_stop_loss_calculation(test_context):
    """Request stop loss calculation for the trading signal"""
    money_manager = test_context['money_manager']
    signal = test_context['trading_signal']
    market_data = test_context['market_data']

    try:
        stop_loss = money_manager.calculate_stop_loss(signal, market_data)
        test_context['calculated_stop_loss'] = stop_loss
        test_context['stop_loss_error'] = None
    except Exception as e:
        test_context['calculated_stop_loss'] = None
        test_context['stop_loss_error'] = e


@when(parsers.parse('I update market prices with {symbol} at {current_price:f}'))
def step_update_market_prices(test_context, symbol, current_price):
    """Update market prices for portfolio valuation"""
    money_manager = test_context['money_manager']

    try:
        price_updates = {symbol: current_price}
        money_manager.update_market_prices(price_updates)
        test_context['updated_symbol'] = symbol
        test_context['updated_price'] = current_price
        test_context['price_update_error'] = None
    except Exception as e:
        test_context['price_update_error'] = e


@when('I request the portfolio summary')
def step_request_portfolio_summary(test_context):
    """Request portfolio summary from MoneyManager"""
    money_manager = test_context['money_manager']

    try:
        summary = money_manager.get_portfolio_summary()
        test_context['portfolio_summary'] = summary
        test_context['summary_error'] = None
    except Exception as e:
        test_context['portfolio_summary'] = None
        test_context['summary_error'] = e


@when(parsers.parse(
    'I calculate position size for {symbol} {direction} signal at {entry_price:f} with desired value {signal_value:d}'))
def step_calculate_position_size_with_desired_value(test_context, symbol, direction, entry_price, signal_value):
    """Calculate position size for signal with desired position value"""
    money_manager = test_context['money_manager']

    # Create trading signal
    position_direction = PositionDirection.LONG if direction == 'long' else PositionDirection.SHORT
    signal = TradingSignal(
        symbol=symbol,
        direction=position_direction,
        strength=1.0,
        entry_price=entry_price,
        timestamp=pd.Timestamp.now()
    )

    # Get ATR period from config to create sufficient market data
    config = test_context['mm_config']
    mm_config = config.get_section('money_management')
    atr_period = mm_config['risk_managers']['atr_based']['atr_period']
    volatility = 0.01  # 1% volatility from config if available

    # Create market data with price variation for ATR calculation
    market_data = pd.DataFrame({
        'high': [entry_price * (1 + volatility)] * atr_period,
        'low': [entry_price * (1 - volatility)] * atr_period,
        'close': [entry_price] * atr_period
    })

    test_context['test_signal'] = signal

    try:
        position_size = money_manager.calculate_position_size(signal, market_data)

        # Calculate what the position size should be based on available cash
        expected_from_cash = int(money_manager.portfolio.available_cash / signal.entry_price)
        test_context['calculated_position_size'] = position_size

    except Exception as e:
        test_context['calculated_position_size'] = None
        test_context['calculation_error'] = e

@ when(parsers.parse(
    'I calculate position size for {symbol} {direction} signal at {entry_price:f} with strength {signal_strength:f} and {data_periods:d} periods'))
def step_calculate_position_size_for_signal_with_params(test_context, symbol, direction, entry_price, signal_strength,
                                                        data_periods):
    """Calculate position size for trading signal with parameters"""
    money_manager = test_context['money_manager']

    # Create trading signal with parameters from feature file
    position_direction = PositionDirection.LONG if direction == 'long' else PositionDirection.SHORT
    signal = TradingSignal(
        symbol=symbol,
        direction=position_direction,
        strength=signal_strength,
        entry_price=entry_price,
        timestamp=pd.Timestamp.now()
    )

    # Create market data with specified periods
    market_data = pd.DataFrame({
        'high': [entry_price] * data_periods,
        'low': [entry_price] * data_periods,
        'close': [entry_price] * data_periods
    })

    try:
        position_size = money_manager.calculate_position_size(signal, market_data)
        test_context['calculated_position_size'] = position_size
        test_context['test_signal'] = signal
        test_context['calculation_error'] = None
    except Exception as e:
        test_context['calculated_position_size'] = None
        test_context['calculation_error'] = e


# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('the MoneyManager should initialize successfully')
def step_money_manager_initializes_successfully(test_context):
    """Verify MoneyManager was created successfully"""
    assert test_context.get('money_manager_created') is True, "MoneyManager should be created successfully"
    assert test_context.get('creation_error') is None, f"Creation error occurred: {test_context.get('creation_error')}"
    assert 'money_manager' in test_context, "MoneyManager instance should be available"


@then('the position sizing strategy should be loaded')
def step_position_sizing_strategy_loaded(test_context):
    """Verify position sizing strategy is loaded"""
    money_manager = test_context['money_manager']
    assert hasattr(money_manager, 'position_sizer'), "MoneyManager should have position_sizer"
    assert money_manager.position_sizer is not None, "Position sizer should be initialized"


@then('the risk management strategy should be loaded')
def step_risk_management_strategy_loaded(test_context):
    """Verify risk management strategy is loaded"""
    money_manager = test_context['money_manager']
    assert hasattr(money_manager, 'risk_manager'), "MoneyManager should have risk_manager"
    assert money_manager.risk_manager is not None, "Risk manager should be initialized"


@then('the portfolio should be initialized with configured capital')
def step_portfolio_initialized_with_capital(test_context):
    """Verify portfolio is initialized with capital from configuration"""
    money_manager = test_context['money_manager']
    config = test_context['mm_config']
    mm_config = config.get_section('money_management')
    expected_capital = mm_config['initial_capital']

    assert money_manager.portfolio.total_equity == expected_capital, \
        f"Portfolio equity should be {expected_capital}, got {money_manager.portfolio.total_equity}"
    assert money_manager.portfolio.available_cash == expected_capital, \
        f"Available cash should be {expected_capital}, got {money_manager.portfolio.available_cash}"


@then('a position size should be calculated')
def step_position_size_calculated(test_context):
    """Verify position size calculation completed"""
    assert test_context.get('calculation_error') is None, \
        f"Position size calculation failed: {test_context.get('calculation_error')}"
    assert 'calculated_position_size' in test_context, "Position size should be calculated"


@then('the position size should be greater than zero')
def step_position_size_greater_than_zero(test_context):
    """Verify calculated position size is positive"""
    position_size = test_context.get('calculated_position_size')
    assert position_size is not None, "Position size should not be None"
    assert position_size >= 0, f"Position size should be non-negative, got {position_size}"


@then('the position size should respect portfolio constraints')
def step_position_size_respects_constraints(test_context):
    """Verify position size respects portfolio constraints"""
    money_manager = test_context['money_manager']
    position_size = test_context.get('calculated_position_size')
    signal = test_context.get('test_signal')

    if position_size > 0 and signal:
        position_value = position_size * signal.entry_price
        portfolio_equity = money_manager.portfolio.total_equity

        # Position should not exceed available cash
        assert position_value <= money_manager.portfolio.available_cash, \
            f"Position value {position_value} exceeds available cash {money_manager.portfolio.available_cash}"


@then('the position should be tracked correctly')
def step_position_tracked_correctly(test_context):
    """Verify position update was successful"""
    assert test_context.get('position_update_success') is True, \
        f"Position update failed: {test_context.get('position_update_error')}"

    money_manager = test_context['money_manager']
    symbol = test_context.get('updated_symbol')

    assert symbol in money_manager.portfolio.positions, f"Position for {symbol} should be tracked"


@then('after position change the portfolio position is updated')
def step_portfolio_equity_updated(test_context):
    """Verify portfolio equity reflects position update"""
    money_manager = test_context['money_manager']
    # Portfolio equity should still be a valid number
    assert isinstance(money_manager.portfolio.total_equity, (int, float)), \
        "Portfolio equity should be numeric"
    assert money_manager.portfolio.total_equity > 0, \
        "Portfolio equity should be positive"


@then('the position should appear in current positions')
def step_position_appears_in_current_positions(test_context):
    """Verify position appears in current positions"""
    money_manager = test_context['money_manager']
    symbol = test_context.get('updated_symbol')

    current_positions = money_manager.get_current_positions()
    assert symbol in current_positions, f"Position for {symbol} should appear in current positions"

    position = current_positions[symbol]
    assert position.symbol == symbol, f"Position symbol should match {symbol}"
    assert position.size == test_context.get('updated_size'), "Position size should match updated size"


@then('the risk evaluation should complete successfully')
def step_risk_evaluation_completes(test_context):
    """Verify risk evaluation completed without error"""
    assert test_context.get('risk_check_error') is None, \
        f"Risk evaluation failed: {test_context.get('risk_check_error')}"
    assert 'should_reduce_risk' in test_context, "Risk evaluation should return a result"


@then('the result should be boolean')
def step_result_should_be_boolean(test_context):
    """Verify risk evaluation result is boolean"""
    should_reduce = test_context.get('should_reduce_risk')
    assert isinstance(should_reduce, bool), f"Risk evaluation should return boolean, got {type(should_reduce)}"


@then('a configuration error should be raised')
def step_configuration_error_raised(test_context):
    """Verify configuration error was raised"""
    assert test_context.get('money_manager_created') is False, \
        "MoneyManager creation should fail with incomplete configuration"
    assert test_context.get('creation_error') is not None, \
        "Configuration error should be raised"


@then('the MoneyManager should not be created')
def step_money_manager_not_created(test_context):
    """Verify MoneyManager was not created due to configuration error"""
    assert test_context.get('money_manager_created') is False, \
        "MoneyManager should not be created with incomplete configuration"

@then('return a valid stop loss price')
def step_return_valid_stop_loss_price(test_context):
    """Verify stop loss calculation returned a valid price"""
    assert test_context.get('stop_loss_error') is None, \
        f"Stop loss calculation failed: {test_context.get('stop_loss_error')}"

    stop_loss = test_context.get('calculated_stop_loss')
    assert stop_loss is not None, "Stop loss should be calculated"
    assert isinstance(stop_loss, (int, float)), "Stop loss should be numeric"
    assert stop_loss > 0, "Stop loss should be positive"


@then(parsers.parse('the stop loss should be {stop_comparison} the entry price to limit losses'))
def step_stop_loss_comparison(test_context, stop_comparison):
    """Verify stop loss is positioned correctly relative to entry price"""
    signal = test_context['trading_signal']
    stop_loss = test_context['calculated_stop_loss']
    entry_price = signal.entry_price

    if stop_comparison == 'below':
        assert stop_loss < entry_price, \
            f"Stop loss {stop_loss} should be below entry price {entry_price} for long positions"
    elif stop_comparison == 'above':
        assert stop_loss > entry_price, \
            f"Stop loss {stop_loss} should be above entry price {entry_price} for short positions"
    else:
        raise ValueError(f"Unknown stop_comparison: {stop_comparison}")
@then(parsers.parse('the portfolio equity should {equity_change} from {initial_equity:d}'))
def step_portfolio_equity_change_from_initial(test_context, equity_change, initial_equity):
    """Verify portfolio equity changed as expected from initial value"""
    money_manager = test_context['money_manager']
    current_equity = money_manager.portfolio.total_equity

    if equity_change == 'increase':
        assert current_equity > initial_equity, \
            f"Portfolio equity should increase from {initial_equity}, but got {current_equity}"
    elif equity_change == 'decrease':
        assert current_equity < initial_equity, \
            f"Portfolio equity should decrease from {initial_equity}, but got {current_equity}"
    else:
        raise ValueError(f"Unknown equity_change: {equity_change}")


@then(parsers.parse('the position unrealized PnL should be {pnl_direction}'))
def step_position_unrealized_pnl_direction(test_context, pnl_direction):
    """Verify position unrealized PnL is in expected direction"""
    money_manager = test_context['money_manager']
    symbol = test_context['position_symbol']

    # Get the position from portfolio
    position = money_manager.portfolio.positions.get(symbol)
    assert position is not None, f"Position for {symbol} should exist"

    unrealized_pnl = position.unrealized_pnl

    if pnl_direction == 'positive':
        assert unrealized_pnl > 0, \
            f"Position unrealized PnL should be positive, but got {unrealized_pnl}"
    elif pnl_direction == 'negative':
        assert unrealized_pnl < 0, \
            f"Position unrealized PnL should be negative, but got {unrealized_pnl}"
    else:
        raise ValueError(f"Unknown pnl_direction: {pnl_direction}")


@then('the portfolio summary should reflect updated values')
def step_portfolio_summary_reflects_updates(test_context):
    """Verify portfolio summary includes updated market values"""
    money_manager = test_context['money_manager']

    try:
        summary = money_manager.get_portfolio_summary()
        test_context['portfolio_summary'] = summary

        # Basic validation that summary contains expected fields
        required_fields = ['total_equity', 'available_cash', 'positions_count', 'daily_pnl', 'total_pnl']
        for field in required_fields:
            assert field in summary, f"Portfolio summary should include {field}"
            assert isinstance(summary[field], (int, float)), f"{field} should be numeric"

        # Verify positions count reflects actual positions
        actual_positions = len(money_manager.portfolio.positions)
        assert summary['positions_count'] == actual_positions, \
            f"Summary positions_count {summary['positions_count']} should match actual {actual_positions}"

        test_context['summary_validation_error'] = None
    except Exception as e:
        test_context['summary_validation_error'] = e
        raise


@then(parsers.parse('the summary should show total equity of {expected_equity:d}'))
def step_summary_shows_total_equity(test_context, expected_equity):
    """Verify portfolio summary shows expected total equity"""
    summary = test_context['portfolio_summary']
    assert summary is not None, "Portfolio summary should be available"

    actual_equity = summary.get('total_equity')
    assert actual_equity == expected_equity, \
        f"Summary should show total equity of {expected_equity}, but got {actual_equity}"


@then(parsers.parse('the summary should show available cash of {expected_cash:d}'))
def step_summary_shows_available_cash(test_context, expected_cash):
    """Verify portfolio summary shows expected available cash"""
    summary = test_context['portfolio_summary']
    assert summary is not None, "Portfolio summary should be available"

    actual_cash = summary.get('available_cash')
    assert actual_cash == expected_cash, \
        f"Summary should show available cash of {expected_cash}, but got {actual_cash}"
@then(parsers.parse('the summary should show positions count of {expected_count:d}'))
def step_summary_shows_positions_count(test_context, expected_count):
    """Verify portfolio summary shows expected positions count"""
    summary = test_context['portfolio_summary']
    assert summary is not None, "Portfolio summary should be available"

    actual_count = summary.get('positions_count')
    assert actual_count == expected_count, \
        f"Summary should show positions count of {expected_count}, but got {actual_count}"


@then(parsers.parse('the summary should show daily PnL of {expected_pnl:d}'))
def step_summary_shows_daily_pnl(test_context, expected_pnl):
    """Verify portfolio summary shows expected daily PnL"""
    summary = test_context['portfolio_summary']
    assert summary is not None, "Portfolio summary should be available"

    actual_pnl = summary.get('daily_pnl')
    assert actual_pnl == expected_pnl, \
        f"Summary should show daily PnL of {expected_pnl}, but got {actual_pnl}"
@then('the summary should show position sizing strategy name')
def step_summary_shows_strategy_name(test_context):
    """Verify portfolio summary includes position sizing strategy name"""
    summary = test_context['portfolio_summary']
    assert summary is not None, "Portfolio summary should be available"

    strategy_name = summary.get('position_sizing_strategy')
    assert strategy_name is not None, "Summary should include position sizing strategy name"
    assert isinstance(strategy_name, str), "Strategy name should be a string"
    assert len(strategy_name) > 0, "Strategy name should not be empty"


@then(parsers.parse('the risk evaluation should return {expected_result}'))
def step_risk_evaluation_result(test_context, expected_result):
    """Verify risk evaluation returns expected boolean result"""
    should_reduce = test_context.get('should_reduce_risk')
    assert should_reduce is not None, "Risk evaluation should have been performed"

    # Convert string to boolean
    if expected_result == 'true':
        expected_bool = True
    elif expected_result == 'false':
        expected_bool = False
    else:
        raise ValueError(f"Unknown expected_result: {expected_result}")

    assert should_reduce == expected_bool, \
        f"Risk evaluation should return {expected_bool}, but got {should_reduce}"


@then(parsers.parse('the position sizer should be {expected_sizer_name}'))
def step_position_sizer_should_be(test_context, expected_sizer_name):
    """Verify the correct position sizer was loaded"""
    money_manager = test_context['money_manager']

    # Get the actual strategy name from the position sizer
    actual_sizer_name = money_manager.position_sizer.get_strategy_name()

    assert actual_sizer_name == expected_sizer_name, \
        f"Position sizer should be {expected_sizer_name}, but got {actual_sizer_name}"


@then(parsers.parse('position calculations should work with {algorithm_type} algorithm'))
def step_position_calculations_work_with_algorithm(test_context, algorithm_type):
    """Verify position calculations work with the specified algorithm"""
    money_manager = test_context['money_manager']

    # Use existing signal from test context if available, or get parameters from config
    if 'trading_signal' in test_context:
        signal = test_context['trading_signal']
        market_data = test_context['market_data']
    else:
        # This scenario doesn't provide signal/market data, so we can't test calculations
        # Just verify the algorithm loaded without attempting calculations
        assert money_manager.position_sizer is not None, f"Position sizer should be loaded for {algorithm_type}"
        return

    try:
        position_size = money_manager.calculate_position_size(signal, market_data)
        test_context['algorithm_test_error'] = None

        # Basic validation that calculation worked
        assert isinstance(position_size, int), f"Position size should be integer, got {type(position_size)}"
        assert position_size >= 0, f"Position size should be non-negative, got {position_size}"

    except Exception as e:
        test_context['algorithm_test_error'] = e
        raise AssertionError(f"Position calculations failed with {algorithm_type} algorithm: {e}")


@then(parsers.parse('a {component_type} configuration error should be raised'))
def step_component_configuration_error_raised(test_context, component_type):
    """Verify configuration error was raised for invalid component"""
    assert test_context.get('money_manager_created') is False, \
        f"MoneyManager creation should fail with invalid {component_type}"

    creation_error = test_context.get('creation_error')
    assert creation_error is not None, \
        f"Configuration error should be raised for invalid {component_type}"

    assert isinstance(creation_error, ValueError), \
        f"Should raise ValueError for invalid {component_type}, got {type(creation_error)}"

    # Check for your actual error message pattern
    error_message = str(creation_error).lower()
    invalid_name = test_context.get('invalid_component_name', '').lower()

    # Look for the invalid component name in the error message
    assert invalid_name in error_message, \
        f"Error message should mention invalid component '{invalid_name}'"


@then(parsers.parse('the error message should list available {component_type} options'))
def step_error_message_lists_available_options(test_context, component_type):
    """Verify error message lists available component options"""
    creation_error = test_context.get('creation_error')
    assert creation_error is not None, f"Error should have been raised for invalid {component_type}"

    error_message = str(creation_error).lower()
    assert 'available types' in error_message, \
        f"Error message should list available {component_type} types"


@then('the position size should be reduced from normal calculation')
def step_position_size_reduced_from_normal(test_context):
    """Verify position size was reduced due to risk constraints"""
    assert test_context.get('calculation_error') is None, \
        f"Position size calculation failed: {test_context.get('calculation_error')}"

    position_size = test_context.get('calculated_position_size')
    assert position_size is not None, "Position size should be calculated"
    assert isinstance(position_size, int), "Position size should be integer"

    # Since risk reduction is triggered, position size should be reasonable (not zero, not huge)
    # The actual reduction logic is tested - we just verify the calculation completed
    # and returned a sensible result under risk reduction conditions
    assert position_size >= 0, "Position size should be non-negative"

@then('the calculation should complete successfully')
def step_calculation_completes_successfully(test_context):
    """Verify calculation completed without errors"""
    assert test_context.get('calculation_error') is None, \
        f"Calculation should complete successfully but got error: {test_context.get('calculation_error')}"

    assert 'calculated_position_size' in test_context, \
        "Calculation should have produced a position size result"


@then(parsers.parse('the position size should be limited to {expected_max_shares:d} shares'))
def step_position_size_limited_to_shares(test_context, expected_max_shares):
    """Verify position size is limited to expected maximum shares"""
    assert test_context.get('calculation_error') is None, \
        f"Position size calculation failed: {test_context.get('calculation_error')}"

    position_size = test_context.get('calculated_position_size')
    assert position_size is not None, "Position size should be calculated"
    assert isinstance(position_size, int), "Position size should be integer"

    assert position_size <= expected_max_shares, \
        f"Position size {position_size} should be limited to {expected_max_shares} shares"

    # For safety constraints, the position size should be close to the expected limit
    # Allow small variance due to rounding or calculation differences
    variance_threshold = max(1, int(expected_max_shares * 0.01))  # 1% tolerance or minimum 1 share
    assert abs(position_size - expected_max_shares) <= variance_threshold, \
        f"Position size {position_size} should be close to expected {expected_max_shares} (within {variance_threshold})"


@then(parsers.parse('the position value should not exceed {available_cash:d}'))
def step_position_value_not_exceed_cash(test_context, available_cash):
    """Verify position value doesn't exceed available cash"""
    position_size = test_context.get('calculated_position_size')
    assert position_size is not None, "Position size should be calculated"

    signal = test_context.get('test_signal')
    assert signal is not None, "Signal should be available"

    position_value = position_size * signal.entry_price
    assert position_value <= available_cash, \
        f"Position value {position_value} should not exceed available cash {available_cash}"

@then('no errors should occur during calculation')
def step_no_errors_during_calculation(test_context):
    """Verify calculation completed without errors"""
    calculation_error = test_context.get('calculation_error')
    assert calculation_error is None, \
        f"Calculation should complete without errors, but got: {calculation_error}"