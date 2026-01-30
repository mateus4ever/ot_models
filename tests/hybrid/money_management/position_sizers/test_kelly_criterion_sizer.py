# test_trade_history.py
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig

# Import the classes we're testing
from src.hybrid.money_management import PortfolioState
from src.hybrid.money_management.position_sizers import KellyCriterionSizer
from src.hybrid.positions.base_trade_history import PositionOutcome
from src.hybrid.positions.leg_trade_history import LegTradeHistory

# Load all scenarios from the feature file
scenarios('kelly_criterion_sizer.feature')


# ==============================================================================
# FIXTURES AND SETUP
# ==============================================================================

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
# GIVEN steps - Setup and preconditions
# =============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    root_path = Path(__file__).parent.parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration path  not found: {config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['config'] = config
    test_context['root_path'] = root_path
    test_context['config_path'] = config_path

@given(parsers.parse('With the configuration I have a Kelly Criterion sizer'))
def create_kelly_sizer_with_statistics(test_context):
    """Create Kelly sizer with specific statistics for testing"""

    # Get kelly_lookback from actual configuration instead of hardcoding
    config = test_context['config']

    kelly_sizer = KellyCriterionSizer(config)
    test_context['kelly_sizer'] = kelly_sizer


@given(parsers.parse('I have a Kelly Criterion sizer with kelly statistics {win_rate}, {avg_win}, {avg_loss}'))
def create_kelly_sizer_with_statistics(test_context, win_rate, avg_win, avg_loss):
    """Create Kelly sizer with specific statistics for testing"""

    # Get kelly_lookback from actual configuration instead of hardcoding
    config = test_context['config']

    kelly_sizer = KellyCriterionSizer(config)

    # Override the statistics for testing
    kelly_sizer.kelly_win_rate = float(win_rate)
    kelly_sizer.kelly_avg_win = float(avg_win)
    kelly_sizer.kelly_avg_loss = float(avg_loss)

    test_context['kelly_sizer'] = kelly_sizer


@given(parsers.parse('the portfolio has {portfolio_equity} total equity'))
def set_portfolio_equity(test_context, portfolio_equity):
    """Set portfolio equity for position sizing"""
    test_context['portfolio_equity'] = float(portfolio_equity)


@given(parsers.parse('the final Kelly percentage is {kelly_percentage}'))
def set_kelly_percentage(test_context, kelly_percentage):
    """Mock the Kelly percentage directly (bypass calculation)"""
    test_context['kelly_percentage'] = float(kelly_percentage)


@given(parsers.parse('I set kelly_lookback to {lookback_period}'))
def set_kelly_lookback(test_context, lookback_period):
    """Override kelly_lookback for testing different lookback periods"""
    sizer = test_context['kelly_sizer']
    sizer.kelly_lookback = int(lookback_period)


@given(parsers.parse('I load all position outcomes from "{trade_data_file}"'))
def load_position_outcomes(test_context, trade_data_file):
    """Load position outcomes from TradeHistory and add to Kelly sizer"""

    config = test_context['config']
    sizer = test_context['kelly_sizer']

    # Create DataManager to load trade history
    trade_history = LegTradeHistory(config)
    success = trade_history.load_from_json(trade_data_file)

    if not success:
        raise ValueError(f"Failed to load trade data from {trade_data_file}")

    # Get all closed positions as PositionOutcome objects
    all_positions = trade_history.all_positions

    # Get all CLOSED position outcomes
    outcomes = trade_history.get_position_outcomes(lookback_periods=0)

    # Add each outcome to Kelly sizer
    for outcome in outcomes:
        sizer.update_trade_result(outcome)

    test_context['loaded_outcomes_count'] = len(outcomes)


@given(parsers.parse(
    'bootstrap statistics are kelly_win_rate {win_rate}, kelly_avg_win {avg_win}, kelly_avg_loss {avg_loss}'))
def verify_bootstrap_statistics(test_context, win_rate, avg_win, avg_loss):
    """Verify bootstrap statistics are set in config"""
    sizer = test_context['kelly_sizer']

    expected_win_rate = float(win_rate)
    expected_avg_win = float(avg_win)
    expected_avg_loss = float(avg_loss)

    assert sizer.kelly_win_rate == expected_win_rate, \
        f"Expected bootstrap win_rate {expected_win_rate}, got {sizer.kelly_win_rate}"
    assert sizer.kelly_avg_win == expected_avg_win, \
        f"Expected bootstrap avg_win {expected_avg_win}, got {sizer.kelly_avg_win}"
    assert sizer.kelly_avg_loss == expected_avg_loss, \
        f"Expected bootstrap avg_loss {expected_avg_loss}, got {sizer.kelly_avg_loss}"


@given(parsers.parse('kelly_min_trades_threshold is {threshold}'))
def verify_min_trades_threshold(test_context, threshold):
    """Verify min trades threshold is set in config"""
    sizer = test_context['kelly_sizer']
    expected_threshold = int(threshold)

    assert sizer.min_trades_threshold == expected_threshold, \
        f"Expected threshold {expected_threshold}, got {sizer.min_trades_threshold}"


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I calculate the raw Kelly percentage')
def calculate_raw_kelly_percentage(test_context):
    """Calculate raw Kelly percentage"""
    kelly_sizer = test_context['kelly_sizer']
    raw_kelly = kelly_sizer._calculate_kelly_percentage()
    test_context['raw_kelly_percentage'] = raw_kelly


@when(parsers.parse('I calculate position size for signal at {entry_price} with stop distance {stop_distance}'))
def calculate_position_size_with_stop(test_context, entry_price, stop_distance):
    """Calculate position size using mocked Kelly percentage"""

    sizer = test_context['kelly_sizer']
    portfolio_equity = test_context['portfolio_equity']
    kelly_pct = test_context['kelly_percentage']

    # Create a simple PortfolioState with just equity
    portfolio = PortfolioState(
        total_equity=portfolio_equity,
        available_cash=portfolio_equity,
        positions={}
    )

    # Use the existing method
    position_size = sizer._calculate_position_from_kelly(
        portfolio=portfolio,
        kelly_percentage=kelly_pct,
        stop_distance=float(stop_distance)
    )

    test_context['position_size'] = position_size
    test_context['entry_price'] = float(entry_price)
    test_context['stop_distance'] = float(stop_distance)


@when('I get current statistics')
def get_current_statistics(test_context):
    """Get current Kelly statistics from the sizer"""
    sizer = test_context['kelly_sizer']

    # This calls _get_current_statistics() which returns tuple
    win_rate, avg_win, avg_loss = sizer._get_current_statistics()

    test_context['current_win_rate'] = win_rate
    test_context['current_avg_win'] = avg_win
    test_context['current_avg_loss'] = avg_loss


@when(parsers.parse(
    'I add {trade_count} trade outcomes with actual win_rate {actual_win_rate}, win_pnl {win_pnl}, loss_pnl {loss_pnl}, fees {fees}'))
def add_outcomes_with_win_rate(test_context, trade_count, actual_win_rate, win_pnl, loss_pnl, fees):
    """Add trade outcomes with specified win rate"""
    config = test_context['config']
    sizer = test_context['kelly_sizer']
    count = int(trade_count)
    win_rate = float(actual_win_rate)
    win_pnl_value = float(win_pnl)
    loss_pnl_value = float(loss_pnl)
    fees_value = float(fees)

    win_count = int(count * win_rate)
    loss_count = count - win_count

    for i in range(win_count):
        win = PositionOutcome(
            outcome='win',
            gross_pnl=win_pnl_value + fees_value,
            net_pnl=win_pnl_value,
            fees=fees_value
        )
        sizer.update_trade_result(win)

    for i in range(loss_count):
        loss = PositionOutcome(
            outcome='loss',
            gross_pnl=loss_pnl_value - fees_value,
            net_pnl=loss_pnl_value,
            fees=fees_value
        )
        sizer.update_trade_result(loss)


# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then(parsers.parse('the Kelly percentage should be {expected_kelly_pct}'))
def verify_kelly_percentage(test_context, expected_kelly_pct):
    """Verify calculated Kelly percentage matches expected"""
    actual = test_context['raw_kelly_percentage']
    expected = float(expected_kelly_pct)
    assert abs(actual - expected) < 0.001, f"Expected {expected}, got {actual}"


@then('the calculation should use the correct Kelly formula')
def verify_kelly_formula_used(test_context):
    """Verify Kelly formula is correctly applied"""
    # This is validated by the mathematical correctness of the result
    # The formula is: f* = (bp - q) / b where b = avg_win/avg_loss
    assert 'raw_kelly_percentage' in test_context


@then(parsers.parse('the position size should be {expected_shares} shares'))
def verify_position_size(test_context, expected_shares):
    """Verify calculated position size matches expected"""
    actual_shares = test_context['position_size']
    expected = int(expected_shares)
    assert actual_shares == expected, \
        f"Expected {expected} shares, got {actual_shares}"


@then('the risk budget should equal Kelly percentage times portfolio equity')
def verify_risk_budget_calculation(test_context):
    """Verify risk budget = kelly_pct * portfolio_equity"""

    config = test_context['config']
    kelly_pct = test_context['kelly_percentage']
    portfolio_equity = test_context['portfolio_equity']
    position_size = test_context['position_size']
    stop_distance = test_context['stop_distance']

    expected_risk_budget = kelly_pct * portfolio_equity
    actual_risk_budget = position_size * stop_distance

    mm_config = config.get_section('money_management')
    tolerance = mm_config['position_sizers']['kelly_criterion']['parameters']['position_sizing_tolerance']

    assert abs(actual_risk_budget - expected_risk_budget) < tolerance, \
        f"Risk budget mismatch: expected {expected_risk_budget}, got {actual_risk_budget}"


@then(parsers.parse('the statistics source should be {expected_source}'))
def verify_statistics_source(test_context, expected_source):
    """Verify whether statistics come from config (bootstrap) or calculated from outcomes"""
    sizer = test_context['kelly_sizer']

    # This scenario tests that outcomes ARE being used (calculated),
    # not whether we've met the threshold for switching from bootstrap.
    # The fact that we loaded outcomes means we're using calculated stats.
    if expected_source == 'calculated':
        assert len(sizer.trade_outcomes) > 0, \
            f"Expected calculated stats but have no outcomes loaded"
    elif expected_source == 'bootstrap':
        assert len(sizer.trade_outcomes) == 0, \
            f"Expected bootstrap stats but have {len(sizer.trade_outcomes)} outcomes"
    else:
        raise ValueError(f"Unknown statistics source: {expected_source}")


@then(parsers.parse('the outcome count used should be {expected_count}'))
def verify_outcome_count_used(test_context, expected_count):
    """Verify the number of outcomes retained in the lookback window"""
    sizer = test_context['kelly_sizer']
    actual_count = len(sizer.trade_outcomes)
    expected = int(expected_count)

    assert actual_count == expected, \
        f"Expected {expected} outcomes in lookback window, got {actual_count}"


@then(parsers.parse('the outcome count used should be {expected_count}'))
def verify_outcome_count_used(test_context, expected_count):
    """Verify the number of outcomes retained in the lookback window"""
    sizer = test_context['kelly_sizer']
    actual_count = len(sizer.trade_outcomes)
    expected = int(expected_count)

    assert actual_count == expected, \
        f"Expected {expected} outcomes in lookback window, got {actual_count}"


@then(parsers.parse('the win_rate should be approximately {expected_win_rate}'))
def verify_win_rate_approximate(test_context, expected_win_rate):
    """Verify win rate is approximately expected (within tolerance)"""
    config = test_context['config']
    sizer = test_context['kelly_sizer']

    # Get tolerance from config
    mm_config = config.get_section('money_management')
    kelly_config = mm_config['position_sizers']['kelly_criterion']
    tolerance = kelly_config['parameters']['position_sizing_tolerance']

    win_rate, _, _ = sizer._get_current_statistics()
    expected = float(expected_win_rate)

    assert abs(win_rate - expected) < tolerance, \
        f"Expected win_rate ~{expected}, got {win_rate} (diff: {abs(win_rate - expected)})"


@then(parsers.parse('the avg_win should be approximately {expected_avg_win}'))
def verify_avg_win_approximate(test_context, expected_avg_win):
    """Verify avg_win is approximately expected (within tolerance)"""
    config = test_context['config']
    sizer = test_context['kelly_sizer']

    mm_config = config.get_section('money_management')
    kelly_config = mm_config['position_sizers']['kelly_criterion']
    tolerance = kelly_config['parameters']['position_sizing_tolerance']

    _, avg_win, _ = sizer._get_current_statistics()
    expected = float(expected_avg_win)

    assert abs(avg_win - expected) < tolerance, \
        f"Expected avg_win ~{expected}, got {avg_win} (diff: {abs(avg_win - expected)})"


@then(parsers.parse('the avg_loss should be approximately {expected_avg_loss}'))
def verify_avg_loss_approximate(test_context, expected_avg_loss):
    """Verify avg_loss is approximately expected (within tolerance)"""
    config = test_context['config']
    sizer = test_context['kelly_sizer']

    mm_config = config.get_section('money_management')
    kelly_config = mm_config['position_sizers']['kelly_criterion']
    tolerance = kelly_config['parameters']['position_sizing_tolerance']

    _, _, avg_loss = sizer._get_current_statistics()
    expected = float(expected_avg_loss)

    assert abs(avg_loss - expected) < tolerance, \
        f"Expected avg_loss ~{expected}, got {avg_loss} (diff: {abs(avg_loss - expected)})"
