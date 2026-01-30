# test_trade_history.py
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
# Import the classes we're testing
from src.hybrid.positions.leg_trade_history import LegTradeHistory

# Load all scenarios from the feature file
scenarios('leg_trade_history.feature')


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

@given(parsers.parse('{config_file} is available in {config_directory} and loaded'))
def load_configuration_file(test_context, config_file, config_directory):
    """Load and validate configuration file from specified directory"""
    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory
    full_config_path = config_path / config_file

    assert full_config_path.exists(), f"Configuration file not found: {full_config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")
    test_context['dm_config'] = config
    test_context['root_path'] = root_path
    test_context['config_path'] = full_config_path

@given('I have a LegTradeHistory instance with base_currency "USD"')
def trade_history_instance_with_usd(test_context):
    """Create LegTradeHistory instance with USD currency"""
    config = test_context['dm_config']
    test_context['leg_trade_history'] = LegTradeHistory(config)

@given(parsers.parse('I have loaded trade data from "{file_path}"'))
def loaded_trade_data(test_context, file_path):
    success = test_context['leg_trade_history'].load_from_json(file_path)
    test_context['load_success'] = success


@given(parsers.parse(
    'I have a position with entry_price {entry_price}, exit_price {exit_price}, quantity {quantity}, direction {direction}'))
def create_test_position_with_cost_model(test_context, entry_price, exit_price, quantity, direction):
    """Create test position using real cost model"""
    from datetime import datetime

    # Use relative dates (same day trade)
    now = datetime.now()
    entry_date = now.replace(hour=10, minute=0, second=0, microsecond=0)
    exit_date = now.replace(hour=15, minute=0, second=0, microsecond=0)

    test_context['test_position'] = {
        'entry_price': float(entry_price),
        'exit_price': float(exit_price),
        'quantity': int (quantity),
        'direction': direction,
        'entry_date': entry_date,
        'exit_date': exit_date,
        'status': 'closed',
        'timestamp': entry_date.isoformat() + 'Z',
        'symbol': 'TEST'
    }

@given('I have loaded trade history data for persistence testing')
def prepare_trade_history_for_persistence_test(test_context):
    leg_trade_history = test_context['leg_trade_history']
    original_stats = leg_trade_history.get_trade_statistics(lookback_periods=0)
    original_count = leg_trade_history.get_trade_count()
    test_context['original_stats'] = original_stats
    test_context['original_count'] = original_count
    test_context['original_trades'] = list(leg_trade_history.trades.values()).copy()

@given(parsers.parse('I have a new trade with timestamp "{timestamp}"'))
def create_new_trade_with_timestamp(test_context, timestamp):
    test_context['new_trade_timestamp'] = timestamp
    test_context['original_count'] = test_context['leg_trade_history'].get_trade_count()

@given(parsers.parse(
    'the trade has position with symbol {symbol}, type {position_type}, entry_price {entry_price}, quantity {quantity}'))
def add_position_to_new_trade(test_context, symbol, position_type, entry_price, quantity):
    """Add trade data in new flat format"""
    from datetime import datetime

    timestamp_str = test_context['new_trade_timestamp']
    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

    test_context['new_trade_data'] = {
        'uuid': f"test-trade-{timestamp_str}",
        'timestamp': timestamp_str,
        'entry_date': timestamp_str,
        'exit_date': None,
        'entry_price': float(entry_price),
        'exit_price': None,
        'quantity': int(quantity),
        'direction': 'LONG',
        'symbol': symbol,
        'type': position_type,
        'currency': 'USD',
        'status': 'open'
    }

@given(parsers.parse('I have a LegTradeHistory with trade pattern from "{file_path}"'))
def create_trade_history_with_pattern(test_context, file_path):
    """Load LegTradeHistory with specific trade pattern from configurable file path"""
    config = test_context['dm_config']  # Load real config files
    test_trade_history = LegTradeHistory(config)

    success = test_trade_history.load_from_json(file_path)

    assert success, f"Failed to load edge case data from {file_path}"
    test_context['test_trade_history'] = test_trade_history
    test_context['file_path'] = file_path

# =============================================================================
# WHEN steps - Actions
# =============================================================================


@when('I calculate the position outcome')
def calculate_position_outcome(test_context):
    position = test_context['test_position']
    leg_trade_history = test_context['leg_trade_history']

    success = leg_trade_history.add_trade(position)
    assert success, "Failed to add test position to trade history"

    timestamp = leg_trade_history._parse_timestamp(position['timestamp'])
    stored_trade = leg_trade_history.trades[timestamp]
    outcome = leg_trade_history._calculate_position_outcome(stored_trade)
    test_context['calculated_outcome'] = outcome

@when('I calculate trade statistics from all positions')
def calculate_trade_statistics_all_positions(test_context):
    statistics = test_context['leg_trade_history'].get_trade_statistics(lookback_periods=0)
    test_context['calculated_statistics'] = statistics

@when(parsers.parse('I save the trade history to "{file_path}"'))
def save_trade_history_to_file(test_context, file_path):
    success = test_context['leg_trade_history'].save_to_json(file_path)
    test_context['save_success'] = success
    test_context['save_file_path'] = file_path

@when('I create a new LegTradeHistory instance')
def create_new_trade_history_instance(test_context):
    dm_config = test_context['dm_config']
    test_context['new_leg_trade_history'] = LegTradeHistory(dm_config)

@when(parsers.parse('I load trade data from "{file_path}"'))
def load_trade_data_from_file(test_context, file_path):
    success = test_context['new_leg_trade_history'].load_from_json(file_path)
    test_context['load_success'] = success

@when(parsers.parse('I calculate trade statistics with lookback {lookback_periods}'))
def calculate_trade_statistics_with_lookback(test_context, lookback_periods):
    lookback = int(lookback_periods)
    statistics = test_context['leg_trade_history'].get_trade_statistics(lookback_periods=lookback)
    test_context['calculated_statistics'] = statistics
    test_context['lookback_used'] = lookback

@when('I add the trade to history')
def add_trade_to_history(test_context):
    success = test_context['leg_trade_history'].add_trade(test_context['new_trade_data'])
    test_context['add_success'] = success

@when('I calculate trade statistics')
def calculate_edge_case_statistics(test_context):
    """For edge case tests - may throw ValueError"""
    trade_history = test_context['test_trade_history']
    try:
        statistics = trade_history.get_trade_statistics(lookback_periods=0)
        test_context['calculated_statistics'] = statistics
        test_context['statistics_exception'] = None
    except ValueError as e:
        test_context['calculated_statistics'] = None
        test_context['statistics_exception'] = e

@when('I identify open and closed trades')
def identify_open_closed_trades(test_context):
    test_context['open_trades'] = []
    test_context['closed_trades'] = []
    for trade_data in test_context['leg_trade_history'].trades.values():
        if trade_data.get('status') == 'open':
            test_context['open_trades'].append(trade_data)
        elif trade_data.get('status') == 'closed':
            test_context['closed_trades'].append(trade_data)

@when(parsers.parse('I load from nonexistent file "{file_path}"'))
def load_nonexistent_file(test_context, file_path):
    """Attempt to load from nonexistent file"""
    config = test_context['dm_config']
    trade_history = LegTradeHistory(config)
    result = trade_history.load_from_json(file_path)
    test_context['load_result'] = result

@when(parsers.parse('I load from malformed file "{file_path}"'))
def load_malformed_file(test_context, file_path):
    """Attempt to load from malformed JSON file"""
    config = test_context['dm_config']
    trade_history = LegTradeHistory(config)
    result = trade_history.load_from_json(file_path)
    test_context['load_result'] = result

@when(parsers.parse('I load from file with invalid timestamp "{file_path}"'))
def load_invalid_timestamp_file(test_context, file_path):
    """Attempt to load from file with invalid timestamps"""
    config = test_context['dm_config']
    trade_history = LegTradeHistory(config)
    result = trade_history.load_from_json(file_path)
    test_context['load_result'] = result


# =============================================================================
# THEN steps - Assertions
# =============================================================================
@then('each trade should have required fields')
def verify_trade_has_required_fields(test_context):
    required_fields = ['timestamp', 'entry_price', 'exit_price', 'quantity',
                       'direction', 'entry_date', 'exit_date', 'symbol']
    for trade_data in test_context['leg_trade_history'].trades.values():
        for field in required_fields:
            assert field in trade_data
@then(parsers.parse('each trade should have symbol "{expected_symbol}"'))
def verify_trade_symbol(test_context, expected_symbol):
    for trade_data in test_context['leg_trade_history'].trades.values():
        assert trade_data.get('symbol') == expected_symbol

@then(parsers.parse('each trade should have {field1}, {field2}, {field3}, and {field4}'))
def verify_trade_has_fields(test_context, field1, field2, field3, field4):
    required_fields = [field1, field2, field3, field4]
    for trade_data in test_context['leg_trade_history'].trades.values():
        for field in required_fields:
            assert field in trade_data

@then(parsers.parse('closed trades should have {field1} and {field2}'))
def verify_closed_trades_have_fields(test_context, field1, field2):
    for trade_data in test_context['leg_trade_history'].trades.values():
        if trade_data.get('status') == 'closed':
            assert field1 in trade_data, f"Closed trade missing {field1}"
            assert field2 in trade_data, f"Closed trade missing {field2}"
            assert trade_data[field1] is not None, f"Closed trade has null {field1}"
            assert trade_data[field2] is not None, f"Closed trade has null {field2}"


@then(parsers.parse('{expected_count:d} trades should be loaded successfully'))
def verify_trades_loaded_successfully(test_context, expected_count):
    assert test_context['load_success'] == True
    actual_count = test_context['leg_trade_history'].get_trade_count()
    assert actual_count == expected_count

@then(parsers.parse('the outcome should be {expected_outcome}'))
def verify_position_outcome(test_context, expected_outcome):
    """Verify position outcome classification"""
    actual_outcome = test_context['calculated_outcome'].outcome
    assert actual_outcome == expected_outcome


@then(parsers.parse('the net P&L should be {expected_pnl:g}'))
def verify_net_pnl(test_context, expected_pnl):
    """Verify net P&L calculation"""
    actual_pnl = test_context['calculated_outcome'].net_pnl
    assert actual_pnl == expected_pnl

@then('fees should be properly subtracted from gross P&L')
def verify_fees_subtracted(test_context):
    """Verify fee handling in P&L calculation"""
    outcome = test_context['calculated_outcome']
    position = test_context['test_position']

    direction = position['direction']
    entry_price = position['entry_price']
    exit_price = position['exit_price']
    quantity = position['quantity']

    # Calculate expected gross P&L
    if direction.upper() == 'SHORT':
        expected_gross = (entry_price - exit_price) * quantity
    else:
        expected_gross = (exit_price - entry_price) * quantity

    # Verify gross P&L matches
    assert outcome.gross_pnl == expected_gross, \
        f"Gross P&L mismatch: {outcome.gross_pnl} != {expected_gross}"

    # Verify net P&L = gross - fees
    expected_net = expected_gross - outcome.fees
    assert outcome.net_pnl == expected_net, \
        f"Net P&L mismatch: {outcome.net_pnl} != {expected_net}"

    # Verify fees were calculated (not zero for non-zero trades)
    if quantity > 0 and entry_price > 0:
        assert outcome.fees > 0, "Fees should be calculated for real trades"

# For integer counts - "should contain"
@then(parsers.parse('the statistics should contain {count} {stat_type}'))
def verify_statistics_count(test_context, count, stat_type):
    """Verify statistics count values"""
    stats = test_context['calculated_statistics']
    count = int(count)

    if stat_type == 'total_positions':
        assert stats.total_positions == count
    elif stat_type == 'winning_positions':
        assert stats.winning_positions == count
    elif stat_type == 'losing_positions':
        assert stats.losing_positions == count
    elif stat_type == 'break_even_positions':
        assert stats.break_even_positions == count


# For float amounts - "should have"
@then(parsers.parse('the statistics should have {stat_name} of {value}'))
def verify_statistics_amount(test_context, stat_name, value):
    """Verify statistics amount values"""
    stats = test_context['calculated_statistics']
    value = float(value)

    if stat_name == 'total_fees':
        assert round(stats.total_fees, 2) == value
    elif stat_name == 'total_pnl':
        assert round(stats.total_pnl, 2) == value


# For list verification

@then('all original trade data should be preserved')
def verify_trade_data_preserved(test_context):
    assert test_context['save_success'] == True
    assert test_context['load_success'] == True
    original_count = test_context['original_count']
    new_count = test_context['new_leg_trade_history'].get_trade_count()
    assert new_count == original_count


@then('all position data should be preserved')
def verify_position_data_preserved(test_context):
    original_trades = test_context['original_trades']
    new_trades = list(test_context['new_leg_trade_history'].trades.values())

    assert len(new_trades) == len(original_trades)

    # Compare key fields for each trade
    for orig, new in zip(original_trades, new_trades):
        assert orig['symbol'] == new['symbol']
        assert orig['entry_price'] == new['entry_price']
        assert orig['quantity'] == new['quantity']
        assert orig['direction'] == new['direction']


@then('timestamp ordering should be maintained')
def verify_timestamp_ordering(test_context):
    trades = list(test_context['new_leg_trade_history'].trades.values())
    timestamps = [trade['timestamp'] for trade in trades]

    # Verify chronological order
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


@then('trade statistics should be identical to original')
def verify_statistics_identical(test_context):
    original_stats = test_context['original_stats']
    new_stats = test_context['new_leg_trade_history'].get_trade_statistics(lookback_periods=0)

    assert new_stats.total_positions == original_stats.total_positions
    assert new_stats.winning_positions == original_stats.winning_positions
    assert new_stats.losing_positions == original_stats.losing_positions
    assert round(new_stats.total_fees, 2) == round(original_stats.total_fees, 2)
    assert round(new_stats.total_pnl, 2) == round(original_stats.total_pnl, 2)


@then(parsers.parse('only the most recent {expected_positions} positions should be used'))
def verify_positions_count_with_lookback(test_context, expected_positions):
    """Verify correct number of positions used in calculation"""
    stats = test_context['calculated_statistics']
    expected = int(expected_positions)
    assert stats.total_positions == expected


@then('the statistics should reflect the limited dataset')
def verify_statistics_reflect_limited_dataset(test_context):
    stats = test_context['calculated_statistics']
    lookback = test_context['lookback_used']
    outcomes = test_context['leg_trade_history'].get_position_outcomes(lookback_periods=lookback)

    # Verify statistics match the limited outcome set
    wins = [o for o in outcomes if o.outcome == 'win']
    losses = [o for o in outcomes if o.outcome == 'loss']
    break_evens = [o for o in outcomes if o.outcome == 'break_even']

    assert stats.winning_positions == len(wins)
    assert stats.losing_positions == len(losses)
    assert stats.break_even_positions == len(break_evens)

    # Verify P&L and fees match the limited outcome set
    expected_pnl = sum(o.net_pnl for o in outcomes)
    expected_fees = sum(o.fees for o in outcomes)

    assert round(stats.total_pnl, 2) == round(expected_pnl, 2)
    assert round(stats.total_fees, 2) == round(expected_fees, 2)

@then('older positions should be excluded from the calculation')
def verify_older_positions_excluded(test_context):
    lookback = test_context['lookback_used']
    if lookback == 0:
        return
    all_outcomes = test_context['leg_trade_history'].get_position_outcomes(lookback_periods=0)
    limited_outcomes = test_context['leg_trade_history'].get_position_outcomes(lookback_periods=lookback)

    # Verify we have fewer positions with lookback than without
    assert len(limited_outcomes) <= len(all_outcomes)
    assert len(limited_outcomes) == lookback

    # If we have more total positions than lookback, verify exclusion occurred
    if len(all_outcomes) > lookback:
        assert len(limited_outcomes) < len(all_outcomes)

@then('the trade count should increase by 1')
def verify_trade_count_increased(test_context):
    """Verify trade count increased by exactly 1"""
    assert test_context['add_success'] == True
    new_count = test_context['leg_trade_history'].get_trade_count()
    original_count = test_context['original_count']
    assert new_count == original_count + 1


@then('the new trade should be stored in chronological order')
def verify_chronological_order(test_context):
    trades = list(test_context['leg_trade_history'].trades.values())
    timestamps = [trade['timestamp'] for trade in trades]

    # Verify ordering is maintained
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]

@then('the trade should be accessible by timestamp')
def verify_trade_accessible_by_timestamp(test_context):
    timestamp_str = test_context['new_trade_timestamp']
    leg_trade_history = test_context['leg_trade_history']
    timestamp = leg_trade_history._parse_timestamp(timestamp_str)
    assert timestamp in leg_trade_history.trades
    retrieved_trade = leg_trade_history.trades[timestamp]
    assert retrieved_trade['uuid'] == test_context['new_trade_data']['uuid']

@then(parsers.parse('trade "{trade_uuid}" should be open'))
def verify_trade_is_open(test_context, trade_uuid):
    trade_found = False  # Fixed
    for trade_data in test_context['leg_trade_history'].trades.values():
        if trade_data.get('uuid') == trade_uuid:
            trade_found = True
            assert trade_data.get('status') == 'open', f"Trade {trade_uuid} is not open"
            test_context['current_trade'] = trade_data
            break

    assert trade_found, f"Trade {trade_uuid} not found"


@then(parsers.parse('the open trade should have {field} as null'))
def verify_open_trade_field_is_null(test_context, field):
    """Verify open trade has null field"""
    trade_data = test_context.get('current_trade')
    assert trade_data is not None, "No current trade in context"
    assert trade_data.get(field) is None, f"Open trade has non-null {field}: {trade_data.get(field)}"

@then(parsers.parse('ValueError should be raised with message "{message}"'))
def verify_value_error_raised(test_context, message):
    """Verify ValueError was raised with expected message"""
    exception = test_context.get('statistics_exception')
    assert exception is not None, "Expected ValueError but none was raised"
    assert message in str(exception), f"Expected '{message}' in '{str(exception)}'"

@then('average loss should be 0')
def verify_average_loss_zero(test_context):
    """Verify average loss is zero (all winning trades)"""
    stats = test_context['calculated_statistics']
    losses = [o for o in stats.outcomes if o.outcome == 'loss']
    assert len(losses) == 0, f"Expected 0 losses, got {len(losses)}"

@then('average win should be 0')
def verify_average_win_zero(test_context):
    """Verify average win is zero (all losing trades)"""
    stats = test_context['calculated_statistics']
    wins = [o for o in stats.outcomes if o.outcome == 'win']
    assert len(wins) == 0, f"Expected 0 wins, got {len(wins)}"

@then('total net P&L should be 0')
def verify_total_net_pnl_zero(test_context):
    """Verify total net P&L is zero (all break even)"""
    stats = test_context['calculated_statistics']
    assert abs(stats.total_pnl) < 0.01, f"Expected ~0 P&L, got {stats.total_pnl}"

@then('load should return false')
def verify_load_returns_false(test_context):
    """Verify load operation returned false"""
    assert test_context['load_result'] == False

@then('the statistics should include position outcomes list')
def verify_position_outcomes_list(test_context):
    """Verify position outcomes list is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'outcomes')
    assert isinstance(stats.outcomes, list)
    assert len(stats.outcomes) > 0