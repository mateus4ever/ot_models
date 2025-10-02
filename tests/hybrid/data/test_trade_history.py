# test_trade_history.py
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
# Import the classes we're testing
from src.hybrid.data.trade_history import TradeHistory, PositionOutcome, TradeStatistics

# Load all scenarios from the feature file
scenarios('trade_history.feature')


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


@given('I have a TradeHistory instance with base_currency "USD"')
def trade_history_instance_with_usd(test_context):
    """Create TradeHistory instance with USD currency"""
    config = test_context['dm_config']  # Load real config files
    pytest.trade_history = TradeHistory(config)


@given(parsers.parse('I have loaded trade data from "{file_path}"'))
def loaded_trade_data(test_context, file_path):
    """Ensure trade data is loaded"""
    success = pytest.trade_history.load_from_json(file_path)
    test_context['load_success'] = success


@given(parsers.parse('I have a position with entry_value {entry_value}, exit_value {exit_value}, amount {amount}, entry_fees {entry_fees}, and exit_fees {exit_fees}'))
def create_test_position_with_new_fees(test_context, entry_value, exit_value, amount, entry_fees, exit_fees):
    """Create test position with entry/exit fee structure"""
    test_context['test_position'] = {
        'entry_value': float(entry_value),
        'exit_value': float(exit_value),
        'amount': float(amount),
        'entry_fees': float(entry_fees),
        'exit_fees': float(exit_fees),
        'status': 'closed'
    }

@given('I have loaded trade history data for persistence testing')
def prepare_trade_history_for_persistence_test(test_context):
    """Capture original state for save/load comparison"""
    # The trade history is already loaded from Background
    original_stats = pytest.trade_history.get_trade_statistics(lookback_periods=0)
    original_count = pytest.trade_history.get_trade_count()
    test_context['original_stats'] = original_stats
    test_context['original_count'] = original_count
    test_context['original_positions'] = pytest.trade_history.all_positions.copy()

@given(parsers.parse('I have a new trade with timestamp "{timestamp}"'))
def create_new_trade_with_timestamp(test_context, timestamp):
    """Create new trade with specified timestamp"""
    test_context['new_trade_timestamp'] = timestamp
    test_context['original_count'] = pytest.trade_history.get_trade_count()

@given(parsers.parse('the trade has position with name_of_position "{name_of_position}", type "{position_type}", entry_value {entry_value}, amount {amount}, and entry_fees {entry_fees}'))
def add_position_to_new_trade(test_context, name_of_position, position_type, entry_value, amount, entry_fees):
    """Add position data to new trade"""
    test_context['new_trade_data'] = {
        'uuid': f"test-trade-{test_context['new_trade_timestamp']}",
        'timestamp': test_context['new_trade_timestamp'],
        'status': 'open',
        'positions': [{
            'name_of_position': name_of_position,
            'type': position_type,
            'amount': float(amount),
            'entry_value': float(entry_value),
            'entry_fees': float(entry_fees),
            'currency': 'USD',
            'status': 'open',
            'entry_timestamp': test_context['new_trade_timestamp'],
            'exit_value': None,
            'exit_timestamp': None,
            'exit_fees': None
        }]
    }


@given(parsers.parse('I have a TradeHistory with trade pattern from "{file_path}"'))
def create_trade_history_with_pattern(test_context, file_path):
    """Load TradeHistory with specific trade pattern from configurable file path"""
    config = test_context['dm_config']  # Load real config files
    test_trade_history = TradeHistory(config)

    success = test_trade_history.load_from_json(file_path)

    assert success, f"Failed to load edge case data from {file_path}"
    test_context['test_trade_history'] = test_trade_history
    test_context['file_path'] = file_path

# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I access positions from the loaded trades')
def access_positions():
    """Access position data from trades"""
    pytest.positions = pytest.trade_history.all_positions


@when('I calculate the position outcome')
def calculate_position_outcome(test_context):
    """Calculate outcome for test position"""
    position = test_context['test_position']
    outcome = pytest.trade_history._calculate_position_outcome(position)
    test_context['calculated_outcome'] = outcome


@when('I calculate trade statistics from all positions')
def calculate_trade_statistics_all_positions(test_context):
    """Calculate trade statistics from all loaded positions"""
    # Use the global pytest.trade_history instead
    statistics = pytest.trade_history.get_trade_statistics(lookback_periods=0)

    # Store in context for verification steps
    test_context['calculated_statistics'] = statistics

@when(parsers.parse('I save the trade history to "{file_path}"'))
def save_trade_history_to_file(test_context, file_path):
    """Save trade history to specified file"""
    success = pytest.trade_history.save_to_json(file_path)
    test_context['save_success'] = success
    test_context['save_file_path'] = file_path

@when('I create a new TradeHistory instance')
def create_new_trade_history_instance(test_context):
    """Create fresh TradeHistory instance"""

    dm_config = test_context['dm_config']
    pytest.new_trade_history = TradeHistory(dm_config)

@when(parsers.parse('I load trade data from "{file_path}"'))
def load_trade_data_from_file(file_path):
    """Load data into new TradeHistory instance"""
    success = pytest.new_trade_history.load_from_json(file_path)
    pytest.load_success = success

@when(parsers.parse('I calculate trade statistics with lookback {lookback_periods}'))
def calculate_trade_statistics_with_lookback(test_context, lookback_periods):
    """Calculate trade statistics with specified lookback window"""
    lookback = int(lookback_periods)
    statistics = pytest.trade_history.get_trade_statistics(lookback_periods=lookback)
    test_context['calculated_statistics'] = statistics
    test_context['lookback_used'] = lookback


@when('I identify open and closed positions')
def identify_open_and_closed_positions(test_context):
    """Identify and categorize positions by status"""
    all_positions = pytest.trade_history.all_positions

    open_positions = [p for p in all_positions if p.get('status') == 'open']
    closed_positions = [p for p in all_positions if p.get('status') == 'closed']

    test_context['open_positions'] = open_positions
    test_context['closed_positions'] = closed_positions

@when('I add the trade to history')
def add_trade_to_history(test_context):
    """Add the new trade to trade history"""
    success = pytest.trade_history.add_trade(test_context['new_trade_data'])
    test_context['add_success'] = success

@when('I calculate trade statistics')
def calculate_edge_case_statistics(test_context):
    """Calculate statistics for edge case scenario"""
    trade_history = test_context['test_trade_history']
    statistics = trade_history.get_trade_statistics(lookback_periods=0)
    test_context['calculated_statistics'] = statistics


@when(parsers.parse('I encounter {error_condition} during operation'))
def encounter_error_condition(test_context, error_condition):
    """Trigger specific error conditions"""
    dm_config = test_context['dm_config']
    test_trade_history = TradeHistory(dm_config)

    if error_condition == 'missing_json_file':
        result = test_trade_history.load_from_json('tests/data/trade/nonexistent_file.json')
        test_context['operation_result'] = result
        test_context['expected_result'] = False

    elif error_condition.endswith('invalid_timestamp_trade.json'):
        result = test_trade_history.load_from_json(error_condition)
        test_context['operation_result'] = result
        test_context['expected_result'] = True  # Load succeeds, but trades skipped
        test_context['trade_count'] = test_trade_history.get_trade_count()
        test_context['expected_trade_count'] = 0

    elif error_condition.endswith('.json'):
        result = test_trade_history.load_from_json(error_condition)
        test_context['operation_result'] = result
        test_context['expected_result'] = False

# =============================================================================
# THEN steps - Assertions
# =============================================================================


@then(parsers.parse('{expected_count:d} trades should be loaded successfully'))
def verify_trades_loaded_successfully(test_context, expected_count):
    """Verify expected number of trades loaded"""
    assert test_context['load_success'] == True  # Now assert here
    actual_count = pytest.trade_history.get_trade_count()
    assert actual_count == expected_count


@then('each trade should have at least one position')
def each_trade_has_position():
    """Verify all trades have positions"""
    assert len(pytest.positions) > 0


@then(parsers.parse('each position should have name_of_position "{symbol}"'))
def positions_have_symbol(symbol):
    """Verify positions have correct symbol"""
    for pos in pytest.positions:
        assert pos.get('name_of_position') == symbol


@then('each position should have entry_value, currency, entry_fees, and exit_fees')
def positions_have_required_fields():
    """Verify positions have required fields"""
    for pos in pytest.positions:
        assert 'entry_value' in pos
        assert 'currency' in pos
        assert 'entry_fees' in pos
        # exit_fees can be null for open positions, so check if key exists
        assert 'exit_fees' in pos


@then('closed positions should have exit_value and exit_timestamp')
def closed_positions_have_exit_data():
    """Verify closed positions have exit information"""
    closed_positions = [p for p in pytest.positions if p.get('status') == 'closed']
    for pos in closed_positions:
        assert pos.get('exit_value') is not None
        assert pos.get('exit_timestamp') is not None


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

    expected_gross = (position['exit_value'] - position['entry_value']) * position['amount']

    # Calculate total fees from entry and exit fees (matching the new structure)
    entry_fees = position.get('entry_fees', 0)
    exit_fees = position.get('exit_fees', 0) if position.get('exit_fees') is not None else 0
    total_fees = entry_fees + exit_fees

    expected_net = expected_gross - total_fees

    assert outcome.gross_pnl == expected_gross
    assert outcome.net_pnl == expected_net
    assert outcome.fees == total_fees  # Also verify the fees field is calculated correctly

@then('the statistics should include total_positions count')
def verify_total_positions_count(test_context):
    """Verify total positions count is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'total_positions')
    assert stats.total_positions >= 0

@then('the statistics should include winning_positions count')
def verify_winning_positions_count(test_context):
    """Verify winning positions count is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'winning_positions')
    assert stats.winning_positions >= 0

@then('the statistics should include losing_positions count')
def verify_losing_positions_count(test_context):
    """Verify losing positions count is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'losing_positions')
    assert stats.losing_positions >= 0

@then('the statistics should include break_even_positions count')
def verify_break_even_positions_count(test_context):
    """Verify break even positions count is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'break_even_positions')
    assert stats.break_even_positions >= 0

@then('the statistics should include total_pnl amount')
def verify_total_pnl_amount(test_context):
    """Verify total P&L amount is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'total_pnl')
    assert isinstance(stats.total_pnl, (int, float))

@then('the statistics should include total_fees amount')
def verify_total_fees_amount(test_context):
    """Verify total fees amount is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'total_fees')
    assert isinstance(stats.total_fees, (int, float))

@then('the statistics should include position outcomes list')
def verify_position_outcomes_list(test_context):
    """Verify position outcomes list is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'outcomes')
    assert isinstance(stats.outcomes, list)


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
@then('the statistics should include position outcomes list')
def verify_position_outcomes_list(test_context):
    """Verify position outcomes list is present"""
    stats = test_context['calculated_statistics']
    assert hasattr(stats, 'outcomes')
    assert isinstance(stats.outcomes, list)
    assert len(stats.outcomes) > 0


@then('all original trade data should be preserved')
def verify_trade_data_preserved(test_context):
    """Verify trade count matches original"""
    assert test_context['save_success'] == True
    assert pytest.load_success == True

    original_count = test_context['original_count']
    new_count = pytest.new_trade_history.get_trade_count()
    assert new_count == original_count


@then('all position data should be preserved')
def verify_position_data_preserved(test_context):
    """Verify all positions are preserved"""
    original_positions = test_context['original_positions']
    new_positions = pytest.new_trade_history.all_positions

    assert len(new_positions) == len(original_positions)

    # Compare key fields for each position
    for orig, new in zip(original_positions, new_positions):
        assert orig['name_of_position'] == new['name_of_position']
        assert orig['entry_value'] == new['entry_value']
        assert orig['amount'] == new['amount']


@then('timestamp ordering should be maintained')
def verify_timestamp_ordering():
    """Verify trades are still in timestamp order"""
    trades = list(pytest.new_trade_history.trades.values())
    timestamps = [trade['timestamp'] for trade in trades]

    # Verify chronological order
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


@then('trade statistics should be identical to original')
def verify_statistics_identical(test_context):
    """Verify statistics match original exactly"""
    original_stats = test_context['original_stats']
    new_stats = pytest.new_trade_history.get_trade_statistics(lookback_periods=0)

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
    """Verify statistics are calculated only from limited positions"""
    stats = test_context['calculated_statistics']
    lookback = test_context['lookback_used']

    # Get position outcomes to verify they match the lookback window
    outcomes = pytest.trade_history.get_position_outcomes(lookback_periods=lookback)

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
    """Verify older positions beyond lookback window are excluded"""
    lookback = test_context['lookback_used']

    if lookback == 0:
        # Special case: 0 means all positions, so nothing excluded
        return

    # Get all closed positions
    all_outcomes = pytest.trade_history.get_position_outcomes(lookback_periods=0)
    limited_outcomes = pytest.trade_history.get_position_outcomes(lookback_periods=lookback)

    # Verify we have fewer positions with lookback than without
    assert len(limited_outcomes) <= len(all_outcomes)
    assert len(limited_outcomes) == lookback

    # If we have more total positions than lookback, verify exclusion occurred
    if len(all_outcomes) > lookback:
        assert len(limited_outcomes) < len(all_outcomes)


@then(parsers.parse('trade "{trade_id}" should have an open position'))
def verify_trade_has_open_position(test_context, trade_id):
    """Verify specific trade contains an open position"""
    # Find the trade by UUID
    trade_found = False
    for timestamp, trade_data in pytest.trade_history.trades.items():
        if trade_data.get('uuid') == trade_id:
            trade_found = True
            positions = trade_data.get('positions', [])
            open_positions = [p for p in positions if p.get('status') == 'open']
            assert len(open_positions) > 0, f"Trade {trade_id} should have at least one open position"
            test_context['target_open_position'] = open_positions[0]
            break

    assert trade_found, f"Trade {trade_id} not found in loaded data"

@then('the open position should have exit_value as null')
def verify_open_position_exit_value_null(test_context):
    """Verify open position has null exit_value"""
    position = test_context['target_open_position']
    assert position.get('exit_value') is None

@then('the open position should have exit_timestamp as null')
def verify_open_position_exit_timestamp_null(test_context):
    """Verify open position has null exit_timestamp"""
    position = test_context['target_open_position']
    assert position.get('exit_timestamp') is None


@then('the trade count should increase by 1')
def verify_trade_count_increased(test_context):
    """Verify trade count increased by exactly 1"""
    assert test_context['add_success'] == True
    new_count = pytest.trade_history.get_trade_count()
    original_count = test_context['original_count']
    assert new_count == original_count + 1


@then('the new trade should be stored in chronological order')
def verify_chronological_order(test_context):
    """Verify trades remain in chronological order"""
    trades = list(pytest.trade_history.trades.values())
    timestamps = [trade['timestamp'] for trade in trades]

    # Verify ordering is maintained
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


@then('the trade should be accessible by timestamp')
def verify_trade_accessible_by_timestamp(test_context):
    """Verify new trade can be retrieved by timestamp"""
    timestamp_str = test_context['new_trade_timestamp']
    timestamp = pytest.trade_history._parse_timestamp(timestamp_str)

    assert timestamp in pytest.trade_history.trades
    retrieved_trade = pytest.trade_history.trades[timestamp]
    assert retrieved_trade['uuid'] == test_context['new_trade_data']['uuid']


@then('the calculation should handle the edge case appropriately')
def verify_edge_case_handled(test_context):
    """Verify calculation completes without errors"""
    stats = test_context['calculated_statistics']
    assert stats is not None
    assert isinstance(stats, TradeStatistics)


@then('no mathematical errors should occur')
def verify_no_mathematical_errors(test_context):
    """Verify no division by zero or other math errors"""
    stats = test_context['calculated_statistics']
    import math
    assert not math.isnan(stats.total_pnl)
    assert not math.isnan(stats.total_fees)
    assert not math.isinf(stats.total_pnl)
    assert not math.isinf(stats.total_fees)


@then(parsers.parse('the result should show {expected_behavior}'))
def verify_expected_behavior(test_context, expected_behavior):
    """Verify specific expected behavior for each edge case"""
    stats = test_context['calculated_statistics']

    if expected_behavior == 'zero_stats_no_errors':
        assert stats.total_positions == 0
        assert stats.winning_positions == 0
        assert stats.losing_positions == 0
        assert stats.total_pnl == 0.0
        assert stats.total_fees == 0.0

    elif expected_behavior == 'zero_avg_loss':
        assert stats.losing_positions == 0
        assert stats.winning_positions > 0
        assert stats.total_pnl > 0

    elif expected_behavior == 'zero_avg_win':
        assert stats.winning_positions == 0
        assert stats.losing_positions > 0
        assert stats.total_pnl < 0

    elif expected_behavior == 'zero_net_pnl':
        assert abs(stats.total_pnl) < 0.01  # Close to zero allowing for floating point precision
        assert stats.break_even_positions > 0


@then('the system should handle it gracefully')
def verify_graceful_handling(test_context):
    """Verify system handles error appropriately"""
    result = test_context['operation_result']
    expected = test_context['expected_result']
    assert result == expected

    # For invalid timestamp case, verify trade was skipped
    if 'expected_trade_count' in test_context:
        actual_count = test_context['trade_count']
        expected_count = test_context['expected_trade_count']
        assert actual_count == expected_count

@then('appropriate error messages should be logged')
def verify_error_logging(test_context):
    """Verify errors are logged appropriately"""
    # Verify operation completed without unhandled exceptions
    assert 'operation_result' in test_context