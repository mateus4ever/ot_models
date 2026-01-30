# test_trade_history.py
import math
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data import DataManager
from src.hybrid.positions.base_trade_history import TradeStatistics
# Import the classes we're testing
from src.hybrid.positions.spread_trade_history import SpreadTradeHistory

# Load all scenarios from the feature file
scenarios('spread_trade_history.feature')


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
# GIVEN steps
# =============================================================================

@given(parsers.parse('{config_file} is available in {config_directory} and loaded'))
def load_configuration_file(test_context, config_file, config_directory):
    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory
    full_config_path = config_path / config_file
    assert full_config_path.exists(), f"Config not found: {full_config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")
    test_context['config'] = config
    test_context['root_path'] = root_path


@given('I have a SpreadTradeHistory instance with base_currency "USD"')
def create_spread_trade_history(test_context):
    config = test_context['config']
    test_context['spread_trade_history'] = SpreadTradeHistory(config)

@given('I have a spread trade with:')
def create_spread_trade_from_table(test_context, datatable):
    import json

    trade_data = {}
    # datatable[0] is header ['field', 'value'], rest are data rows
    for row in datatable[1:]:
        field = row[0]
        value = row[1]

        # Parse JSON arrays
        if value.startswith('['):
            value = json.loads(value)
        # Parse numbers
        elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
            value = float(value)
        elif value.isdigit():
            value = int(value)

        trade_data[field] = value

    test_context['spread_trade'] = trade_data


@given(parsers.parse('I load base trade from "{file_path}"'))
def load_base_trade(test_context, file_path):
    import json

    root_path = test_context['root_path']
    with open(root_path / file_path) as f:
        data = json.load(f)

    test_context['spread_trade'] = data['trades'][0].copy()

@given(parsers.parse('I load base trade from "{file_path}"'))
def load_base_trade(test_context, file_path):
    import json

    root_path = test_context['root_path']
    with open(root_path / file_path) as f:
        data = json.load(f)

    test_context['spread_trade'] = data['trades'][0].copy()

@given(parsers.parse('I modify trade with {invalid_condition}'))
def modify_trade_with_condition(test_context, invalid_condition):
    trade = test_context['spread_trade']

    if invalid_condition == 'missing leg_trades':
        del trade['leg_trades']
    elif invalid_condition == 'empty leg_trades':
        trade['leg_trades'] = []
    elif invalid_condition == 'missing gross_pnl':
        del trade['gross_pnl']

@given(parsers.parse('I set gross_pnl to {gross_pnl}'))
def set_gross_pnl(test_context, gross_pnl):
    trade = test_context['spread_trade']
    trade['gross_pnl'] = float(gross_pnl)
    if 'net_pnl' in trade:
        del trade['net_pnl']
# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I add the spread trade to history')
def add_spread_trade_to_history(test_context):
    spread_history = test_context['spread_trade_history']
    trade_data = test_context['spread_trade']
    result = spread_history.add_trade(trade_data)
    test_context['add_result'] = result


@when(parsers.parse('I load spread history from "{file_path}"'))
def load_spread_data(test_context, file_path):
    # Use new instance if exists (for round-trip test), otherwise original
    if 'new_spread_trade_history' in test_context:
        spread_history = test_context['new_spread_trade_history']
    else:
        spread_history = test_context['spread_trade_history']

    success = spread_history.load_from_json(file_path)
    test_context['load_success'] = success
    assert success, f"Failed to load from {file_path}"

@when(parsers.parse('I update timestamp on trade "{trade_uuid}" to current time'))
def update_trade_timestamp(test_context, trade_uuid):
    from datetime import datetime

    spread_history = test_context['spread_trade_history']
    now = datetime.now()

    for timestamp, trade in spread_history.trades.items():
        if trade.get('uuid') == trade_uuid:
            trade['exit_date'] = now.isoformat() + 'Z'
            test_context['modified_trade_uuid'] = trade_uuid
            return

    raise ValueError(f"Trade {trade_uuid} not found")

@when(parsers.parse('I save spread history to "{file_path}"'))
def save_spread_history(test_context, file_path):
    spread_history = test_context['spread_trade_history']
    success = spread_history.save_to_json(file_path)
    test_context['save_success'] = success
    test_context['save_path'] = file_path

@when('I create a new SpreadTradeHistory instance')
def create_new_spread_history(test_context):
    config = test_context['config']
    test_context['new_spread_trade_history'] = SpreadTradeHistory(config)

@when('I calculate trade statistics')
def calculate_statistics(test_context):
    spread_history = test_context['spread_trade_history']
    stats = spread_history.get_trade_statistics(lookback_periods=0)
    test_context['statistics'] = stats

# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then(parsers.parse('trade count should be {count:d}'))
def verify_trade_count(test_context, count):
    spread_history = test_context['spread_trade_history']
    assert spread_history.get_trade_count() == count


@then(parsers.parse('the trade should have {count} leg references'))
def verify_leg_references(test_context, count):
    spread_history = test_context['spread_trade_history']
    last_trade = list(spread_history.trades.values())[-1]
    leg_trades = last_trade.get('leg_trades', [])
    assert len(leg_trades) == int(count), f"Expected {count} legs, got {len(leg_trades)}"

@then('leg_trades references should be preserved')
def verify_leg_trades_preserved(test_context):
    spread_history = test_context['spread_trade_history']
    for trade in spread_history.trades.values():
        assert 'leg_trades' in trade, "Trade missing leg_trades"
        assert len(trade['leg_trades']) > 0, "leg_trades is empty"


@then('gross_pnl should be preserved')
def verify_gross_pnl_preserved(test_context):
    spread_history = test_context['spread_trade_history']
    for trade in spread_history.trades.values():
        assert 'gross_pnl' in trade, "Trade missing gross_pnl"

@then('the file should be created successfully')
def verify_file_created(test_context):
    assert test_context['save_success'] == True

@then('the trade should be rejected')
def verify_trade_rejected(test_context):
    assert test_context['add_result'] == False

@then(parsers.parse('the stored trade should have net_pnl {expected_pnl}'))
def verify_stored_net_pnl(test_context, expected_pnl):
    spread_history = test_context['spread_trade_history']
    last_trade = list(spread_history.trades.values())[-1]
    assert last_trade['net_pnl'] == float(expected_pnl), \
        f"Expected net_pnl {expected_pnl}, got {last_trade['net_pnl']}"

@then('trades should be stored in chronological order')
def verify_chronological_order(test_context):
    spread_history = test_context['spread_trade_history']
    timestamps = list(spread_history.trades.keys())

    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], \
            f"Trades not in order: {timestamps[i - 1]} > {timestamps[i]}"

@then('the loaded trade should match all original fields')
def verify_all_fields_match(test_context):
    original = test_context['spread_trade']
    new_history = test_context['new_spread_trade_history']
    loaded = list(new_history.trades.values())[0]

    for field in original.keys():
        assert field in loaded, f"Field {field} missing after load"
        assert loaded[field] == original[field], \
            f"Field {field} mismatch: {original[field]} != {loaded[field]}"

@then(parsers.parse('total_positions should be {count:d}'))
def verify_total_positions(test_context, count):
    stats = test_context['statistics']
    assert stats.total_positions == count


@then(parsers.parse('winning_positions should be {count:d}'))
def verify_winning_positions(test_context, count):
    stats = test_context['statistics']
    assert stats.winning_positions == count


@then(parsers.parse('losing_positions should be {count:d}'))
def verify_losing_positions(test_context, count):
    stats = test_context['statistics']
    assert stats.losing_positions == count


@then(parsers.parse('break_even_positions should be {count:d}'))
def verify_break_even_positions(test_context, count):
    stats = test_context['statistics']
    assert stats.break_even_positions == count


@then(parsers.parse('total_pnl should be {expected_pnl}'))
def verify_total_pnl(test_context, expected_pnl):
    stats = test_context['statistics']
    assert round(stats.total_pnl, 2) == float(expected_pnl)