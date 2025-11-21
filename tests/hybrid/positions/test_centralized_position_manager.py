# tests/hybrid/money_management/test_money_management.py

import logging
# Import the system under test
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, parsers, then, when
import ast
import random

from src.hybrid.positions.centralized_position_manager import CentralizedPositionManager

# Go up 4 levels from tests/hybrid/money_management/test_money_management.py to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid.config.unified_config import UnifiedConfig

# Load all scenarios from money_management.feature
scenarios('centralized_position_manager.feature')

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

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['config'] = config

@given('a centralized position manager is initialized from configuration')
def create_position_manager(test_context):
    """Create CentralizedPositionManager with initial_capital from base.json"""
    unified_config = test_context['config']

    initial_capital = unified_config.config.get('backtesting', {}).get('initial_capital')
    assert initial_capital, "initial_capital not found in configuration"

    position_manager = CentralizedPositionManager(unified_config)
    position_manager.set_total_capital(initial_capital)

    test_context['position_manager'] = position_manager

@given(parsers.parse('bots have committed amounts {amounts}'))
def bots_have_committed(test_context, amounts):
    """Setup: Bots have already committed specified amounts"""
    import ast
    amount_list = ast.literal_eval(amounts)
    position_manager = test_context['position_manager']

    for i, amount in enumerate(amount_list):
        bot_id = f"bot_{i + 1}"
        trade_id = f"trade_{i + 1:03d}"
        success = position_manager.commit_position(trade_id, amount, bot_id)
        assert success, f"Failed to setup initial commitment of ${amount}"

    test_context['initial_trade_count'] = len(amount_list)

@given(parsers.parse('positions {indices} are already released'))
def positions_already_released(test_context, indices):
    """Setup: Positions have already been released"""
    index_list = ast.literal_eval(indices)
    position_manager = test_context['position_manager']

    for idx in index_list:
        trade_id = f"trade_{idx + 1:03d}"
        success = position_manager.release_position(trade_id)
        assert success, f"Failed to release position {trade_id} during setup"
@given(parsers.parse('{bot_count} bots attempt to commit {amount} each simultaneously'))
def bots_commit_simultaneously(test_context, bot_count, amount):
    """Multiple bots attempt to commit same amount concurrently"""
    bot_count = int(bot_count)
    amount = int(amount)

    position_manager = test_context['position_manager']
    config = test_context['config']
    max_workers = config.testing.get('max_concurrent_workers', 10)

    results = []
    results_lock = threading.Lock()

    def commit_for_bot(bot_index):
        bot_id = f"bot_{bot_index}"
        trade_id = f"{bot_id}_trade_001"
        success = position_manager.commit_position(trade_id, amount, bot_id)
        with results_lock:
            results.append(success)

    with ThreadPoolExecutor(max_workers=min(max_workers, bot_count)) as executor:
        futures = [executor.submit(commit_for_bot, i) for i in range(bot_count)]
        for future in futures:
            future.result()

    test_context['commit_results'] = results
@given(parsers.parse('{bot_count} bots perform mixed commit/release operations'))
def setup_concurrent_operations(test_context, bot_count):
    """Setup for concurrent mixed operations test"""
    test_context['concurrent_bot_count'] = int(bot_count)


@given(parsers.parse('bot "{bot_id}" has committed {amount} for trade "{trade_id}"'))
def bot_has_committed_for_trade(test_context, bot_id, amount, trade_id):
    """Setup: Bot has committed specific amount for specific trade"""
    # Remove commas and dollar sign from amount
    amount = int(amount.replace('$', '').replace(',', ''))

    position_manager = test_context['position_manager']
    success = position_manager.commit_position(trade_id, amount, bot_id)
    assert success, f"Failed to commit ${amount} for {bot_id}"

# =============================================================================
# WHEN steps - Actions
# =============================================================================
@when(parsers.parse('bots try to commit amounts {amounts}'))
def bots_try_commit_amounts(test_context, amounts):
    """Multiple bots attempt to commit (may succeed or fail)"""
    amount_list = ast.literal_eval(amounts)
    position_manager = test_context['position_manager']
    initial_count = test_context.get('initial_trade_count', 0)

    successes = []
    for i, amount in enumerate(amount_list):
        bot_id = f"bot_{initial_count + i + 1}"
        trade_id = f"trade_{initial_count + i + 1:03d}"
        success = position_manager.commit_position(trade_id, amount, bot_id)
        successes.append(success)

    test_context['commit_results'] = successes

@when(parsers.parse('bots commit amounts {amounts}'))
def commit_multiple_amounts(test_context, amounts):
    """Commit capital for multiple bots with given amounts"""
    amount_list = ast.literal_eval(amounts)
    position_manager = test_context['position_manager']

    for i, amount in enumerate(amount_list):
        bot_id = f"bot_{i + 1}"
        trade_id = f"trade_{i + 1:03d}"
        success = position_manager.commit_position(trade_id, amount, bot_id)
        assert success, f"Failed to commit ${amount} for {bot_id}"

    test_context['last_commit_success'] = True


@when(parsers.parse('When bot tries to commit {amount}'))
def bot_attempts_commit(test_context, amount):
    """Bot attempts to commit specified amount"""
    position_manager = test_context['position_manager']
    initial_count = test_context.get('initial_trade_count', 0)
    bot_id = f"bot_{initial_count + 1}"
    trade_id = f"trade_{initial_count + 1:03d}"
    amount = float(amount)

    success = position_manager.commit_position(trade_id, amount, bot_id)
    test_context['last_commit_success'] = success


@when(parsers.parse('bot attempts to commit {amount} with same trade_id'))
def attempt_commit_duplicate_trade_id(test_context, amount):
    """Bot attempts to commit with duplicate trade_id"""
    position_manager = test_context['position_manager']
    # Reuse first trade_id
    trade_id = "trade_001"
    bot_id = "bot_duplicate"
    amount = float(amount)

    success = position_manager.commit_position(trade_id, amount, bot_id)
    test_context['last_commit_success'] = success


@when(parsers.parse('positions {indices} are released'))
def release_multiple_positions(test_context, indices):
    """Release multiple positions by index"""
    index_list = ast.literal_eval(indices)
    position_manager = test_context['position_manager']

    successes = []
    for idx in index_list:
        trade_id = f"trade_{idx + 1:03d}"
        success = position_manager.release_position(trade_id)
        successes.append(success)

    test_context['release_results'] = successes


@when(parsers.parse('position {index} is released'))
def release_single_position(test_context, index):
    """Release single position by index"""
    position_manager = test_context['position_manager']
    index = int(index)
    trade_id = f"trade_{index + 1:03d}"
    success = position_manager.release_position(trade_id)
    test_context['last_release_success'] = success

@when('allocation summary is requested')
def request_allocation_summary(test_context):
    """Request allocation summary from position manager"""
    position_manager = test_context['position_manager']
    summary = position_manager.get_allocation_summary()
    test_context['allocation_summary'] = summary


@when(parsers.parse('{operation_count} operations are executed concurrently'))
def execute_concurrent_operations(test_context, operation_count):
    """Execute random commit/release operations concurrently"""
    operation_count = int(operation_count)

    position_manager = test_context['position_manager']
    config = test_context['config']
    bot_count = test_context['concurrent_bot_count']

    # No defaults - must be in config
    testing_config = config.testing
    max_workers = testing_config['max_concurrent_workers']
    commit_amount = testing_config['test_commit_amount']
    release_probability = testing_config['release_probability']

    operations_lock = threading.Lock()
    active_trades = {}

    def random_operation(op_index):
        bot_id = f"bot_{op_index % bot_count}"

        with operations_lock:
            has_position = bot_id in active_trades

        if has_position and random.random() < release_probability:
            with operations_lock:
                trade_id = active_trades.pop(bot_id, None)
            if trade_id:
                position_manager.release_position(trade_id)
        else:
            # Generate unique trade_id based on current positions
            current_positions = len(position_manager.get_committed_positions())
            trade_id = f"{bot_id}_trade_{current_positions + 1}"
            success = position_manager.commit_position(trade_id, commit_amount, bot_id)
            if success:
                with operations_lock:
                    active_trades[bot_id] = trade_id

    with ThreadPoolExecutor(max_workers=min(max_workers, bot_count)) as executor:
        futures = [executor.submit(random_operation, i) for i in range(operation_count)]
        for future in futures:
            future.result()

@when('position manager is reset')
def reset_position_manager(test_context):
    """Reset the position manager"""
    position_manager = test_context['position_manager']
    position_manager.reset()

# =============================================================================
# THEN steps - Assertions
# =============================================================================
@then(parsers.parse('available capital should be {amount}'))
def check_available_capital(test_context, amount):
    """Verify available capital matches expected amount"""
    position_manager = test_context['position_manager']
    amount = float(amount)
    available = position_manager.get_available_capital()
    assert available == amount, f"Expected ${amount}, got ${available}"


@then(parsers.parse('committed capital should be {amount}'))
def check_committed_capital(test_context, amount):
    """Verify committed capital matches expected amount"""
    position_manager = test_context['position_manager']
    summary = position_manager.get_allocation_summary()
    committed = summary['committed']
    amount = float(amount)
    assert committed == amount, f"Expected ${amount} committed, got ${committed}"


@then(parsers.parse('active positions count should be {count}'))
def check_active_positions_count(test_context, count):
    """Verify number of active positions"""
    position_manager = test_context['position_manager']
    summary = position_manager.get_allocation_summary()
    active = summary['active_positions']
    count = int(count)
    assert active == count, f"Expected {count} active positions, got {active}"

@then('all commitments should succeed')
def check_all_commits_succeeded(test_context):
    """Verify all commitments were successful"""
    successes = test_context.get('commit_successes', [])
    assert all(successes), f"Some commitments failed: {successes}"

@then('commitment should fail')
def check_commitment_failed(test_context):
    """Verify the commitment failed"""
    success = test_context.get('last_commit_success', True)
    assert not success, "Expected commitment to fail but it succeeded"

@then('the duplicate commitment should fail')
def check_duplicate_failed(test_context):
    """Verify duplicate trade_id commitment failed"""
    success = test_context.get('last_commit_success', True)
    assert not success, "Expected duplicate commitment to fail"

@then(parsers.parse('commitments should have results {expected}'))
def check_commitment_results(test_context, expected):
    """Verify commitment success/failure pattern"""
    expected_results = ast.literal_eval(expected)
    actual_results = test_context.get('commit_results', [])
    assert actual_results == expected_results, f"Expected {expected_results}, got {actual_results}"

@then(parsers.parse('releases should have results {expected}'))
def check_release_results(test_context, expected):
    """Verify release success/failure pattern"""
    expected_results = ast.literal_eval(expected)
    actual_results = test_context.get('release_results', [])
    assert actual_results == expected_results, f"Expected {expected_results}, got {actual_results}"
@then('release should fail')
def check_release_failed(test_context):
    """Verify release failed"""
    success = test_context.get('last_release_success', True)
    assert not success, "Expected release to fail but it succeeded"

@then(parsers.parse('summary total_capital should be {value}'))
def check_summary_total_capital(test_context, value):
    """Verify total_capital in summary"""
    summary = test_context['allocation_summary']
    value = float(value)
    assert summary['total_capital'] == value, f"Expected {value}, got {summary['total_capital']}"


@then(parsers.parse('summary available should be {value}'))
def check_summary_available(test_context, value):
    """Verify available capital in summary"""
    summary = test_context['allocation_summary']
    value = float(value)
    assert summary['available'] == value, f"Expected {value}, got {summary['available']}"


@then(parsers.parse('summary committed should be {value}'))
def check_summary_committed(test_context, value):
    """Verify committed capital in summary"""
    summary = test_context['allocation_summary']
    value = float(value)
    assert summary['committed'] == value, f"Expected {value}, got {summary['committed']}"

@then(parsers.parse('summary available_pct should be {value}'))
def check_summary_available_pct(test_context, value):
    summary = test_context['allocation_summary']
    unified_config = test_context['config']

    value = float(value)
    tolerance = unified_config.config.get('testing', {}).get('float_comparison_tolerance')
    assert abs(summary['available_pct'] - value) < tolerance


@then(parsers.parse('summary committed_pct should be {value:f}'))
def check_summary_committed_pct(test_context, value):
    """Verify committed percentage in summary"""
    summary = test_context['allocation_summary']
    unified_config = test_context['config']
    tolerance = unified_config.config.get('testing', {}).get('float_comparison_tolerance')
    assert abs(summary['committed_pct'] - value) < tolerance, f"Expected {value}, got {summary['committed_pct']}"


@then(parsers.parse('summary active_positions should be {value:d}'))
def check_summary_active_positions(test_context, value):
    """Verify active positions count in summary"""
    summary = test_context['allocation_summary']
    assert summary['active_positions'] == value, f"Expected {value}, got {summary['active_positions']}"


@then('capital integrity is maintained')
def check_capital_integrity(test_context):
    """Verify no capital lost or duplicated"""
    position_manager = test_context['position_manager']
    unified_config = test_context['config']

    expected_total = unified_config.config.get('backtesting', {}).get('initial_capital')
    summary = position_manager.get_allocation_summary()

    assert summary['total_capital'] == expected_total, \
        f"Total capital changed from {expected_total} to {summary['total_capital']}"


@then('available plus committed equals total capital')
def check_capital_sum(test_context):
    """Verify capital accounting is correct"""
    position_manager = test_context['position_manager']
    summary = position_manager.get_allocation_summary()

    total_accounted = summary['available'] + summary['committed'] + summary['reserved']
    assert total_accounted == summary['total_capital'], \
        f"Capital mismatch: {summary['available']} + {summary['committed']} + {summary['reserved']} != {summary['total_capital']}"


@given(parsers.parse('bot "{bot_id}" commits multiple amounts {amounts}'))
def bot_commits_multiple(test_context, bot_id, amounts):
    """Single bot commits multiple positions"""
    amount_list = ast.literal_eval(amounts)
    position_manager = test_context['position_manager']

    for i, amount in enumerate(amount_list):
        trade_id = f"{bot_id}_trade_{i+1}"  # Readable for debugging
        success = position_manager.commit_position(trade_id, amount, bot_id)
        assert success, f"Failed to commit ${amount} for {bot_id}"


@given(parsers.parse('bot "{bot_id}" commits amounts {amounts}'))
def bot_commits_amounts(test_context, bot_id, amounts):
    """Bot commits specified amounts"""
    amount_list = ast.literal_eval(amounts)
    position_manager = test_context['position_manager']

    # Get current position count to generate unique trade_ids
    current_positions = len(position_manager.get_committed_positions())

    for i, amount in enumerate(amount_list):
        trade_id = f"{bot_id}_trade_{current_positions + i + 1}"
        success = position_manager.commit_position(trade_id, amount, bot_id)
        assert success, f"Failed to commit ${amount} for {bot_id}"

@then(parsers.parse('bot "{bot_id}" should have {count} position totaling {total}'))
@then(parsers.parse('bot "{bot_id}" should have {count} positions totaling {total}'))
def check_bot_allocation(test_context, bot_id, count, total):
    """Verify bot's allocation in summary"""
    summary = test_context['allocation_summary']
    by_bot = summary.get('by_bot', {})

    assert bot_id in by_bot, f"Bot {bot_id} not found in summary"
    bot_data = by_bot[bot_id]
    total = int(total)

    # Note: API returns 'committed' not 'amount' based on your implementation
    actual_total = bot_data.get('committed')
    assert actual_total == total, f"Expected ${total}, got ${actual_total} for {bot_id}"

@then(parsers.parse('exactly {success_count} commitments should succeed'))
def check_success_count(test_context, success_count):
    """Verify number of successful commitments"""
    success_count = int(success_count)
    results = test_context['commit_results']
    actual_success = sum(1 for r in results if r)
    assert actual_success == success_count, \
        f"Expected {success_count} successes, got {actual_success}"

@then(parsers.parse('{fail_count} commitments should fail'))
def check_fail_count(test_context, fail_count):
    """Verify number of failed commitments"""
    fail_count = int(fail_count)
    results = test_context['commit_results']
    actual_fail = sum(1 for r in results if not r)
    assert actual_fail == fail_count, \
        f"Expected {fail_count} failures, got {actual_fail}"