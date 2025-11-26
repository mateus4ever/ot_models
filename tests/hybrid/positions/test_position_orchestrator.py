# tests/hybrid/money_management/test_money_management.py

import logging
# Import the system under test
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, parsers, then, when

from src.hybrid.data import DataManager
from src.hybrid.positions.position_orchestrator import PositionOrchestrator
from src.hybrid.products.product_types import PositionDirection

# Go up 4 levels from tests/hybrid/money_management/test_money_management.py to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid.config.unified_config import UnifiedConfig

# Load all scenarios from money_management.feature
scenarios('position_orchestrator.feature')

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

@given('a PositionOrchestrator is initialized from configuration')
def create_position_orchestrator(test_context):
    """Create PositionOrchestrator from config"""
    config = test_context['config']
    orchestrator = PositionOrchestrator(config)
    orchestrator.set_initial_capital(config.config.get('backtesting').get('initial_capital'))
    test_context['orchestrator'] = orchestrator


@given(parsers.parse(
    'DataManager is initialized with market data "{data_file}" from "{data_directory}" with training window {training_window}'))
def create_data_manager_with_data(test_context, data_file, data_directory, training_window):
    """Create DataManager and load specified test data"""
    training_window = int(training_window)
    config = test_context['config']
    data_manager = DataManager(config)

    root_path = Path(__file__).parent.parent.parent.parent
    data_directory= root_path / data_directory

    # Build path and load
    data_path = Path(data_directory) / data_file
    success = data_manager.load_market_data_with_temporal_setup(str(data_path), training_window)

    assert success, f"Failed to load test market data from {data_path}"

    test_context['data_manager'] = data_manager


@given('PositionTracker is registered as DataManager listener')
def register_position_tracker_as_listener(test_context):
    """Register PositionTracker with DataManager"""
    data_manager = test_context['data_manager']
    orchestrator = test_context['orchestrator']

    data_manager.register_listener(orchestrator.position_tracker)
    test_context['listener_registered'] = True


@given('PositionTracker is registered as DataManager listener')
def register_position_tracker_as_listener(test_context):
    """Register PositionTracker with DataManager"""
    data_manager = test_context['data_manager']
    orchestrator = test_context['orchestrator']

    data_manager.register_listener(orchestrator.position_tracker)
    test_context['listener_registered'] = True


@given(parsers.parse('a position is opened for "{symbol}" '
                     'with entry price {entry_price}'))
def open_test_position(test_context, symbol, entry_price):
    """Open test position in PositionTracker"""
    entry_price = float(entry_price)
    orchestrator = test_context['orchestrator']

    success = orchestrator.position_tracker.open_position(
        trade_id="test_trade_001",
        symbol=symbol,
        direction=PositionDirection.LONG,
        quantity=1000,
        entry_price=entry_price
    )

    assert success, f"Failed to open position for {symbol}"
    test_context['test_trade_id'] = "test_trade_001"
    test_context['test_symbol'] = symbol

@given(parsers.parse('PositionOrchestrator has {capital} initial capital'))
def set_orchestrator_capital(test_context, capital):
    """Set initial capital in PositionOrchestrator"""
    capital = float(capital)
    orchestrator = test_context['orchestrator']
    orchestrator.set_initial_capital(capital)


@given(parsers.parse('a position "{trade_id}" is open for {symbol} at {entry_price}'))
def setup_open_position(test_context, trade_id, symbol, entry_price):
    """Setup an open position for testing"""
    entry_price = float(entry_price)
    orchestrator = test_context['orchestrator']

    success = orchestrator.open_position(
        trade_id=trade_id,
        symbol=symbol,
        direction=PositionDirection.LONG,
        quantity=1000,
        entry_price=entry_price,
        capital_required=entry_price * 1000
    )

    assert success, f"Failed to open test position {trade_id}"
    test_context['last_trade_id'] = trade_id


@given(parsers.parse('position "{trade_id}" is open: {quantity} {symbol} at {entry_price}, current {current_price}'))
def setup_open_position_with_price(test_context, trade_id,
                                   quantity, symbol, entry_price,
                                   current_price):
    """Setup open position and update current price"""
    quantity = int(quantity)
    entry_price = float(entry_price)
    current_price = float(current_price)
    orchestrator = test_context['orchestrator']

    # Open position
    success = orchestrator.open_position(
        trade_id=trade_id,
        symbol=symbol,
        direction=PositionDirection.LONG,
        quantity=quantity,
        entry_price=entry_price,
        capital_required=entry_price * quantity
    )
    assert success

    # Update to current price
    orchestrator.position_tracker.update_position_price(trade_id, current_price)


@given(parsers.parse(
    'a closed trade exists with symbol "{symbol}" quantity {quantity} at entry {entry_price} exit {exit_price} with net P&L {pnl}'))
def setup_closed_trade_from_params(test_context, symbol, quantity, entry_price, exit_price, pnl):
    """Setup closed trade with all parameters from feature"""
    quantity = int(quantity)
    entry_price = float(entry_price)
    exit_price = float(exit_price)
    pnl = float(pnl)

    orchestrator = test_context['orchestrator']

    trade_data = {
        'uuid': 'closed_trade_001',
        'timestamp': datetime.now().isoformat() + 'Z',
        'symbol': symbol,
        'direction': 'LONG',
        'quantity': quantity,
        'entry_price': entry_price,
        'entry_date': (datetime.now() - timedelta(days=1)).isoformat() + 'Z',
        'exit_price': exit_price,
        'exit_date': datetime.now().isoformat() + 'Z',
        'status': 'closed'
    }

    added_trade = orchestrator.trade_history.add_trade(trade_data)
    print(added_trade)

#TODO: hardcoded values must be removed
@given(parsers.parse('an open position has unrealized P&L of {pnl}'))
def setup_position_with_unrealized_pnl(test_context, pnl):
    """Setup open position with specific unrealized P&L"""
    pnl = float(pnl)
    orchestrator = test_context['orchestrator']

    # Open position where price movement creates desired P&L
    # For long: P&L = (current - entry) * quantity
    # If quantity = 1000, need price movement of pnl/1000
    entry_price = 1.1000
    quantity = 1000
    current_price = entry_price + (pnl / quantity)

    success = orchestrator.open_position(
        trade_id='pnl_test_001',
        symbol='EURUSD',
        direction=PositionDirection.LONG,
        quantity=quantity,
        entry_price=entry_price,
        capital_required=entry_price * quantity
    )
    assert success

    # Update to current price
    orchestrator.position_tracker.update_position_price('pnl_test_001', current_price)


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when(parsers.parse('DataManager advances to next bar'))
def advance_data_manager(test_context):
    """Advance DataManager temporal pointer - price will come from actual data"""
    data_manager = test_context['data_manager']

    success = data_manager.next()
    assert success, "Failed to advance DataManager"


@when(parsers.parse(
    'I open position "{trade_id}" for {symbol} {direction} with {quantity} shares at {entry_price} requiring {capital}'))
def open_position_via_orchestrator(test_context, trade_id, symbol, direction, quantity, entry_price, capital):
    """Open position through PositionOrchestrator"""
    quantity = int(quantity)
    entry_price = float(entry_price)
    capital = float(capital)

    # Convert string to enum
    direction_enum = PositionDirection[direction.upper()]  # 'long' -> PositionDirection.LONG

    orchestrator = test_context['orchestrator']
    success = orchestrator.open_position(trade_id, symbol, direction_enum, quantity, entry_price, capital)

    assert success, f"Failed to open position {trade_id}"
    test_context['last_trade_id'] = trade_id


@when(parsers.parse('I close position "{trade_id}" at exit price {exit_price}'))
def close_position_via_orchestrator(test_context, trade_id, exit_price):
    """Close position through PositionOrchestrator"""
    exit_price = float(exit_price)
    orchestrator = test_context['orchestrator']

    success = orchestrator.close_position(trade_id, exit_price)
    assert success, f"Failed to close position {trade_id}"

    test_context['closed_trade_id'] = trade_id
    test_context['exit_price'] = exit_price

@when('I request portfolio state')
def request_portfolio_state(test_context):
    """Request portfolio state from orchestrator"""
    orchestrator = test_context['orchestrator']

    stats = orchestrator.trade_history.get_trade_statistics()
    # Call without parameters - prices already updated via listener
    portfolio_state = orchestrator.get_portfolio_state()
    test_context['portfolio_state'] = portfolio_state

# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('PositionManager should be initialized')
def check_position_manager_initialized(test_context):
    """Verify PositionManager is initialized in orchestrator"""
    orchestrator = test_context['orchestrator']
    assert orchestrator.position_manager is not None
    assert hasattr(orchestrator.position_manager, 'get_allocation_summary')


@then('PositionTracker should be initialized')
def check_position_tracker_initialized(test_context):
    """Verify PositionTracker is initialized in orchestrator"""
    orchestrator = test_context['orchestrator']
    assert orchestrator.position_tracker is not None
    assert hasattr(orchestrator.position_tracker, 'get_all_positions')


@then('TradeHistory should be initialized')
def check_trade_history_initialized(test_context):
    """Verify TradeHistory is initialized in orchestrator"""
    orchestrator = test_context['orchestrator']
    assert orchestrator.trade_history is not None
    assert hasattr(orchestrator.trade_history, 'get_trade_statistics')


@then(parsers.parse('position current price should be updated to {expected_price}'))
def check_position_price_updated(test_context, expected_price):
    """Verify position price was updated by listener"""
    expected_price = float(expected_price)
    orchestrator = test_context['orchestrator']
    trade_id = test_context['test_trade_id']

    position = orchestrator.position_tracker.get_position(trade_id)
    assert position is not None, f"Position {trade_id} not found"
    assert position.current_price == expected_price, \
        f"Expected price {expected_price}, got {position.current_price}"


@then('capital should be committed in PositionManager')
def check_capital_committed(test_context):
    """Verify capital was committed"""
    orchestrator = test_context['orchestrator']
    trade_id = test_context['last_trade_id']

    committed = orchestrator.position_manager.get_committed_positions()
    assert trade_id in committed, f"Position {trade_id} not found in committed positions"


@then('position should be tracked in PositionTracker')
def check_position_tracked(test_context):
    """Verify position is tracked"""
    orchestrator = test_context['orchestrator']
    trade_id = test_context['last_trade_id']

    position = orchestrator.position_tracker.get_position(trade_id)
    assert position is not None, f"Position {trade_id} not found in tracker"


@then('trade should be recorded in TradeHistory as open')
def check_trade_recorded(test_context):
    """Verify trade is recorded in history"""
    orchestrator = test_context['orchestrator']
    trade_id = test_context['last_trade_id']

    # Get open trades from history
    open_trades = orchestrator.trade_history.get_open_trades()
    assert trade_id in [t['trade_id'] for t in open_trades], \
        f"Trade {trade_id} not found in open trades"

@then('capital should be released in PositionManager')
def check_capital_released(test_context):
    """Verify capital was released"""
    orchestrator = test_context['orchestrator']
    trade_id = test_context['closed_trade_id']

    # Verify position no longer in committed positions
    committed = orchestrator.position_manager.get_committed_positions()
    assert trade_id not in committed, \
        f"Position {trade_id} still in committed positions after closing"

@then('position should be removed from PositionTracker')
def check_position_removed_from_tracker(test_context):
    """Verify position was removed from tracker"""
    orchestrator = test_context['orchestrator']
    trade_id = test_context['closed_trade_id']

    position = orchestrator.position_tracker.get_position(trade_id)
    assert position is None, \
        f"Position {trade_id} still in PositionTracker after closing"


@then('trade should be updated in TradeHistory with exit data')
def check_trade_updated_in_history(test_context):
    """Verify trade was updated with exit data in history"""
    orchestrator = test_context['orchestrator']
    trade_id = test_context['closed_trade_id']
    exit_price = test_context['exit_price']

    # Get trade from history
    trade = orchestrator.trade_history.get_trade_by_id(trade_id)
    assert trade is not None, f"Trade {trade_id} not found in TradeHistory"

    # Verify it's closed with exit data
    assert trade.get('status') == 'closed', \
        f"Trade {trade_id} status is {trade.get('status')}, expected 'closed'"
    assert trade.get('exit_price') == exit_price, \
        f"Trade exit_price is {trade.get('exit_price')}, expected {exit_price}"
    assert trade.get('exit_date') is not None, \
        f"Trade {trade_id} missing exit_date"


@then('total equity should reflect unrealized P&L')
def check_total_equity_reflects_pnl(test_context):
    """Verify total equity includes unrealized P&L"""
    portfolio_state = test_context['portfolio_state']
    orchestrator = test_context['orchestrator']

    # Calculate expected equity
    allocation = orchestrator.position_manager.get_allocation_summary()
    initial_capital = allocation['total_capital']

    # Calculate unrealized P&L from positions
    positions = orchestrator.position_tracker.get_all_positions()
    unrealized_pnl = 0.0
    for pos in positions.values():
        if pos.direction == 'long':
            unrealized_pnl += (pos.current_price - pos.entry_price) * pos.size
        else:
            unrealized_pnl += (pos.entry_price - pos.current_price) * pos.size

    expected_equity = initial_capital + unrealized_pnl

    assert abs(portfolio_state.total_equity - expected_equity) < 0.01, \
        f"Total equity {portfolio_state.total_equity} doesn't match expected {expected_equity}"

@then('total P&L should combine realized and unrealized')
def check_total_pnl_combined(test_context):
    """Verify total P&L correctly combines realized and unrealized"""
    portfolio_state = test_context['portfolio_state']
    orchestrator = test_context['orchestrator']

    # Get realized P&L from trade history
    stats = orchestrator.trade_history.get_trade_statistics()
    realized = stats.total_pnl

    # Calculate unrealized from positions
    unrealized = 0.0
    for pos in orchestrator.position_tracker.get_all_positions().values():
        if pos.direction == 'long':
            unrealized += (pos.current_price - pos.entry_price) * pos.size
        else:
            unrealized += (pos.entry_price - pos.current_price) * pos.size

    expected_total = realized + unrealized

    assert abs(portfolio_state.total_pnl - expected_total) < 0.01, \
        f"Total P&L {portfolio_state.total_pnl} doesn't match calculated {expected_total} (realized={realized}, unrealized={unrealized})"
@then('available cash should reflect committed capital')
def check_available_cash(test_context):
    """Verify available cash reflects committed capital"""
    portfolio_state = test_context['portfolio_state']
    orchestrator = test_context['orchestrator']

    # Get allocation from position manager
    allocation = orchestrator.position_manager.get_allocation_summary()
    expected_available = allocation['available']

    assert abs(portfolio_state.available_cash - expected_available) < 0.01, \
        f"Available cash {portfolio_state.available_cash} doesn't match expected {expected_available}"

@then('positions should include both open trades')
def check_positions_include_open_trades(test_context):
    """Verify portfolio state includes all open positions"""
    portfolio_state = test_context['portfolio_state']
    orchestrator = test_context['orchestrator']

    # Get all open positions from tracker
    open_positions = orchestrator.position_tracker.get_all_positions()

    # Verify portfolio state has same positions
    assert len(portfolio_state.positions) == len(open_positions), \
        f"Portfolio state has {len(portfolio_state.positions)} positions, expected {len(open_positions)}"

    # Verify specific trade IDs exist
    for trade_id in open_positions.keys():
        assert trade_id in portfolio_state.positions, \
            f"Trade {trade_id} not found in portfolio state positions"

@then('daily P&L should be calculated from price changes')
def check_daily_pnl_calculated(test_context):
    """Verify daily P&L is calculated"""
    portfolio_state = test_context['portfolio_state']

    # Daily P&L should be set (not zero if there are positions with price changes)
    assert portfolio_state.daily_pnl is not None, "Daily P&L is None"

    # If there are open positions, daily P&L should equal unrealized P&L
    orchestrator = test_context['orchestrator']
    positions = orchestrator.position_tracker.get_all_positions()

    if len(positions) > 0:
        # Daily P&L currently equals unrealized P&L in implementation
        expected_daily = portfolio_state.total_pnl - orchestrator.trade_history.get_trade_statistics().total_pnl
        assert abs(portfolio_state.daily_pnl - expected_daily) < 0.01, \
            f"Daily P&L {portfolio_state.daily_pnl} doesn't match expected {expected_daily}"
@then('drawdown should be calculated from peak equity')
def check_drawdown_calculated(test_context):
    """Verify drawdown is calculated from peak equity"""
    portfolio_state = test_context['portfolio_state']

    # Verify drawdown is calculated
    assert portfolio_state.max_drawdown is not None, "Drawdown is None"

    # Verify drawdown formula: (peak - current) / peak
    if portfolio_state.peak_equity > 0:
        expected_drawdown = (portfolio_state.peak_equity - portfolio_state.total_equity) / portfolio_state.peak_equity
        assert abs(portfolio_state.max_drawdown - expected_drawdown) < 0.01, \
            f"Drawdown {portfolio_state.max_drawdown} doesn't match expected {expected_drawdown}"

@then('peak equity should be tracked correctly')
def check_peak_equity_tracked(test_context):
    """Verify peak equity is tracked"""
    portfolio_state = test_context['portfolio_state']

    # Peak equity should be at least current equity
    assert portfolio_state.peak_equity >= portfolio_state.total_equity, \
        f"Peak equity {portfolio_state.peak_equity} is less than current equity {portfolio_state.total_equity}"

    # Peak should be positive
    assert portfolio_state.peak_equity > 0, \
        f"Peak equity {portfolio_state.peak_equity} should be positive"