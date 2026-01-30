import logging
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, parsers, then, when

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data import DataManager
from src.hybrid.money_management import MoneyManager
from src.hybrid.positions.position_orchestrator import PositionOrchestrator
from src.hybrid.positions.spread_trade_history import SpreadTradeHistory
from src.hybrid.predictors.vasicek.triangular_arbitrage_predictor import TriangularArbitragePredictor
from src.hybrid.strategies.implementation import TriangularStrategy

# Load scenarios from the strategy_factory.feature
scenarios('triangular_strategy.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
# GIVEN steps - Setup
# =============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    root_path = Path(__file__).parent.parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    unified_config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['unified_config'] = unified_config

@given(parsers.parse('data source is set to {data_path}'))
def set_data_source(test_context, data_path):
    """Set data source path in config"""
    test_context['data_path'] = data_path


@given('a data manager is initialized with three markets:')
def initialize_data_manager(test_context, datatable):
    """Initialize data manager with specified markets"""
    config = test_context['unified_config']
    data_path = test_context['data_path']

    root_path = Path(__file__).parent.parent.parent.parent.parent

    data_manager = DataManager(config, root_path)

    # Get folder names from table (EURUSD, EURGBP, GBPUSD)
    folder_names = [row[0] for row in datatable[1:]]

    for folder in folder_names:
        market_path = f"{data_path}/{folder}"
        success = data_manager.load_market_data(market_path)
        assert success, f"Failed to load market from {market_path}"

    # Verify markets loaded (IDs may differ from folder names)
    loaded_markets = data_manager.get_available_markets()
    assert len(loaded_markets) == len(folder_names), \
        f"Expected {len(folder_names)} markets, got {len(loaded_markets)}: {loaded_markets}"

    loaded_markets = data_manager.get_available_markets()
    data_manager.align_market_data(loaded_markets)

    test_context['data_manager'] = data_manager
    test_context['loaded_markets'] = loaded_markets

@given(parsers.parse('temporal pointer is initialized with lookback window {window:d}'))
def initialize_temporal_pointer(test_context, window):
    """Initialize temporal pointer with specified lookback"""
    data_manager = test_context['data_manager']
    data_manager.initialize_temporal_pointer(window)

@given('a triangular arbitrage predictor is initialized')
def initialize_predictor(test_context):
    """Initialize triangular arbitrage predictor"""
    config = test_context['unified_config']
    predictor = TriangularArbitragePredictor(config)
    test_context['predictor'] = predictor


@given('a position orchestrator is initialized')
def initialize_position_orchestrator(test_context):
    """Initialize position orchestrator"""
    config = test_context['unified_config']
    position_orchestrator = PositionOrchestrator(config)
    position_orchestrator.set_initial_capital(100000)  # From config ideally
    test_context['position_orchestrator'] = position_orchestrator

@given('I have a TriangularStrategy instance')
def create_triangular_strategy(test_context):
    """Create TriangularStrategy and wire dependencies"""
    config = test_context['unified_config']

    strategy = TriangularStrategy(name='triangular_arbitrage', config=config)
    strategy.set_data_manager(test_context['data_manager'])
    strategy.set_position_orchestrator(test_context['position_orchestrator'])
    strategy.add_predictor(test_context['predictor'])

    # Add spread trade history
    spread_trade_history = SpreadTradeHistory(config)
    strategy.set_spread_trade_history(spread_trade_history)

    test_context['strategy'] = strategy

@given('a money manager is initialized')
def initialize_money_manager(test_context):
    config = test_context['unified_config']
    money_manager = MoneyManager(config)

    # Need to inject position orchestrator
    position_orchestrator = test_context['position_orchestrator']
    money_manager.set_position_orchestrator(position_orchestrator)

    test_context['money_manager'] = money_manager

@given(parsers.parse('a triangular strategy "{name}" is created'))
def create_named_triangular_strategy(test_context, name):
    """Create TriangularStrategy and wire dependencies"""
    config = test_context['unified_config']

    strategy = TriangularStrategy(name=name, config=config)
    strategy.set_data_manager(test_context['data_manager'])
    strategy.set_position_orchestrator(test_context['position_orchestrator'])
    strategy.set_money_manager(test_context['money_manager'])
    strategy.add_predictor(test_context['predictor'])

    # Add spread trade history
    spread_trade_history = SpreadTradeHistory(config)
    strategy.set_spread_trade_history(spread_trade_history)

    test_context['strategy'] = strategy

# =============================================================================
# WHEN steps - Actions
# =============================================================================
@when('strategy runs')
def run_strategy(test_context):
    strategy = test_context['strategy']
    result = strategy.run()
    test_context['result'] = result

# =============================================================================
# THEN steps - Assertions
# =============================================================================
@then('strategy should have all required dependencies')
def assert_strategy_has_dependencies(test_context):
    """Verify all dependencies are wired"""
    strategy = test_context['strategy']
    assert strategy.data_manager is not None, "data_manager not set"
    assert strategy.position_orchestrator is not None, "position_orchestrator not set"
    assert strategy.money_manager is not None, "money_manager not set"
    assert len(strategy.predictors) > 0, "no predictors added"

@then('strategy should accept predictor interface')
def assert_predictor_interface(test_context):
    """Verify predictor implements interface"""
    strategy = test_context['strategy']
    predictor = strategy.predictors[0]
    assert hasattr(predictor, 'train'), "predictor missing train()"
    assert hasattr(predictor, 'predict'), "predictor missing predict()"
    assert hasattr(predictor, 'is_trained'), "predictor missing is_trained"

@then('strategy should complete without error')
def assert_no_error(test_context):
    result = test_context['result']
    assert 'error' not in result, f"Strategy failed: {result.get('error')}"

@then(parsers.parse('trade history should have at least {count:d} trade'))
def assert_min_trade_count(test_context, count):
    strategy = test_context['strategy']
    trade_count = strategy.position_orchestrator.trade_history.get_trade_count()
    assert trade_count >= count, f"Expected at least {count} trades, got {trade_count}"

@then('final result should be logged')
def log_final_result(test_context):
    result = test_context['result']
    logger.warning(f"BACKTEST: trades={result.get('total_trades')}, "
                   f"pnl={result.get('net_pnl')}, "
                   f"win_rate={result.get('win_rate')}")