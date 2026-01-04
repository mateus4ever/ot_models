# tests/hybrid/signals/test_signal_factory.py
import io
import pytest
import logging
import pandas

from pathlib import Path

from pandas import Series, Timestamp
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data import DataManager
from src.hybrid.money_management import MoneyManager
from src.hybrid.signals.market_signal_enum import MarketSignal
from src.hybrid.signals.signal_factory import SignalFactory

# Load scenarios from the signal_factory.feature
scenarios('sma_crossover_signal.feature')

# Set up debug logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


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
# GIVEN steps - Setup
# =============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    root_path = Path(__file__).parent.parent.parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['config'] = config
    test_context['root_path'] = root_path
    test_context['config_path'] = config_path

    print("\n--- Registered Given Steps ---")


@given(parsers.parse('SMA signal with fast and slow period {fast_period:d}, {slow_period:d} is initialized'))
def step_create_sma_signal_configured(test_context, fast_period, slow_period):
    """
    Modifies the live UnifiedConfig instance using the existing update_config method
    to set the test parameters. This verifies the full configuration path.
    """
    config = test_context['config']

    # Construct the nested update dictionary payload
    # This must match the structure of your JSON config files.
    update_payload = {
        'signals': {
            'trend_following': {
                'simplemovingaveragecrossover': {
                    'parameters': {
                        'fast_period': fast_period,
                        'slow_period': slow_period
                    }
                }
            }
        }
    }

    # Use the existing update_config method to inject the test parameters
    config.update_config(update_payload)

    # Create the SignalFactory instance, passing the MODIFIED config
    factory = SignalFactory(config)

    # Create the signal instance. The signal's __init__ must read the updated values.
    # (Assuming the SignalFactory can be accessed or created here)
    signal = factory.create_signal('trend_following.simplemovingaveragecrossover')

    test_context['signal'] = signal

    # Verification: Ensure the signal actually loaded the expected values from config.
    assert signal.fast_period == fast_period
    assert signal.slow_period == slow_period


@given(parsers.parse('market data is loaded from {data_file}'))
def load_market_data(test_context, data_file):
    """Load market data using DataManager"""

    config = test_context['config']

    data_manager = DataManager(config)

    root_path = Path(__file__).parent.parent.parent.parent.parent.parent
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


@given(parsers.parse('historical market data with {data_points:d} periods'))
def set_data_points(test_context, data_points):
    """Set the number of data points for training"""
    test_context['data_points'] = data_points


@given('the recent historical price data is updated:')
def step_load_historical_prices(test_context, docstring):
    """
    Parses the Doc String containing complete time-series price rows (timestamp;open;high;low;close;volume)
    robustly using Pandas, converts it to a time-series DataFrame, and passes each row (pd.Series)
    sequentially to the signal's update method.
    """
    signal = test_context['signal']

    # Use io.StringIO to treat the docstring as a file-like object
    data = io.StringIO(docstring)

    # Read the data using pandas, setting the timestamp as the index and parsing dates
    try:
        df = pandas.read_csv(
            data,
            sep=';',
            header=None,
            names=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            index_col=0,  # Set 'timestamp' column as the DataFrame index
            parse_dates=True,
            dtype={'open': float, 'high': float, 'low': float, 'close': float, 'volume': int}
        )
    except Exception as e:
        pytest.fail(f"Could not parse market data from Doc String: {e}")
        return

    # Iterate over the DataFrame rows. Each 'row' is a pd.Series indexed by timestamp.
    for index, row in df.iterrows():
        # Pass the entire row (pd.Series) to the signal's update method,
        # satisfying the data_point: pd.Series requirement.
        signal.update_with_new_data(row)


@given(parsers.parse('crossover confirmation is {confirmation_periods:d}'))
def set_crossover_confirmation(test_context, confirmation_periods):
    """Set crossover confirmation periods"""
    signal = test_context['signal']
    signal.crossover_confirmation = confirmation_periods


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when(parsers.parse('signal is trained'))
def train_signal(test_context):
    """Train the signal with historical data"""
    data_manager = test_context['data_manager']
    signal = test_context['signal']

    period = test_context['data_points']

    try:
        data_manager.initialize_temporal_pointer(int(period))
        past_data = data_manager.get_past_data()
        market_id = data_manager._active_market
        market_data = past_data[market_id]

        signal.train(market_data)
        test_context['training_error'] = None

    except Exception as e:
        test_context['training_error'] = e
        test_context['signal_ready'] = False


@when(parsers.parse('the current price {current_price:g} is processed'))
def process_current_price(test_context, current_price: float):
    """
    Process the current price by creating a synthetic OHLCV bar and generating a signal.
    The timestamp is set as the series name/index.
    """
    signal = test_context['signal']

    next_timestamp = Timestamp.now()

    # Create synthetic bar where open=high=low=close=current_price (zero-range bar)
    final_data_point = Series({
        'open': current_price,
        'high': current_price,
        'low': current_price,
        'close': current_price,
    }, name=next_timestamp)

    # Update the signal with the new data point
    signal.update_with_new_data(final_data_point)

    # Generate and store the signal for assertion
    test_context['generated_signal'] = signal.generate_signal()

@when('I try to create a MoneyManager instance')
def try_create_invalid_money_manager(test_context):
    config = test_context['config']
    try:
        MoneyManager(config)
        test_context['creation_error'] = None
    except Exception as e:
        test_context['creation_error'] = e

# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then(parsers.parse('signal has readiness {state}'))
def verify_signal_readiness(test_context, state):
    """Verify signal readiness state"""
    signal = test_context['signal']

    expected_ready = (state == 'ready')
    actual_ready = signal.is_ready

    assert actual_ready == expected_ready, \
        f"Expected signal to be {'ready' if expected_ready else 'not ready'}, but it was {'ready' if actual_ready else 'not ready'}"


@then(parsers.parse('signal direction is {expected_direction}'))
def verify_signal_direction(test_context, expected_direction):
    """Assert that the generated signal matches expected direction"""
    actual_signal = test_context.get('generated_signal')

    if actual_signal is None:
        pytest.fail("No signal was generated in the When step")

    expected_enum = MarketSignal[expected_direction]

    assert actual_signal == expected_enum, \
        f"Expected {expected_enum} signal, but received {actual_signal}"

@then(parsers.parse('a configuration error should be raised for {component_type}'))
def check_config_error(test_context, component_type):
    error = test_context['creation_error']
    assert error is not None, "Expected error but MoneyManager created successfully"
    assert component_type in str(error).lower()