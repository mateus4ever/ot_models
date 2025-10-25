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
                    'fast_period': fast_period,
                    'slow_period': slow_period
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



@given(parsers.parse('historical market data with {data_points} periods'))
def load_market_data(test_context, data_points):
    """Load market data using DataManager"""

    test_context['data_points'] = data_points


@given('the recent historical price data is updated:')
def step_load_historical_prices(test_context,docstring):
    """
    Parses the Doc String containing complete time-series price rows (timestamp;open;high;low;close;volume)
    robustly using Pandas, converts it to a time-series DataFrame, and passes each row (pd.Series)
    sequentially to the signal's update method.
    """
    signal  = test_context['signal']

    # Use io.StringIO to treat the docstring as a file-like object
    data = io.StringIO(docstring)

    # Read the data using pandas, settg the timestamp as the index and parsing dates
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

@given(parsers.parse('crossover confirmation is {confirmation_periods}'))
def set_crossover_confirmation(test_context, confirmation_periods):
    """Set crossover confirmation periods"""
    signal = test_context['signal']
    signal.crossover_confirmation = int(confirmation_periods)
# =============================================================================
# WHEN steps - Actions
# =============================================================================
@when(parsers.parse('signal is trained'))
def train_signal(test_context):
    data_manager = test_context['data_manager']
    signal = test_context['signal']

    period = test_context['data_points']

    try:
        data_manager.initialize_temporal_pointer(
            data_manager._active_market_data,
            int(period)
        )
        past_data = data_manager.get_past_data()
        market_id = data_manager._active_market  # Get active market name
        market_data = past_data[market_id]  # Extract single DataFrame

        signal.train(market_data)
        test_context['training_error'] = None

    except Exception as e:
        test_context['training_error'] = e
        test_context['signal_ready'] = False

@when('the current price {current_price:g} is processed')
def step_process_current_price(context, current_price):
    """
    Processes the final, current price, which is expected to trigger the signal event.
    The ':g' type converter handles floats/decimal numbers.
    """
    if not hasattr(context, 'signal'):
        raise AssertionError("Signal object is missing in context. Check preceding Given steps.")

    # 1. Process the price (this will update the MAs for the final time)
    context.signal.update_price(current_price)

    # 2. Generate the signal based on the current state (MAs cross)
    # Store the result in the context for the 'Then' step to assert
    context.generated_signal = context.signal.get_current_signal()


@when(parsers.parse('the current price {current_price:g} is processed'))
def process_current_price(test_context, current_price: float):
    """
    The final price update that should cause the Golden Cross.
    It creates a full, single-row pd.Series for the final update to satisfy the API.
    """
    # ** THE ANSWER: The timestamp is the index (the series name). **
    # We use a placeholder unique timestamp, which must be set using the 'name' argument.
    signal = test_context['signal']

    next_timestamp = Timestamp.now()

    final_data_point = Series({
        'open': current_price,
        'high': current_price,
        'low': current_price,
        'close': current_price,
    }, name=next_timestamp)  # <-- Timestamp is set here as the series index

    signal.update_with_new_data(final_data_point)


# =============================================================================
# THEN steps - Assertions
# =============================================================================


@then(parsers.parse('signal should be {expected_readiness}'))
def verify_signal_readiness(test_context, expected_readiness):
    """Verify signal readiness state"""
    signal = test_context['signal']

    expected_ready = (expected_readiness == 'ready')
    actual_ready = signal.is_ready

    assert actual_ready == expected_ready, \
        f"Expected signal to be {'ready' if expected_ready else 'not ready'}, but it was {'ready' if actual_ready else 'not ready'}"


@then('signal should be BUY')
def step_assert_signal_is_buy(context):
    """
    Asserts that the signal generated in the 'When' step matches the expected BUY signal.
    """
    expected_signal = 'BUY'

    if not hasattr(context, 'generated_signal'):
        raise AssertionError("Signal was not generated in the 'When' step.")

    # 1. Perform the final assertion
    assert context.generated_signal == expected_signal, (
        f"Expected signal to be '{expected_signal}', but received '{context.generated_signal}'."
    )

    from pytest_bdd import given, then

@given('the SMA signal is initialized')
def step_init_sma(test_context):
    test_context['sma_initialized'] = True

@given('the historical price data is:')
def step_load_historical_prices(test_context, docstring):
    print("DocString content:", docstring)  # Debug output
    test_context['prices'] = [float(p.strip()) for p in docstring.split(',')]

@then('the signal should process the data')
def step_process_data(test_context):
    assert hasattr(test_context, 'prices')
    assert len(test_context.prices) == 5

@then('signal should be BUY')
def signal_should_be_buy(test_context):

    signal = test_context['signal']
    """Asserts that the signal generated is 'BUY' by calling the final method."""
    # Call the final generation method and assert the result
    actual_signal = signal.generate_signal()
    assert actual_signal == 'BUY', f"Expected BUY signal, but received {actual_signal}"
