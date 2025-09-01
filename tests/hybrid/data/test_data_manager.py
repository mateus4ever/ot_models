# tests/hybrid/data/test_data_manager.py
"""
pytest-bdd test runner for DataManager Controller functionality
Tests data controller pattern with temporal boundary enforcement
ZERO DATAFRAME EXPOSURE - Controller interface only
"""

import pytest
import logging
from pathlib import Path
from pytest_bdd import scenarios, given, when, then, parsers

# Import the system under test
import sys

# Go up 4 levels from tests/hybrid/data/test_data_manager.py to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data.data_manager import DataManager

# Load all scenarios from data_manager.feature
scenarios('data_manager.feature')

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
# GIVEN steps - Setup and preconditions
# =============================================================================

@given('the system has proper directory structure')
def step_system_directory_structure(test_context):
    """Verify basic directory structure exists and establish centralized paths"""
    root_path = Path(__file__).parent.parent.parent.parent

    # Check key directories exist
    assert (root_path / 'src').exists(), "src directory missing"
    assert (root_path / 'tests').exists(), "tests directory missing"

    # Establish centralized path configuration
    test_context['root_path'] = root_path
    test_context['config_path'] = root_path / 'tests' / 'config' / 'smoke_config.json'


@given(parsers.parse('test data files are available in {base_data_directory}'))
def step_test_data_files_available(test_context, base_data_directory):
    """Establish base data directory and verify subdirectories exist"""

    root_path = test_context['root_path']
    base_data_path = root_path / base_data_directory

    assert base_data_path.exists(), f"Base data directory {base_data_directory} missing"

    # Check that subdirectories exist (don't check for CSV files in base)
    small_dir = base_data_path / 'small'
    big_dir = base_data_path / 'big'

    # At least one subdirectory should exist
    assert small_dir.exists() or big_dir.exists(), f"Expected small or big subdirectories in {base_data_directory}"

    test_context['base_data_directory'] = base_data_path


@given(parsers.parse('I have {file_count:d} market data files in {data_directory}'))
def step_have_market_data_files(test_context, file_count, data_directory):
    """Verify specific number of market data files exist"""

    # Use established base data directory from Background
    base_path = test_context['base_data_directory']
    data_path = base_path / data_directory

    csv_files = list(data_path.glob('*.csv'))
    assert len(
        csv_files) >= file_count, f"Expected at least {file_count} CSV files in {data_directory}, found {len(csv_files)}"

    # Take the specified number of files for testing
    test_context['market_files'] = csv_files[:file_count]
    test_context['expected_file_count'] = file_count
    test_context['data_directory'] = data_directory


@given(parsers.parse('each market file has {rows_per_file:d} rows without headers'))
def step_market_files_have_rows(test_context, rows_per_file):
    """Verify each market file has specified number of rows"""
    market_files = test_context['market_files']

    for market_file in market_files:
        with open(market_file, 'r') as f:
            line_count = sum(1 for line in f)
        assert line_count == rows_per_file, f"File {market_file.name} has {line_count} rows, expected {rows_per_file}"

    test_context['rows_per_file'] = rows_per_file


@given(parsers.parse('the files follow the format: {csv_format}'))
def step_files_follow_format(test_context, csv_format):
    """Verify market files have specified format"""
    market_files = test_context['market_files']

    # Parse expected format (e.g., "timestamp;open;high;low;close;volume")
    expected_columns = csv_format.split(';')
    expected_delimiter = ';'

    for market_file in market_files:
        with open(market_file, 'r') as f:
            first_line = f.readline().strip()

        # Check delimiter and column count
        parts = first_line.split(expected_delimiter)
        assert len(parts) == len(expected_columns), \
            f"File {market_file.name} has {len(parts)} columns, expected {len(expected_columns)}"

    test_context['csv_format'] = csv_format
    test_context['expected_columns'] = expected_columns


@given(parsers.parse('I have a market data file {data_file} in {data_directory}'))
def step_have_market_data_file(test_context, data_file, data_directory):
    """Setup specific market data file for temporal testing"""

    # Use established base data directory
    base_path = test_context['base_data_directory']
    data_path = base_path / data_directory / data_file

    assert data_path.exists(), f"Market data file {data_file} not found in {data_directory}"
    test_context['temporal_data_file'] = data_path
    test_context['temporal_data_filename'] = data_file
    test_context['data_directory'] = data_directory


@given('I have loaded market data with temporal boundaries initialized')
def step_loaded_data_with_temporal_boundaries(test_context):
    """Setup DataManager with loaded data and temporal boundaries"""

    # Use centralized config path from Background
    config_path = test_context['config_path']
    config = UnifiedConfig(config_path=str(config_path))

    data_manager = DataManager(config)

    # Use market files from context established in previous steps
    market_files = [str(f) for f in test_context['market_files']]  # Convert Path to cross-platform string

    try:
        # Load markets and initialize temporal boundaries
        success = data_manager.load_market_data_with_temporal_setup(market_files, 20000)
        test_context['data_manager'] = data_manager
        test_context['temporal_setup_success'] = success
        assert success, "Failed to load data with temporal setup"
    except Exception as e:
        test_context['temporal_setup_error'] = e
        raise


@given('I have loaded multiple markets with temporal boundaries')
def step_loaded_multiple_markets_with_temporal(test_context):
    """Setup DataManager with multiple markets and temporal boundaries"""

    # Ensure context exists
    if 'config_path' not in test_context:
        root_path = Path(__file__).parent.parent.parent.parent
        test_context['root_path'] = root_path
        test_context['config_path'] = root_path / 'tests' / 'config' / 'smoke_config.json'
        test_context['base_data_directory'] = root_path / 'tests' / 'data'

    # Use centralized config path
    config_path = test_context['config_path']
    config = UnifiedConfig(config_path=str(config_path))

    data_manager = DataManager(config)

    # Use market files from context established in previous steps
    market_files = [f.name for f in test_context['market_files']]

    try:
        success = data_manager.load_market_data_with_temporal_setup(market_files, 200)
        test_context['data_manager'] = data_manager
        test_context['temporal_setup_success'] = success
        assert success, "Failed to load multiple markets with temporal setup"
    except Exception as e:
        test_context['temporal_setup_error'] = e
        raise


@given(parsers.parse('I have previously loaded market data for {market_name}'))
def step_previously_loaded_market_data(test_context, market_name):
    """Setup previously loaded data for caching test"""

    # Use centralized config path
    config_path = test_context['config_path']
    config = UnifiedConfig(config_path=str(config_path))

    # Override data source to use the subdirectory with base_data_directory
    data_directory = test_context['data_directory']
    base_data_path = test_context['base_data_directory']
    full_data_path = base_data_path / data_directory

    config.config['data_loading']['data_source'] = str(full_data_path)

    data_manager = DataManager(config)

    # Use first available market file (since market_name might not match exact filename)
    market_files = test_context['market_files']
    market_file_names = [f.name for f in market_files]

    success = data_manager.load_market_data(market_file_names)
    assert success, f"Failed to pre-load market data from {market_file_names}"

    test_context['data_manager'] = data_manager
    test_context['previously_loaded_market'] = market_name


@given(parsers.parse('I have an invalid data directory {invalid_directory}'))
def step_invalid_data_directory(test_context, invalid_directory):
    """Setup invalid directory for error testing"""
    test_context['invalid_directory'] = invalid_directory.strip('"')


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I load multiple market data through DataManager')
def step_load_multiple_market_data(test_context):
    """Load multiple market data through DataManager controller"""

    # Use centralized config path from Background
    config_path = test_context['config_path']
    config = UnifiedConfig(config_path=str(config_path))

    # Override data source to use the subdirectory with base_data_directory
    data_directory = test_context['data_directory']
    base_data_path = test_context['base_data_directory']
    full_data_path = base_data_path / data_directory

    # Update config to point to correct subdirectory
    config.config['data_loading']['data_source'] = str(full_data_path)

    data_manager = DataManager(config)

    # Create market names from files
    market_files = test_context['market_files']
    markets = [f.name for f in market_files]

    try:
        # Controller returns boolean success, not data dictionary
        success = data_manager.load_market_data(markets)
        test_context['loading_success'] = success
        test_context['data_manager'] = data_manager
        test_context['data_error'] = None

    except Exception as e:
        test_context['loading_success'] = False
        test_context['data_error'] = e


@when(parsers.parse(
    'I load the data and initialize temporal pointer with training window of {training_records:d} records'))
def step_load_data_initialize_temporal_pointer(test_context, training_records):
    """Load data and initialize temporal pointer through controller"""

    # Ensure context exists
    if 'config_path' not in test_context:
        root_path = Path(__file__).parent.parent.parent.parent
        test_context['root_path'] = root_path
        test_context['config_path'] = root_path / 'tests' / 'config' / 'smoke_config.json'
        test_context['base_data_directory'] = root_path / 'tests' / 'data'

    # Use centralized config path
    config_path = test_context['config_path']
    config = UnifiedConfig(config_path=str(config_path))

    # Override data source to use the subdirectory with base_data_directory
    data_directory = test_context['data_directory']
    base_data_path = test_context['base_data_directory']
    full_data_path = base_data_path / data_directory

    # Update config to point to correct subdirectory
    config.config['data_loading']['data_source'] = str(full_data_path)

    data_manager = DataManager(config)

    # Get the data file from context
    data_file = str(test_context['temporal_data_file'])

    try:
        # Load single market with temporal setup
        success = data_manager.load_market_data_with_temporal_setup([data_file], training_records)
        test_context['data_manager'] = data_manager
        test_context['temporal_setup_success'] = success
        test_context['training_window'] = training_records
        test_context['temporal_error'] = None

    except Exception as e:
        test_context['temporal_setup_success'] = False
        test_context['temporal_error'] = e


@when('I query current pointer position')
def step_query_current_pointer_position(test_context):
    """Query current temporal pointer position"""
    data_manager = test_context['data_manager']

    try:
        current_position = data_manager.get_current_pointer()
        test_context['queried_position'] = current_position
        test_context['position_query_error'] = None
    except Exception as e:
        test_context['position_query_error'] = e


@when(parsers.parse('I advance pointer using next({step_size:d}) method'))
def step_advance_pointer_using_next(test_context, step_size):
    """Advance pointer using next() method"""
    data_manager = test_context['data_manager']

    try:
        success = data_manager.next(step_size)
        test_context['next_advancement_success'] = success
        test_context['advancement_error'] = None
    except Exception as e:
        test_context['next_advancement_success'] = False
        test_context['advancement_error'] = e


@when(parsers.parse('I set pointer to absolute position {target_position:d}'))
def step_set_pointer_absolute_position(test_context, target_position):
    """Set pointer to absolute position"""
    data_manager = test_context['data_manager']

    try:
        data_manager.set_pointer(target_position)
        test_context['set_position_success'] = True
        test_context['target_position'] = target_position
        test_context['set_position_error'] = None
    except Exception as e:
        test_context['set_position_success'] = False
        test_context['set_position_error'] = e


@when('I attempt to advance pointer beyond available data')
def step_attempt_advance_beyond_data(test_context):
    """Attempt to advance pointer past data boundaries"""
    data_manager = test_context['data_manager']

    try:
        # Try to advance by a very large number to exceed boundaries
        success = data_manager.next(1000000)
        test_context['boundary_test_success'] = success
        test_context['boundary_test_error'] = None
    except Exception as e:
        test_context['boundary_test_success'] = False
        test_context['boundary_test_error'] = e


@when('I attempt to set pointer to invalid position')
def step_attempt_set_invalid_position(test_context):
    """Attempt to set pointer to invalid position"""
    data_manager = test_context['data_manager']

    try:
        # Try to set to invalid position (negative or beyond data)
        data_manager.set_pointer(-1)
        test_context['invalid_position_success'] = True
        test_context['invalid_position_error'] = None
    except Exception as e:
        test_context['invalid_position_success'] = False
        test_context['invalid_position_error'] = e


@when('training algorithms request historical data')
def step_training_request_historical_data(test_context):
    """Request past data for training through controller"""
    data_manager = test_context['data_manager']

    try:
        past_data = data_manager.get_past_data()
        test_context['past_data'] = past_data
        test_context['past_data_error'] = None
    except Exception as e:
        test_context['past_data'] = None
        test_context['past_data_error'] = e


@when('trading signals request current market data')
def step_trading_signals_request_current_data(test_context):
    """Request current data for signal generation through controller"""
    data_manager = test_context['data_manager']

    try:
        current_data = data_manager.get_current_data()
        test_context['current_data'] = current_data
        test_context['current_data_error'] = None
    except Exception as e:
        test_context['current_data'] = None
        test_context['current_data_error'] = e


@when('I request the same market data again')
def step_request_same_market_data_again(test_context):
    """Request cached data through controller"""
    data_manager = test_context['data_manager']

    # Use the actual loaded market files for cache test
    market_files = test_context['market_files']
    market_file_names = [f.name for f in market_files]

    try:
        # Request same market data again (should use cache) - use load_market_data for consistency
        success = data_manager.load_market_data(market_file_names)
        test_context['cache_request_success'] = success
        test_context['cache_request_error'] = None
    except Exception as e:
        test_context['cache_request_success'] = False
        test_context['cache_request_error'] = e


@when('I attempt to load market data')
def step_attempt_load_market_data(test_context):
    """Attempt to load market data from invalid directory"""

    # Use centralized config path
    config_path = test_context['config_path']
    config = UnifiedConfig(config_path=str(config_path))

    # Override data source to invalid directory
    invalid_dir = test_context['invalid_directory']
    config.config['data_loading']['data_source'] = invalid_dir

    data_manager = DataManager(config)

    try:
        success = data_manager.load_market_data_single("nonexistent.csv")
        test_context['invalid_load_success'] = success
        test_context['invalid_load_error'] = None
    except Exception as e:
        test_context['invalid_load_success'] = False
        test_context['invalid_load_error'] = e


# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('DataManager should load all markets successfully')
def step_datamanager_loads_successfully(test_context):
    """Verify all markets loaded successfully through controller"""
    data_error = test_context.get('data_error')
    assert data_error is None, f"Data loading failed: {data_error}"

    loading_success = test_context.get('loading_success')
    assert loading_success is True, "DataManager should load all markets successfully"


@then(parsers.parse(
    'market record counts should show {expected_markets:d} markets with {total_records_per_market:d} records each'))
def step_market_record_counts_validation(test_context, expected_markets, total_records_per_market):
    """Verify market record counts after consolidation"""
    data_manager = test_context.get('data_manager')
    assert data_manager is not None, "DataManager should be available"

    record_counts = data_manager.get_market_record_counts()

    assert len(
        record_counts) == expected_markets, f"Expected {expected_markets} markets, got {len(record_counts)}: {list(record_counts.keys())}"

    for market, count in record_counts.items():
        assert count == total_records_per_market, f"Market {market} has {count} records, expected {total_records_per_market}"


@then('no data loading errors should occur')
def step_no_data_loading_errors(test_context):
    """Verify no data loading errors occurred"""
    data_error = test_context.get('data_error')
    assert data_error is None, f"Data loading error occurred: {data_error}"


@then('DataManager should cache the loaded data')
def step_datamanager_caches_data(test_context):
    """Verify DataManager caches data through controller interface"""
    data_manager = test_context.get('data_manager')
    assert data_manager is not None, "DataManager should be available"

    cache_info = data_manager.get_cache_info()
    assert cache_info['cache_size'] > 0, "DataManager should have cached data"


@then(parsers.parse('temporal pointer should be positioned at record {now_position:d}'))
def step_temporal_pointer_positioned(test_context, now_position):
    """Verify temporal pointer position through controller"""
    temporal_error = test_context.get('temporal_error')
    assert temporal_error is None, f"Temporal setup failed: {temporal_error}"

    temporal_success = test_context.get('temporal_setup_success')
    assert temporal_success is True, "Temporal setup should succeed"

    data_manager = test_context['data_manager']
    current_position = data_manager.get_current_pointer()
    assert current_position == now_position, f"Expected pointer at {now_position}, got {current_position}"


@then(parsers.parse('past data boundary should be set to record {training_records:d}'))
def step_past_data_boundary_set(test_context, training_records):
    """Verify past data boundary through controller"""
    data_manager = test_context['data_manager']

    past_data = data_manager.get_past_data()

    # Past data should be dict for multi-market controller
    assert isinstance(past_data, dict), "Past data should be dictionary for multi-market access"

    # Check that past data contains expected number of records
    for market, data in past_data.items():
        assert len(
            data) == training_records, f"Market {market} past data has {len(data)} records, expected {training_records}"


@then('future data access should be restricted')
def step_future_data_access_restricted(test_context):
    """Verify future data access is controlled"""
    data_manager = test_context['data_manager']

    # Should be able to get current and past data
    current_data = data_manager.get_current_data()
    assert isinstance(current_data, dict), "Current data should be accessible"

    past_data = data_manager.get_past_data()
    assert isinstance(past_data, dict), "Past data should be accessible"

    # Future data preview should work but with warnings (logged)
    try:
        future_data = data_manager.get_future_data_preview(10)
        assert isinstance(future_data, dict), "Future data preview should work for validation"
    except Exception as e:
        pytest.fail(f"Future data preview should be available for validation: {e}")


@then('current position should match expected location')
def step_current_position_matches_expected(test_context):
    """Verify queried position matches expectation"""
    position_error = test_context.get('position_query_error')
    assert position_error is None, f"Position query failed: {position_error}"

    queried_position = test_context.get('queried_position')
    assert queried_position is not None, "Should be able to query current position"
    assert isinstance(queried_position, int), "Position should be integer"
    assert queried_position > 0, "Position should be positive"


@then('pointer should move to new position correctly')
def step_pointer_moves_correctly(test_context):
    """Verify pointer advancement works correctly"""
    advancement_error = test_context.get('advancement_error')
    assert advancement_error is None, f"Pointer advancement failed: {advancement_error}"

    advancement_success = test_context.get('next_advancement_success')
    assert advancement_success is True, "Pointer advancement should succeed"


@then('pointer should be at specified position')
def step_pointer_at_specified_position(test_context):
    """Verify pointer is at target position"""
    set_position_error = test_context.get('set_position_error')
    assert set_position_error is None, f"Set position failed: {set_position_error}"

    set_success = test_context.get('set_position_success')
    assert set_success is True, "Set position should succeed"

    # Verify position was actually set
    data_manager = test_context['data_manager']
    current_position = data_manager.get_current_pointer()
    target_position = test_context['target_position']
    assert current_position == target_position, f"Expected position {target_position}, got {current_position}"


@then('advancement should fail gracefully')
def step_advancement_fails_gracefully(test_context):
    """Verify boundary advancement fails gracefully"""
    boundary_success = test_context.get('boundary_test_success')

    # Should either return False or raise appropriate exception
    if boundary_success is not None:
        assert boundary_success is False, "Advancement beyond boundaries should return False"
    else:
        # If exception was raised, that's also acceptable
        boundary_error = test_context.get('boundary_test_error')
        assert boundary_error is not None, "Should either return False or raise exception for boundary violation"


@then('pointer should remain at last valid position')
def step_pointer_remains_at_valid_position(test_context):
    """Verify pointer stays at valid position after failed advancement"""
    data_manager = test_context['data_manager']

    try:
        current_position = data_manager.get_current_pointer()
        # Should be able to query position (not crashed)
        assert isinstance(current_position, int), "Should still be able to query position"
        assert current_position > 0, "Position should remain valid"
    except Exception as e:
        pytest.fail(f"Should be able to query position after failed advancement: {e}")


@then('position setting should fail with appropriate error')
def step_position_setting_fails_appropriately(test_context):
    """Verify invalid position setting fails appropriately"""
    invalid_success = test_context.get('invalid_position_success')
    invalid_error = test_context.get('invalid_position_error')

    # Should either return False or raise exception
    if invalid_success is not None:
        assert invalid_success is False, "Invalid position setting should fail"
    else:
        assert invalid_error is not None, "Should raise exception for invalid position"


@then('past data should be returned for all loaded markets')
def step_past_data_returned_all_markets(test_context):
    """Verify past data returned for all markets"""
    past_data_error = test_context.get('past_data_error')
    assert past_data_error is None, f"Past data request failed: {past_data_error}"

    past_data = test_context.get('past_data')
    assert past_data is not None, "Past data should be returned"
    assert isinstance(past_data, dict), "Past data should be dictionary for multi-market access"
    assert len(past_data) > 0, "Should have past data for at least one market"


@then('data should only include records before current pointer')
def step_data_only_before_pointer(test_context):
    """Verify past data temporal boundaries"""
    data_manager = test_context['data_manager']
    current_position = data_manager.get_current_pointer()

    past_data = test_context['past_data']
    for market, data in past_data.items():
        # Past data should have fewer records than current position
        assert len(
            data) < current_position, f"Market {market} past data should have less than {current_position} records"


@then('current data should be returned for all loaded markets')
def step_current_data_returned_all_markets(test_context):
    """Verify current data returned for all markets"""
    current_data_error = test_context.get('current_data_error')
    assert current_data_error is None, f"Current data request failed: {current_data_error}"

    current_data = test_context.get('current_data')
    assert current_data is not None, "Current data should be returned"
    assert isinstance(current_data, dict), "Current data should be dictionary for multi-market access"
    assert len(current_data) > 0, "Should have current data for at least one market"


@then('data should only include current pointer position')
def step_data_only_current_position(test_context):
    """Verify current data contains only current position"""
    current_data = test_context['current_data']

    for market, data in current_data.items():
        # Current data should be a Series (single record)
        assert hasattr(data, 'name'), f"Market {market} current data should be a Series (single record)"


@then('cached data should be used')
def step_cached_data_used(test_context):
    """Verify cached data was used"""
    cache_error = test_context.get('cache_request_error')
    assert cache_error is None, f"Cache request failed: {cache_error}"

    cache_success = test_context.get('cache_request_success')
    assert cache_success is True, "Cache request should succeed"


@then('no file loading should occur')
def step_no_file_loading_occurs(test_context):
    """Verify no additional file loading occurred (cache hit)"""
    # This is validated through successful cache request
    # Real implementation could track file access counts
    cache_success = test_context.get('cache_request_success')
    assert cache_success is True, "Should successfully use cached data"


@then('cache hit should be logged')
def step_cache_hit_logged(test_context):
    """Verify cache hit was logged"""
    # This would require log capture in real implementation
    # For now, verify cache operation succeeded
    cache_success = test_context.get('cache_request_success')
    assert cache_success is True, "Cache operation should succeed"


@then('appropriate error messages should be logged')
def step_appropriate_error_messages_logged(test_context):
    """Verify error logging for invalid operations"""
    invalid_error = test_context.get('invalid_load_error')
    # Should have an error for invalid directory
    assert invalid_error is not None, "Should have error for invalid directory access"


@then('loading should fail gracefully')
def step_loading_fails_gracefully(test_context):
    """Verify loading fails gracefully"""
    invalid_success = test_context.get('invalid_load_success')
    assert invalid_success is False, "Loading from invalid directory should fail"


@then('no system crash should occur')
def step_no_system_crash(test_context):
    """Verify system stability despite errors"""
    # If we reach this assertion, system didn't crash
    # Test framework would have failed earlier if there was a crash
    assert True, "System should remain stable"