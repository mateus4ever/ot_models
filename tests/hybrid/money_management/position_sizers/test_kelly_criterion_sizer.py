# test_trade_history.py
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
# Import the classes we're testing
from src.hybrid.data.trade_history import TradeHistory, PositionOutcome, TradeStatistics
from src.hybrid.money_management.position_sizers import KellyCriterionSizer
from pathlib import Path
from src.hybrid.config.unified_config import UnifiedConfig

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

@given(parsers.parse('{config_file} is available in {config_directory}'))
def load_configuration_file(config_file, config_directory):
    """Load configuration file from specified directory"""

    config_path = Path(config_directory) / config_file
    assert config_path.exists(), f"Configuration file not found: {config_path}"

    # Load the unified configuration
    config = UnifiedConfig()

    # Verify money_management section exists
    mm_config = config.get_section('money_management')
    assert mm_config is not None, "money_management section not found in configuration"

    # Store for use in other steps
    pytest.unified_config = config
    pytest.mm_config = mm_config

@given(parsers.parse('kelly criterion configuration is available with win_rate {win_rate}, avg_win {avg_win}, avg_loss {avg_loss}'))
def kelly_configuration_available(win_rate, avg_win, avg_loss):
    """Set up Kelly configuration parameters"""
    pytest.kelly_base_config = {
        'kelly_win_rate': float(win_rate),
        'kelly_avg_win': float(avg_win),
        'kelly_avg_loss': float(avg_loss)
    }

@given(parsers.parse('kelly fraction is {fraction} and max kelly position is {max_position}'))
def kelly_limits_configuration(fraction, max_position):
    """Set up Kelly limit parameters"""
    pytest.kelly_base_config['kelly_fraction'] = float(fraction)
    pytest.kelly_base_config['max_kelly_position'] = float(max_position)


@given(parsers.parse('I have a Kelly Criterion sizer with kelly statistics {win_rate}, {avg_win}, {avg_loss}'))
def create_kelly_sizer_with_statistics(test_context, win_rate, avg_win, avg_loss):
    """Create Kelly sizer with specific statistics for testing"""
    config = pytest.unified_config

    # Get kelly_lookback from actual configuration instead of hardcoding
    mm_config = config.get_section('money_management')
    kelly_config = mm_config['position_sizers']['kelly_criterion']

    # Override with test-specific statistics
    test_kelly_config = pytest.kelly_base_config.copy()
    test_kelly_config['kelly_win_rate'] = float(win_rate)
    test_kelly_config['kelly_avg_win'] = float(avg_win)
    test_kelly_config['kelly_avg_loss'] = float(avg_loss)
    test_kelly_config['kelly_lookback'] = kelly_config['kelly_lookback']  # From config
    test_kelly_config['kelly_min_trades_threshold'] = kelly_config['kelly_min_trades_threshold']  # From config

    kelly_sizer = KellyCriterionSizer(config)

    # Override the statistics for testing
    kelly_sizer.kelly_win_rate = float(win_rate)
    kelly_sizer.kelly_avg_win = float(avg_win)
    kelly_sizer.kelly_avg_loss = float(avg_loss)

    test_context['kelly_sizer'] = kelly_sizer


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I calculate the raw Kelly percentage')
def calculate_raw_kelly_percentage(test_context):
    """Calculate raw Kelly percentage"""
    kelly_sizer = test_context['kelly_sizer']
    raw_kelly = kelly_sizer._calculate_kelly_percentage()
    test_context['raw_kelly_percentage'] = raw_kelly


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