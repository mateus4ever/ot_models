"""
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_bdd import scenarios, given, parsers, when, then

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.predictors.vasicek.triangular_arbitrage_predictor import TriangularArbitragePredictor
from src.hybrid.predictors.vasicek.vasicek_model import VasicekModel

# Load scenarios from the strategy_factory.feature
scenarios('triangular_arbitrage_predictor.feature')

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
# GIVEN steps
# =============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""
    root_path = Path(__file__).parent.parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    unified_config = UnifiedConfig(config_path=str(config_path), environment="test")
    test_context['unified_config'] = unified_config


@given('I have a TriangularArbitrage model instance')
def create_predictor_instance(test_context):
    """Create predictor instance"""
    config = test_context['unified_config']
    test_context['predictor'] = TriangularArbitragePredictor(config)


@given(parsers.parse('predictor is calibrated with theta={theta}, sigma={sigma}, kappa={kappa}'))
def set_predictor_calibrated(test_context, theta, sigma, kappa):
    predictor = test_context['predictor']

    predictor.vasicek_model.theta = float(theta)
    predictor.vasicek_model.sigma = float(sigma)
    predictor.vasicek_model.kappa = float(kappa)
    predictor.vasicek_model.half_life = np.log(2) / float(kappa)
    predictor.vasicek_model.is_calibrated = True

    predictor._is_trained = True


@given('predictor has no open position')
def set_no_position(test_context):
    """Ensure predictor has no open position"""
    predictor = test_context['predictor']
    predictor.state.position = None
    predictor.state.entry_spread_pips = None
    predictor.state.entry_z_score = None


@given(parsers.parse('predictor has open position "{position}" with entry_z_score={entry_z}'))
def set_open_position_with_z(test_context, position, entry_z):
    """Set predictor to have open position with entry z-score"""
    predictor = test_context['predictor']
    predictor.state.position = position
    predictor.state.entry_z_score = float(entry_z)
    predictor.state.entry_spread_pips = float(entry_z) * predictor.vasicek_model.sigma * predictor.pip_multiplier


@given(parsers.parse('predictor has position state "{position}"'))
def set_position_state(test_context, position):
    """Set predictor position state (None or position name)"""
    predictor = test_context['predictor']
    if position == "None":
        predictor.state.position = None
        predictor.state.entry_spread_pips = None
        predictor.state.entry_z_score = None
    else:
        predictor.state.position = position
        predictor.state.entry_z_score = predictor.entry_threshold
        predictor.state.entry_spread_pips = predictor.state.entry_z_score * predictor.vasicek_model.sigma * predictor.pip_multiplier


@given('predictor is NOT calibrated')
def set_predictor_not_calibrated(test_context):
    predictor = test_context['predictor']
    predictor._is_trained = False
    predictor.vasicek_model.is_calibrated = False


# =============================================================================
# WHEN steps
# =============================================================================

@when('get_required_markets is called')
def call_get_required_markets(test_context):
    """Call get_required_markets"""
    predictor = test_context['predictor']
    test_context['required_markets'] = predictor.get_required_markets()


@when(parsers.parse('synthetic price is calculated for leg1={leg1} and leg2={leg2}'))
def calculate_synthetic(test_context, leg1, leg2):
    """Calculate synthetic price"""
    predictor = test_context['predictor']
    result = predictor.calculate_synthetic_price(float(leg1), float(leg2))
    test_context['synthetic_price'] = result


@when(parsers.parse('spread is calculated for target={target}, leg1={leg1}, leg2={leg2}'))
def calculate_spread(test_context, target, leg1, leg2):
    """Calculate spread"""
    predictor = test_context['predictor']
    result = predictor.calculate_spread(float(target), float(leg1), float(leg2))
    test_context['spread'] = result


@when(parsers.parse('signal is generated for z_score={z_score}'))
def generate_signal(test_context, z_score):
    """Generate signal for given z_score"""
    predictor = test_context['predictor']
    z = float(z_score)

    signal, confidence = predictor._generate_signal(z)

    spread_pips = z * predictor.vasicek_model.sigma * predictor.pip_multiplier
    predictor._update_state(signal, spread_pips, z)

    test_context['signal'] = signal
    test_context['confidence'] = confidence


@when('predictor is reset')
def reset_predictor(test_context):
    """Reset predictor state"""
    predictor = test_context['predictor']
    predictor.reset()

@when('predictor generates prediction')
def generate_prediction(test_context):
    """Generate prediction (for uncalibrated test)"""
    predictor = test_context['predictor']
    prediction = predictor._create_neutral_prediction()
    test_context['signal'] = prediction['signal']
    test_context['confidence'] = prediction['confidence']

@when('predictions are generated for z_score sequence:')
def generate_prediction_sequence(test_context, datatable):
    """Generate predictions for sequence of z_scores"""
    predictor = test_context['predictor']
    results = []

    # Convert list of lists to list of dicts
    headers = datatable[0]
    rows = [dict(zip(headers, row)) for row in datatable[1:]]

    for row in rows:
        z_score = float(row['z_score'])
        expected_signal = row['expected_signal']
        expected_position = row['expected_position']

        signal, confidence = predictor._generate_signal(z_score)
        spread_pips = z_score * predictor.vasicek_model.sigma * predictor.pip_multiplier
        predictor._update_state(signal, spread_pips, z_score)

        actual_position = predictor.state.position

        results.append({
            'step': row['step'],
            'z_score': z_score,
            'expected_signal': expected_signal,
            'actual_signal': signal,
            'expected_position': None if expected_position == "None" else expected_position,
            'actual_position': actual_position
        })

    test_context['sequence_results'] = results


# =============================================================================
# THEN steps - Assertions
# =============================================================================

# =============================================================================
# THEN steps
# =============================================================================

@then(parsers.parse('predictor should have target_market "{expected}"'))
def assert_target_market(test_context, expected):
    """Assert target market"""
    predictor = test_context['predictor']
    assert predictor.target_market == expected, \
        f"Expected target_market '{expected}', got '{predictor.target_market}'"


@then(parsers.parse('predictor should have leg1_market "{expected}"'))
def assert_leg1_market(test_context, expected):
    """Assert leg1 market"""
    predictor = test_context['predictor']
    assert predictor.leg1_market == expected, \
        f"Expected leg1_market '{expected}', got '{predictor.leg1_market}'"


@then(parsers.parse('predictor should have leg2_market "{expected}"'))
def assert_leg2_market(test_context, expected):
    """Assert leg2 market"""
    predictor = test_context['predictor']
    assert predictor.leg2_market == expected, \
        f"Expected leg2_market '{expected}', got '{predictor.leg2_market}'"


@then(parsers.parse('predictor should have entry_threshold {expected}'))
def assert_entry_threshold(test_context, expected):
    """Assert entry threshold"""
    predictor = test_context['predictor']
    assert predictor.entry_threshold == float(expected), \
        f"Expected entry_threshold {expected}, got {predictor.entry_threshold}"


@then(parsers.parse('predictor should have exit_threshold {expected}'))
def assert_exit_threshold(test_context, expected):
    """Assert exit threshold"""
    predictor = test_context['predictor']
    assert predictor.exit_threshold == float(expected), \
        f"Expected exit_threshold {expected}, got {predictor.exit_threshold}"


@then(parsers.parse('predictor should have pip_multiplier {expected}'))
def assert_pip_multiplier(test_context, expected):
    """Assert pip multiplier"""
    predictor = test_context['predictor']
    assert predictor.pip_multiplier == float(expected), \
        f"Expected pip_multiplier {expected}, got {predictor.pip_multiplier}"


@then(parsers.parse('predictor should have lookback_window {expected}'))
def assert_lookback_window(test_context, expected):
    """Assert lookback window"""
    predictor = test_context['predictor']
    assert predictor.lookback_window == int(expected), \
        f"Expected lookback_window {expected}, got {predictor.lookback_window}"


@then('predictor should be marked as not calibrated')
def assert_not_calibrated(test_context):
    """Assert predictor is not trained"""
    predictor = test_context['predictor']
    assert not predictor.is_trained, "Expected predictor to be not trained"

@then(parsers.parse('result should contain "{market}"'))
def assert_result_contains_market(test_context, market):
    """Assert result contains market"""
    markets = test_context['required_markets']
    assert market in markets, f"Expected '{market}' in {markets}"


@then(parsers.parse('result should have exactly {count} markets'))
def assert_market_count(test_context, count):
    """Assert market count"""
    markets = test_context['required_markets']
    assert len(markets) == int(count), f"Expected {count} markets, got {len(markets)}"


@then(parsers.parse('synthetic price should be approximately {expected} within {tolerance}'))
def assert_synthetic_price(test_context, expected, tolerance):
    """Assert synthetic price within tolerance"""
    actual = test_context['synthetic_price']
    assert abs(actual - float(expected)) < float(tolerance), \
        f"Expected {expected}, got {actual}"


@then(parsers.parse('spread should be approximately {expected} within {tolerance}'))
def assert_spread(test_context, expected, tolerance):
    """Assert spread within tolerance"""
    actual = test_context['spread']
    assert abs(actual - float(expected)) < float(tolerance), \
        f"Expected {expected}, got {actual}"


@then(parsers.parse('signal should be "{expected}"'))
def assert_signal(test_context, expected):
    """Assert signal"""
    actual = test_context['signal']
    assert actual == expected, f"Expected signal '{expected}', got '{actual}'"


@then(parsers.parse('confidence should be approximately {expected} within {tolerance}'))
def assert_confidence(test_context, expected, tolerance):
    """Assert confidence within tolerance"""
    actual = test_context['confidence']
    assert abs(actual - float(expected)) < float(tolerance), \
        f"Expected confidence {expected}, got {actual}"

@then(parsers.parse('predictor state position should be "{expected}"'))
def assert_state_position(test_context, expected):
    """Assert predictor state position"""
    predictor = test_context['predictor']
    expected_val = None if expected == "None" else expected
    assert predictor.state.position == expected_val, \
        f"Expected position '{expected_val}', got '{predictor.state.position}'"


@then('predictor state position should be None')
def assert_state_position_none(test_context):
    """Assert predictor state position is None"""
    predictor = test_context['predictor']
    assert predictor.state.position is None, \
        f"Expected position None, got '{predictor.state.position}'"


@then('predictor state entry_z_score should be None')
def assert_state_entry_z_none(test_context):
    """Assert predictor state entry z-score is None"""
    predictor = test_context['predictor']
    assert predictor.state.entry_z_score is None, \
        f"Expected entry_z_score None, got {predictor.state.entry_z_score}"


@then('predictor state entry_spread_pips should be None')
def assert_state_entry_spread_none(test_context):
    """Assert predictor state entry spread pips is None"""
    predictor = test_context['predictor']
    assert predictor.state.entry_spread_pips is None, \
        f"Expected entry_spread_pips None, got {predictor.state.entry_spread_pips}"


@then('all signals should match expected')
def assert_all_signals_match(test_context):
    """Assert all signals in sequence match expected"""
    results = test_context['sequence_results']
    for r in results:
        assert r['actual_signal'] == r['expected_signal'], \
            f"Step {r['step']}: Expected signal '{r['expected_signal']}', got '{r['actual_signal']}'"


@then('all position states should match expected')
def assert_all_positions_match(test_context):
    """Assert all position states in sequence match expected"""
    results = test_context['sequence_results']
    for r in results:
        assert r['actual_position'] == r['expected_position'], \
            f"Step {r['step']}: Expected position '{r['expected_position']}', got '{r['actual_position']}'"