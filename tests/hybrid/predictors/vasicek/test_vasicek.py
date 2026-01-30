"""
Vasicek Model BDD Step Definitions

Step definitions for vasicek.feature testing:
- Model calibration (successful and edge cases)
- Z-score calculations
- Mean reversion validation
- Half-life calculations
- Trading threshold generation

Uses UnifiedConfig for configuration, no data_manager.
Fixtures loaded from tests/config/prediction/vasicek/fixtures.json
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_bdd import scenarios, given, parsers, when, then

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.predictors.vasicek.vasicek_model import VasicekModel

# Load scenarios from the strategy_factory.feature
scenarios('vasicek.feature')

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

@given('I have a Vasicek model instance')
def create_vasicek_model_instance(test_context):
    """Create a fresh VasicekModel instance."""

    unified_config = test_context.get('unified_config')
    if not unified_config:
        raise ValueError("UnifiedConfig is required to create VasicekModel")

    test_context['model'] = VasicekModel(unified_config)

@given(parsers.parse('a synthetic O-U series with n={n}, kappa={kappa}, theta={theta}, sigma={sigma}, seed={seed}'))
def create_ou_series(test_context, n, kappa, theta, sigma, seed):
    """Generate synthetic Ornstein-Uhlenbeck series."""
    test_context['series'] = generate_ou_series(int(n), float(kappa), float(theta), float(sigma), int(seed))

@given(parsers.parse('I generate non-stationary {series_type} series with n={n}, seed={seed}'))
def create_non_stationary_series(test_context, series_type, n, seed):
    """Generate non-stationary series."""
    if series_type == "random_walk":
        test_context['series'] = generate_random_walk(int(n), int(seed))
    elif series_type == "trending":
        test_context['series'] = generate_trending_series(int(n), int(seed))
    else:
        raise ValueError(f"Unknown series type: {series_type}")


@given(parsers.parse(
    'test series of type "{series_type}" with {n_points} points, start {start}, end {end}, noise {noise}, seed {seed}'))
def create_test_series(test_context, series_type, n_points, start, end, noise, seed):
    """Create test series from parameters"""
    np.random.seed(int(seed))

    n = int(n_points)
    start_val = float(start)
    end_val = float(end)
    noise_val = float(noise)

    if series_type == 'linear_trend':
        series = pd.Series(
            np.linspace(start_val, end_val, n) + np.random.normal(0, noise_val, n)
        )
    elif series_type == 'random_walk':
        series = pd.Series(
            np.cumsum(np.random.normal(0, noise_val, n)) + start_val
        )
    elif series_type == 'explosive':
        series = pd.Series(
            np.exp(np.linspace(start_val, end_val, n)) * 0.001 + np.random.normal(0, noise_val, n)
        )
    else:
        raise ValueError(f"Unknown series type: {series_type}")

    test_context['series'] = series

@given(parsers.parse('Vasicek model is calibrated with theta = {theta}, sigma = {sigma} and kappa = {kappa}'))
def set_model_theta_sigma_kappa(test_context, theta, sigma, kappa):
    """Set theta, sigma and kappa directly for threshold testing."""
    model = test_context['model']
    model.theta = float(theta)
    model.sigma = float(sigma)
    model.kappa = float(kappa)
    model.is_calibrated = True


# ============================================================================
# ERROR HANDLING
# ============================================================================

@given('Vasicek model is NOT calibrated')
def create_uncalibrated_model(test_context):
    """Ensure model is not calibrated"""
    model = test_context['model']
    assert not model.is_calibrated, "Model should not be calibrated"


# =============================================================================
# HALF-LIFE STEPS
# =============================================================================

@given(parsers.parse('Vasicek model is calibrated with kappa = {kappa}'))
def set_model_kappa_directly(test_context, kappa):
    """Set kappa directly for half-life testing (bypasses full calibration)."""
    model = test_context['model']
    kappa_val = float(kappa)

    model.kappa = kappa_val
    model.theta = 0.0  # Not relevant for half-life calculation
    model.sigma = 1.0  # Not relevant for half-life calculation

    # Calculate half-life: ln(2) / kappa
    if kappa_val > 0:
        model.half_life = np.log(2) / kappa_val
    else:
        model.half_life = float('inf')

    model.is_calibrated = True


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('Vasicek model is calibrated on the series')
def calibrate_model(test_context):
    """Calibrate the Vasicek model on the loaded series."""
    model = test_context['model']
    series = test_context['series']

    try:
        result = model.calibrate(series)
        test_context['calibration_result'] = result
        test_context['calibration_error'] = None
        test_context['predictor'] = model
    except Exception as e:
        test_context['calibration_result'] = None
        test_context['calibration_error'] = e
        logger.debug(f"Calibration raised exception: {type(e).__name__}: {e}")

@when(parsers.parse('Z-score is calculated for value {current_value}'))
def calculate_z_score(test_context, current_value):
    """Calculate Z-score for given value"""
    model = test_context['model']
    z_score = model.calculate_z_score(float(current_value))
    test_context['z_score'] = z_score


@when(parsers.parse('Z-score calculation is attempted for value {value}'))
def attempt_z_score_calculation(test_context, value):
    """Attempt to calculate Z-score (expect error)"""
    model = test_context['model']

    try:
        z_score = model.calculate_z_score(float(value))
        test_context['z_score'] = z_score
        test_context['exception'] = None
    except Exception as e:
        test_context['exception'] = e
        test_context['z_score'] = None


@when(parsers.parse(
    'next value is predicted for current={current_value} after {hours} hours with near_mean_threshold={threshold} and convergence_pct={convergence_pct}'))
def predict_next_value_after_hours(test_context, current_value, hours, threshold, convergence_pct):
    """Predict the next value using O-U process dynamics."""
    model = test_context['model']

    if not model.is_calibrated:
        raise ValueError("Model must be calibrated before making predictions")

    current = float(current_value)
    dt = float(hours)
    near_mean_threshold = float(threshold)
    conv_pct = float(convergence_pct)

    # Call model method instead of duplicating formula
    expected_value = model.predict_next_value(current, dt)

    # Time to convergence: t = -ln(1 - convergence_pct) / kappa
    time_to_equilibrium = -np.log(1 - conv_pct) / model.kappa

    # Determine direction
    initial_distance = abs(current - model.theta)
    predicted_distance = abs(expected_value - model.theta)

    if predicted_distance < initial_distance * near_mean_threshold:
        direction = "near mean"
    elif predicted_distance < initial_distance:
        direction = "towards mean"
    else:
        direction = "away from mean"

    test_context['prediction'] = {
        'expected_value': expected_value,
        'time_to_equilibrium': time_to_equilibrium,
        'direction': direction,
        'initial_distance': initial_distance,
        'predicted_distance': predicted_distance,
    }

@when(parsers.parse('get_trading_threshold() is called with z_threshold {z_threshold}'))
def call_trading_threshold(test_context, z_threshold):
    """Calculate absolute threshold from Z-score."""
    model = test_context['model']
    result = model.get_trading_threshold(float(z_threshold))
    test_context['threshold_result'] = result


# =============================================================================
# THEN steps - Assertions
# =============================================================================
@then('model should be marked as calibrated')
def assert_model_calibrated(test_context):
    """Assert that the model is marked as calibrated."""
    error = test_context.get('calibration_error')
    if error:
        pytest.fail(f"Calibration failed with error: {error}")

    model = test_context['model']
    assert model.is_calibrated, "Model should be marked as calibrated"


@then(parsers.parse('kappa should be approximately {expected_kappa} within {kappa_tol}'))
def assert_kappa_approximate(test_context, expected_kappa, kappa_tol):
    """Assert calibrated kappa matches expected value within tolerance."""
    model = test_context['model']
    actual_kappa = model.kappa

    assert abs(actual_kappa - float(expected_kappa)) <= float(kappa_tol), \
        f"Kappa {actual_kappa:.4f} not within {kappa_tol} of expected {expected_kappa:.4f}"


@then(parsers.parse('theta should be approximately {expected_theta} within {theta_tol}'))
def assert_theta_approximate(test_context, expected_theta, theta_tol):
    """Assert calibrated theta matches expected value within tolerance."""
    model = test_context['model']
    actual_theta = model.theta

    assert abs(actual_theta - float(expected_theta)) <= float(theta_tol), \
        f"Theta {actual_theta:.6f} not within {theta_tol} of expected {expected_theta:.6f}"


@then('mean reversion should be statistically significant')
def assert_mean_reversion_significant(test_context):
    """Assert that mean reversion is statistically significant."""
    model = test_context['model']

    # Ensure model is calibrated
    assert model.is_calibrated, "Model must be calibrated before checking significance"

    # Get validation thresholds from model's config
    vasicek_config = model.config.get_section('vasicek_prediction')
    profile_config = vasicek_config.get(model.profile)
    params = profile_config.get('parameters', {})
    validation_config = params.get('validation', {})

    max_p_value = validation_config.get('significance_level')

    # Check p-value is significant
    assert model.p_value is not None, "Model p_value should be set after calibration"
    assert model.p_value <= max_p_value, \
        f"P-value {model.p_value:.4f} exceeds significance level {max_p_value}"

    # Verify is_mean_reverting() agrees
    assert model.is_mean_reverting(), \
        f"Model should indicate mean reversion (p={model.p_value:.4f}, kappa={model.kappa:.4f})"


@then('mean reversion should NOT be statistically significant')
def assert_mean_reversion_not_significant(test_context):
    """Assert that mean reversion is NOT statistically significant."""
    model = test_context['model']

    # Ensure model is calibrated
    assert model.is_calibrated, "Model must be calibrated before checking significance"

    # Get validation thresholds from model's config
    vasicek_config = model.config.get_section('vasicek_prediction')
    profile_config = vasicek_config.get(model.profile)
    params = profile_config.get('parameters', {})
    validation_config = params.get('validation', {})

    max_p_value = validation_config.get('significance_level')
    max_half_life = validation_config.get('max_half_life')
    min_kappa = validation_config.get('min_kappa')

    # Check that is_mean_reverting() returns False
    assert not model.is_mean_reverting(), \
        f"Model should NOT indicate mean reversion (p={model.p_value:.4f}, kappa={model.kappa:.4f}, half_life={model.half_life:.2f})"

    # Verify at least one rejection criterion is met
    conditions_failed = []

    if model.p_value is not None and model.p_value > max_p_value:
        conditions_failed.append(f"p-value {model.p_value:.4f} > {max_p_value}")

    if model.kappa is not None and model.kappa <= 0:
        conditions_failed.append(f"kappa {model.kappa:.4f} <= 0 (no mean reversion)")

    if model.kappa is not None and 0 < model.kappa < min_kappa:
        conditions_failed.append(f"kappa {model.kappa:.4f} < {min_kappa} (too weak)")

    if model.half_life is not None and model.half_life > max_half_life:
        conditions_failed.append(f"half-life {model.half_life:.2f} > {max_half_life}")

    assert len(conditions_failed) > 0, \
        f"At least one mean-reversion condition should fail, but all passed. " \
        f"Expected failures but got: p={model.p_value:.4f}, kappa={model.kappa:.4f}, half_life={model.half_life:.2f}"


@then(parsers.parse('is_mean_reverting() should return {expected}'))
def assert_mean_reverting(test_context, expected):
    """Assert is_mean_reverting returns expected value"""
    model = test_context['model']
    result = model.is_mean_reverting()

    expected_bool = expected == 'True'

    assert result == expected_bool, \
        f"Expected is_mean_reverting()={expected_bool}, got {result}"

@then(parsers.parse('Z-score should be approximately {expected_z_score:f}'))
def assert_z_score_approximately(test_context, expected_z_score):
    """Assert Z-score is approximately equal to expected value"""
    actual_z_score = test_context['z_score']
    tolerance = 0.1

    assert abs(actual_z_score - expected_z_score) < tolerance, \
        f"Z-score {actual_z_score:.2f} not within {tolerance} of expected {expected_z_score:.2f}"

@then(parsers.parse('interpretation should be "{interpretation}"'))
def assert_interpretation(test_context, interpretation):
    """Store interpretation (documentation only)"""
    test_context['interpretation'] = interpretation
    # No assertion - just for feature file documentation

@then(parsers.parse('{exception_type} should be raised'))
def assert_exception_raised(test_context, exception_type):
    """Assert that specific exception was raised"""
    exception = test_context.get('exception')

    assert exception is not None, "Expected exception but none was raised"

    exception_map = {
        'ValueError': ValueError,
        'TypeError': TypeError,
        'RuntimeError': RuntimeError
    }

    expected_type = exception_map.get(exception_type, ValueError)
    assert isinstance(exception, expected_type), \
        f"Expected {exception_type} but got {type(exception).__name__}"

@then(parsers.parse('error message should contain "{expected_text}"'))
def assert_error_message_contains(test_context, expected_text):
    """Assert error message contains expected text"""
    exception = test_context.get('exception')

    assert exception is not None, "No exception to check message for"

    error_message = str(exception).lower()
    expected_text_lower = expected_text.lower()

    assert expected_text_lower in error_message, \
        f"Error message '{error_message}' does not contain '{expected_text}'"


@then(parsers.parse(
    'half-life should be approximately {half_life} periods within {tolerance_pct} percent or {min_tolerance} minimum'))
def assert_half_life(test_context, half_life, tolerance_pct, min_tolerance):
    """Assert half-life matches expected value."""
    model = test_context['model']
    expected = float(half_life)
    actual = model.half_life

    tol_pct = float(tolerance_pct) / 100.0
    min_tol = float(min_tolerance)
    tolerance = max(expected * tol_pct, min_tol)

    assert abs(actual - expected) < tolerance, \
        f"Half-life {actual:.2f} not approximately {expected:.2f} (kappa={model.kappa})"

@then(parsers.parse('reversion speed category should be {category}'))
def assert_reversion_category(test_context, category):
    """Assert reversion speed category."""
    model = test_context['model']
    actual = model.get_reversion_category()

    assert actual == category, \
        f"Expected category '{category}', got '{actual}' (half_life={model.half_life:.2f})"


@then('half-life should be infinity')
def assert_half_life_infinity(test_context):
    """Assert half-life is infinite (no mean reversion)."""
    model = test_context['model']

    assert model.half_life == float('inf'), \
        f"Expected infinite half-life, got {model.half_life}"

def generate_ou_series(n: int, kappa: float, theta: float, sigma: float, seed: int) -> pd.Series:
    """
    Generate synthetic Ornstein-Uhlenbeck (mean-reverting) series.

    dX = kappa * (theta - X) * dt + sigma * dW

    Args:
        n: Number of data points
        kappa: Speed of mean reversion
        theta: Long-term mean
        sigma: Volatility
        seed: Random seed for reproducibility

    Returns:
        pd.Series with mean-reverting values
    """
    np.random.seed(seed)

    dt = 1.0
    series = np.zeros(n)
    series[0] = theta  # Start at equilibrium

    for t in range(1, n):
        dW = np.random.normal(0, np.sqrt(dt))
        series[t] = series[t - 1] + kappa * (theta - series[t - 1]) * dt + sigma * dW

    return pd.Series(series, name='spread')


@then(parsers.parse('predicted value should be approximately {predicted_value} within {tolerance}'))
def assert_predicted_value_within_tolerance(test_context, predicted_value, tolerance):
    """Assert predicted value matches expected within tolerance."""
    actual = test_context['prediction']['expected_value']

    assert abs(actual - float(predicted_value)) <= float(tolerance), \
        f"Predicted value {actual:.6f} not within {tolerance} of expected {predicted_value}"

@then(parsers.parse('direction should be {expected_direction}'))
def assert_prediction_direction(test_context, expected_direction):
    """Assert the direction of predicted movement."""
    actual = test_context['prediction']['direction']

    assert actual == expected_direction, \
        f"Expected direction '{expected_direction}', got '{actual}'"

@then(parsers.parse(
    'time to equilibrium should be approximately {time_to_eq} hours within {tolerance_pct} percent or {min_tolerance} hours'))
def assert_time_to_equilibrium_hours(test_context, time_to_eq, tolerance_pct, min_tolerance):
    """Assert time to equilibrium matches expected within tolerance."""
    actual = test_context['prediction']['time_to_equilibrium']
    expected = float(time_to_eq)
    tol_pct = float(tolerance_pct) / 100.0
    min_tol = float(min_tolerance)

    tolerance = max(expected * tol_pct, min_tol)

    assert abs(actual - expected) <= tolerance, \
        f"Time to equilibrium {actual:.1f}h not within {tolerance:.1f}h of expected {expected:.1f}h"


@then(parsers.parse('absolute threshold should be {absolute_value}'))
def assert_absolute_threshold(test_context, absolute_value):
    """Assert absolute threshold matches expected."""
    actual = test_context['threshold_result']
    expected = float(absolute_value)

    assert abs(actual - expected) < 0.00001, \
        f"Expected {expected}, got {actual}"

def generate_random_walk(n: int, seed: int, step_std: float = 0.0003) -> pd.Series:
    """
    Generate non-stationary random walk series.

    X(t) = X(t-1) + noise

    Args:
        n: Number of data points
        seed: Random seed for reproducibility
        step_std: Standard deviation of each step

    Returns:
        pd.Series with random walk values
    """
    np.random.seed(seed)

    steps = np.random.normal(0, step_std, n)
    series = np.cumsum(steps)

    return pd.Series(series, name='spread')


def generate_trending_series(n: int, seed: int, drift: float = 0.0001, noise_std: float = 0.0002) -> pd.Series:
    """
    Generate trending (non-stationary) series.

    X(t) = drift * t + noise

    Args:
        n: Number of data points
        seed: Random seed for reproducibility
        drift: Linear trend per period
        noise_std: Standard deviation of noise

    Returns:
        pd.Series with trending values
    """
    np.random.seed(seed)

    trend = np.arange(n) * drift
    noise = np.random.normal(0, noise_std, n)
    series = trend + noise

    return pd.Series(series, name='spread')