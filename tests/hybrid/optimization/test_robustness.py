# tests/hybrid/test_strategy_factory.py

import logging
from pathlib import Path

import numpy as np
import pytest
from pytest_bdd import scenarios, parsers, given, when, then

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.optimization.robustness import RobustnessAnalyzer

# Only when you need project root path:

# Import the system under test

# Load scenarios from the strategy_factory.feature
scenarios('robustness.feature')

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
# GIVEN steps - Setup
# =============================================================================
@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"
    unified_config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['unified_config'] = unified_config

@given('a RobustnessAnalyzer with standard configuration')
def step_create_robustness_analyzer(test_context):
    """Create RobustnessAnalyzer instance with config"""
    unified_config = test_context['unified_config']
    analyzer = RobustnessAnalyzer(unified_config)
    test_context['analyzer'] = analyzer


@given('robustness thresholds are set:')
def step_set_robustness_thresholds(test_context, datatable):
    """Override robustness thresholds in config"""
    analyzer = test_context['analyzer']

    # Parse datatable and update analyzer config
    for row in datatable[1:]:
        threshold_name = row[0]
        threshold_value = float(row[1])

        # Update the cached config values in analyzer
        if threshold_name == 'cv_threshold_robust':
            analyzer.cv_threshold_robust = threshold_value
        elif threshold_name == 'cv_threshold_sensitive':
            analyzer.cv_threshold_sensitive = threshold_value
        elif threshold_name == 'top_performers_percentile':
            analyzer.top_performers_percentile = threshold_value
        elif threshold_name == 'plateau_threshold':
            analyzer.plateau_threshold = threshold_value
        elif threshold_name == 'strong_correlation_threshold':
            # Store for later use in correlation analysis
            if 'correlation_thresholds' not in test_context:
                test_context['correlation_thresholds'] = {}
            test_context['correlation_thresholds']['strong'] = threshold_value

@given(parsers.parse(
    '{n_results} optimization results where {param_name} has mean {mean_value} and std {std_value}'))
def given_optimization_results_with_param_stats(test_context, n_results, param_name, mean_value, std_value):
    """Generate optimization results with specified parameter statistics"""
    n = int(n_results)
    mean_val = float(mean_value)
    std_val = float(std_value)

    # Generate values with target mean and std
    np.random.seed(42)
    values = np.random.normal(mean_val, std_val, n)

    # Build optimization results
    results = []
    for i in range(n):
        results.append({
            'params': {param_name: values[i]},
            'fitness': 0
        })

    test_context['optimization_results'] = results
    test_context['param_name'] = param_name

@given(parsers.parse('I have {n_results} optimization results'))
def given_n_optimization_results(test_context, n_results):
    """Set up optimization results"""
    n = int(n_results)
    test_context['n_results'] = n

    if n == 0:
        test_context['optimization_results'] = []
    else:
        # Generate placeholder results
        results = []
        for i in range(n):
            results.append({
                'params': {},
                'fitness': 0
            })
        test_context['optimization_results'] = results


@given(parsers.parse('top {top_percentage} of results have fitness within {tolerance} of maximum {max_fitness}'))
def given_plateau_shape(test_context, top_percentage, tolerance, max_fitness):
    n = test_context['n_results']
    top_pct = float(top_percentage.strip('%')) / 100
    tol = float(tolerance.strip('%')) / 100
    max_fit = float(max_fitness)

    n_plateau = int(n * top_pct)
    n_rest = n - n_plateau

    # Plateau: values within tolerance of max
    plateau_min = max_fit * (1 - tol)
    plateau_values = np.linspace(max_fit, plateau_min, n_plateau)

    # Rest: below plateau, evenly spread to zero
    rest_values = np.linspace(plateau_min, 0, n_rest + 1)[1:]

    fitness_values = np.concatenate([plateau_values, rest_values])

    results = [{'params': {}, 'fitness': f} for f in fitness_values]
    test_context['optimization_results'] = results


@given(parsers.parse('{n_results} optimization results with {robust_pct} robust parameters'))
def given_results_with_robust_percentage(test_context, n_results, robust_pct):
    """Generate optimization results where specified percentage of parameters are ROBUST"""
    n = int(n_results)
    pct = float(robust_pct.strip('%')) / 100

    np.random.seed(42)

    analyzer = test_context['analyzer']
    cv_threshold_robust = analyzer.cv_threshold_robust
    cv_threshold_sensitive = analyzer.cv_threshold_sensitive

    mean_val = n
    std_robust = mean_val * (cv_threshold_robust / 2)
    std_sensitive = mean_val * (cv_threshold_sensitive * 2)

    n_robust = int(n * pct)
    n_sensitive = n - n_robust

    results = []
    for i in range(n):
        params = {}
        for j in range(n_robust):
            params[f'robust_param_{j}'] = np.random.normal(mean_val, std_robust)

        for j in range(n_sensitive):
            params[f'sensitive_param_{j}'] = np.random.normal(mean_val, std_sensitive)

        results.append({
            'params': params,
            'fitness': 0
        })

    test_context['optimization_results'] = results
    test_context['n_results'] = n

@given(parsers.parse('fitness landscape is {landscape_type}'))
def given_fitness_landscape(test_context, landscape_type):
    """Adjust results to produce specified landscape type"""
    results = test_context['optimization_results']
    n = len(results)

    analyzer = test_context['analyzer']
    gradient_flat = analyzer.robustness_config['gradient_threshold_flat']
    gradient_steep = analyzer.robustness_config['gradient_threshold_steep']

    if landscape_type == 'PLATEAU_DOMINATED':
        # Gradient below flat threshold
        step = gradient_flat / 2
    elif landscape_type == 'PEAKY':
        # Gradient above steep threshold
        step = gradient_steep * 2
    else:  # MIXED
        # Gradient between thresholds
        step = (gradient_flat + gradient_steep) / 2

    fitness_values = [n - (i * step) for i in range(n)]

    for i, result in enumerate(results):
        result['fitness'] = fitness_values[i]

    test_context['optimization_results'] = results


# =============================================================================
# WHEN steps - Actions
# =============================================================================
@when('I analyze parameter stability')
def when_analyze_parameter_stability(test_context):
    """Run parameter stability analysis"""
    analyzer = test_context['analyzer']
    results = test_context['optimization_results']

    test_context['stability_analysis'] = analyzer.analyze_parameter_stability(results)


@when('I analyze fitness landscape')
def when_analyze_fitness_landscape(test_context):
    """Run fitness landscape analysis"""
    analyzer = test_context['analyzer']
    results = test_context['optimization_results']

    test_context['landscape_analysis'] = analyzer.analyze_fitness_landscape(results)

@when('I find robust parameter ranges')
def when_find_robust_ranges(test_context):
    analyzer = test_context['analyzer']
    results = test_context['optimization_results']
    test_context['robust_ranges'] = analyzer.find_robust_parameter_ranges(results)

@when('I generate robustness report')
def when_generate_robustness_report(test_context):
    """Generate comprehensive robustness report"""
    analyzer = test_context['analyzer']
    results = test_context['optimization_results']

    test_context['robustness_report'] = analyzer.generate_robustness_report(results)

# =============================================================================
# THEN steps - Assertions
# =============================================================================
@then(parsers.parse('parameter {param_name} should be classified as {robustness_class}'))
def then_parameter_classified_as(test_context, param_name, robustness_class):
    """Verify parameter robustness classification"""
    analysis = test_context['stability_analysis']

    assert param_name in analysis, f"Parameter {param_name} not in analysis"
    assert analysis[param_name]['robustness_class'] == robustness_class

@then('the analysis should indicate insufficient data')
def then_analysis_indicates_insufficient_data(test_context):
    """Verify analysis returns empty for insufficient data"""
    analysis = test_context['stability_analysis']
    assert analysis == {}


@then('analysis should fail with insufficient samples error')
def then_analysis_fails_insufficient_samples(test_context):
    """Verify analysis raises ValueError for insufficient samples"""
    analyzer = test_context['analyzer']
    results = test_context['optimization_results']

    with pytest.raises(ValueError) as exc_info:
        analyzer.analyze_parameter_stability(results)

    assert 'Insufficient samples' in str(exc_info.value)

@then(parsers.parse('landscape type should be {landscape_type}'))
def then_landscape_type_should_be(test_context, landscape_type):
    """Verify landscape classification"""
    analysis = test_context['landscape_analysis']

    assert 'landscape_type' in analysis, "No landscape_type in analysis"
    assert analysis['landscape_type'] == landscape_type

@then(parsers.parse('{param_name} should have confidence level {confidence}'))
def then_param_confidence_level(test_context, param_name, confidence):
    robust_ranges = test_context['robust_ranges']

    if confidence == 'NONE':
        assert param_name not in robust_ranges, f"{param_name} should not have robust range"
    else:
        assert param_name in robust_ranges, f"{param_name} missing from robust ranges"
        assert robust_ranges[param_name]['confidence'] == confidence

@then(parsers.parse('recommendation should be {recommendation}'))
def then_recommendation_should_be(test_context, recommendation):
    """Verify report recommendation"""
    report = test_context['robustness_report']

    assert 'summary' in report, "Report missing summary"
    assert 'recommendation' in report['summary'], "Summary missing recommendation"

    actual = report['summary']['recommendation']
    assert recommendation in actual, f"Expected {recommendation} in '{actual}'"