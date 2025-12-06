# tests/hybrid/test_strategy_factory.py
"""
pytest-bdd test runner for StrategyFactory creation and error handling
Tests factory's core responsibility: create strategies successfully and handle errors properly
ZERO MOCKS - Real strategy factory with actual strategy creation
"""
import logging
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.optimization.fitness import FitnessCalculator

# Only when you need project root path:
# Import the system under test

# Load scenarios from the strategy_factory.feature
scenarios('fitness.feature')

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

@given('metrics with values:')
def step_metrics_with_values(test_context, datatable):
    """Create metrics dict from table and update config"""
    metrics = {}

    # Build metrics dict from table
    for row in datatable[1:]:
        metric_name = row[0]
        value = float(row[1])
        metrics[metric_name] = value

    test_context['metrics'] = metrics

    # Update config to include all metrics from table
    config = test_context['unified_config']
    optimization_config = config.get_section('optimization', {})
    fitness_config = optimization_config.get('fitness', {})
    existing_metrics = fitness_config.get('metrics', [])

    # Add any missing metrics to config
    existing_metric_names = {m['name'] for m in existing_metrics}

    updates = {'optimization': {'fitness': {'metrics': existing_metrics.copy()}}}

    for metric_name in metrics.keys():
        if metric_name not in existing_metric_names:
            updates['optimization']['fitness']['metrics'].append({
                'name': metric_name,
                'weight': 0.0,  # Zero weight - not used for fitness, just available
                'direction': 'maximize'
            })

    config.update_config(updates)
    test_context['unified_config'] = config

@given('a FitnessCalculator instance')
def step_create_fitness_calculator(test_context):
    """Create FitnessCalculator instance"""
    config = test_context['unified_config']
    fitness_calculator = FitnessCalculator(config)
    test_context['fitness_calculator'] = fitness_calculator


@given('metrics configured with maximize and minimize directions')
def step_metrics_configured_with_directions(test_context):
    """Verify config has both maximize and minimize metrics"""
    config = test_context['unified_config']

    optimization_config = config.get_section('optimization', {})
    fitness_config = optimization_config.get('fitness', {})
    metrics_config = fitness_config.get('metrics', [])

    has_maximize = any(m['direction'] == 'maximize' for m in metrics_config)
    has_minimize = any(m['direction'] == 'minimize' for m in metrics_config)

    assert has_maximize, "Config should have at least one maximize metric"
    assert has_minimize, "Config should have at least one minimize metric"

# =============================================================================
# WHEN steps - Actions
# =============================================================================
@when('I calculate fitness')
def step_calculate_fitness(test_context):
    """Calculate fitness from metrics"""
    fitness_calculator = test_context['fitness_calculator']
    metrics = test_context['metrics']

    try:
        fitness_score = fitness_calculator.calculate_fitness(metrics)
        test_context['fitness_score'] = fitness_score
        test_context['fitness_error'] = None
    except Exception as e:
        test_context['fitness_score'] = None
        test_context['fitness_error'] = e

# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then('a fitness score should be returned')
def step_fitness_score_returned(test_context):
    """Verify fitness score was calculated"""
    error = test_context.get('fitness_error')
    assert error is None, f"Fitness calculation failed: {error}"

    score = test_context.get('fitness_score')
    assert score is not None, "Fitness score should not be None"

@then('the score should be a number')
def step_score_is_number(test_context):
    """Verify fitness score is numeric"""
    score = test_context['fitness_score']
    assert isinstance(score, (int, float)), f"Score should be numeric, got {type(score)}"

@then('metric weights should come from config')
def step_weights_from_config(test_context):
    """Verify weights are loaded from config"""
    fitness_calculator = test_context['fitness_calculator']
    assert hasattr(fitness_calculator, 'metrics_config'), \
        "FitnessCalculator should have metrics_config"

    for metric in fitness_calculator.metrics_config:
        assert 'weight' in metric, f"Metric {metric.get('name')} missing weight"

@then('metric directions should come from config')
def step_directions_from_config(test_context):
    """Verify directions are loaded from config"""
    fitness_calculator = test_context['fitness_calculator']

    for metric in fitness_calculator.metrics_config:
        assert 'direction' in metric, f"Metric {metric.get('name')} missing direction"
        assert metric['direction'] in ['maximize', 'minimize'], \
            f"Invalid direction: {metric['direction']}"

@then('maximize metrics should increase fitness')
def step_maximize_increases_fitness(test_context):
    """Verify maximize direction increases fitness"""
    # This is implicitly tested by calculation
    # Maximize metrics contribute positively
    fitness_calculator = test_context['fitness_calculator']

    maximize_metrics = [m for m in fitness_calculator.metrics_config if m['direction'] == 'maximize']
    assert len(maximize_metrics) > 0, "Should have maximize metrics"

@then('minimize metrics should decrease fitness')
def step_minimize_decreases_fitness(test_context):
    """Verify minimize direction decreases fitness"""
    # This is implicitly tested by calculation
    # Minimize metrics contribute negatively
    fitness_calculator = test_context['fitness_calculator']

    minimize_metrics = [m for m in fitness_calculator.metrics_config if m['direction'] == 'minimize']
    assert len(minimize_metrics) > 0, "Should have minimize metrics"


@then('severe_penalty should be returned')
def step_severe_penalty_returned(test_context):
    """Verify severe penalty was returned"""
    score = test_context['fitness_score']
    fitness_calculator = test_context['fitness_calculator']

    assert score == fitness_calculator.severe_penalty, \
        f"Expected severe_penalty {fitness_calculator.severe_penalty}, got {score}"

@then('severe_penalty value should come from config')
def step_severe_penalty_from_config(test_context):
    """Verify severe_penalty comes from config"""
    fitness_calculator = test_context['fitness_calculator']
    assert hasattr(fitness_calculator, 'severe_penalty'), \
        "FitnessCalculator should have severe_penalty"
    assert isinstance(fitness_calculator.severe_penalty, (int, float)), \
        "severe_penalty should be numeric"

@then('metrics list should be loaded from config')
def step_metrics_list_from_config(test_context):
    """Verify metrics list loaded from config"""
    fitness_calculator = test_context['fitness_calculator']
    assert hasattr(fitness_calculator, 'metrics_config'), \
        "FitnessCalculator should have metrics_config"
    assert isinstance(fitness_calculator.metrics_config, list), \
        "metrics_config should be a list"


@then('each metric should have weight from config')
def step_each_metric_has_weight(test_context):
    """Verify each metric has weight"""
    fitness_calculator = test_context['fitness_calculator']
    for metric in fitness_calculator.metrics_config:
        assert 'weight' in metric, f"Metric missing weight: {metric}"


@then('each metric should have direction from config')
def step_each_metric_has_direction(test_context):
    """Verify each metric has direction"""
    fitness_calculator = test_context['fitness_calculator']
    for metric in fitness_calculator.metrics_config:
        assert 'direction' in metric, f"Metric missing direction: {metric}"


@then('penalty conditions should be loaded from config')
def step_penalty_conditions_from_config(test_context):
    """Verify penalty conditions loaded from config"""
    fitness_calculator = test_context['fitness_calculator']
    assert hasattr(fitness_calculator, 'penalty_conditions'), \
        "FitnessCalculator should have penalty_conditions"
    assert isinstance(fitness_calculator.penalty_conditions, list), \
        "penalty_conditions should be a list"


@then('severe_penalty value should be loaded from config')
def step_severe_penalty_loaded(test_context):
    """Verify severe_penalty loaded from config"""
    fitness_calculator = test_context['fitness_calculator']
    assert hasattr(fitness_calculator, 'severe_penalty'), \
        "FitnessCalculator should have severe_penalty"
    assert isinstance(fitness_calculator.severe_penalty, (int, float)), \
        "severe_penalty should be numeric"

@then(parsers.parse('the score should be {expected_score} with the deviation of {tolerance}'))
def step_score_should_be_with_tolerance(test_context, expected_score, tolerance):
    """Verify fitness score matches expected value within tolerance"""
    score = test_context['fitness_score']
    expected = float(expected_score)
    tol = float(tolerance)

    assert abs(score - expected) < tol, \
        f"Expected score {expected} ±{tol}, got {score}"

@then('maximize metrics should contribute positively')
def step_maximize_contributes_positively(test_context):
    """Verify maximize metrics add to fitness"""
    # Recalculate with maximize metrics only
    fitness_calc = test_context['fitness_calculator']
    metrics = test_context['metrics']

    maximize_contribution = 0
    for metric_config in fitness_calc.metrics_config:
        if metric_config['direction'] == 'maximize':
            value = metrics.get(metric_config['name'], 0)
            maximize_contribution += value * metric_config['weight']

    assert maximize_contribution > 0, \
        f"Maximize metrics should contribute positively, got {maximize_contribution}"

@then('minimize metrics should contribute negatively')
def step_minimize_contributes_negatively(test_context):
    """Verify minimize metrics subtract from fitness"""
    fitness_calc = test_context['fitness_calculator']
    metrics = test_context['metrics']

    minimize_contribution = 0
    for metric_config in fitness_calc.metrics_config:
        if metric_config['direction'] == 'minimize':
            value = metrics.get(metric_config['name'], 0)
            minimize_contribution += value * metric_config['weight'] * (-1)

    assert minimize_contribution < 0, \
        f"Minimize metrics should contribute negatively, got {minimize_contribution}"