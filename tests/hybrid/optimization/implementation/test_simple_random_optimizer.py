import logging
# Import the system under test
import sys
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, parsers, when, then

from src.hybrid.optimization import OptimizerType
from src.hybrid.optimization.implementation.simple_optimizer import SimpleRandomOptimizer
from src.hybrid.strategies import StrategyFactory

# Go up 4 levels from tests/hybrid/money_management/test_money_management.py to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid.config.unified_config import UnifiedConfig

# Load all scenarios from money_management.feature
scenarios('simple_random_optimizer.feature')

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

    root_path = Path(__file__).parent.parent.parent.parent.parent
    config_path = root_path / config_directory

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['config'] = config

@given('optimization configuration is set')
def step_optimization_config_set(test_context):
    """Verify optimization section exists in config"""
    config = test_context['config']
    opt_config = config.get_section('optimization', {})
    assert opt_config is not None, "Optimization config missing"


@given('a SimpleRandomOptimizer')
def step_given_simple_optimizer(test_context):
    """Create SimpleRandomOptimizer with base strategy"""
    config = test_context['config']

    # Create factory instance first
    strategy_factory = StrategyFactory()

    # Create base strategy (no dependencies needed for parameter extraction)
    strategy = strategy_factory.create_strategy('base', config)

    # Create optimizer with strategy
    optimizer = SimpleRandomOptimizer(config, strategy)
    test_context['optimizer'] = optimizer
    test_context['strategy'] = strategy

# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when('I create a SimpleRandomOptimizer')
def step_create_simple_optimizer(test_context):
    """Create SimpleRandomOptimizer instance"""
    config = test_context['config']

    # Create factory instance
    strategy_factory = StrategyFactory()
    strategy = strategy_factory.create_strategy('base', config)

    optimizer = SimpleRandomOptimizer(config, strategy)
    test_context['optimizer'] = optimizer
    test_context['strategy'] = strategy
@when(parsers.parse('I generate {count} random parameter combinations'))
def step_generate_random_combinations(test_context, count):
    """Generate random parameter combinations"""
    count = int(count)
    optimizer = test_context['optimizer']
    combinations = optimizer.generate_random_parameters(count)
    test_context['combinations'] = combinations
    test_context['expected_count'] = count

# =============================================================================
# THEN steps - Assertions
# =============================================================================

@then(parsers.parse('optimizer type should be "{expected_type}"'))
def step_check_optimizer_type(test_context, expected_type):
    """Verify optimizer type matches expected"""
    optimizer = test_context['optimizer']
    actual_type = optimizer.get_optimization_type()
    assert actual_type == OptimizerType[expected_type], \
        f"Expected {expected_type}, got {actual_type}"


@then(parsers.parse('description should mention "{expected_text}"'))
def step_check_description_contains(test_context, expected_text):
    """Verify description contains expected text"""
    optimizer = test_context['optimizer']
    description = optimizer.get_description()
    assert expected_text in description, \
        f"Expected '{expected_text}' in description, got: {description}"

@then(parsers.parse('{count} unique combinations should be created'))
def step_verify_unique_combinations(test_context, count):
    """Verify correct number of unique combinations created"""
    count = int(count)
    combinations = test_context['combinations']
    assert len(combinations) == count, \
        f"Expected {count} combinations, got {len(combinations)}"


@then(parsers.parse('{count} unique combinations should be created'))
def step_verify_unique_combinations(test_context, count):
    """Verify correct number of unique combinations created"""
    count = int(count)
    combinations = test_context['combinations']
    assert len(combinations) == count, \
        f"Expected {count} combinations, got {len(combinations)}"


@then('each combination should have all configured parameters')
def step_verify_all_parameters_present(test_context):
    """Verify each combination has all expected parameters"""
    combinations = test_context['combinations']
    optimizer = test_context['optimizer']
    expected_params = set(optimizer.param_ranges.keys())

    for i, combo in enumerate(combinations):
        combo_params = set(combo.keys())
        assert combo_params == expected_params, \
            f"Combination {i} has parameters {combo_params}, expected {expected_params}"


@then('all parameter values should be within their configured ranges')
def step_verify_all_values_within_ranges(test_context):
    """Verify all parameter values are within their configured min/max ranges"""
    combinations = test_context['combinations']
    optimizer = test_context['optimizer']
    param_ranges = optimizer.param_ranges

    for i, combo in enumerate(combinations):
        for param_name, param_value in combo.items():
            assert param_name in param_ranges, \
                f"Unknown parameter {param_name} in combination {i}"

            min_val = param_ranges[param_name]['min']
            max_val = param_ranges[param_name]['max']

            assert min_val <= param_value <= max_val, \
                f"Combination {i}: {param_name}={param_value} outside range [{min_val}, {max_val}]"