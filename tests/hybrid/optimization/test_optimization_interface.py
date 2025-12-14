import logging
# Import the system under test
import sys
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, parsers, when, then

from src.hybrid.optimization import OptimizerType
from src.hybrid.optimization.implementation.bayesian_optimizer import BayesianOptimizer
from src.hybrid.optimization.implementation.cached_optimizer import CachedRandomOptimizer
from src.hybrid.optimization.implementation.simple_optimizer import SimpleRandomOptimizer
from src.hybrid.optimization.optimizer_factory import OptimizerFactory
from src.hybrid.strategies import StrategyFactory

# Go up 4 levels from tests/hybrid/money_management/test_money_management.py to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid.config.unified_config import UnifiedConfig

# Load all scenarios from money_management.feature
scenarios('optimization_interface.feature')

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

    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory
    test_root = Path(__file__).parent.parent.parent

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    config = UnifiedConfig(config_path=str(config_path), environment="test")
    test_context['config'] = config
    test_context['test_root'] = test_root

@given(parsers.parse('an optimizer of type "{optimizer_type}"'))
def step_create_optimizer_by_type(test_context, optimizer_type):
    """Create optimizer instance by type name"""

    config = test_context['config']
    initial_capital = config.config['testing']['initial_capital']
    test_root = test_context['test_root']

    strategy_factory = StrategyFactory()
    strategy = strategy_factory.create_strategy_isolated(
        'base', config,initial_capital,test_root)

    # Create optimizer directly based on type
    if optimizer_type == "SIMPLE_RANDOM":
        optimizer = SimpleRandomOptimizer(config, strategy)
    elif optimizer_type == "CACHED_RANDOM":
        optimizer = CachedRandomOptimizer(config, strategy)
    elif optimizer_type == "BAYESIAN":
        optimizer = BayesianOptimizer(config, strategy)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    test_context['optimizer'] = optimizer
    test_context['optimizer_type'] = optimizer_type
# =============================================================================
# WHEN steps - Actions
# =============================================================================

# =============================================================================
# THEN steps - Assertions
# =============================================================================
@then(parsers.parse('the optimizer should have method "{method_name}"'))
def step_verify_optimizer_has_method(test_context, method_name):
    """Verify optimizer has specified method"""
    optimizer = test_context['optimizer']
    optimizer_type = test_context['optimizer_type']

    assert hasattr(optimizer, method_name), \
        f"Optimizer {optimizer_type} missing method: {method_name}"

    # Verify it's actually callable
    method = getattr(optimizer, method_name)
    assert callable(method), \
        f"Optimizer {optimizer_type} has {method_name} but it's not callable"


@then(parsers.parse('it should have attribute "{attribute_name}"'))
def step_verify_optimizer_has_attribute(test_context, attribute_name):
    """Verify optimizer has specified attribute from base class"""
    optimizer = test_context['optimizer']

    assert hasattr(optimizer, attribute_name), \
        f"Optimizer missing base attribute: {attribute_name}"

    # Verify attribute is not None
    attribute_value = getattr(optimizer, attribute_name)
    assert attribute_value is not None, \
        f"Base attribute {attribute_name} is None"