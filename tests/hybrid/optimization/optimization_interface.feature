@optimizer_interface
Feature: Optimizer Interface Requirements
  All optimizer implementations must provide complete optimization capabilities
  including execution, monitoring, persistence, and validation

  Background:
    Given a valid UnifiedConfig is loaded

  @core_methods
  Scenario Outline: Optimizer implements core interface methods
    Given an optimizer of type "<optimizer_type>"
    Then the optimizer should have method "run_optimization"
    And the optimizer should have method "get_optimization_type"
    And the optimizer should have method "get_description"

    Examples:
      | optimizer_type    |
      | RANDOM           |
      | CACHED_RANDOM    |
      | BAYESIAN         |

  @base_attributes
  Scenario: Optimizer base class provides common attributes
    Given an optimizer inheriting from IOptimizerBase
    Then it should have attribute "zero_value"
    And it should have attribute "unity_value"
    And it should have attribute "severe_penalty"
    And it should have attribute "min_trades_required"
    And it should have attribute "max_drawdown_limit"

  @constraint_validation
  Scenario: Optimizer must validate parameter constraints
    Given an optimizer implementation
    Then it should have method "validate_parameters"
    And it should check parameter value ranges
    And it should check parameter relationships
    And invalid parameters should be rejected before backtest

  @constraint_enforcement
  Scenario: Common parameter constraints must be enforced
    Given parameter constraints "fast_period < slow_period"
    When optimizer generates parameter combinations
    Then all combinations should satisfy "fast_period < slow_period"