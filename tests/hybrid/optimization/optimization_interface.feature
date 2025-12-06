@optimizer_interface
Feature: Optimizer Interface Requirements
  All optimizer implementations must provide complete optimization capabilities
  including execution, monitoring, persistence, and validation

  Background:
    Given config files are available in tests/config/optimization

  @core_methods
  Scenario Outline: Optimizer implements core interface methods
    Given an optimizer of type "<optimizer_type>"
    Then the optimizer should have method "run_optimization"
    And the optimizer should have method "get_optimization_type"
    And the optimizer should have method "get_description"

    Examples:
      | optimizer_type    |
      | SIMPLE_RANDOM    |
      | CACHED_RANDOM    |
      | BAYESIAN         |

    @base_attributes
    Scenario Outline: Optimizer base class provides common attributes
      Given an optimizer of type "<optimizer_type>"
      Then it should have attribute "config"
      Examples:
        | optimizer_type    |
        | SIMPLE_RANDOM    |
        | CACHED_RANDOM    |
        | BAYESIAN         |
