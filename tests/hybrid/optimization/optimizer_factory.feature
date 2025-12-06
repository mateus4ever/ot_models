@optimizer_factory
Feature: Optimizer Factory
  Factory for creating and managing optimizer instances

  Background:
    Given config files are available in tests/config/strategies

  @factory_creation
  Scenario Outline: Create optimizer by type
    Given optimizer type "<optimizer_type>"
    When I create an optimizer using the factory
    Then an optimizer instance should be returned
    And the optimizer should implement IOptimizer interface
    And the optimizer type should be "<optimizer_type>"

    Examples:
      | optimizer_type    |
      | SIMPLE_RANDOM    |
      | CACHED_RANDOM    |
      | BAYESIAN         |

  @factory_invalid_type
  Scenario: Reject invalid optimizer type
    Given an invalid optimizer type "NONEXISTENT"
    When I attempt to create an optimizer
    Then a ValueError should be raised
    And error message should include "Unsupported optimizer type"
    And error message should include "NONEXISTENT"

  @factory_available_optimizers
  Scenario: List available optimizers
    When I request available optimizer types
    Then the list should contain "SIMPLE_RANDOM"
    And the list should contain "CACHED_RANDOM"
    And the list should contain "BAYESIAN"

  @factory_invalid_inputs
  Scenario Outline: Throw errors for invalid inputs
    When I try to create an optimizer with <invalid_input>
    Then a ValueError should be raised
    And the error message should be informative

    Examples:
      | invalid_input |
      | None         |
      | ""           |
      | "   "        |