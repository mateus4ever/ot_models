@optimizer_factory
Feature: Optimizer Factory
  Factory for creating and managing optimizer instances

  Background:
    Given a valid UnifiedConfig is loaded

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

  @factory_missing_dependency
  Scenario: Handle missing dependencies gracefully
    Given optimizer type "BAYESIAN"
    And scikit-optimize is not installed
    When I attempt to create an optimizer
    Then an ImportError should be raised
    And error message should include "scikit-optimize is required"
    And error message should include installation instructions

  @factory_available_optimizers
  Scenario: List available optimizers
    When I request available optimizer types
    Then the list should contain "SIMPLE_RANDOM"
    And the list should contain "CACHED_RANDOM"

  @factory_available_optimizers_conditional
  Scenario: List includes BAYESIAN only if dependency available
    When I request available optimizer types
    And scikit-optimize is installed
    Then the list should contain "BAYESIAN"

  @factory_available_optimizers_missing_dependency
  Scenario: List excludes BAYESIAN if dependency missing
    When I request available optimizer types
    And scikit-optimize is not installed
    Then the list should not contain "BAYESIAN"

  @factory_descriptions
  Scenario: Get optimizer descriptions
    When I request optimizer descriptions
    Then descriptions should be a dictionary
    And each available optimizer should have a description
    And all descriptions should be non-empty strings
    And descriptions should not be "Description unavailable"

  @factory_descriptions_graceful_failure
  Scenario: Handle description retrieval failures gracefully
    Given an optimizer that fails during instantiation
    When I request optimizer descriptions
    Then the failing optimizer should have description "Description unavailable"
    And other optimizer descriptions should be normal

  @factory_config_validation
  Scenario: Validate config before creating optimizer
    Given an invalid UnifiedConfig with missing required sections
    When I attempt to create an optimizer
    Then a ConfigurationError should be raised
    And error message should indicate which section is missing

  @factory_legacy_functions
  Scenario Outline: Legacy functions still work
    When I call legacy function "<legacy_function>"
    Then it should delegate to run_optimization
    And it should use optimizer type "<expected_type>"
    And a deprecation warning should be issued

    Examples:
      | legacy_function                  | expected_type    |
      | run_fast_optimization           | SIMPLE_RANDOM    |
      | run_optimized_fast_optimization | CACHED_RANDOM    |
      | run_bayesian_optimization       | BAYESIAN         |

  @factory_config_reuse
  Scenario: Factory can create multiple optimizers with same config
    Given a UnifiedConfig instance
    When I create optimizer "SIMPLE_RANDOM"
    And I create optimizer "CACHED_RANDOM"
    Then both optimizers should be created successfully
    And both should use the same config instance
    And both should be independent objects

  @factory_registry_pattern
  Scenario: Support dynamic optimizer registration (future)
    Given a custom optimizer class "MyCustomOptimizer"
    When I register it with OptimizerFactory as "CUSTOM"
    Then factory should accept type "CUSTOM"
    And factory should create "MyCustomOptimizer" instances
    And "CUSTOM" should appear in available optimizers