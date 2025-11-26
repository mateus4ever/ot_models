@simple_random_optimizer
Feature: Simple Random Optimizer
  Basic random parameter search without caching - educational baseline

  Background:
    Given config files are available in tests/config/optimization

  @simple_creation
  Scenario: Create Simple Random optimizer instance
    When I create a SimpleRandomOptimizer
    Then optimizer type should be "SIMPLE_RANDOM"
    And description should mention "full data loading per iteration"

  @simple_random_parameters
  Scenario: Generate random parameter combinations
    Given a SimpleRandomOptimizer
    When I generate 100 random parameter combinations
    Then 100 unique combinations should be created
    And each combination should have all configured parameters
    And all parameter values should be within their configured ranges