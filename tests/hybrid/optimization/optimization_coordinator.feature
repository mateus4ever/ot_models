@optimization_coordinator
Feature: Optimization Coordinator
  Coordinates parameter generation, worker distribution, and result aggregation

  Background:
    Given config files are available in tests/config/optimization
    And data source is set to data/stock/

  @coordinator_initialization
  Scenario: Initialize optimization coordinator
    Given I create an OptimizationCoordinator
    Then coordinator should be initialized
    And checkpoint settings should be loaded from config

  @coordinator_optimize
  Scenario: Run optimization with strategy factory
    Given I create an OptimizationCoordinator
    And initial capital is set to 100000
    And a strategy factory function
    When I run optimization with SIMPLE_RANDOM optimizer
      | parameter      | value |
      | n_combinations | 3     |
      | n_workers      | 2     |
    Then parameter combinations should be generated
    And work should be distributed to workers
    And results should be aggregated
    And best result should be identified

#  @coordinator_result_aggregation
#  Scenario: Aggregate results and identify best
#    Given a UnifiedConfig is loaded
#    And an OptimizationCoordinator
#    And completed optimization results
#    When results are aggregated
#    Then valid results should be separated from failed
#    And results should be sorted by fitness descending
#    And best result should have highest fitness
#    And total and valid counts should be correct
#
#  @coordinator_config_defaults
#  Scenario: Use config defaults when parameters not provided
#    Given a UnifiedConfig is loaded
#    And an OptimizationCoordinator
#    When I run optimization without specifying n_combinations
#    And without specifying n_workers
#    Then n_combinations should default from config
#    And n_workers should default from config