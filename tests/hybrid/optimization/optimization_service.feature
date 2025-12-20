@optimization_service
Feature: Optimization Service
  Entry point for running optimizations

  Background:
    Given config files are available in tests/config/optimization

  @basic_optimization
  Scenario: Run optimization through service
    Given an OptimizationService
    And a strategy factory
    When I run optimization with optimizer type SIMPLE_RANDOM
    Then optimization results should be returned
    And results should contain best_result
    And results should contain all_results