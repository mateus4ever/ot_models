@cached_random_optimizer
Feature: Cached Random Optimizer
  Optimized random search with data and ML caching for performance

  Background:
    Given a valid UnifiedConfig is loaded
    And optimization configuration is set

  @cached_creation
  Scenario: Create Cached Random optimizer instance
    When I create a CachedRandomOptimizer
    Then optimizer type should be "CACHED_RANDOM"
    And description should mention "optimized data/ML reuse"
    And cached_data should be None initially
    And cached_strategy should be None initially
    And cached_signals should be None initially

  @cached_initialization
  Scenario: Initialize cache with data and ML training
    Given a CachedRandomOptimizer
    And market data path
    When I initialize cache
    Then cached_data should be loaded
    And cached_strategy should be trained
    And cached_signals should be generated
    And cache should be ready for reuse

  @cached_data_loading
  Scenario: Load and preprocess data once
    Given a CachedRandomOptimizer
    When I initialize cache
    Then data should be loaded exactly once
    And preprocessing should occur once
    And data should be stored in cached_data

  @cached_ml_training
  Scenario: Train ML models once
    Given a CachedRandomOptimizer
    When I initialize cache
    Then ML models should be trained exactly once
    And training time should be logged
    And trained strategy should be cached

  @cached_signal_generation
  Scenario: Generate signals once
    Given a CachedRandomOptimizer
    When I initialize cache
    Then signals should be generated exactly once
    And signals should match data length
    And signals should be stored in cached_signals

  @cached_config_creation
  Scenario: Create optimized config for backtesting
    Given a CachedRandomOptimizer
    And parameter combination
    When I create optimized config
    Then new config should have parameters
    And verbose should be False
    And debug_mode should be False
    And all debug flags should be disabled

  @cached_single_backtest
  Scenario: Run single backtest with cached data
    Given a CachedRandomOptimizer with initialized cache
    And parameter combination
    When I run single backtest
    Then cached_data should be reused
    And cached_signals should be reused
    And only risk parameters should vary
    And backtest should execute quickly

  @cached_random_parameters
  Scenario: Generate random parameter combinations
    Given a CachedRandomOptimizer
    When I generate 50 random combinations
    Then 50 unique combinations should be created
    And parameters should be within configured ranges

  @cached_optimization_run
  Scenario: Run cached random optimization
    Given a CachedRandomOptimizer
    When I run optimization with 100 combinations
    Then cache should be initialized once
    And 100 backtests should be executed
    And data should be loaded only once
    And ML should be trained only once
    And signals should be generated only once

  @cached_performance_gain
  Scenario: Cached optimizer is faster than simple
    Given a CachedRandomOptimizer
    And a SimpleRandomOptimizer
    When both run optimization with 20 combinations
    Then cached optimizer should complete significantly faster
    And speedup should be measurable

  @cached_progress_reporting
  Scenario: Report progress during cached optimization
    Given a CachedRandomOptimizer
    When I run optimization with 100 combinations
    Then progress should be reported every 10 combinations
    And user should see "Progress: X/100 combinations tested"

  @cached_error_handling
  Scenario: Handle backtest errors in cached mode
    Given a CachedRandomOptimizer with initialized cache
    When a backtest fails
    Then error should be caught
    And severe_penalty should be assigned
    And error message should be stored
    And optimization should continue

  @cached_results_collection
  Scenario: Collect results from all backtests
    Given completed cached optimization
    When I process results
    Then each result should include params
    And each result should include fitness
    And each result should include return, sharpe, trades
    And failed results should be marked

  @cached_results_filtering
  Scenario: Filter valid results from failures
    Given optimization results with some failures
    When I filter valid results
    Then only results without severe_penalty should remain
    And results should be sorted by fitness
    And best result should be identified

  @cached_results_summary
  Scenario: Generate cached optimization summary
    Given completed cached optimization
    Then results should include optimizer_type
    And results should include total_combinations
    And results should include valid_results count
    And results should include best_result
    And results should include all_results
    And cache_used flag should be true

  @cached_default_combinations
  Scenario: Use default n_combinations from config
    Given a CachedRandomOptimizer
    And n_combinations is not specified
    When I run optimization
    Then n_combinations should default from config defaults

  @cached_zero_hardcoded
  Scenario: No hardcoded values in cached optimizer
    Given a CachedRandomOptimizer
    Then all parameter ranges should come from config
    And data_source path should come from config
    And n_combinations default should come from config
    And severe_penalty should come from config