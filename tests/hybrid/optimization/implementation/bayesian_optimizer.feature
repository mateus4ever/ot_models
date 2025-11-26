@bayesian_optimizer
Feature: Bayesian Optimizer
  Smart parameter optimization using Bayesian methods with Gaussian Process

  Background:
    Given a valid UnifiedConfig is loaded
    And scikit-optimize library is installed
    And optimization configuration is set

  @bayesian_basic
  Scenario: Create Bayesian optimizer instance
    When I create a BayesianOptimizer
    Then optimizer type should be "BAYESIAN"
    And description should mention "Bayesian optimization"
    And n_calls should come from config
    And n_initial_points should come from config
    And acquisition_function should come from config

  @bayesian_missing_dependency
  Scenario: Handle missing scikit-optimize dependency
    Given scikit-optimize is not installed
    When I attempt to run Bayesian optimization
    Then ImportError should be raised
    And error message should include "scikit-optimize is required"
    And error message should include installation command

  @bayesian_initialization
  Scenario: Initialize cache with train/test split
    Given a BayesianOptimizer
    And market data available
    When I initialize cache
    Then cached_train_data should be set
    And cached_test_data should be set
    And base_trained_strategy should be trained on train data
    And temporal ordering should be preserved
    And no data leakage should occur

  @bayesian_temporal_validation
  Scenario: Validate temporal split prevents look-ahead bias
    Given a BayesianOptimizer
    When I initialize cache
    Then training data should be older than test data
    And train end date should be before test start date
    And temporal validation should be logged

  @bayesian_fresh_signals
  Scenario: Generate fresh signals for each evaluation
    Given a BayesianOptimizer with initialized cache
    When I evaluate a parameter combination
    Then new strategy instance should be created
    And ML models should be copied from base_trained_strategy
    And fresh signals should be generated with optimized parameters
    And no signal caching should occur

  @bayesian_objective_function
  Scenario: Objective function evaluates parameter combination
    Given a BayesianOptimizer with initialized cache
    And parameter combination [0.02, 0.04, 0.10]
    When I call objective_function
    Then parameters should be unpacked correctly
    And new config should be created with parameters
    And fresh strategy should generate signals
    And backtest should run on test data
    And fitness should be calculated
    And negative fitness should be returned for minimization

  @bayesian_evaluation_tracking
  Scenario: Track all evaluations during optimization
    Given a BayesianOptimizer
    When I run optimization with 10 calls
    Then all_evaluations should contain 10 entries
    And each evaluation should have evaluation number
    And each evaluation should have params
    And each evaluation should have fitness
    And each evaluation should have success flag

  @bayesian_error_handling
  Scenario: Handle evaluation errors gracefully
    Given a BayesianOptimizer
    When an evaluation fails with exception
    Then error should be caught
    And severe_penalty should be assigned as fitness
    And evaluation should be marked as unsuccessful
    And error message should be stored
    And optimization should continue

  @bayesian_search_space
  Scenario: Define parameter search space from config
    Given parameter ranges in config
    When I create search space
    Then dimensions should include stop_loss_pct Real range
    And dimensions should include take_profit_pct Real range
    And dimensions should include max_position_size Real range
    And all ranges should come from config

  @bayesian_acquisition_function
  Scenario Outline: Use different acquisition functions
    Given a BayesianOptimizer
    And acquisition_function is set to "<acq_func>"
    When I run optimization
    Then gp_minimize should use "<acq_func>" acquisition function

    Examples:
      | acq_func |
      | EI       |
      | PI       |
      | LCB      |

  @bayesian_initial_points
  Scenario: Random exploration in initial phase
    Given a BayesianOptimizer
    And n_calls is 50
    And n_initial_points is 10
    When I run optimization
    Then first 10 evaluations should be random
    And remaining 40 should use Bayesian acquisition

  @bayesian_gaussian_process
  Scenario: Build Gaussian Process model from evaluations
    Given a BayesianOptimizer
    When I run optimization
    Then GP model should be fit to evaluation history
    And model should predict fitness for unseen parameters
    And acquisition function should use GP predictions

  @bayesian_convergence
  Scenario: Optimization converges to best parameters
    Given a BayesianOptimizer
    When I run optimization with 100 calls
    Then best_result should be identified
    And best parameters should have highest fitness
    And convergence should occur before 100 calls

  @bayesian_reproducibility
  Scenario: Results are reproducible with same random seed
    Given a BayesianOptimizer with random_state 42
    When I run optimization twice
    Then both runs should produce identical results
    And same evaluations should occur in same order

  @bayesian_results_summary
  Scenario: Generate comprehensive results summary
    Given completed Bayesian optimization
    When I generate results
    Then results should include optimizer_type
    And results should include skopt_result
    And results should include all_evaluations
    And results should include valid_evaluations
    And results should include best_result
    And results should include total_duration
    And results should include fresh_signals flag

  @bayesian_valid_evaluations
  Scenario: Filter valid evaluations from failed ones
    Given optimization with some failed evaluations
    When I filter valid_evaluations
    Then only successful evaluations should be included
    And evaluations should be sorted by fitness descending
    And best result should be first

  @bayesian_results_display
  Scenario: Display optimization results in formatted table
    Given completed Bayesian optimization
    When I print optimization results
    Then table should include rank, parameters, and metrics
    And top N performers should be displayed
    And best combination summary should be shown
    And ML component quality should be displayed

  @bayesian_ml_quality_metrics
  Scenario: Include ML component quality in results
    Given a BayesianOptimizer with trained base_strategy
    When I display results
    Then regime detection accuracy should be shown
    And volatility prediction accuracy should be shown
    And duration prediction accuracy should be shown
    And number of ML features should be shown

  @bayesian_parameter_unpacking
  Scenario: Unpack parameter list to dictionary
    Given parameter list [0.02, 0.04, 0.10]
    And parameter indices from config
    When I unpack parameters
    Then stop_loss_pct should be 0.02
    And take_profit_pct should be 0.04
    And max_position_size should be 0.10

  @bayesian_config_update
  Scenario: Update config with optimized parameters
    Given a BayesianOptimizer
    And parameter combination
    When I create new config
    Then risk_management section should be updated
    And debug settings should be preserved
    And no hardcoded values should be used

  @bayesian_ml_model_reuse
  Scenario: Reuse trained ML models across evaluations
    Given base_trained_strategy with trained ML models
    When I create fresh strategy for evaluation
    Then ML models should be copied from base_strategy
    And no retraining should occur
    And only signals should be regenerated

  @bayesian_evaluation_counter
  Scenario: Track evaluation count
    Given a BayesianOptimizer
    When I run multiple evaluations
    Then evaluation_count should increment
    And each evaluation should be numbered sequentially

  @bayesian_zero_hardcoded_values
  Scenario: All configuration comes from config
    Given a BayesianOptimizer
    Then n_calls should come from config
    And n_initial_points should come from config
    And acquisition_function should come from config
    And data_sample_size should come from config
    And random_state should come from config
    And parameter ranges should come from config
    And zero_value should come from config
    And unity_value should come from config

  @bayesian_temporal_split_validation
  Scenario: Validate train/test split sizes
    Given walk_forward config with train_window_size 10000
    And test_window_size 2000
    When I initialize cache
    Then cached_train_data should have 10000 rows
    And cached_test_data should have 2000 rows

  @bayesian_insufficient_data
  Scenario: Handle insufficient data gracefully
    Given data with only 5000 rows
    And required train + test is 12000 rows
    When I initialize cache
    Then ValueError should be raised
    And error should indicate insufficient data

  @bayesian_optimization_complete
  Scenario: Complete optimization workflow
    Given a BayesianOptimizer
    And market data
    When I run optimization with 20 calls
    Then cache should be initialized
    And 20 evaluations should be executed
    And results should be sorted by fitness
    And best parameters should be returned
    And duration should be recorded