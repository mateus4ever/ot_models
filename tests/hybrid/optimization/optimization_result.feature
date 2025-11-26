@optimization_result
Feature: Optimization Result
  Complete result structure from an optimization run

  Background:
    Given an optimization has been executed

  @result_metadata
  Scenario: Optimization result contains metadata
    Given a completed OptimizationResult
    Then result should have job_id
    And result should have optimizer_type
    And result should have strategy_name
    And result should have started_at timestamp
    And result should have completed_at timestamp
    And result should have duration_seconds
    And result should have status

  @result_status_values
  Scenario Outline: Result status indicates completion state
    Given an OptimizationResult with status "<status>"
    Then status should be one of: completed, failed, cancelled, running

    Examples:
      | status    |
      | completed |
      | failed    |
      | cancelled |

  @result_configuration
  Scenario: Result preserves optimization configuration
    Given a completed OptimizationResult
    Then result should have config_snapshot
    And result should have parameter_space definition
    And result should have n_combinations
    And result should have n_workers
    And result should have execution_mode

  @result_best_params
  Scenario: Result contains best parameter combination
    Given a completed OptimizationResult
    Then result should have best_params dictionary
    And best_params should contain all optimized parameters
    And result should have best_fitness score
    And result should have best_metrics dictionary

  @result_best_metrics
  Scenario: Best metrics include comprehensive performance data
    Given a completed OptimizationResult with best result
    Then best_metrics should include total_return
    And best_metrics should include sharpe_ratio
    And best_metrics should include max_drawdown
    And best_metrics should include num_trades
    And best_metrics should include win_rate
    And best_metrics should include profit_factor

  @result_all_evaluations
  Scenario: Result contains all parameter combinations tried
    Given an optimization with 100 combinations
    When I access all_evaluations
    Then all_evaluations should contain 100 entries
    And each entry should be an EvaluationResult
    And entries should be in chronological order

  @result_valid_evaluations
  Scenario: Result separates valid from failed evaluations
    Given an optimization with 100 total evaluations
    And 8 evaluations failed
    When I access valid_evaluations
    Then valid_evaluations should contain 92 entries
    And all should have success flag as true

  @result_failed_evaluations
  Scenario: Result tracks failed evaluations
    Given an optimization with failed evaluations
    When I access failed_evaluations
    Then each should have success flag as false
    And each should have error_message

  @result_fitness_statistics
  Scenario: Result includes fitness distribution statistics
    Given a completed OptimizationResult
    Then fitness_statistics should include mean
    And fitness_statistics should include std
    And fitness_statistics should include min and max
    And fitness_statistics should include percentiles

  @result_robustness_analysis
  Scenario: Result includes robustness analysis
    Given a completed OptimizationResult
    Then result should have parameter_stability
    And result should have robust_ranges
    And result should have landscape_type
    And result should have robustness_score

  @result_checkpoint_info
  Scenario: Result includes checkpoint information
    Given an OptimizationResult that used checkpoints
    Then result should have checkpoint_path
    And if resumed, result should have resumed_from

  @result_walk_forward_validation
  Scenario: Result includes walk-forward validation if used
    Given an optimization with train/test split
    When I access validation data
    Then result should have train_period
    And result should have test_period
    And result should have train_fitness
    And result should have test_fitness
    And result should have degradation metric

  @result_serialization
  Scenario: OptimizationResult can be serialized to JSON
    Given a completed OptimizationResult
    When I serialize to JSON
    Then all fields should be JSON-serializable
    And datetime should be ISO format strings
    And nested structures should be preserved

  @result_deserialization
  Scenario: OptimizationResult can be loaded from JSON
    Given a serialized OptimizationResult JSON
    When I deserialize
    Then all fields should be restored
    And types should be correct
    And result should be equivalent to original

  @result_comparison
  Scenario: Compare results from different optimization runs
    Given two OptimizationResult instances
    When I compare them
    Then I should be able to compare best_fitness
    And I should be able to compare robustness_score
    And I should be able to compare execution time

  @evaluation_result_structure
  Scenario: EvaluationResult contains single combination data
    Given an EvaluationResult from optimization
    Then it should have evaluation_id
    And it should have params dictionary
    And it should have fitness score
    And it should have all backtest metrics
    And it should have execution_time
    And it should have success flag

  @evaluation_result_metrics
  Scenario: EvaluationResult includes comprehensive metrics
    Given a successful EvaluationResult
    Then metrics should include total_return
    And metrics should include sharpe_ratio
    And metrics should include sortino_ratio
    And metrics should include max_drawdown
    And metrics should include calmar_ratio
    And metrics should include win_rate
    And metrics should include profit_factor
    And metrics should include num_trades
    And metrics should include expectancy

  @evaluation_result_failure
  Scenario: Failed EvaluationResult includes error information
    Given a failed EvaluationResult
    Then success should be false
    And fitness should be severe_penalty
    And error_message should explain failure
    And metrics should be zero or null

  @evaluation_result_timestamp
  Scenario: Each evaluation includes execution timestamp
    Given an EvaluationResult
    Then it should have evaluated_at timestamp
    And timestamp should be in ISO format

  @result_termination_reason
  Scenario: Result explains why optimization stopped
    Given a completed OptimizationResult
    Then result should have termination_reason
    And reason should be one of: completed, time_limit, no_improvement, cancelled, failed
    And if stopped early, result should include trigger_details