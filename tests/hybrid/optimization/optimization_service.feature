@optimization_service
Feature: Optimization Service
  Unified optimization service that runs locally or in cloud with same interface

  Background:
    Given a valid UnifiedConfig is loaded
    And market data is available

  @simple_optimization_run
  #TODO: this is moved from simple_random_optimizer. check if it's make sense to test it here
  Scenario: Run simple random optimization
    Given a SimpleRandomOptimizer
    And market data path
    When I run optimization with 10 combinations
    Then 10 backtests should be executed
    And each backtest should load data fresh
    And each backtest should train strategy fresh
    And results should be collected

  @job_submission
  Scenario Outline: Submit optimization job
    Given an OptimizationService in "<execution_mode>" mode
    And a strategy "base" with parameter space
    And optimizer type "BAYESIAN"
    When I submit an optimization job with 16 workers
    Then a job_id should be returned immediately
    And the job_id should be a valid string
    And the job should be tracked in the service

    Examples:
      | execution_mode |
      | local         |
      | cloud         |

  @job_status_check
  Scenario: Check job status during execution
    Given an optimization job is submitted
    And the job is currently running
    When I check the job status
    Then status should be "running"
    And progress should show "X/Y" format
    And best_score_so_far should be a number
    And eta_seconds should be estimated time remaining
    And started_at should be ISO timestamp

  @job_status_completed
  Scenario: Check status of completed job
    Given an optimization job is submitted
    And the job has completed successfully
    When I check the job status
    Then status should be "completed"
    And completed_at timestamp should be present
    And total_iterations should match requested count

  @job_status_failed
  Scenario: Check status of failed job
    Given an optimization job is submitted
    And the job has failed with an error
    When I check the job status
    Then status should be "failed"
    And error message should be present
    And failed_at timestamp should be present

  @results_retrieval_blocking
  Scenario: Retrieve results with blocking call
    Given an optimization job is submitted
    When I call get_results with the job_id
    And the job is still running
    Then the call should block and wait
    When the job completes
    Then OptimizationResult should be returned
    And best_params should be present
    And fitness_score should be present

  @results_retrieval_async
  Scenario: Retrieve results with non-blocking call
    Given an optimization job is submitted
    When I call get_results_async with the job_id
    And the job is still running
    Then None should be returned immediately
    When the job completes
    And I call get_results_async again
    Then OptimizationResult should be returned

  @results_not_ready
  Scenario: Attempt to retrieve results before completion
    Given an optimization job is submitted
    And the job is still running
    When I call get_results_async with the job_id
    Then None should be returned
    And no exception should be raised

  @results_failed_job
  Scenario: Attempt to retrieve results from failed job
    Given an optimization job is submitted
    And the job has failed
    When I call get_results with the job_id
    Then OptimizationFailedError should be raised
    And error message should explain the failure

  @job_cancellation
  Scenario: Cancel running optimization job
    Given an optimization job is submitted
    And the job is currently running
    When I cancel the job
    Then the job status should change to "cancelled"
    And worker processes should be terminated
    And partial results should be saved

  @local_execution
  Scenario: Local execution spawns process pool
    Given an OptimizationService in "local" mode
    When I submit an optimization job with 16 workers
    Then 16 worker processes should be spawned locally
    And each worker should run backtests independently
    And results should be collected in main process

  @cloud_execution
  Scenario: Cloud execution submits to k8s
    Given an OptimizationService in "cloud" mode
    When I submit an optimization job with 16 workers
    Then optimizer package should be serialized
    And job should be submitted to k8s cluster
    And job_id should be trackable in cloud

  @execution_mode_consistency
  Scenario: Same interface works for local and cloud
    Given an OptimizationService in "local" mode
    And an OptimizationService in "cloud" mode
    When I submit identical optimization jobs to both
    Then both should return job_id immediately
    And both should support get_status
    And both should support get_results
    And both should support cancel_job

  @checkpoint_persistence
  Scenario: Job checkpoints are saved during execution
    Given an optimization job is running
    When checkpoint interval is reached
    Then current progress should be saved to storage
    And checkpoint should include iteration number
    And checkpoint should include best_params_so_far
    And checkpoint should include all results so far

  @job_resume
  Scenario: Resume optimization from checkpoint
    Given an optimization job was interrupted at iteration 500
    And checkpoint file exists
    When I resume the optimization job
    Then execution should continue from iteration 501
    And previous results should be preserved
    And final results should be complete

  @job_id_generation
  Scenario: Generate unique job identifiers
    When I submit multiple optimization jobs
    Then each job_id should be unique
    And job_id should include timestamp
    And job_id should be human-readable

  @invalid_job_id
  Scenario: Handle invalid job_id gracefully
    Given an invalid or non-existent job_id
    When I check status with that job_id
    Then JobNotFoundError should be raised
    And error message should include the invalid job_id

  @concurrent_jobs
  Scenario: Support multiple concurrent optimization jobs
    Given an OptimizationService
    When I submit 5 optimization jobs
    Then all 5 jobs should run concurrently
    And each job should be independently trackable
    And resources should be distributed across jobs

  @results_storage
  Scenario: Optimization results are persisted
    Given an optimization job has completed
    When results are generated
    Then results should be saved to persistent storage
    And results should include job_id
    And results should include all parameter combinations tried
    And results should include fitness scores
    And results should include best_params

  @results_retrieval_multiple_times
  Scenario: Results can be retrieved multiple times
    Given an optimization job has completed
    When I call get_results twice with same job_id
    Then both calls should return identical results
    And results should be loaded from storage
    And no re-execution should occur

    @early_stopping_no_improvement
    Scenario: Stop early when no improvement detected
      Given an optimization job with no_improvement_threshold of 50 iterations
      When 50 consecutive iterations show no fitness improvement
      Then optimization should stop automatically
      And status should be "completed_early"
      And reason should be "no_improvement"

    @early_stopping_time_limit
    Scenario: Stop when time budget exhausted
      Given an optimization job with max_runtime of 2 hours
      When 2 hours have elapsed
      Then optimization should stop gracefully
      And status should be "completed_early"
      And reason should be "time_limit_reached"