@optimization_coordinator
Feature: Optimization Coordinator
  Central coordinator manages workers, collects results, and handles checkpointing

  Background:
    Given a valid UnifiedConfig is loaded
    And optimization parameters are configured

  @coordinator_initialization
  Scenario: Initialize optimization coordinator
    Given an OptimizationCoordinator
    Then coordinator should have empty evaluations list
    And coordinator should have checkpoint_interval from config
    And coordinator should have checkpoint_time_interval from config
    And coordinator should have storage backend configured

  @coordinator_spawn_workers
  Scenario: Coordinator spawns worker processes
    Given an OptimizationCoordinator
    And n_workers is set to 16
    When I start optimization
    Then 16 worker processes should be spawned
    And each worker should have unique worker_id
    And each worker should have result queue

  @coordinator_worker_assignment
  Scenario: Coordinator assigns parameter combinations to workers
    Given an OptimizationCoordinator with 4 workers
    And 100 parameter combinations to evaluate
    When I start optimization
    Then combinations should be distributed to workers
    And each worker should receive approximately 25 combinations
    And no combination should be assigned twice

  @coordinator_collect_result
  Scenario: Coordinator collects result from worker
    Given an OptimizationCoordinator with running workers
    When a worker completes evaluation
    And worker sends EvaluationResult to coordinator
    Then coordinator should receive result
    And result should be added to all_evaluations
    And evaluation count should increment

  @coordinator_checkpoint_by_count
  Scenario: Coordinator checkpoints every N evaluations
    Given an OptimizationCoordinator
    And checkpoint_interval is 50
    When coordinator collects 50 results
    Then checkpoint should be saved
    And checkpoint should contain all 50 evaluations

  @coordinator_checkpoint_by_time
  Scenario: Coordinator checkpoints every M minutes
    Given an OptimizationCoordinator
    And checkpoint_time_interval is 300 seconds
    When 300 seconds elapse since last checkpoint
    Then checkpoint should be saved
    And checkpoint should contain all evaluations so far

  @coordinator_checkpoint_combines_triggers
  Scenario: Either time or count triggers checkpoint
    Given an OptimizationCoordinator
    And checkpoint_interval is 100
    And checkpoint_time_interval is 300 seconds
    When coordinator collects 100 results
    Then checkpoint should be saved
    When 300 seconds elapse with only 50 more results
    Then another checkpoint should be saved

  @coordinator_checkpoint_content
  Scenario: Checkpoint contains complete state
    Given an OptimizationCoordinator with 150 evaluations
    When checkpoint is saved
    Then checkpoint should include job_id
    And checkpoint should include current iteration
    And checkpoint should include total iterations
    And checkpoint should include all evaluations
    And checkpoint should include best result so far
    And checkpoint should include elapsed time
    And checkpoint should include optimizer state

  @coordinator_storage_local
  Scenario: Coordinator saves checkpoint to local filesystem
    Given an OptimizationCoordinator with local storage
    When checkpoint is saved
    Then checkpoint file should be created in checkpoints directory
    And filename should include job_id and iteration
    And file should be valid JSON

  @coordinator_storage_s3
  Scenario: Coordinator saves checkpoint to S3
    Given an OptimizationCoordinator with S3 storage
    When checkpoint is saved
    Then checkpoint should be uploaded to S3 bucket
    And object key should include job_id and iteration
    And object should be retrievable

  @coordinator_resume_from_checkpoint
  Scenario: Resume optimization from checkpoint
    Given a checkpoint file exists with 342 completed evaluations
    And total iterations is 1000
    When I create OptimizationCoordinator and resume
    Then all_evaluations should be loaded from checkpoint
    And evaluation count should start at 343
    And remaining work should be 658 iterations
    And workers should continue from iteration 343

  @coordinator_resume_validation
  Scenario: Validate checkpoint before resuming
    Given a checkpoint file
    When I attempt to resume
    Then checkpoint should be validated for corruption
    And job_id should match if specified
    And optimizer_type should match
    And parameter_space should match

  @coordinator_resume_failure
  Scenario: Handle corrupted checkpoint gracefully
    Given a corrupted checkpoint file
    When I attempt to resume
    Then error should be raised
    And error should indicate checkpoint corruption
    And user should be prompted to start fresh

  @coordinator_progress_tracking
  Scenario: Track optimization progress
    Given an OptimizationCoordinator
    When workers complete evaluations
    Then progress should be calculated as completed/total
    And elapsed time should be tracked
    And estimated time remaining should be calculated
    And progress should be logged periodically

  @coordinator_best_result_tracking
  Scenario: Track best result during optimization
    Given an OptimizationCoordinator
    When evaluations are collected
    Then best_result should be updated when better fitness found
    And best_result should always contain highest fitness
    And best_params should be current best

  @coordinator_worker_failure
  Scenario: Handle worker process failure
    Given an OptimizationCoordinator with 16 workers
    When worker 7 crashes
    Then coordinator should detect worker failure
    And failed worker's pending work should be reassigned
    And optimization should continue with remaining workers
    And failure should be logged

  @coordinator_graceful_shutdown
  Scenario: Gracefully shutdown optimization
    Given an OptimizationCoordinator with running workers
    When shutdown signal is received
    Then coordinator should signal all workers to stop
    And coordinator should wait for workers to finish current evaluations
    And final checkpoint should be saved
    And workers should be terminated

  @coordinator_cancellation
  Scenario: Cancel running optimization
    Given an OptimizationCoordinator with 1000 total iterations
    And 342 iterations completed
    When user cancels optimization
    Then coordinator should stop assigning new work
    And workers should finish current evaluations
    And checkpoint should be saved with status "cancelled"
    And partial results should be preserved

  @coordinator_result_aggregation
  Scenario: Aggregate results from all workers
    Given an OptimizationCoordinator with completed optimization
    When I request final results
    Then all_evaluations should be aggregated
    And valid_evaluations should be filtered
    And evaluations should be sorted by fitness
    And fitness_statistics should be calculated
    And robustness_analysis should be performed

  @coordinator_final_results
  Scenario: Save final optimization results
    Given a completed optimization
    When coordinator saves final results
    Then OptimizationResult should be created
    And result should be saved to storage
    And result should include all metadata
    And result should include all evaluations
    And result should include analysis

  @coordinator_multi_objective
  Scenario: Coordinate multi-objective optimization
    Given an OptimizationCoordinator for multi-objective optimization
    When evaluations are collected
    Then Pareto front should be calculated
    And non-dominated solutions should be tracked
    And diversity metrics should be calculated

  @coordinator_bayesian_state
  Scenario: Coordinate Bayesian optimization with GP model
    Given an OptimizationCoordinator for Bayesian optimization
    When evaluations are collected
    Then GP model should be updated
    And next parameter combinations should be suggested
    And acquisition function values should be tracked
    And GP model state should be saved in checkpoint

  @coordinator_parallel_safety
  Scenario: Coordinator is thread-safe for result collection
    Given an OptimizationCoordinator with 16 workers
    When multiple workers send results simultaneously
    Then all results should be collected without data loss
    And evaluation count should be accurate
    And no race conditions should occur

  @coordinator_memory_management
  Scenario: Coordinator manages memory for large optimizations
    Given an OptimizationCoordinator with 10000 iterations
    When many evaluations are collected
    Then memory usage should remain reasonable
    And old checkpoints should be cleaned up
    And only recent checkpoints should be retained

  @coordinator_checkpoint_retention
  Scenario: Manage checkpoint retention policy
    Given an OptimizationCoordinator
    And max_checkpoints_retained is 5
    When 10 checkpoints have been created
    Then only most recent 5 checkpoints should be kept
    And oldest checkpoints should be deleted

  @coordinator_status_query
  Scenario: Query optimization status
    Given a running OptimizationCoordinator
    When I query status
    Then status should include job_id
    And status should include progress percentage
    And status should include elapsed time
    And status should include ETA
    And status should include best fitness so far
    And status should include worker states

  @coordinator_zero_hardcoded
  Scenario: All coordinator settings from config
    Given an OptimizationCoordinator
    Then checkpoint_interval should come from config
    And checkpoint_time_interval should come from config
    And max_checkpoints_retained should come from config
    And storage_backend should come from config
    And n_workers should come from config

    @train_test_split
    Scenario: Split data for proper validation
      Given an OptimizationCoordinator
      And market data from 2020-2025
      When I configure train_test_split with 0.7 ratio
      Then training data should be 2020-2023.5
      And test data should be 2023.5-2025
      And test data should never be used during optimization

    @test_validation
    Scenario: Validate best parameters on test set
      Given optimization completed on training data
      And best parameters found
      When I validate on test data
      Then backtest should run with best parameters
      And test results should be separate from train results
      And test performance should indicate overfitting if worse