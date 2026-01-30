Feature: StrategyFactory Creation and Error Handling
  As a trading system orchestrator
  I want the StrategyFactory to create all implemented strategies successfully
  So that I can reliably instantiate any available strategy

  # NOTE: create_strategy_shared() is tested via BacktestOrchestrator tests
  # NOTE: create_strategy_isolated() and create_strategy_with_params() are tested via OptimizationCoordinator tests
  # These integration tests provide sufficient coverage for factory methods

  Background:
    Given config files are available in tests/config/strategies
    And I have a StrategyFactory instance

  @factory @creation @success
  Scenario Outline: Successfully create all implemented strategies
    Given the factory is properly initialized
    When I create a <strategy_name> strategy
    Then a valid strategy instance should be created
    And the strategy should implement StrategyInterface
    And the strategy name should be <strategy_name>
    And no creation errors should occur

    Examples:
      | strategy_name |
      | base          |
      | chained       |
      | triangular_arbitrage|

  @factory @creation @with_config
  Scenario Outline: Successfully create strategies with configuration
    Given the factory is properly initialized
    When I create a <strategy_name> strategy with configuration
    Then a valid strategy instance should be created with config
    And the configuration should be passed to the strategy
    And no creation errors should occur

    Examples:
      | strategy_name |
      | base          |
      | chained       |
      | triangular_arbitrage|

  @factory @error @unknown_strategy
  Scenario: Throw error for unknown strategy
    Given the factory is properly initialized
    When I try to create a strategy with name "nonexistent_strategy"
    Then a ValueError should be thrown
    And the error message should mention "Unknown strategy: name "nonexistent_strategy"
    And the error message should list available strategies
    And the available strategies should include "base"
    And the available strategies should include "chained"

  @factory @error @invalid_inputs
  Scenario Outline: Throw errors for invalid inputs
    Given the factory is properly initialized
    When I try to create a strategy with <invalid_input>
    Then a ValueError should be thrown
    And the error message should be informative

    Examples:
      | invalid_input |
      | None         |
      | ""           |
      | "   "        |

  @factory @available_strategies
  Scenario: Get list of available strategies
    Given the factory is properly initialized
    When I request the list of available strategies
    Then the list should contain "base"
    And the list should contain "chained"
    And the list should contain "triangular_arbitrage"
    And the list should have exactly 3 strategies
