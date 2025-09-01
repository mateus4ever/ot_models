Feature: StrategyFactory Creation and Error Handling
  As a trading system orchestrator
  I want the StrategyFactory to create all implemented strategies successfully
  So that I can reliably instantiate any available strategy

  Background:
    Given the system has proper directory structure
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
      | base         |
      | hybrid       |

  @factory @creation @with_config
  Scenario Outline: Successfully create strategies with configuration
    Given the factory is properly initialized
    And I have a valid configuration object
    When I create a <strategy_name> strategy with configuration
    Then a valid strategy instance should be created with config
    And the configuration should be passed to the strategy
    And no creation errors should occur

    Examples:
      | strategy_name |
      | base         |
      | hybrid       |

  @factory @error @unknown_strategy
  Scenario: Throw error for unknown strategy
    Given the factory is properly initialized
    When I try to create a strategy with name "nonexistent_strategy"
    Then a ValueError should be thrown
    And the error message should mention "Unknown strategy: name "nonexistent_strategy"
    And the error message should list available strategies
    And the available strategies should include "base"
    And the available strategies should include "hybrid"

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
    And the list should contain "hybrid"
    And the list should have exactly 2 strategies

    @factory @resilience @import_errors
  Scenario: Handle import errors gracefully
    Given the factory is properly initialized
    And there is a strategy file with syntax errors
    When the factory performs auto-discovery
    Then the factory should continue discovering other strategies
    And the broken strategy should be skipped
    And other valid strategies should still be available
    And import errors should be logged appropriately

  @factory @resilience @missing_dependencies
  Scenario: Handle missing dependencies gracefully
    Given the factory is properly initialized
    And there is a strategy with missing dependencies
    When the factory performs auto-discovery
    Then the factory should continue discovering other strategies
    And the strategy with missing dependencies should be skipped
    And other valid strategies should remain functional
    And dependency errors should be logged appropriately

  @factory @resilience @partial_failure
  Scenario: Handle partial discovery failures
    Given the factory is properly initialized
    And multiple strategies have various import issues
    When the factory performs auto-discovery
    Then valid strategies should be discovered successfully
    And invalid strategies should be gracefully skipped
    And the factory should remain functional
    And all errors should be properly logged
    And no exceptions should propagate to the caller