Feature: BacktestOrchestrator Methods Testing
  As a developer
  I want to test only the methods in backtest.py
  So that I can verify the BacktestOrchestrator logic works independently

  Background:
    Given the system has proper directory structure
    And smoke_config.json is available in tests/config

  @orchestrator @initialization @backtest_only
  Scenario Outline: BacktestOrchestrator initialization and configuration caching
    Given I have the <config_file> file in <config_directory>
    When I initialize a BacktestOrchestrator with the config
    Then the orchestrator should be properly initialized
    And configuration values should be cached correctly
    And backtesting method should be set to <expected_method> from config
    And mathematical constants should have unity=<expected_unity> from config
    And verbose mode should be <expected_verbose> from config

    Examples:
      | config_file      | config_directory | expected_method | expected_unity | expected_verbose |
      | smoke_config.json| tests/config     | walk_forward    | 1              | true            |

  @multi_strategy @backtest_method @success
  Scenario Outline: Successful multi-strategy backtest execution
    Given I have the <config_file> file in <config_directory>
    And I have test CSV files in <data_directory>
    When I initialize a BacktestOrchestrator with the config
    And I run the multi-strategy backtest
    Then I should receive aggregated backtest results
    And no exceptions should be thrown
    And the results should contain portfolio metrics
    And the execution should complete successfully

    Examples:
      | config_file       | config_directory | data_directory |
      | smoke_config.json | tests/config     | tests/data/big |

  @multi_strategy @backtest_method @error_handling
  Scenario Outline: Backtest error handling
    Given I have <error_condition>
    When I initialize a BacktestOrchestrator with the config
    And I run the multi-strategy backtest
    Then the results should contain an error about <error_type>
    And the error message should be informative
    And the system should fail gracefully

    Examples:
      | error_condition              | error_type           |
      | invalid config file          | configuration        |
      | missing CSV data files       | data loading         |
      | invalid strategy setup       | strategy             |