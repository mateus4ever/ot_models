Feature: Trading System Backtest Orchestration
  As a trading strategy developer
  I want to test the backtesting orchestrator's ability to handle multiple strategies and markets
  So that I can validate the orchestration capabilities before live trading

  Background:
    Given the system has proper directory structure
    And test data files are available in tests/data
    And smoke_config.json is available in tests/config

  @config @orchestrator @initialization @smoke
  Scenario Outline: BacktestOrchestrator initialization with managers
    Given I have the <config_file> file in <config_directory>
    When I initialize a BacktestOrchestrator with the config
    Then the orchestrator should be properly initialized
    And a DataManager should be created and configured
    And a MoneyManager should be created and configured
    And configuration values should be cached correctly
    And backtesting method should be set to <expected_method> from config
    And mathematical constants should have unity=<expected_unity> from config
    And verbose mode should be <expected_verbose> from config

    Examples:
      | config_file      | config_directory | expected_method | expected_unity | expected_verbose |
      | smoke_config.json| tests/config     | walk_forward    | 1              | true            |

  @data @loading @multi-market
  Scenario Outline: Multi-market data loading coordination
    Given I have <market_count> market data files in <data_directory>
    And each market file has <rows_per_file> rows without headers
    And the files follow the format: <csv_format>
    When I load multiple market data through DataManager
    Then exactly <expected_total_rows> rows should be loaded across all markets
    And the data should have required <column_format> columns
    And the data should be properly sorted by timestamp per market
    And no data loading errors should occur
    And DataManager should cache the loaded data

    Examples:
      | market_count | data_directory | rows_per_file | csv_format                           | expected_total_rows | column_format |
      | 3           | tests/data     | 100           | timestamp;open;high;low;close;volume | 300                | OHLCV         |

  @strategy @initialization @injection
  Scenario Outline: Strategy initialization with dependency injection
    Given I have a BacktestOrchestrator with configured managers
    And I have <strategy_count> mock strategies to test
    When I initialize the strategies with dependency injection
    Then each strategy should receive a DataManager instance
    And each strategy should receive a MoneyManager instance
    And strategies should be ready for execution
    And no initialization errors should occur

    Examples:
      | strategy_count |
      | 2             |

  @execution @serial @orchestration
  Scenario Outline: Serial strategy execution orchestration
    Given I have a BacktestOrchestrator with <strategy_count> initialized mock strategies
    And market data is loaded and prepared
    When I execute strategies in serial mode
    Then each strategy should execute in sequence
    And DataManager should provide training data to each strategy
    And MoneyManager should calculate position sizes for each strategy
    And execution results should be collected for each strategy
    And total execution time should be recorded

    Examples:
      | strategy_count |
      | 2             |

  @execution @parallel @orchestration
  Scenario Outline: Parallel strategy execution orchestration
    Given I have a BacktestOrchestrator with <strategy_count> initialized mock strategies
    And market data is loaded and prepared
    When I execute strategies in parallel mode
    Then strategies should execute concurrently
    And DataManager should handle concurrent data access safely
    And MoneyManager should handle concurrent position calculations safely
    And execution results should be collected from all strategies
    And parallel execution should be faster than serial

    Examples:
      | strategy_count |
      | 2             |

  @data @training @preparation
  Scenario Outline: Training data preparation coordination
    Given I have strategies requiring <training_window> training periods
    And market data with <total_periods> periods available
    When DataManager prepares training data for strategies
    Then training data should contain <training_window> periods per strategy
    And training data should not leak future information
    And multiple strategies should share prepared data efficiently
    And training data preparation should complete without errors

    Examples:
      | training_window | total_periods |
      | 50             | 100           |

  @money @position @sizing
  Scenario Outline: Money management coordination
    Given I have strategies with different risk profiles
    And an initial capital of <initial_capital>
    And maximum risk per trade of <max_risk_pct>%
    When MoneyManager calculates position sizes
    Then position sizes should respect the <max_risk_pct>% risk limit
    And total allocated capital should not exceed <initial_capital>
    And position sizes should be appropriate for strategy risk profiles
    And no position sizing errors should occur

    Examples:
      | initial_capital | max_risk_pct |
      | 10000          | 2            |

  @results @aggregation @analysis
  Scenario Outline: Results aggregation and analysis
    Given I have completed backtest runs for <strategy_count> strategies
    And each strategy has generated trade results
    When I aggregate and analyze the results
    Then results should include performance metrics for each strategy
    And aggregate portfolio performance should be calculated
    And strategy comparison metrics should be available
    And results should be formatted for <output_format> output

    Examples:
      | strategy_count | output_format |
      | 2             | json          |

  @error @handling @orchestration
  Scenario Outline: Error handling during orchestration
    Given I have a BacktestOrchestrator configured for <error_scenario>
    When a <error_type> error occurs during strategy execution
    Then the error should be handled gracefully
    And other strategies should continue execution if possible
    And meaningful error messages should be logged with <log_level>
    And execution should complete with partial results
    And error recovery actions should be performed

    Examples:
      | error_scenario | error_type        | log_level |
      | robust        | strategy_failure  | ERROR     |
      | robust        | data_corruption   | ERROR     |

  @performance @monitoring @orchestration
  Scenario Outline: Performance monitoring during execution
    Given I have strategies executing on <market_count> markets
    When I monitor execution performance
    Then memory usage should be tracked per strategy
    And execution time should be measured per strategy
    And data loading performance should be monitored
    And resource utilization should stay within <max_memory_mb>MB
    And performance metrics should be logged

    Examples:
      | market_count | max_memory_mb |
      | 2           | 1000          |

  @mock @strategy @validation
  Scenario Outline: Mock strategy execution validation
    Given I have <mock_strategy_type> mock strategies for testing
    When I execute the mock strategies
    Then mock strategies should generate predictable signals
    And mock strategy results should be deterministic
    And orchestrator should handle mock strategies same as real strategies
    And mock execution should validate orchestration without strategy complexity

    Examples:
      | mock_strategy_type |
      | always_buy        |
      | always_sell       |
      | random_signals    |