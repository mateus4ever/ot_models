Feature: Backtest Result Container
  As a trading system
  I want to store and retrieve backtest results
  So that I can analyze performance and optimize strategies

  Background:
    Given a unified configuration is loaded
    And a trade history is initialized with sample trades

  # ============================================================================
  # RESULT CREATION
  # ============================================================================

  Scenario: Create backtest result with core fields
    Given a strategy name "TestStrategy"
    And a market identifier "TSLA"
    And a trade history with 5 closed trades
    And a configuration dictionary
    When a backtest result is created
    Then result should have strategy name "TestStrategy"
    And result should have market identifier "TSLA"
    And result should have a trade history instance
    And result should have a configuration
    And result should have a timestamp

  Scenario: Create backtest result with optional fields
    Given a strategy name "FullStrategy"
    And a market identifier "AAPL"
    And a trade history with 10 closed trades
    And an equity curve with 50 data points
    And execution time of 5.3 seconds
    When a backtest result is created with optional fields
    Then result should have equity curve with 50 points
    And result should have execution time 5.3 seconds

  # ============================================================================
  # LAZY METRICS CALCULATION
  # ============================================================================

  Scenario: Metrics are calculated on first access
    Given a backtest result with 8 trades
    And metrics have not been calculated yet
    When metrics property is accessed
    Then metrics should be calculated from trade history
    And metrics should contain win_rate
    And metrics should contain total_pnl
    And metrics should contain sharpe_ratio
    And subsequent access should use cached metrics

  Scenario: Metrics calculation with equity curve
    Given a backtest result with 10 trades
    And an equity curve is provided
    When metrics are calculated
    Then metrics should include max_drawdown
    And metrics should include profit_factor
    And metrics should include sortino_ratio

  # ============================================================================
  # CUSTOM DATA EXTENSIBILITY
  # ============================================================================

  Scenario: Add custom metrics to result
    Given a backtest result is created
    When custom metric "max_consecutive_losses" with value 3 is added
    And custom metric "overnight_exposure" with value 0.35 is added
    Then result should contain custom metric "max_consecutive_losses" with value 3
    And result should contain custom metric "overnight_exposure" with value 0.35

  Scenario: Custom data persists in serialization
    Given a backtest result with custom metrics
    When result is serialized to dictionary
    Then dictionary should contain custom_data section
    And custom_data should have all custom metrics

  # ============================================================================
  # FITNESS FOR OPTIMIZER
  # ============================================================================

  Scenario Outline: Get fitness value for optimization
    Given a backtest result with calculated metrics
    And metrics show sharpe_ratio of 1.5
    And metrics show total_return of 0.25
    And metrics show profit_factor of 2.3
    When fitness is requested for "<metric_name>"
    Then fitness value should be <expected_value>

    Examples:
      | metric_name   | expected_value |
      | sharpe_ratio  | 1.5            |
      | total_return  | 0.25           |
      | profit_factor | 2.3            |

  Scenario: Get fitness for non-existent metric fails
    Given a backtest result with calculated metrics
    When fitness is requested for "invalid_metric"
    Then an AttributeError should be raised

  # ============================================================================
  # SERIALIZATION - TO DICT
  # ============================================================================

  Scenario: Serialize result to dictionary
    Given a backtest result with all fields populated
    When result is converted to dictionary
    Then dictionary should contain strategy_name
    And dictionary should contain market_id
    And dictionary should contain timestamp
    And dictionary should contain execution_time_seconds
    And dictionary should contain equity_curve
    And dictionary should contain config
    And dictionary should contain custom_data
    And dictionary should contain trade_count

  Scenario: Serialization includes metrics if calculated
    Given a backtest result with calculated metrics
    When result is converted to dictionary
    Then dictionary should contain metrics section
    And metrics section should have all metric values

  # ============================================================================
  # SAVE - FULL FORMAT
  # ============================================================================

  Scenario: Save complete backtest result
    Given a backtest result with 15 trades
    And metrics have been calculated
    And custom data has been added
    When result is saved to "test_full_result.json" in full format
    Then file "test_full_result.json" should exist
    And file should contain strategy_name
    And file should contain all trades
    And file should contain pre-calculated metrics
    And file should contain equity_curve
    And file should contain custom_data
    And file should contain metadata section

  Scenario: Full save creates directory if not exists
    Given a backtest result is created
    When result is saved to "new_dir/subdir/result.json" in full format
    Then directory "new_dir/subdir" should exist
    And file "new_dir/subdir/result.json" should exist

  # ============================================================================
  # SAVE - TRADES ONLY
  # ============================================================================

  Scenario: Save trades only in lightweight format
    Given a backtest result with 20 trades
    When result is saved to "test_trades.json" in trades-only format
    Then file "test_trades.json" should exist
    And file should contain trades array
    And file should contain metadata
    And file should NOT contain equity_curve
    And file should NOT contain metrics

  # ============================================================================
  # EXPORT TO CSV
  # ============================================================================

  Scenario: Export trades to CSV
    Given a backtest result with 12 trades
    When trades are exported to "test_trades.csv"
    Then file "test_trades.csv" should exist
    And CSV should have headers
    And CSV should have 12 data rows
    And CSV should contain columns: timestamp, direction, entry_price, exit_price, quantity, gross_pnl, net_pnl

  Scenario: Export empty result to CSV
    Given a backtest result with 0 trades
    When trades are exported to "empty_trades.csv"
    Then file "empty_trades.csv" should exist
    And CSV should have headers only
    And CSV should have 0 data rows

  # ============================================================================
  # LOAD - FROM FULL
  # ============================================================================

  Scenario: Load complete backtest from file
    Given a full result file "saved_result.json" exists
    When result is loaded from "saved_result.json"
    Then loaded result should have original strategy_name
    And loaded result should have original market_id
    And loaded result should have all trades restored
    And loaded result should have equity_curve restored
    And loaded result should have custom_data restored
    And loaded result should have pre-calculated metrics

  Scenario: Load from full file with missing metrics
    Given a full result file without pre-calculated metrics exists
    When result is loaded from file
    Then metrics should be recalculated on first access
    And metrics should match expected values

  Scenario: Load from non-existent file fails
    When result is loaded from "missing_file.json"
    Then FileNotFoundError should be raised

  # ============================================================================
  # LOAD - FROM TRADES ONLY
  # ============================================================================

  Scenario: Load from trades-only file
    Given a trades-only file "trades.json" exists with 25 trades
    And a configuration dictionary
    When result is loaded from trades file
    Then result should have 25 trades in trade history
    And result should have strategy_name "Loaded"
    And result should have market_id "Unknown"
    And metrics should be calculated on first access

  Scenario: Load from trades with custom strategy info
    Given a trades-only file exists
    When result is loaded with strategy_name "CustomStrategy" and market_id "EURUSD"
    Then result should have strategy_name "CustomStrategy"
    And result should have market_id "EURUSD"

  # ============================================================================
  # ROUND-TRIP PERSISTENCE
  # ============================================================================

  Scenario: Save and load full result preserves all data
    Given a backtest result with complete data
    When result is saved in full format
    And result is loaded from saved file
    Then loaded result should match original result
    And all trades should be identical
    And all metrics should be identical
    And custom data should be identical

  Scenario: Save trades-only and reload recalculates correctly
    Given a backtest result with known metrics
    When result is saved in trades-only format
    And result is loaded from trades file
    Then recalculated metrics should match original metrics

  # ============================================================================
  # ERROR HANDLING
  # ============================================================================

  Scenario: Handle save failure gracefully
    Given a backtest result is created
    When result is saved to invalid path "/invalid/path/result.json"
    Then save should return False
    And error should be logged

  Scenario: Handle invalid JSON during load
    Given a file with invalid JSON exists
    When result is loaded from invalid file
    Then ValueError should be raised

  Scenario: Handle corrupted trade history during load
    Given a result file with corrupted trade data exists
    When result is loaded from corrupted file
    Then appropriate error should be raised
