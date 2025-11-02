Feature: Base Strategy Execution
  As a trading system
  I want to execute strategies with different product types
  So that I can generate trades and collect results based on signals

  Background:
    Given a unified configuration is loaded
    And a data manager is initialized with market data
    And a money manager is initialized
    And a trade history is initialized

  # ============================================================================
  # DEPENDENCY INJECTION
  # ============================================================================

  Scenario: Strategy initialization with all dependencies
    Given a base strategy "TestStrategy" is created
    When data manager is injected into strategy
    And money manager is injected into strategy
    And trade history is injected into strategy
    And a signal generator is added to strategy
    Then strategy should have all required dependencies

  # ============================================================================
  # STOCK PRODUCT - LONG ONLY
  # ============================================================================

  Scenario: Stock product generates LONG trades only
    Given a base strategy "StockStrategy" is created with stock product
    And all dependencies are injected into strategy
    And market data has 30 data points
    And signal generator produces the following signals:
      """
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL
      """
    When strategy runs
    Then trade history should contain 6 trades
    And all trades should have direction "LONG"
    And all trades should have exit_reason "signal"
    And result should show 6 total trades
    And result should contain trade history
    And result should contain performance metrics

  # ============================================================================
  # FOREX PRODUCT - LONG AND SHORT
  # ============================================================================

  Scenario: Forex product generates LONG and SHORT trades
    Given a base strategy "ForexStrategy" is created with forex product
    And all dependencies are injected into strategy
    And market data has 25 data points
    And signal generator produces the following signals:
      """
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BEARISH,NEUTRAL,NEUTRAL,BULLISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BEARISH,NEUTRAL,NEUTRAL,BULLISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL
      """
    When strategy runs
    Then trade history should contain 10 trades
    And trade history should contain 5 LONG trades
    And trade history should contain 5 SHORT trades
    And all trades should have exit_reason "signal"
    And result should show 10 total trades

  # ============================================================================
  # CFD PRODUCT - LONG, SHORT, AND STOP LOSS
  # ============================================================================

  Scenario: CFD product with stop loss exits
    Given a base strategy "CFDStrategy" is created with CFD product
    And all dependencies are injected into strategy
    And market data has 30 data points with volatility
    And stop loss is configured with tight parameters
    And signal generator produces the following signals:
      """
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BEARISH,NEUTRAL,NEUTRAL,BULLISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BEARISH,NEUTRAL,NEUTRAL,BULLISH,NEUTRAL,
      BULLISH,NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,
      BEARISH,NEUTRAL,NEUTRAL,BULLISH,NEUTRAL
      """
    When strategy runs
    Then trade history should contain at least 10 trades
    And trade history should contain both LONG and SHORT trades
    And trade history should contain trades with exit_reason "stop_loss"
    And trade history should contain trades with exit_reason "signal"
    And result should contain performance metrics
    And all trades in trade history should have valid timestamps
    And all trades in trade history should have entry_price and exit_price
    And all trades in trade history should have gross_pnl and net_pnl calculated

  # ============================================================================
  # NEUTRAL SIGNALS - NO TRADES
  # ============================================================================

  Scenario: Only NEUTRAL signals generate no trades
    Given a base strategy "NeutralStrategy" is created with forex product
    And all dependencies are injected into strategy
    And market data has 20 data points
    And signal generator produces only NEUTRAL signals for 20 bars
    When strategy runs
    Then trade history should contain 0 trades
    And result should show 0 total trades
    And result should show 0.0 total pnl

  # ============================================================================
  # PRODUCT VALIDATION
  # ============================================================================

  Scenario: Stock product prevents SHORT positions
    Given a base strategy "StockValidation" is created with stock product
    And all dependencies are injected into strategy
    And market data has 15 data points
    And signal generator produces the following signals:
      """
      BEARISH,BEARISH,BEARISH,BULLISH,NEUTRAL,
      NEUTRAL,BEARISH,NEUTRAL,NEUTRAL,BULLISH,
      NEUTRAL,NEUTRAL,BEARISH,NEUTRAL,NEUTRAL
      """
    When strategy runs
    Then trade history should contain 2 trades
    And all trades should have direction "LONG"
    And trade history should not contain any SHORT trades
    And result should show 2 total trades

  # ============================================================================
  # RESULT STRUCTURE VALIDATION
  # ============================================================================

  Scenario: Strategy result contains all required components
    Given a base strategy "ResultValidation" is created with CFD product
    And all dependencies are injected into strategy
    And market data has 20 data points
    And signal generator produces mixed signals resulting in 5 trades
    When strategy runs
    Then result should have strategy name "ResultValidation"
    And result should have market identifier
    And result should contain trade history instance
    And result should contain performance metrics with:
      | metric            |
      | total_trades      |
      | winning_trades    |
      | losing_trades     |
      | total_pnl         |
      | win_rate          |
    And result should be serializable to JSON
    And result should be serializable to CSV

  # ============================================================================
  # SIGNAL COUNT VALIDATION
  # ============================================================================

  Scenario Outline: Signal distribution analysis
    Given a base strategy "SignalAnalysis" is created with <product> product
    And all dependencies are injected into strategy
    And market data has <data_points> data points
    And signal generator produces <bullish> BULLISH, <bearish> BEARISH, <neutral> NEUTRAL signals
    When strategy runs
    Then trade history should contain <expected_trades> trades
    And result should show signal distribution

    Examples:
      | product | data_points | bullish | bearish | neutral | expected_trades |
      | stock   | 25          | 10      | 5       | 10      | 5               |
      | forex   | 30          | 8       | 8       | 14      | 8               |
      | cfd     | 35          | 10      | 10      | 15      | 10              |
