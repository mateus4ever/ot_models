Feature: Triangular Arbitrage Strategy Execution
  As a trading system
  I want to execute triangular arbitrage spread positions
  So that I can trade mean-reverting spreads across 3 markets

  Background:
    Given config files are available in tests/config/strategies/implementation/triangular_strategy
    And data source is set to tests/data/forex
    And a data manager is initialized with three markets:
      | market  |
      | EURUSD  |
      | EURGBP  |
      | GBPUSD  |
    And temporal pointer is initialized with lookback window 93000
    And a triangular arbitrage predictor is initialized
    And a position orchestrator is initialized
    And a money manager is initialized
    And I have a TriangularStrategy instance

  # ============================================================================
  # DEPENDENCY INJECTION
  # ============================================================================

  Scenario: TriangularStrategy initialization with all dependencies
    Given a triangular strategy "TriangularArbitrage" is created
    Then strategy should have all required dependencies
    And strategy should accept predictor interface

  # ============================================================================
  # STRATEGY RUN
  # ============================================================================
  Scenario: Strategy runs complete backtest
    Given a triangular strategy "TriangularArbitrage" is created
    When strategy runs
    Then strategy should complete without error
    And trade history should have at least 1 trade
    And final result should be logged

  # ============================================================================
  # 3-LEG POSITION OPENING - LONG_SPREAD
  # ============================================================================

#  Scenario: LONG_SPREAD signal opens 3 positions atomically
#    Given a triangular strategy is running
#    And current prices are:
#      | market | price   |
#      | EURUSD | 1.0900  |
#      | EURGBP | 0.8500  |
#      | GBPUSD | 1.2941  |
#    And predictor generates prediction:
#      """
#      {
#        "signal": "LONG_SPREAD",
#        "confidence": 0.85,
#        "z_score": -2.5,
#        "spread_pips": -80.0
#      }
#      """
#    And position size is configured as 1.0 lot
#    When strategy processes the prediction
#    Then 3 positions should be opened:
#      | market | direction | quantity |
#      | EURUSD | LONG      | 1.0      |
#      | EURGBP | SHORT     | 1.0      |
#      | GBPUSD | SHORT     | 1.0      |
#    And spread position should be tracked with:
#      | field              | value  |
#      | signal             | LONG_SPREAD |
#      | entry_z_score      | -2.5   |
#      | entry_spread_pips  | -80.0  |
#    And all 3 positions should have same timestamp
#
#  Scenario: Strategy handles multiple triangles
#  Given a triangular strategy with 2 predictors:
#    | target | leg1   | leg2   |
#    | EURUSD | EURGBP | GBPUSD |
#    | EURJPY | EURUSD | USDJPY |
#  When strategy processes one tick
#  Then both predictors should be called
#  And positions from both triangles should be tracked
#  And money manager should see combined exposure


#  # ============================================================================
#  # 3-LEG POSITION OPENING - SHORT_SPREAD
#  # ============================================================================
#
#  Scenario: SHORT_SPREAD signal opens 3 positions atomically
#    Given a triangular strategy is running
#    And current prices are:
#      | market | price   |
#      | EURUSD | 1.1100  |
#      | EURGBP | 0.8500  |
#      | GBPUSD | 1.2941  |
#    And predictor generates prediction:
#      """
#      {
#        "signal": "SHORT_SPREAD",
#        "confidence": 0.90,
#        "z_score": 2.8,
#        "spread_pips": 100.0
#      }
#      """
#    When strategy processes the prediction
#    Then 3 positions should be opened:
#      | market | direction | quantity |
#      | EURUSD | SHORT     | 1.0      |
#      | EURGBP | LONG      | 1.0      |
#      | GBPUSD | LONG      | 1.0      |
#    And spread position should be tracked
#
#  # ============================================================================
#  # ATOMIC EXECUTION - ROLLBACK ON FAILURE
#  # ============================================================================
#
#  Scenario: Failed leg execution triggers rollback
#    Given a triangular strategy is running
#    And predictor generates LONG_SPREAD signal
#    And EURUSD position opens successfully
#    And EURGBP position opens successfully
#    But GBPUSD position fails to open
#    When strategy attempts atomic 3-leg opening
#    Then EURUSD position should be closed (rollback)
#    And EURGBP position should be closed (rollback)
#    And no spread position should be tracked
#    And error should be logged
#
#  Scenario: Insufficient capital prevents all positions
#    Given a triangular strategy is running
#    And predictor generates LONG_SPREAD signal
#    And required capital is 30,000
#    But available capital is 15,000
#    When strategy attempts to open positions
#    Then no positions should be opened
#    And no spread position should be tracked
#    And capital check should fail before any execution
#
#  # ============================================================================
#  # 3-LEG POSITION CLOSING - SIGNAL
#  # ============================================================================
#
#  Scenario: CLOSE signal exits all 3 positions
#    Given a triangular strategy is running
#    And LONG_SPREAD position is open with 3 legs
#    And current prices are:
#      | market | price   |
#      | EURUSD | 1.1000  |
#      | EURGBP | 0.8500  |
#      | GBPUSD | 1.2941  |
#    And predictor generates prediction:
#      """
#      {
#        "signal": "CLOSE",
#        "confidence": 1.0,
#        "z_score": -0.3
#      }
#      """
#    When strategy processes the prediction
#    Then all 3 leg positions should be closed
#    And trades should be recorded with exit_reason "signal"
#    And spread position should be cleared
#    And P&L should be calculated
#
#  # ============================================================================
#  # Z-SCORE STOP LOSS
#  # ============================================================================
#
#  Scenario: Stop loss triggered when Z-score exceeds threshold
#    Given a triangular strategy is running
#    And LONG_SPREAD position is open
#    And entry Z-score was -2.5
#    And stop loss threshold is configured as 3.0
#    And predictor generates prediction with z_score 3.5
#    When strategy checks stop loss condition
#    Then stop loss should trigger
#    And all 3 positions should be closed with exit_reason "stop_loss"
#    And spread position should be cleared
#
#  Scenario: Stop loss NOT triggered when within threshold
#    Given a triangular strategy is running
#    And SHORT_SPREAD position is open
#    And entry Z-score was 2.5
#    And stop loss threshold is 3.0
#    And predictor generates prediction with z_score 2.8
#    When strategy checks stop loss condition
#    Then stop loss should NOT trigger
#    And position should remain open
#    And no trades should be closed
#
#  # ============================================================================
#  # COMPLETE STRATEGY RUN
#  # ============================================================================
#
#  Scenario: Full strategy run with multiple trades
#    Given a triangular strategy "TriangularBacktest" is created
#    And all dependencies are injected
#    And market data has 500 data points
#    And predictor is configured and calibrated
#    And predictor generates the following signal sequence:
#      """
#      HOLD × 50, LONG_SPREAD, HOLD × 30, CLOSE,
#      HOLD × 40, SHORT_SPREAD, HOLD × 35, CLOSE,
#      HOLD × 50, LONG_SPREAD, HOLD × 28, CLOSE,
#      HOLD × 45, SHORT_SPREAD, HOLD × 32, CLOSE,
#      HOLD × 105
#      """
#    When strategy runs
#    Then trade history should contain 4 completed spread trades
#    And each trade should have 3 leg positions
#    And trade history should show:
#      | metric          | value |
#      | total_trades    | 4     |
#      | long_spreads    | 2     |
#      | short_spreads   | 2     |
#    And all trades should have exit_reason "signal"
#
#  # ============================================================================
#  # HOLD SIGNAL - NO ACTION
#  # ============================================================================
#
#  Scenario: HOLD signals generate no trades
#    Given a triangular strategy is running
#    And market data has 100 data points
#    And predictor generates only HOLD signals
#    When strategy runs
#    Then no positions should be opened
#    And trade history should contain 0 trades
#    And strategy should complete successfully
#
#  Scenario: HOLD signal while position open maintains position
#    Given a triangular strategy is running
#    And LONG_SPREAD position is open
#    And predictor generates HOLD signal
#    When strategy processes the prediction
#    Then position should remain open
#    And no trades should be closed
#    And no new positions should be opened
#
#  # ============================================================================
#  # SEQUENTIAL SIGNALS
#  # ============================================================================
#
#  Scenario: Cannot open new position while one is open
#    Given a triangular strategy is running
#    And LONG_SPREAD position is currently open
#    And predictor generates SHORT_SPREAD signal
#    When strategy processes the prediction
#    Then SHORT_SPREAD signal should be ignored
#    And no new positions should be opened
#    And only CLOSE or stop loss can exit current position
#
#  # ============================================================================
#  # RESULT STRUCTURE
#  # ============================================================================
#
#  Scenario: Strategy result contains all required components
#    Given a triangular strategy "ResultTest" is created
#    And all dependencies are injected
#    And strategy completes 5 spread trades
#    When strategy run completes
#    Then result should have strategy name "ResultTest"
#    And result should contain trade history
#    And result should contain performance metrics:
#      | metric          |
#      | total_trades    |
#      | winning_trades  |
#      | losing_trades   |
#      | total_pnl       |
#      | win_rate        |
#    And result should be serializable to JSON
#    And each trade should list all 3 leg positions
#
#  # ============================================================================
#  # TRADE RECORDING
#  # ============================================================================
#
#  Scenario: Each spread trade records all leg details
#    Given a triangular strategy completes one LONG_SPREAD trade
#    When trade is recorded in history
#    Then trade record should contain:
#      | field           | present |
#      | trade_id        | yes     |
#      | signal          | yes     |
#      | entry_z_score   | yes     |
#      | exit_z_score    | yes     |
#      | entry_timestamp | yes     |
#      | exit_timestamp  | yes     |
#      | exit_reason     | yes     |
#      | leg_count       | 3       |
#    And each leg should record:
#      | field           |
#      | market          |
#      | direction       |
#      | entry_price     |
#      | exit_price      |
#      | quantity        |
#      | pnl             |
#
#  # ============================================================================
#  # P&L CALCULATION
#  # ============================================================================
#
#  Scenario: P&L calculated from spread convergence
#    Given LONG_SPREAD position opened at:
#      | market | direction | price   | quantity |
#      | EURUSD | LONG      | 1.0900  | 1.0      |
#      | EURGBP | SHORT     | 0.8500  | 1.0      |
#      | GBPUSD | SHORT     | 1.2941  | 1.0      |
#    And position closed at:
#      | market | price   |
#      | EURUSD | 1.1000  |
#      | EURGBP | 0.8500  |
#      | GBPUSD | 1.2941  |
#    When strategy calculates P&L
#    Then EURUSD leg P&L should be (1.1000 - 1.0900) × 1.0 × 100000 = 1000
#    And EURGBP leg P&L should be (0.8500 - 0.8500) × 1.0 × 100000 = 0
#    And GBPUSD leg P&L should be (1.2941 - 1.2941) × 1.0 × 100000 = 0
#    And total P&L should be sum of all legs
#
#  # ============================================================================
#  # CONFIGURATION
#  # ============================================================================
#
#  Scenario: Strategy reads configuration correctly
#    Given configuration specifies:
#      | parameter         | value   |
#      | position_size     | 2.0     |
#      | stop_loss_z_score | 3.5     |
#    When triangular strategy is initialized
#    Then strategy should use position size 2.0 lots per leg
#    And strategy should use stop loss threshold 3.5
#
#  Scenario: Missing configuration causes initialization failure
#    Given configuration is missing triangular_arbitrage section
#    When triangular strategy is initialized
#    Then initialization should fail
#    And error should indicate missing configuration