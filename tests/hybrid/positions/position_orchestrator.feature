Feature: PositionOrchestrator coordination
  As a trading system
  I want to coordinate capital, positions, and trade history
  So that portfolio state is accurate and consistent

Background:
  Given config files are available in tests/config/positions
  And a PositionOrchestrator is initialized from configuration
  And DataManager is initialized with market data "DAT_ASCII_EURUSD_M1_2021_100000.csv" from "tests/data/big" with training window 20000

@orchestrator @initialization
Scenario: PositionOrchestrator initializes all components
  Then PositionManager should be initialized
  And PositionTracker should be initialized
  And TradeHistory should be initialized

@orchestrator @listener
Scenario: PositionTracker receives price updates from DataManager
  Given PositionTracker is registered as DataManager listener
  And a position is opened for "EURUSD" with quantity 1000 at entry price 1.1000
  When DataManager advances to next bar
  Then position current price should be updated to 1.1

@orchestrator @open_position
Scenario Outline: Open position coordinates all components
  Given PositionOrchestrator has <initial_capital> initial capital
  When I open position "<trade_id>" for <symbol> <direction> with <quantity> shares at <entry_price> requiring <capital>
  Then capital should be committed in PositionManager
  And position should be tracked in PositionTracker
  And trade should be recorded in TradeHistory as open

  Examples:
    | initial_capital | trade_id  | symbol | direction | quantity | entry_price | capital |
    | 100000          | trade_001 | EURUSD | long      | 1000     | 1.1000      | 11000   |
    | 100000          | trade_002 | GBPUSD | short     | 500      | 1.2500      | 6250    |

@orchestrator @close_position
Scenario: Close position coordinates all components
  Given a position "trade_001" is open for EURUSD with quantity 1000 at 1.1000
  When I close position "trade_001" at exit price 1.1050
  Then capital should be released in PositionManager
  And position should be removed from PositionTracker
  And trade should be updated in TradeHistory with exit data

@orchestrator @portfolio_state
Scenario: Get portfolio state aggregates all sources
  Given PositionOrchestrator has 100000 initial capital
  And position "trade_001" is open: 1000 EURUSD at 1.1000, current 1.1050
  And position "trade_002" is open: 500 GBPUSD at 1.2500, current 1.2450
  When I request portfolio state
  Then total equity should reflect unrealized P&L
  And available cash should reflect committed capital
  And positions should include both open trades
  And daily P&L should be calculated from price changes

@orchestrator @portfolio_metrics
Scenario: Portfolio state includes calculated metrics
  Given PositionOrchestrator has 100000 initial capital
  And a closed trade exists with symbol "EURUSD" quantity 1000 at entry 1.1000 exit 1.1500 with net P&L 500
  And an open position has unrealized P&L of 200
  When I request portfolio state
  Then total P&L should combine realized and unrealized
  And drawdown should be calculated from peak equity
  And peak equity should be tracked correctly