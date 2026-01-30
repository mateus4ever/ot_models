Feature: SpreadTradeHistory
  As a trading system
  I want to track spread trades from triangular arbitrage
  So that I can persist and reload spread P&L separately from leg costs

Background:
  Given data_management.json is available in tests/config/positions and loaded
  And I have a SpreadTradeHistory instance with base_currency "USD"

# CORE WORKFLOW
# ============================================================================

@spread_history @core
Scenario: Add spread trade with leg references
  Given I have a spread trade with:
    | field        | value                    |
    | timestamp    | 2024-04-25T10:00:00Z     |
    | entry_price  | 0.0007                   |
    | exit_price   | 0.0001                   |
    | quantity     | 1.5                      |
    | direction    | SHORT_SPREAD             |
    | leg_trades   | ["leg1", "leg2", "leg3"] |
    | gross_pnl    | 95.25                    |
    | status       | closed                   |
    | entry_date   | 2024-04-25T08:00:00Z     |
    | exit_date    | 2024-04-25T10:00:00Z     |
  When I add the spread trade to history
  Then trade count should be 1
  And the trade should have 3 leg references

@spread_history @core
Scenario: Load spread trades from JSON
  When I load spread history from "tests/data/trade/spread_trades.json"
  Then trade count should be 5
  And leg_trades references should be preserved
  And gross_pnl should be preserved

@spread_history @core
Scenario: Load, modify and save spread trades
  When I load spread history from "tests/data/trade/spread_trades.json"
  And I update timestamp on trade "spread-005" to current time
  And I save spread history to "tests/data/trade/spread_output.json"
  Then trade count should be 5
  And the file should be created successfully

## ============================================================================
## SPREAD TRADE SCENARIOS
## ============================================================================

@spread_history @validation @error
Scenario Outline: Reject invalid spread trades
  Given I load base trade from "tests/data/trade/spread_trades.json"
  And I modify trade with <invalid_condition>
  When I add the spread trade to history
  Then the trade should be rejected

  Examples:
    | invalid_condition  |
    | missing leg_trades |
    | empty leg_trades   |
    | missing gross_pnl  |

@spread_history @finalization @core
Scenario: Spread trade net_pnl defaults to gross_pnl (no fees)
  Given I load base trade from "tests/data/trade/spread_trades.json"
  And I set gross_pnl to 95.25
  When I add the spread trade to history
  Then the stored trade should have net_pnl 95.25

@spread_history @ordering @core
Scenario: Multiple spread trades stored in chronological order
  When I load spread history from "tests/data/trade/spread_trades.json"
  Then trade count should be 5
  And trades should be stored in chronological order

@spread_history @statistics @core
Scenario: Calculate statistics from spread trades
  When I load spread history from "tests/data/trade/spread_trades.json"
  And I calculate trade statistics
  Then total_positions should be 5
  And winning_positions should be 3
  And losing_positions should be 1
  And break_even_positions should be 1
  And total_pnl should be 259.75

@spread_history @persistence @core
Scenario: Round-trip save and load preserves all fields
  Given I load base trade from "tests/data/trade/spread_trades.json"
  When I add the spread trade to history
  And I save spread history to "tests/data/trade/spread_roundtrip.json"
  And I create a new SpreadTradeHistory instance
  And I load spread history from "tests/data/trade/spread_roundtrip.json"
  Then the loaded trade should match all original fields
