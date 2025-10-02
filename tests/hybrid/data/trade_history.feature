Feature: TradeHistory Basic Test
  As a trading system
  I want to test TradeHistory initialization
  So that I can verify the basic setup works

Background:
  Given data_management.json is available in tests/config and loaded
  And  I have a TradeHistory instance with base_currency "USD"
  And I have loaded trade data from "tests/data/trade/base_trade.json"

Scenario: Load mock trade data from JSON file
  Then 50 trades should be loaded successfully

@trade_history @position_access @core
Scenario: Access positions from loaded trades
  When I access positions from the loaded trades
  Then each trade should have at least one position
  And each position should have name_of_position "MOCK"
  Then each position should have entry_value, currency, entry_fees, and exit_fees
  And closed positions should have exit_value and exit_timestamp

@trade_history @position_outcomes @core
Scenario Outline: Calculate position outcomes with different P&L scenarios
  Given I have a position with entry_value <entry>, exit_value <exit>, amount <amount>, entry_fees <entry_fees>, and exit_fees <exit_fees>
  When I calculate the position outcome
  Then the outcome should be <expected_outcome>
  And the net P&L should be <expected_pnl>
  And fees should be properly subtracted from gross P&L

  Examples:
    | entry  | exit   | amount | entry_fees | exit_fees | expected_outcome | expected_pnl |
    | 100.0  | 110.0  | 10     | 2.5        | 2.5       | win             | 95.0         |
    | 100.0  | 95.0   | 10     | 2.5        | 2.5       | loss            | -55.0        |
    | 100.0  | 105.0  | 10     | 25.0       | 25.0      | break_even      | 0.0          |
    | 100.0  | 105.0  | 10     | 0.0        | 0.0       | win             | 50.0         |

@trade_history @statistics @core
Scenario: Calculate trade statistics from loaded positions
  When I calculate trade statistics from all positions
  Then the statistics should contain 49 total_positions
  And the statistics should contain 25 winning_positions
  And the statistics should contain 24 losing_positions
  And the statistics should contain 0 break_even_positions
  And the statistics should have total_fees of 342.51
  And the statistics should have total_pnl of 7.04
  And the statistics should include position outcomes list

  @trade_history @lookback @core
  Scenario Outline: Trade statistics with lookback windows
    When I calculate trade statistics with lookback <lookback_periods>
    Then only the most recent <expected_positions> positions should be used
    And the statistics should reflect the limited dataset
    And older positions should be excluded from the calculation
    Examples:
      | lookback_periods | expected_positions |
      | 30              | 30                 |
      | 10              | 10                 |
      | 0               | 49                 |

  @trade_history @open_positions @core
  Scenario: Handle open and closed positions correctly
    When I identify open and closed positions
    Then trade "trade-050" should have an open position
    And the open position should have exit_value as null
    And the open position should have exit_timestamp as null

  @trade_history @add_trade @core
  Scenario Outline: Add new trade to history
    Given I have a new trade with timestamp "<timestamp>"
    And the trade has position with name_of_position "<name_of_position>", type "<type>", entry_value <entry_value>, amount <amount>, and entry_fees <entry_fees>
    When I add the trade to history
    Then the trade count should increase by 1
    And the new trade should be stored in chronological order
    And the trade should be accessible by timestamp
    Examples:
      | timestamp            | name_of_position | type  | entry_value | amount | entry_fees |
      | 2024-04-25T10:00:00Z | AAPL-LONG-001   | stock | 120.0       | 8      | 3.2        |
      | 2024-04-26T14:30:00Z | MSFT-LONG-002   | stock | 95.5        | 15     | 4.1        |
      | 2024-04-27T09:15:00Z | TSLA-LONG-003   | stock | 200.75      | 5      | 2.8        |
      | 2024-04-28T16:45:00Z | GOOGL-LONG-004  | stock | 45.25       | 25     | 5.3        |
      | 2024-04-29T11:20:00Z | NVDA-LONG-005   | stock | 150.0       | 12     | 3.7        |

  @trade_history @json_persistence @core
  Scenario: Save and reload trade history
    Given I have loaded trade history data for persistence testing
    When I save the trade history to "tests/data/trade/test_output.json"
    And I create a new TradeHistory instance
    And I load trade data from "tests/data/trade/test_output.json"
    Then all original trade data should be preserved
    And all position data should be preserved
    And timestamp ordering should be maintained
    And trade statistics should be identical to original

@trade_history @edge_cases @core
Scenario Outline: Handle edge cases in statistics calculation
  Given I have a TradeHistory with trade pattern from "<file_path>"
  When I calculate trade statistics
  Then the calculation should handle the edge case appropriately
  And no mathematical errors should occur
  And the result should show <expected_behavior>

  Examples:
    | file_path                                      | expected_behavior           |
    | tests/data/trade/edge_cases/no_trades.json     | zero_stats_no_errors       |
    | tests/data/trade/edge_cases/all_winning.json   | zero_avg_loss               |
    | tests/data/trade/edge_cases/all_losing.json    | zero_avg_win                |
    | tests/data/trade/edge_cases/all_break_even.json| zero_net_pnl                |


  @trade_history @error_handling @core
  Scenario Outline: Handle errors gracefully
    When I encounter <error_condition> during operation
    Then the system should handle it gracefully
    And appropriate error messages should be logged

  Examples:
    | error_condition                                          |
    | missing_json_file                                        |
    | tests/data/trade/edge_cases/malformed_trade.json         |
    | tests/data/trade/edge_cases/invalid_timestamp_trade.json |