Feature: TradeHistory Basic Test
  As a trading system
  I want to test TradeHistory initialization
  So that I can verify the basic setup works

Background:
  Given data_management.json is available in tests/config/data and loaded
  And  I have a TradeHistory instance with base_currency "USD"
  And I have loaded trade data from "tests/data/trade/base_trade.json"

Scenario: Load mock trade data from JSON file
  Then 50 trades should be loaded successfully

Scenario: Access trade data from loaded trades
  Then each trade should have required fields
  And each trade should have symbol "MOCK"
  And each trade should have entry_price, exit_price, quantity, and currency
  And closed trades should have exit_price and exit_date

  #TODO: this step must be adapted: Given I have a position with entry_price
Scenario Outline: Calculate position outcomes with different P&L scenarios
  Given I have a position with entry_price <entry>, exit_price <exit>, quantity <quantity>, direction <direction>
  When I calculate the position outcome
  Then the outcome should be <expected_outcome>
  And the net P&L should be <expected_pnl>
  And fees should be properly subtracted from gross P&L

  Examples:
    | entry  | exit   | quantity | direction | expected_outcome | expected_pnl |
    | 100.0  | 110.0  | 10       | LONG      | win             | 96.85        |
    | 100.0  | 95.0   | 10       | LONG      | loss            | -52.925      |
    | 100.0  | 100.0  | 10       | LONG      | loss            | -3.00        |
    | 100.0  | 105.0  | 10       | LONG      | win             | 46.925       |

@trade_history @statistics @core
Scenario: Calculate trade statistics from loaded positions
  When I calculate trade statistics from all positions
  Then the statistics should contain 49 total_positions
  And the statistics should contain 26 winning_positions
  And the statistics should contain 23 losing_positions
  And the statistics should contain 0 break_even_positions
  And the statistics should have total_fees of 140.57
  And the statistics should have total_pnl of 676.68
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

  Scenario: Handle open and closed trades correctly
    When I identify open and closed trades
    Then trade "trade-050" should be open
    And the open trade should have exit_price as null
    And the open trade should have exit_date as null

    Scenario Outline: Add new trade to history
      Given I have a new trade with timestamp "<timestamp>"
      And the trade has position with symbol "<symbol>", type "<type>", entry_price <entry_price>, quantity <quantity>
      When I add the trade to history
      Then the trade count should increase by 1
      And the new trade should be stored in chronological order
      And the trade should be accessible by timestamp
      Examples:
        | timestamp            | symbol        | type  | entry_price | quantity |
        | 2024-04-25T10:00:00Z | AAPL-LONG-001 | stock | 120.0       | 8        |
        | 2024-04-26T14:30:00Z | MSFT-LONG-002 | stock | 95.5        | 15       |
        | 2024-04-27T09:15:00Z | TSLA-LONG-003 | stock | 200.75      | 5        |
        | 2024-04-28T16:45:00Z | GOOGL-LONG-004| stock | 45.25       | 25       |
        | 2024-04-29T11:20:00Z | NVDA-LONG-005 | stock | 150.0       | 12       |

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