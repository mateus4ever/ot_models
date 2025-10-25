Feature: Kelly Criterion Position Sizer
  As a trading system
  I want to validate Kelly Criterion position sizing calculations
  So that position sizes are mathematically correct and financially sound

  Background:
    Given config files are available in tests/config/kelly_criterion_sizer

  @kelly_criterion @calculation @core
  Scenario Outline: Kelly formula mathematical validation
    Given I have a Kelly Criterion sizer with kelly statistics <win_rate>, <avg_win>, <avg_loss>
    When I calculate the raw Kelly percentage
    Then the Kelly percentage should be <expected_kelly_pct>
    And the calculation should use the correct Kelly formula

    Examples:
      | win_rate | avg_win | avg_loss | expected_kelly_pct |
      | 0.60     | 1.5     | 1.0      | 0.33333            |
      | 0.55     | 1.2     | 1.0      | 0.17500            |
      | 0.40     | 2.0     | 1.0      | 0.10               |
      | 0.50     | 1.0     | 1.0      | 0.00               |

  @kelly_criterion @position_sizing @core
  Scenario Outline: Position size calculation with stop distance
    Given With the configuration I have a Kelly Criterion sizer
    And the portfolio has <portfolio_equity> total equity
    And the final Kelly percentage is <kelly_percentage>
    When I calculate position size for signal at <entry_price> with stop distance <stop_distance>
    Then the position size should be <expected_shares> shares
    And the risk budget should equal Kelly percentage times portfolio equity

    Examples:
      | portfolio_equity | kelly_percentage | entry_price | stop_distance | expected_shares |
      | 100000          | 0.0375          | 50.0        | 2.0           | 1875            |
      | 100000          | 0.20            | 100.0       | 5.0           | 4000            |
      | 200000          | 0.125           | 25.0        | 1.0           | 25000           |
      | 100000          | 0.00            | 50.0        | 2.0           | 0               |


  @kelly_criterion @trade_history_integration @core
  Scenario Outline: Hybrid statistics with lookback periods
    Given With the configuration I have a Kelly Criterion sizer
    And I set kelly_lookback to <lookback_period>
    And I load all position outcomes from "tests/data/trade/base_trade.json"
    When I get current statistics
    Then the statistics source should be <expected_source>
    And the outcome count used should be <expected_outcomes_used>

    Examples:
      | lookback_period | expected_source | expected_outcomes_used |
      | 50              | calculated      | 49                     |
      | 30              | calculated      | 30                     |
      | 10              | calculated      | 10                     |

@kelly_criterion @trade_history_integration @bayesian
Scenario Outline: Bayesian weighted transition from bootstrap to calculated statistics
  Given With the configuration I have a Kelly Criterion sizer
  And bootstrap statistics are kelly_win_rate 0.55, kelly_avg_win 120.0, kelly_avg_loss 100.0
  And kelly_min_trades_threshold is 30
  When I add <trade_count> trade outcomes with actual win_rate <actual_win_rate>, win_pnl <win_pnl>, loss_pnl <loss_pnl>, fees <fees>
  And I get current statistics
  Then the win_rate should be approximately <expected_win_rate>
  And the avg_win should be approximately <expected_avg_win>
  And the avg_loss should be approximately <expected_avg_loss>
Examples:
  | trade_count | actual_win_rate | win_pnl | loss_pnl | fees | expected_win_rate | expected_avg_win | expected_avg_loss | description                    |
  | 10          | 0.70            | 145.0   | -105.0   | 5.0  | 0.5875            | 126.25          | 101.25           | mostly_bootstrap_75pct_weight  |
  | 30          | 0.70            | 145.0   | -105.0   | 5.0  | 0.6250            | 132.50          | 102.50           | equal_weight_50pct_each        |
  | 45          | 0.70            | 145.0   | -105.0   | 5.0  | 0.6333            | 135.00          | 103.00           | mostly_calculated_60pct_weight |
  | 100         | 0.70            | 145.0   | -105.0   | 5.0  | 0.45625           | 135.625         | 103.125          | mostly_calculated_77pct_weight |