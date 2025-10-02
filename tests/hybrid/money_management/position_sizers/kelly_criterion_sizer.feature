Feature: Kelly Criterion Position Sizer
  As a trading system
  I want to validate Kelly Criterion position sizing calculations
  So that position sizes are mathematically correct and financially sound

  Background:
    Given money_management_config.json is available in tests/config
    And kelly criterion configuration is available with win_rate 0.55, avg_win 1.2, avg_loss 1.0
    And kelly fraction is 0.25 and max kelly position is 0.2

  @kelly_criterion @calculation @core
  Scenario Outline: Kelly formula mathematical validation
    Given I have a Kelly Criterion sizer with kelly statistics <win_rate>, <avg_win>, <avg_loss>
    When I calculate the raw Kelly percentage
    Then the Kelly percentage should be <expected_kelly_pct>
    And the calculation should use the correct Kelly formula

    Examples:
      | win_rate | avg_win | avg_loss | expected_kelly_pct |
      | 0.60     | 1.5     | 1.0      | 0.35               |
      | 0.55     | 1.2     | 1.0      | 0.375              |
      | 0.40     | 2.0     | 1.0      | 0.10               |
      | 0.50     | 1.0     | 1.0      | 0.00               |

  @kelly_criterion @limits @core
  Scenario Outline: Kelly percentage limits and constraints
    Given I have a Kelly Criterion sizer with configuration
    And the raw Kelly calculation results in <raw_kelly> percentage
    When I apply Kelly limits and constraints
    Then the final Kelly percentage should be <expected_final>
    And the percentage should respect fraction and maximum limits

    Examples:
      | raw_kelly | expected_final | description                    |
      | 0.50      | 0.125         | fraction_0.25_applied          |
      | 0.80      | 0.20          | max_limit_0.20_applied         |
      | -0.10     | 0.00          | negative_kelly_becomes_zero    |
      | 0.15      | 0.0375        | normal_case_with_fraction      |

  @kelly_criterion @position_sizing @core
  Scenario Outline: Position size calculation with stop distance
    Given I have a Kelly Criterion sizer with configuration
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

  @kelly_criterion @edge_cases @core
  Scenario Outline: Edge cases and mathematical safety
    Given I have a Kelly Criterion sizer with kelly statistics <win_rate>, <avg_win>, <avg_loss>
    When I calculate the Kelly percentage
    Then the result should handle the edge case appropriately
    And no mathematical errors should occur
    And the result should be <expected_result>

    Examples:
      | win_rate | avg_win | avg_loss | expected_result | description              |
      | 0.00     | 1.0     | 1.0      | 0.00           | zero_win_rate           |
      | 1.00     | 1.0     | 1.0      | 0.00           | perfect_win_rate_safe   |
      | 0.50     | 0.0     | 1.0      | 0.00           | zero_average_win        |
      | 0.50     | 1.0     | 0.0      | 0.00           | zero_average_loss       |

  @kelly_criterion @configuration @core
  Scenario: Configuration-driven Kelly parameters
    Given I have a Kelly Criterion sizer with configuration
    When I access the Kelly parameters
    Then kelly_win_rate should come from configuration
    And kelly_avg_win should come from configuration
    And kelly_avg_loss should come from configuration
    And kelly_fraction should come from configuration
    And max_kelly_position should come from configuration
    And no hardcoded values should be used

  @kelly_criterion @risk_integration @core
  Scenario: Integration with risk manager calculated stop distance
    Given I have a Kelly Criterion sizer with configuration
    And the portfolio has 100000 total equity
    And a risk manager has calculated stop distance of 3.0
    When I calculate position size for signal at 60.0 with the calculated stop distance
    Then the position should use the stop distance parameter
    And the risk budget should be divided by stop distance to get shares
    And the calculation should not recalculate any risk metrics

  @kelly_criterion @portfolio_constraints @core
  Scenario: Zero position when Kelly indicates no edge
    Given I have a Kelly Criterion sizer with configuration
    And the Kelly statistics indicate no statistical edge
    And the portfolio has 100000 total equity
    When I calculate position size for any signal and stop distance
    Then the position size should be 0 shares
    And no capital should be risked when there is no edge

    @kelly_criterion @trade_history_integration @core
    Scenario Outline: Hybrid statistics with lookback periods
      Given I have a Kelly Criterion sizer with kelly_lookback <lookback_period>
      And I load all position outcomes from "tests/data/trade/base_trade.json"
      When I get current statistics
      Then the statistics source should be <expected_source>
      And the outcome count used should be <expected_outcomes_used>

      Examples:
        | lookback_period | expected_source | expected_outcomes_used |
        | 50              | calculated      | 49                     |
        | 30              | calculated      | 30                     |
        | 0               | calculated      | 49                     |