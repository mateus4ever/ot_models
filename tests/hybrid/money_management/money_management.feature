Feature: MoneyManager Core Service Functionality
  As a trading system
  I want to verify MoneyManager's basic position sizing and risk management
  So that the service loads configuration and calculates position sizes correctly

Background:
  Given config files are available in tests/config/money_management
  And a centralized position orchestrator is initialized from configuration
  And position orchestrator has 100000 initial capital
  And a MoneyManager instance is created and configured with position orchestrator

@position_sizing
Scenario Outline: Calculate position size with available capital
  Given position orchestrator has <available_capital> available capital
  When I calculate position size for <direction> signal at <entry_price> with <data_length> bars and price range <high_mult> to <low_mult>
  Then a position size should be calculated
  And the position size should be greater than zero
  And the position size should not exceed available capital

  Examples:
    | available_capital | direction | entry_price | data_length | high_mult | low_mult |
    | 100000           | long      | 50.00       | 20          | 1.01      | 0.99     |
    | 50000            | short     | 75.00       | 25          | 1.02      | 0.98     |

  @stop_loss
  Scenario Outline: Stop loss calculation delegation
    Given I have a trading signal for <symbol> <direction> at <entry_price> with strength <signal_strength>
    And I have market data with <volatility_percent> volatility and <data_periods> periods
    When I request stop loss calculation for the signal
    Then a valid stop loss price should be returned
    And the stop loss should be <stop_comparison> the entry price

    Examples:
      | symbol | direction | entry_price | signal_strength | volatility_percent | data_periods | stop_comparison |
      | EURUSD | long      | 1.1000      | 1.0            | 0.005             | 15           | below           |
      | GBPUSD | short     | 1.2500      | 0.8            | 0.03              | 20           | above           |
      | USDJPY | long      | 110.00      | 0.6            | 0.015             | 25           | below           |

  @risk_management
  Scenario Outline: Risk reduction based on drawdown and daily P&L
    Given I have portfolio metrics with equity <equity>, cash <cash>, daily_pnl <daily_pnl>, drawdown <drawdown>, and peak <peak>
    When I check if risk should be reduced
    Then the risk evaluation should return <expected_result>

    Examples:
      | equity | cash  | daily_pnl | drawdown | peak   | expected_result |
      | 97000 | 50000 | -2000     | 0.03     | 100000 | false           |
      | 85000 | 50000 | -6000     | 0.15     | 100000 | true            |
      | 75000 | 50000 | -3000     | 0.25     | 100000 | true            |
      | 90000 | 50000 | 1000      | 0.10     | 100000 | false           |

 @position_sizer_selection
  Scenario Outline: Configuration-driven position sizer selection
    Given I have money management configuration with <sizer_type> position sizer
    When I create a MoneyManager instance with the updated configuration
    Then the MoneyManager should initialize successfully
    And the position sizer should be <expected_sizer_name>
    And position calculations should work with <sizer_type> algorithm

    Examples:
      | sizer_type       | expected_sizer_name |
      | fixed_fractional | FixedFractional     |

  @money_management @error_handling @core
  Scenario: Configuration error handling
    Given I have incomplete money management configuration
    When I try to create a MoneyManager instance
    Then a configuration error should be raised
    And the MoneyManager should not be created

  @error_handling
  Scenario Outline: Invalid component configuration error
    Given money management config has invalid <component_type> "<invalid_name>"
    When I try to create a MoneyManager instance
    Then for invalid component type <component_type> a configuration error should be raised

    Examples:
      | component_type | invalid_name     |
      | position_sizer | invalid_sizer    |
      | risk_manager   | invalid_manager  |

  @risk_reduction
  Scenario Outline: Position sizing with risk reduction triggered by drawdown
    Given position orchestrator has <available_capital> available and peak equity was <peak_equity>
    When I calculate position size for <direction> signal at <entry_price> with <data_length> bars and price range <high_mult> to <low_mult>
    Then the position size should be reduced by risk reduction factor
    And the calculation should complete successfully

    Examples:
      | available_capital | peak_equity | direction | entry_price | data_length | high_mult | low_mult |
      | 75000             | 100000      | long      | 1.1000      | 20          | 1.01      | 0.99     |
      | 70000             | 100000      | short     | 1.2500      | 20          | 1.01      | 0.99     |

  @safety_constraints
  Scenario Outline: Safety constraints override strategy calculations
    Given position orchestrator has <available_cash> available capital
    When I calculate position size for <direction> signal at <entry_price> with <data_length> bars and price range <high_mult> to <low_mult>
    Then the position size should be limited to <expected_max_shares> shares
    And the position value should not exceed <available_cash>
    And no errors should occur during calculation

    Examples:
      | available_cash | direction | entry_price | data_length | high_mult | low_mult | expected_max_shares |
      | 50000          | long      | 1.1000      | 20          | 1.01      | 0.99     | 9090                |
      | 25000          | short     | 1.2500      | 20          | 1.01      | 0.99     | 8000                |
      | 10000          | long      | 110.00      | 20          | 1.01      | 0.99     | 90                  |