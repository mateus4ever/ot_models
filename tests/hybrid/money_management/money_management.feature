Feature: MoneyManager Core Service Functionality
  As a trading system
  I want to verify MoneyManager's basic position sizing and risk management
  So that the service loads configuration and calculates position sizes correctly

Background:
  Given money_management.json is available in tests/config and loaded

  @money_management @initialization @core
  Scenario: MoneyManager service initialization
    Given I have money management configuration loaded
    When I create a MoneyManager instance
    Then the MoneyManager should initialize successfully
    And the position sizing strategy should be loaded
    And the risk management strategy should be loaded
    And the portfolio should be initialized with configured capital

  @money_management @position_sizing @core
  Scenario Outline: Basic position size calculation
    Given I have a MoneyManager with fixed fractional sizing
    And the portfolio has <portfolio_equity> equity
    When I calculate position size for a <direction> signal at <entry_price> with market data length <data_length> and price multipliers <high_mult> and <low_mult>
    Then a position size should be calculated
    And the position size should be greater than zero
    And the position size should respect portfolio constraints

    Examples:
      | portfolio_equity | direction | entry_price | data_length | high_mult | low_mult |
      | 100000          | long      | 50.00      | 20          | 1.01      | 0.99     |
      | 150000          | short     | 75.00      | 25          | 1.02      | 0.98     |

  @money_management @stop_loss @core
  Scenario Outline: Stop loss calculation delegation
    Given I have a MoneyManager initialized
    And I have a trading signal for <symbol> <direction> at <entry_price> with strength <signal_strength>
    And I have market data with <volatility_percent> volatility and <data_periods> periods
    When I request stop loss calculation for the signal
    Then return a valid stop loss price
    And the stop loss should be <stop_comparison> the entry price to limit losses

    Examples:
      | symbol | direction | entry_price | signal_strength | volatility_percent | data_periods | stop_comparison |
      | EURUSD | long      | 1.1000      | 1.0            | 0.005             | 15           | below           |
      | GBPUSD | short     | 1.2500      | 0.8            | 0.03              | 20           | above           |
      | USDJPY | long      | 110.00      | 0.6            | 0.015             | 25           | below           |

  @money_management @portfolio_tracking @core
  Scenario Outline: Basic portfolio position tracking
    Given I have a MoneyManager initialized
    When I update position for <symbol> with <size> shares at <price> going <direction>
    Then the position should be tracked correctly
    And after position change the portfolio position is updated
    And the position should appear in current positions

    Examples:
      | symbol | size | price  | direction |
      | EURUSD | 1000 | 1.1000 | long     |
      | GBPUSD | 500  | 1.2500 | short    |

@money_management @market_updates @core
Scenario Outline: Market price updates affect portfolio valuation
  Given I have a MoneyManager with existing positions
  And the portfolio has <initial_equity> total equity
  And I have a <direction> position of <position_size> shares in <symbol> at <entry_price>
  When I update market prices with <symbol> at <current_price>
  Then the portfolio equity should <equity_change> from <initial_equity>
  And the position unrealized PnL should be <pnl_direction>
  And the portfolio summary should reflect updated values

  Examples:
    | initial_equity | symbol | direction | position_size | entry_price | current_price | equity_change | pnl_direction |
    | 100000        | EURUSD | long      | 1000         | 1.1000      | 1.1050        | increase      | positive      |
    | 150000        | GBPUSD | long      | 500          | 1.2500      | 1.2450        | decrease      | negative      |
    | 100000        | EURUSD | short     | 1500         | 1.1000      | 1.0950        | increase      | positive      |
    | 200000        | GBPUSD | short     | 800          | 1.2500      | 1.2550        | decrease      | negative      |

@money_management @portfolio_summary @core
Scenario Outline: Portfolio summary provides complete metrics
  Given I have a MoneyManager initialized
  And I create <position_count> test positions
  And each position has <position_size> shares at <entry_price>
  And the portfolio has <total_equity> total equity
  And the portfolio has <available_cash> available cash
  And the portfolio has <daily_pnl> daily PnL
  When I request the portfolio summary
  Then the summary should show total equity of <total_equity>
  And the summary should show available cash of <available_cash>
  And the summary should show positions count of <position_count>
  And the summary should show daily PnL of <daily_pnl>
  And the summary should show position sizing strategy name

  Examples:
    | position_count | position_size | entry_price | total_equity | available_cash | daily_pnl |
    | 2              | 1000         | 1.0         | 105000       | 25000          | 2500      |
    | 3              | 500          | 1.5         | 98000        | 15000          | -1500     |

  @money_management @risk_management @core
  Scenario Outline: Risk constraint validation with specific conditions
    Given I have a MoneyManager with risk limits
    And the portfolio has <daily_pnl> daily PnL
    And the portfolio has <max_drawdown> maximum drawdown
    When I check if risk should be reduced
    Then the risk evaluation should return <expected_result>

    Examples:
      | daily_pnl | max_drawdown | expected_result |
      | -2000     | 0.03         | false          |
      | -6000     | 0.15         | true           |
      | -3000     | 0.25         | true           |
      | 1000      | 0.10         | false          |

  @money_management @position_sizer_selection @core
  Scenario Outline: Configuration-driven position sizer selection
    Given I have money management configuration with <sizer_type> position sizer
    When I create a MoneyManager instance
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

  @money_management @error_handling @core
  Scenario Outline: Invalid component configuration error
    Given I have money management configuration with unknown <component_type> "<invalid_name>"
    When I try to create a MoneyManager instance
    Then a <component_type> configuration error should be raised
    And the error message should list available <component_type> options

    Examples:
      | component_type | invalid_name     |
      | position_sizer | invalid_sizer    |
      | risk_manager   | invalid_manager  |

  @money_management @integration @core
Scenario Outline: Position sizing with risk reduction
  Given I have a MoneyManager initialized
  And the portfolio has <drawdown_percent> drawdown exceeding risk limits
  When I calculate position size for <symbol> <direction> signal at <entry_price> with strength <signal_strength> and <data_periods> periods
  Then the position size should be reduced from normal calculation
  And the calculation should complete successfully

  Examples:
    | drawdown_percent | symbol | direction | entry_price | signal_strength | data_periods |
    | 0.25            | EURUSD | long      | 1.1000      | 1.0            | 15           |
    | 0.30            | GBPUSD | short     | 1.2500      | 0.8            | 20           |
  @money_management @safety_constraints @core
  Scenario Outline: Safety constraints override strategy calculations
    Given I have a MoneyManager initialized
    And the portfolio has <available_cash> available cash
    When I calculate position size for <symbol> <direction> signal at <entry_price> with desired value <signal_value>
    Then the position size should be limited to <expected_max_shares> shares
    And the position value should not exceed <available_cash>
    And no errors should occur during calculation

    Examples:
      | available_cash | symbol | direction | entry_price | signal_value | expected_max_shares |
      | 50000         | EURUSD | long      | 1.1000      | 100000       | 9090            |
      | 25000         | GBPUSD | short     | 1.2500      | 75000        | 8000            |
      | 10000         | USDJPY | long      | 110.00      | 50000        | 90              |