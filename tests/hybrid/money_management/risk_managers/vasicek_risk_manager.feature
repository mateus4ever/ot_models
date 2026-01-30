Feature: Vasicek Risk Manager
  As a trading system
  I want to calculate stop losses using Z-scores
  So that position risk is based on spread mean reversion

  Background:
    Given config files are available in tests/config/money_management/risk_managers/vasicek_risk_manager

  # ============================================================================
  # CORE FUNCTIONALITY - Stop Loss Calculation
  # ============================================================================

  Scenario Outline: Calculate stop loss using Z-scores for different spread positions
    Given Vasicek parameters are set with theta=<theta>, sigma=<sigma>
    And signal is <direction> with entry price <entry_price>, strength <strength> and entry Z-score <entry_z>
    And stop loss Z-score threshold is <stop_z_threshold>, max daily loss is 0.05, max drawdown is 0.20
    And Vasicek risk manager is initialized
    When stop loss is calculated
    Then stop loss should be approximately <expected_stop_loss> within 0.000001

    Examples: LONG_SPREAD positions (entered at negative Z)
      | direction | entry_price | strength| entry_z | theta  | sigma  | stop_z_threshold | expected_stop_loss |
      | LONG      | 0.0014      | 0.75    |-2.5    | 0.0020 | 0.0003 | 3.5              | 0.0002              |
      | LONG      | 0.0012      | 0.75    |-3.0    | 0.0020 | 0.0003 | 3.5              | 0.00005            |
      | LONG      | 0.0010      | 0.75    |-2.0    | 0.0018 | 0.0004 | 3.0              | -0.0002            |

    Examples: SHORT_SPREAD positions (entered at positive Z)
      | direction | entry_price | strength| entry_z | theta  | sigma  | stop_z_threshold | expected_stop_loss |
      | SHORT     | 0.0026      | 0.75    |2.5     | 0.0020 | 0.0003 | 3.5              | 0.0038             |
      | SHORT     | 0.0028      | 0.75    |3.0     | 0.0020 | 0.0003 | 3.5              | 0.00395            |
      | SHORT     | 0.0030      | 0.75    |2.0     | 0.0018 | 0.0004 | 3.0              | 0.0038             |

  # ============================================================================
  # PARAMETER SETTING - Calibrated Values
  # ============================================================================

  Scenario: Set Vasicek parameters from calibration
    Given parameters are set with theta=0.0020, sigma=0.0003, kappa=0.30
    And Vasicek risk manager is initialized
    Then risk manager should have theta=0.0020
    And risk manager should have sigma=0.0003
    And risk manager should have kappa=0.30

  Scenario: Stop loss calculation raises error without parameters
    Given parameters are NOT set
    And Vasicek risk manager is initialized
    And signal is LONG with entry price 0.0014, strength 1.0 and entry Z-score -2.5
    When stop loss is calculated
    Then calculation should raise error about missing parameters

  # ============================================================================
  # RISK REDUCTION TRIGGERS
  # ============================================================================

  Scenario Outline: Risk reduction triggered by portfolio conditions
    Given portfolio with total equity <total_equity>
    And Vasicek risk manager is initialized
    And portfolio daily P&L is <daily_pnl>
    And portfolio peak equity <peak_equity>
    And max daily loss threshold is <max_daily_loss>
    And max drawdown threshold is <max_drawdown>
    When risk reduction check is performed
    Then risk reduction should be <expected_result>

    Examples: Daily loss triggers
      | total_equity | daily_pnl | peak_equity | max_daily_loss | max_drawdown | expected_result |
      | 100000.00    | -6000.00  | 100000.00   | 0.05           | 0.20         | triggered       |
      | 100000.00    | -4000.00  | 100000.00   | 0.05           | 0.20         | not triggered   |

    Examples: Drawdown triggers
      | total_equity | daily_pnl | peak_equity | max_daily_loss | max_drawdown | expected_result |
      | 75000.00     | 0.00      | 100000.00   | 0.05           | 0.20         | triggered       |
      | 85000.00     | 0.00      | 100000.00   | 0.05           | 0.20         | not triggered   |

    Examples: Combined triggers
      | total_equity | daily_pnl | peak_equity | max_daily_loss | max_drawdown | expected_result |
      | 75000.00     | -3000.00  | 100000.00   | 0.05           | 0.20         | triggered       |
      | 95000.00     | -2000.00  | 100000.00   | 0.05           | 0.20         | not triggered   |

  # ============================================================================
  # EDGE CASES
  # ============================================================================

Scenario Outline: Stop loss with extreme Z-scores
  Given config files are available in config
  And Vasicek parameters are set with theta=<theta>, sigma=<sigma>
  And signal is <direction> with entry price <entry_price>, strength 1.0 and entry Z-score <entry_z>
  And stop loss Z-score threshold is <stop_z_threshold>, max daily loss is 0.05, max drawdown is 0.20
  And Vasicek risk manager is initialized
  When stop loss is calculated
  Then stop loss should be approximately <expected_stop_loss> within 0.000001

  Examples: Extreme entry Z-scores
    | direction | entry_price | entry_z | theta  | sigma  | stop_z_threshold | expected_stop_loss |
    | LONG      | 0.0005      | -5.0    | 0.0020 | 0.0003 | 3.5              | -0.00055           |
    | SHORT     | 0.0035      | 5.0     | 0.0020 | 0.0003 | 3.5              | 0.00455            |

  Examples: Very small sigma (low volatility)
    | direction | entry_price | entry_z | theta  | sigma   | stop_z_threshold | expected_stop_loss |
    | LONG      | 0.0019      | -2.0    | 0.0020 | 0.00001 | 3.5              | 0.001945           |
    | SHORT     | 0.0021      | 2.0     | 0.0020 | 0.00001 | 3.5              | 0.002055           |

  Examples: Very large sigma (high volatility)
    | direction | entry_price | entry_z | theta  | sigma  | stop_z_threshold | expected_stop_loss |
    | LONG      | 0.0000      | -2.0    | 0.0020 | 0.0010 | 3.5              | -0.0035            |
    | SHORT     | 0.0040      | 2.0     | 0.0020 | 0.0010 | 3.5              | 0.0075             |

Scenario: Stop loss with zero sigma raises error
  Given config files are available in config
  And Vasicek parameters are set with theta=0.0020, sigma=0.0
  And signal is LONG with entry price 0.0020, strength 1.0 and entry Z-score 0.0
  And stop loss Z-score threshold is 3.5, max daily loss is 0.05, max drawdown is 0.20
  And Vasicek risk manager is initialized
  When stop loss is calculated
  Then calculation should raise error about invalid sigma

  # ============================================================================
  # CONFIGURATION LOADING
  # ============================================================================

  Scenario: Missing configuration raises error
    Given config files are available in config
    And config file is missing vasicek section
    When Vasicek risk manager initialization is attempted
    Then initialization should raise error
    And error message should contain "vasicek"

  # ============================================================================
  # INTEGRATION WITH TRADING SIGNALS
  # ============================================================================

Scenario: Missing entry_z_score in signal raises error
  Given config files are available in config
  And Vasicek parameters are set with theta=0.0020, sigma=0.0003
  And stop loss Z-score threshold is 3.5, max daily loss is 0.05, max drawdown is 0.20
  And Vasicek risk manager is initialized
  And signal without Z-score is LONG with entry price 0.0014, strength 1.0
  When stop loss is calculated
  Then calculation should raise error about missing Z-score

