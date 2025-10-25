Feature: ATR-Based Risk Manager
  As a trading system
  I want to calculate stop losses using ATR
  So that position risk is based on market volatility

  Background:
    Given config files are available in tests/config/atr_based_risk_manager

  # ============================================================================
  # CORE FUNCTIONALITY - Stop Loss Calculation
  # ============================================================================

  Scenario Outline: Calculate stop loss using ATR for different positions
    Given market data has ATR of <atr>
    And signal is <direction> with entry price <entry_price> and strength <strength>
    And ATR multiplier is <multiplier>
    When stop loss is calculated
    Then stop loss should be <expected_stop_loss>

    Examples:
      | direction | entry_price | strength | atr  | multiplier | expected_stop_loss |
      | LONG      | 100.00      | 0.75     | 2.50 | 2.0        | 95.00              |
      | SHORT     | 100.00      | 0.75     | 2.50 | 2.0        | 105.00             |
      | LONG      | 150.00      | 0.80     | 3.00 | 1.5        | 145.50             |

  # ============================================================================
  # ATR CALCULATION - Core Algorithm
  # ============================================================================

Scenario Outline: Calculate ATR with different data conditions
  Given market data is loaded from <data_file>
  And market data with <periods> periods
  And ATR period is <atr_period>
  When ATR is calculated
  Then ATR should be <expected_atr>

Examples:
  | data_file                                                | periods | atr_period | expected_atr |
  | tests/data/small/DAT_ASCII_EURUSD_M1_2021_smoke.csv     | 20      | 14         | 0.000136   |
  | tests/data/small/DAT_ASCII_EURUSD_M1_2021_smoke.csv     | 5       | 14         | 0.000703   |
  | tests/data/small/DAT_ASCII_EURUSD_M1_2021_smoke.csv     | 50      | 14         |  0.003602  |

  # ============================================================================
  # RISK REDUCTION TRIGGERS
  # ============================================================================

Scenario Outline: Risk reduction triggered by portfolio conditions
  Given portfolio with total equity <total_equity>
  And portfolio daily P&L is <daily_pnl>
  And portfolio peak equity <peak_equity>
  And max daily loss threshold is <max_daily_loss>
  And max drawdown threshold is <max_drawdown>
  When risk reduction check is performed
  Then risk reduction should be <expected_result>

  Examples:
  | total_equity | daily_pnl | peak_equity | max_daily_loss | max_drawdown | expected_result |
  | 100000.00    | -3500.00  | 100000.00   | 0.03          | 0.10         | triggered       |
  | 100000.00    | -2500.00  | 100000.00   | 0.03          | 0.10         | not triggered   |
  | 88000.00     | 0.00      | 100000.00   | 0.03          | 0.10         | triggered       |
  | 92000.00     | 0.00      | 100000.00   | 0.03          | 0.10         | not triggered   |
  | 88000.00     | -2000.00  | 100000.00   | 0.03          | 0.10         | triggered       |

  # ============================================================================
  # EDGE CASES
  # ============================================================================

Scenario Outline: Stop loss calculation with edge case ATR values
  Given market data has ATR of <atr>
  And signal is <direction> with entry price <entry_price> and strength <strength>
  And ATR multiplier is <multiplier>
  When stop loss is calculated
  Then stop loss should be <expected_stop_loss>

  Examples:
    | atr   | direction | entry_price | strength | multiplier | expected_stop_loss |
    | 0.0   | LONG      | 100.00      | 0.75     | 2.0        | 100.00             |
    | 0.01  | LONG      | 100.00      | 0.75     | 2.0        | 99.98              |
    | 15.00 | LONG      | 100.00      | 0.75     | 2.0        | 70.00              |

