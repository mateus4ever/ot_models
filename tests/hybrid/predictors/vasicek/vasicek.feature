Feature: Vasicek Mean-Reversion Model
  As a quantitative trading system
  I want to model mean-reverting time series using Ornstein-Uhlenbeck process
  So that I can detect and quantify mean reversion in spread data

  Background:
    Given config files are available in tests/config/predictors/vasicek
    And I have a Vasicek model instance

  # ============================================================================
  # MODEL CALIBRATION - SUCCESSFUL CASES
  # ============================================================================

  Scenario Outline: Successful calibration on mean-reverting data
    Given a synthetic O-U series with n=<n>, kappa=<kappa>, theta=<theta>, sigma=<sigma>, seed=<seed>
    When Vasicek model is calibrated on the series
    Then model should be marked as calibrated
    And kappa should be approximately <kappa> within <kappa_tol>
    And theta should be approximately <theta> within <theta_tol>
    And mean reversion should be statistically significant

    Examples: Synthetic data with known parameters
      | n   | kappa | theta  | sigma  | seed | kappa_tol | theta_tol | description                 |
      | 500 | 0.30  | 0.0020 | 0.0003 | 42   | 0.10      | 0.0005    | Standard mean-reverting     |
      | 300 | 0.70  | 0.0018 | 0.0002 | 43   | 0.15      | 0.0005    | Fast reversion (high kappa) |
      | 800 | 0.03  | 0.0022 | 0.0004 | 44   | 0.02      | 0.0005    | Slow reversion (low kappa)  |
      | 500 | 0.25  | 0.0020 | 0.0008 | 45   | 0.10      | 0.0005    | High volatility             |

  Scenario Outline: Non-stationary series should fail calibration
    Given I generate non-stationary <series_type> series with n=<n>, seed=<seed>
    When Vasicek model is calibrated on the series
    Then mean reversion should NOT be statistically significant

    Examples:
      | series_type  | n   | seed | description        |
      | random_walk  | 500 | 47   | Random walk        |
      | trending     | 500 | 46   | Trending series    |

  # ============================================================================
  # MODEL CALIBRATION - FAILURE CASES
  # ============================================================================

  Scenario Outline: Calibration on non-stationary data
    Given test series of type "<series_type>" with <n_points> points, start <start>, end <end>, noise <noise>, seed <seed>
    When Vasicek model is calibrated on the series
    Then is_mean_reverting() should return False

    Examples: Explosive
      | series_type   | n_points | start | end   | noise    | seed |
      | explosive     | 400      | 0.0   | 1.0   | 0.00001  | 42   |

  # ============================================================================
  # Z-SCORE CALCULATION
  # ============================================================================

  Scenario Outline: Z-score calculation for various values
    Given a synthetic O-U series with n=500, kappa=0.30, theta=0.0020, sigma=0.0003, seed=42
    When Vasicek model is calibrated on the series
    And Z-score is calculated for value <current_value>
    Then Z-score should be approximately <expected_z_score>
    And interpretation should be "<interpretation>"

    Examples:
      | current_value | expected_z_score | interpretation                |
      | 0.0020        | 0.0              | At equilibrium                |
      | 0.0026        | 2.0              | 2 sigma above mean            |
      | 0.0014        | -2.0             | 2 sigma below mean            |
      | 0.0023        | 1.0              | 1 sigma above mean            |
      | 0.0017        | -1.0             | 1 sigma below mean            |
      | 0.0029        | 3.0              | 3 sigma above mean (extreme)  |

  Scenario: Z-score calculation without calibration raises error
    Given Vasicek model is NOT calibrated
    When Z-score calculation is attempted for value 0.0020
    Then ValueError should be raised
    And error message should contain "not calibrated"

  # ============================================================================
  # PREDICTION - NEXT VALUE
  # ============================================================================

Scenario Outline: Predict convergence timeline for position management
  Given a synthetic O-U series with n=500, kappa=0.30, theta=0.0020, sigma=0.0003, seed=42
  When Vasicek model is calibrated on the series
  And next value is predicted for current=<current_value> after <hours> hours with near_mean_threshold=<threshold> and convergence_pct=<conv_pct>
  Then predicted value should be approximately <predicted_value> within 0.0001
  And direction should be <direction>
  And time to equilibrium should be approximately <time_to_eq> hours within <tol_pct> percent or <min_tol> hours

  Examples: Convergence from below
    | current_value | hours | threshold | conv_pct | predicted_value | direction    | time_to_eq | tol_pct | min_tol |
    | 0.0014        | 4     | 0.10      | 0.95     | 0.0018          | towards mean | 10         | 20      | 1.0     |
    | 0.0014        | 8     | 0.10      | 0.95     | 0.0019          | near mean    | 10         | 20      | 1.0     |
    | 0.0014        | 10    | 0.10      | 0.95     | 0.0020          | near mean    | 10         | 20      | 1.0     |

  Examples: Convergence from above
    | current_value | hours | threshold | conv_pct | predicted_value | direction    | time_to_eq | tol_pct | min_tol |
    | 0.0026        | 4     | 0.10      | 0.95     | 0.0022          | towards mean | 10         | 20      | 1.0     |
    | 0.0026        | 10    | 0.10      | 0.95     | 0.0020          | near mean    | 10         | 20      | 1.0     |


  # ============================================================================
  # HALF-LIFE CALCULATION
  # ============================================================================

  Scenario Outline: Half-life for different reversion speeds
    Given Vasicek model is calibrated with kappa = <kappa>
    Then half-life should be approximately <half_life> periods within <tol_pct> percent or <min_tol> minimum
    And reversion speed category should be <category>

    Examples:
      | kappa | half_life | tol_pct | min_tol | category  |
      | 0.70  | 0.99      | 1       | 0.01    | very_fast |
      | 0.30  | 2.31      | 1       | 0.01    | fast      |
      | 0.10  | 6.93      | 1       | 0.01    | moderate  |
      | 0.03  | 23.10     | 1       | 0.01    | slow      |
      | 0.01  | 69.31     | 1       | 0.01    | very_slow |

  Scenario Outline: Invalid kappa produces infinite half-life
    Given Vasicek model is calibrated with kappa = <kappa>
    Then half-life should be infinity
    And is_mean_reverting() should return False

    Examples:
      | kappa | reason           |
      | -0.10 | Negative kappa   |
      | 0.0   | Zero kappa       |

  # ============================================================================
  # MEAN REVERSION VALIDATION
  # ============================================================================

Scenario Outline: Mean reversion rejection edge cases
  Given Vasicek model is calibrated with kappa = <kappa>
  Then is_mean_reverting() should return <expected>

  Examples:
    | kappa  | expected | description                    |
    | 0.005  | False    | kappa below min_kappa (0.01)   |
    | 0.002  | False    | half_life exceeds max (200)    |
    | 0.10   | True     | valid moderate reversion       |


  # ============================================================================
  # TRADING THRESHOLD CONVERSION
  # ============================================================================

  Scenario Outline: Convert Z-score threshold to absolute spread value
    Given Vasicek model is calibrated with theta = 0.0020, sigma = 0.0003 and kappa = 0.30
    When get_trading_threshold() is called with z_threshold <z_threshold>
    Then absolute threshold should be <absolute_value>

    Examples:
      | z_threshold | absolute_value | usage           |
      | 2.0         | 0.0026         | Entry threshold |
      | 0.5         | 0.00215        | Exit threshold  |
      | 3.0         | 0.0029         | Stop loss       |
      | 1.5         | 0.00245        | Partial exit    |
