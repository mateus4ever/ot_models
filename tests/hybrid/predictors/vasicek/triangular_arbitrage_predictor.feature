@triangular_arbitrage
Feature: Triangular Arbitrage Predictor
  As a trading system
  I want to generate spread-based trading signals using Vasicek mean reversion
  So that I can trade triangular arbitrage opportunities

  Background:
    Given config files are available in tests/config/predictors/vasicek
    And I have a TriangularArbitrage model instance

  # ============================================================================
  # PREDICTOR INITIALIZATION
  # ============================================================================

  Scenario: Predictor initializes with correct configuration
    Then predictor should have target_market "EUR/USD"
    And predictor should have leg1_market "EUR/GBP"
    And predictor should have leg2_market "GBP/USD"
    And predictor should have entry_threshold 2.0
    And predictor should have exit_threshold 0.5
    And predictor should have pip_multiplier 10000.0
    And predictor should have lookback_window 200
    And predictor should be marked as not calibrated

  Scenario: Predictor provides required markets list
    When get_required_markets is called
    Then result should contain "EUR/USD"
    And result should contain "EUR/GBP"
    And result should contain "GBP/USD"
    And result should have exactly 3 markets

  # ============================================================================
  # SPREAD CALCULATION
  # ============================================================================

  Scenario Outline: Calculate synthetic price from legs
    When synthetic price is calculated for leg1=<leg1> and leg2=<leg2>
    Then synthetic price should be approximately <synthetic> within 0.0001

    Examples:
      | leg1   | leg2   | synthetic | description     |
      | 0.8500 | 1.2941 | 1.1000    | typical rates   |
      | 0.8600 | 1.2800 | 1.1008    | different rates |
      | 0.8400 | 1.3000 | 1.0920    | other rates     |

  Scenario Outline: Calculate spread between actual and synthetic
    When spread is calculated for target=<target>, leg1=<leg1>, leg2=<leg2>
    Then spread should be approximately <spread> within 0.0001

    Examples:
      | target | leg1   | leg2   | spread   | description      |
      | 1.1000 | 0.8500 | 1.2941 | 0.0000   | at equilibrium   |
      | 1.0994 | 0.8500 | 1.2941 | -0.0006  | below synthetic  |
      | 1.1010 | 0.8500 | 1.2941 | 0.0010   | above synthetic  |

  # ============================================================================
  # SIGNAL GENERATION - ENTRY
  # ============================================================================

  Scenario Outline: Generate entry signals based on Z-score
    Given predictor is calibrated with theta=<theta>, sigma=<sigma>, kappa=<kappa>
    And predictor has no open position
    When signal is generated for z_score=<z_score>
    Then signal should be "<signal>"
    And confidence should be approximately <confidence> within 0.05

    Examples: Long spread entries (spread underpriced)
      | theta  | sigma  | kappa | z_score | signal       | confidence | description           |
      | 0.0020 | 0.0003 | 0.30  | -2.0    | LONG_SPREAD  | 0.67       | at -2σ threshold      |
      | 0.0020 | 0.0003 | 0.30  | -2.5    | LONG_SPREAD  | 0.83       | below -2σ             |
      | 0.0020 | 0.0003 | 0.30  | -3.0    | LONG_SPREAD  | 1.00       | at -3σ, max conf      |
      | 0.0020 | 0.0003 | 0.30  | -4.0    | LONG_SPREAD  | 1.00       | beyond -3σ, capped    |

    Examples: Short spread entries (spread overpriced)
      | theta  | sigma  | kappa | z_score | signal        | confidence | description           |
      | 0.0020 | 0.0003 | 0.30  | 2.0     | SHORT_SPREAD  | 0.67       | at +2σ threshold      |
      | 0.0020 | 0.0003 | 0.30  | 2.5     | SHORT_SPREAD  | 0.83       | above +2σ             |
      | 0.0020 | 0.0003 | 0.30  | 3.0     | SHORT_SPREAD  | 1.00       | at +3σ, max conf      |

    Examples: Hold signals (inside thresholds)
      | theta  | sigma  | kappa | z_score | signal | confidence | description           |
      | 0.0020 | 0.0003 | 0.30  | -1.9    | HOLD   | 0.0        | just inside threshold |
      | 0.0020 | 0.0003 | 0.30  | -1.0    | HOLD   | 0.0        | halfway to threshold  |
      | 0.0020 | 0.0003 | 0.30  | 0.0     | HOLD   | 0.0        | at equilibrium        |
      | 0.0020 | 0.0003 | 0.30  | 1.0     | HOLD   | 0.0        | halfway to threshold  |
      | 0.0020 | 0.0003 | 0.30  | 1.9     | HOLD   | 0.0        | just inside threshold |

  # ============================================================================
  # SIGNAL GENERATION - EXIT
  # ============================================================================

  Scenario Outline: Generate exit signals for open positions
    Given predictor is calibrated with theta=<theta>, sigma=<sigma>, kappa=<kappa>
    And predictor has open position "<position>" with entry_z_score=<entry_z>
    When signal is generated for z_score=<z_score>
    Then signal should be "<signal>"

    Examples: Long spread exits
      | theta  | sigma  | kappa | position    | entry_z | z_score | signal | description              |
      | 0.0020 | 0.0003 | 0.30  | LONG_SPREAD | -2.5    | -2.0    | HOLD   | still far from mean      |
      | 0.0020 | 0.0003 | 0.30  | LONG_SPREAD | -2.5    | -1.0    | HOLD   | approaching mean         |
      | 0.0020 | 0.0003 | 0.30  | LONG_SPREAD | -2.5    | -0.5    | CLOSE  | at exit threshold        |
      | 0.0020 | 0.0003 | 0.30  | LONG_SPREAD | -2.5    | 0.0     | CLOSE  | at equilibrium           |
      | 0.0020 | 0.0003 | 0.30  | LONG_SPREAD | -2.5    | 0.5     | CLOSE  | past equilibrium         |

    Examples: Short spread exits
      | theta  | sigma  | kappa | position     | entry_z | z_score | signal | description              |
      | 0.0020 | 0.0003 | 0.30  | SHORT_SPREAD | 2.5     | 2.0     | HOLD   | still far from mean      |
      | 0.0020 | 0.0003 | 0.30  | SHORT_SPREAD | 2.5     | 1.0     | HOLD   | approaching mean         |
      | 0.0020 | 0.0003 | 0.30  | SHORT_SPREAD | 2.5     | 0.5     | CLOSE  | at exit threshold        |
      | 0.0020 | 0.0003 | 0.30  | SHORT_SPREAD | 2.5     | 0.0     | CLOSE  | at equilibrium           |
      | 0.0020 | 0.0003 | 0.30  | SHORT_SPREAD | 2.5     | -0.5    | CLOSE  | past equilibrium         |

  # ============================================================================
  # STATE MANAGEMENT
  # ============================================================================

  Scenario Outline: State updates correctly on signals
    Given predictor is calibrated with theta=0.0020, sigma=0.0003, kappa=0.30
    And predictor has position state "<initial_position>"
    When signal is generated for z_score=<z_score>
    Then predictor state position should be "<final_position>"

    Examples:
      | initial_position | z_score | final_position | description              |
      | None             | -2.5    | LONG_SPREAD    | entry updates state      |
      | None             | 2.5     | SHORT_SPREAD   | entry updates state      |
      | None             | 0.0     | None           | hold keeps state         |
      | LONG_SPREAD      | 0.0     | None           | close resets state       |
      | SHORT_SPREAD     | 0.0     | None           | close resets state       |
      | LONG_SPREAD      | -1.5    | LONG_SPREAD    | hold keeps state         |

  Scenario: Reset clears all position state
    Given predictor is calibrated with theta=0.0020, sigma=0.0003, kappa=0.30
    And predictor has open position "LONG_SPREAD" with entry_z_score=-2.5
    When predictor is reset
    Then predictor state position should be None
    And predictor state entry_z_score should be None
    And predictor state entry_spread_pips should be None

  # ============================================================================
  # UNCALIBRATED BEHAVIOR
  # ============================================================================

  Scenario: Uncalibrated predictor returns neutral prediction
    Given predictor is NOT calibrated
    When predictor generates prediction
    Then signal should be "HOLD"
    And confidence should be approximately 0.0 within 0.01

  # ============================================================================
  # COMPLETE TRADING CYCLE
  # ============================================================================

  Scenario: Multi-prediction sequence with state transitions
    Given predictor is calibrated with theta=0.0020, sigma=0.0003, kappa=0.30
    And predictor has no open position
    When predictions are generated for z_score sequence:
      | step | z_score | expected_signal | expected_position |
      | 1    | 0.5     | HOLD            | None              |
      | 2    | 1.5     | HOLD            | None              |
      | 3    | 2.5     | SHORT_SPREAD    | SHORT_SPREAD      |
      | 4    | 2.0     | HOLD            | SHORT_SPREAD      |
      | 5    | 1.0     | HOLD            | SHORT_SPREAD      |
      | 6    | 0.3     | CLOSE           | None              |
      | 7    | -0.5    | HOLD            | None              |
      | 8    | -2.3    | LONG_SPREAD     | LONG_SPREAD       |
      | 9    | -1.5    | HOLD            | LONG_SPREAD       |
      | 10   | -0.4    | CLOSE           | None              |
    Then all signals should match expected
    And all position states should match expected