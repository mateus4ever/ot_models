Feature: SMA Crossover Signal Generator
  As a trading system
  I want to generate signals based on moving average crossovers
  So that I can identify trend direction changes

  Background:
    Given config files are available in tests/config

  # ============================================================================
  # INITIALIZATION AND TRAINING
  # ============================================================================

  #this should be merged with Initializate SMA
Scenario Outline: Train signal with varying amounts of historical data
  Given SMA signal with fast and slow period <fast_period>, <slow_period> is initialized
  And market data is loaded from tests/data/small/DAT_ASCII_EURUSD_M1_2021_smoke.csv
  And historical market data with <data_points> periods
  When signal is trained
  Then signal should be <expected_readiness>
#  And historical buffer should contain <expected_buffer_size> records
  Examples:5
    | fast_period | slow_period | data_points | expected_readiness | expected_buffer_size |
    | 10          | 30          | 100         | ready              | 100                  |
    | 5           | 20          | 50          | ready              | 50                   |
    | 10          | 30          | 20          | not ready          | 20                   |
    | 10          | 30          | 150         | not ready          | 100                  |

  # ============================================================================
  # SIGNAL GENERATION - GOLDEN CROSS (BUY)
  # ============================================================================
  Scenario Outline: Golden cross generates BUY signal
    Given SMA signal with fast and slow period <fast_period>, <slow_period> is initialized
    And the recent historical price data is updated:
    """
    20210103 170000;1.22290;1.22305;1.22290;1.22300;1000
    20210103 170100;1.22300;1.22315;1.22300;1.22310;1100
    20210103 170200;1.22310;1.22325;1.22310;1.22320;1200
    20210103 170300;1.22320;1.22335;1.22320;1.22330;1300
    20210103 170400;1.22330;1.22345;1.22330;1.22340;1400
    """
    # Note: current_price represents a single tick price
    # Signal currently requires full OHLCV bars, so test creates synthetic bar
    # where open=high=low=close=current_price (zero-range bar)
    # TODO: Consider tick-based signal interface for live trading
    When the current price <current_price> is processed
    Then signal should be BUY

    Examples:
      | fast_period | slow_period | current_price |
      | 10          | 30          | 1.22450       |
      | 5           | 20          | 1.22480       |

  # ============================================================================
  # SIGNAL GENERATION - DEATH CROSS (SELL)
  # ============================================================================

  Scenario Outline: Death cross generates SELL signal
    Given SMA signal with fast and slow period <fast_period>, <slow_period> is initialized
    And the recent historical price data is updated:
    """
    20210103 170000;1.22400;1.22400;1.22380;1.22390;1000
    20210103 170100;1.22380;1.22390;1.22360;1.22370;1100
    20210103 170200;1.22360;1.22370;1.22340;1.22350;1200
    20210103 170300;1.22340;1.22350;1.22320;1.22330;1300
    20210103 170400;1.22320;1.22330;1.22300;1.22310;1400
    """
    When the current price <current_price> is processed
    Then signal should be SELL

    Examples:
      | fast_period | slow_period | current_price |
      | 10          | 30          | 1.22250       |
      | 5           | 20          | 1.22200       |

  # ============================================================================
  # CROSSOVER CONFIRMATION
  # ============================================================================

  Scenario Outline: Crossover confirmation periods
    Given SMA signal with fast and slow period 10, 30 is initialized
    And crossover confirmation is <confirmation_periods>
    And the recent historical price data is updated:
    """
    20210103 170000;1.22290;1.22305;1.22290;1.22300;1000
    20210103 170100;1.22300;1.22315;1.22300;1.22310;1100
    20210103 170200;1.22310;1.22325;1.22310;1.22320;1200
    20210103 170300;1.22320;1.22335;1.22320;1.22330;1300
    20210103 170400;1.22330;1.22345;1.22330;1.22340;1400
    """
    When the current price <price1> is processed
    Then signal should be <expected_signal1>
    When the current price <price2> is processed
    Then signal should be <expected_signal2>
    Examples:
      | confirmation_periods | price1  | price2  | expected_signal1 | expected_signal2 |
      | 1                    | 1.22450 | 1.22480 | BUY              | BUY              |
      | 2                    | 1.22450 | 1.22480 | HOLD             | BUY              |
