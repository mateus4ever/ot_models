Feature: SMA Crossover Signal Generator
  As a trading system
  I want to generate signals based on moving average crossovers
  So that I can identify trend direction changes

  Background:
    Given config files are available in tests/config/signals

  # ============================================================================
  # INITIALIZATION AND TRAINING
  # ============================================================================

  #this should be merged with Initializate SMA
Scenario Outline: Train signal with varying amounts of historical data
  Given SMA signal with fast and slow period <fast_period>, <slow_period> is initialized
  And market data is loaded from tests/data/small/DAT_ASCII_EURUSD_M1_2021_smoke.csv
  And historical market data with <data_points> periods
  When signal is trained
  Then signal has readiness <expected_readiness>
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
Scenario: Golden cross with sharp reversal
    Given SMA signal with fast and slow period 5, 10 is initialized
    And the recent historical price data is updated:
    """
    20210103 170000;1.2400;1.2400;1.2400;1.2400;1000
    20210103 170100;1.2380;1.2380;1.2380;1.2380;1000
    20210103 170200;1.2360;1.2360;1.2360;1.2360;1000
    20210103 170300;1.2340;1.2340;1.2340;1.2340;1000
    20210103 170400;1.2320;1.2320;1.2320;1.2320;1000
    20210103 170500;1.2310;1.2310;1.2310;1.2310;1000
    20210103 170600;1.2305;1.2305;1.2305;1.2305;1000
    20210103 170700;1.2315;1.2315;1.2315;1.2315;1000
    20210103 170800;1.2330;1.2330;1.2330;1.2330;1000
    20210103 170900;1.2350;1.2350;1.2350;1.2350;1000
    20210103 171000;1.2370;1.2370;1.2370;1.2370;1000
    """
    When the current price 1.2390 is processed
    Then signal direction is BULLISH

  Scenario: Golden cross with gradual reversal
    Given SMA signal with fast and slow period 5, 10 is initialized
    And the recent historical price data is updated:
    """
    20210103 170000;1.2350;1.2350;1.2350;1.2350;1000
    20210103 170100;1.2345;1.2345;1.2345;1.2345;1000
    20210103 170200;1.2340;1.2340;1.2340;1.2340;1000
    20210103 170300;1.2335;1.2335;1.2335;1.2335;1000
    20210103 170400;1.2330;1.2330;1.2330;1.2330;1000
    20210103 170500;1.2328;1.2328;1.2328;1.2328;1000
    20210103 170600;1.2327;1.2327;1.2327;1.2327;1000
    20210103 170700;1.2329;1.2329;1.2329;1.2329;1000
    20210103 170800;1.2333;1.2333;1.2333;1.2333;1000
    20210103 170900;1.2338;1.2338;1.2338;1.2338;1000
    20210103 171000;1.2344;1.2344;1.2344;1.2344;1000
    """
    When the current price 1.2351 is processed
    Then signal direction is BULLISH
  # ============================================================================
  # SIGNAL GENERATION - DEATH CROSS (SELL)
  # ============================================================================

# ============================================================================
# SIGNAL GENERATION - DEATH CROSS (BEARISH)
# ============================================================================

  Scenario: Death cross with sharp reversal
    Given SMA signal with fast and slow period 5, 10 is initialized
    And the recent historical price data is updated:
    """
    20210103 170000;1.2200;1.2200;1.2200;1.2200;1000
    20210103 170100;1.2220;1.2220;1.2220;1.2220;1000
    20210103 170200;1.2240;1.2240;1.2240;1.2240;1000
    20210103 170300;1.2260;1.2260;1.2260;1.2260;1000
    20210103 170400;1.2280;1.2280;1.2280;1.2280;1000
    20210103 170500;1.2290;1.2290;1.2290;1.2290;1000
    20210103 170600;1.2295;1.2295;1.2295;1.2295;1000
    20210103 170700;1.2285;1.2285;1.2285;1.2285;1000
    20210103 170800;1.2270;1.2270;1.2270;1.2270;1000
    20210103 170900;1.2250;1.2250;1.2250;1.2250;1000
    20210103 171000;1.2230;1.2230;1.2230;1.2230;1000
    """
    When the current price 1.2210 is processed
    Then signal direction is BEARISH

  # ============================================================================
  # CROSSOVER CONFIRMATION
  # ============================================================================

  Scenario: Crossover with confirmation=1 detects immediately
    Given SMA signal with fast and slow period 5, 10 is initialized
    And crossover confirmation is 1
    And the recent historical price data is updated:
    """
    20210103 165900;1.2380;1.2380;1.2380;1.2380;1000
    20210103 170000;1.2370;1.2370;1.2370;1.2370;1000
    20210103 170100;1.2360;1.2360;1.2360;1.2360;1000
    20210103 170200;1.2350;1.2350;1.2350;1.2350;1000
    20210103 170300;1.2340;1.2340;1.2340;1.2340;1000
    20210103 170400;1.2335;1.2335;1.2335;1.2335;1000
    20210103 170500;1.2330;1.2330;1.2330;1.2330;1000
    20210103 170600;1.2335;1.2335;1.2335;1.2335;1000
    20210103 170700;1.2345;1.2345;1.2345;1.2345;1000
    20210103 170800;1.2360;1.2360;1.2360;1.2360;1000
    20210103 170900;1.2375;1.2375;1.2375;1.2375;1000
    """
    When the current price 1.2380 is processed
    Then signal direction is BULLISH