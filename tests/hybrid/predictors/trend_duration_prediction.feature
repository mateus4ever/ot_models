@trend_duration_predictor
Feature: Trend Duration Predictor
  ML-based predictor for how long trends will last

Background:
  Given config files are available in tests/config/predictors
  And data source is set to data/big/

# ==============================================================================
# UNIT TESTS - Fast, verify features work
# ==============================================================================

@unit
Scenario: Train and predict trend duration
  Given create a TrendDurationPredictor and DataManager
  When I train the duration predictor with 500 historical elements
  And I predict duration on the next 100 elements
  Then duration predictions should be 0, 1, 2, or 3 only
  And predictions length should be 100

@unit
Scenario: Insufficient data for training
  Given create a TrendDurationPredictor and DataManager
  When I train the duration predictor with 10 historical elements
  Then predictor should not be marked as trained

@unit
Scenario: Momentum features are generated
  Given create a TrendDurationPredictor and DataManager
  When I train the duration predictor with 2000 historical elements
  Then features should include momentum_5
  And features should include momentum_10
  And features should include momentum_20
  And features should include momentum_40

@unit
Scenario: Volatility features are generated
  Given create a TrendDurationPredictor and DataManager
  When I train the duration predictor with 2000 historical elements
  Then features should include volatility_5
  And features should include volatility_20
  And features should include volatility_60

@unit
Scenario: Trend maturity features are generated
  Given create a TrendDurationPredictor and DataManager
  When I train the duration predictor with 2000 historical elements
  Then features should include trend_age
  And features should include trend_age_normalized
  And features should include trend_strength_short
  And features should include trend_strength_medium

@unit
Scenario: Exhaustion features are generated
  Given create a TrendDurationPredictor and DataManager
  When I train the duration predictor with 2000 historical elements
  Then features should include rsi
  And features should include momentum_exhaustion

@unit
Scenario: Reversion pressure features are generated
  Given create a TrendDurationPredictor and DataManager
  When I train the duration predictor with 2000 historical elements
  Then features should include bb_position
  And features should include bb_overextension

# ==============================================================================
# VALIDATION - Slow, comprehensive check
# ==============================================================================

@validation @slow @comparison
Scenario Outline: Compare duration prediction configurations <config_name>
  Given data source is set to data/eurusd/
  And create a TrendDurationPredictor and DataManager
  When I run duration chunked validation with 10000 training window and 5000 per chunk
  Then duration confusion matrix should be logged

  Examples:
    | config_name      |
    | baseline         |