@volatility_predictor
Feature: Volatility Predictor
  ML-based predictor for HIGH_VOL / LOW_VOL regimes

Background:
  Given config files are available in tests/config/predictors
  And data source is set to data/big/

# ==============================================================================
# UNIT TESTS - Fast, verify features work
# ==============================================================================

@unit
Scenario: Train and predict volatility regime
  Given create a VolatilityPredictor and DataManager
  When I train the predictor with 500 historical elements
  And I predict volatility on the next 100 elements
  Then predictions should be 0 or 1 only
  And predictions length should be 100

@unit
Scenario: Insufficient data for training
  Given create a VolatilityPredictor and DataManager
  When I train the predictor with 10 historical elements
  Then predictor should not be marked as trained

@unit
Scenario: Time features are generated when enabled
  Given time features are enabled
  And create a VolatilityPredictor and DataManager
  When I train the predictor with 500 historical elements
  Then features should include hour_sin
  And features should include hour_cos
  And features should include day_sin
  And features should include day_cos

@unit
Scenario: Efficiency ratio features are generated when enabled
  Given efficiency ratio is enabled
  And create a VolatilityPredictor and DataManager
  When I train the predictor with 500 historical elements
  Then features should include efficiency_ratio_10
  And features should include efficiency_ratio_20
  And features should include efficiency_ratio_40

@unit
Scenario Outline: Session overlap feature generation <config_name>
  Given session overlap is <session_overlap>
  And create a VolatilityPredictor and DataManager
  When I train the predictor with 500 historical elements
  Then features should <expectation> session_overlap

  Examples:
    | config_name | session_overlap | expectation |
    | enabled     | enabled         | include     |
    | disabled    | disabled        | not include |

@unit
Scenario: Features should contain valid numeric data
  Given create a VolatilityPredictor and DataManager
  When I train the predictor with 500 historical elements
  Then all features should be numeric
  And no features should contain NaN values
  And no features should contain inf values

@unit
Scenario: Training should succeed and produce valid model
  Given create a VolatilityPredictor and DataManager
  When I train the predictor with 5000 historical elements
  Then predictor should be marked as trained
  And training should return valid metrics
  And feature importance should sum to approximately 1.0

# ==============================================================================
# COMPARISON - A/B testing different configurations
# ==============================================================================

@validation @slow @comparison
Scenario Outline: Compare predictor configurations <config_name>
  Given data source is set to data/eurusd/
  And time features are <time_features>
  And efficiency ratio is <efficiency_ratio>
  And session overlap is <session_overlap>
  And create a VolatilityPredictor and DataManager
  When I run chunked validation with 10000 training window and 5000 per chunk
  Then confusion matrix should be logged
  And comparison matrix should be logged

  Examples:
    | config_name           | time_features | efficiency_ratio | session_overlap |
    | baseline              | disabled      | disabled         | disabled        |
    | time_only             | enabled       | disabled         | disabled        |
    | efficiency_only       | disabled      | enabled          | disabled        |
    | time_efficiency       | enabled       | enabled          | disabled        |
    | session_only          | disabled      | disabled         | enabled         |
    | all_features          | enabled       | enabled          | enabled         |

@validation @slow @regression
Scenario: Baseline accuracy must not regress
  Given baseline configuration (all features disabled)
  And data source is set to data/eurusd/
  When I run chunked validation with 10000 training window and 5000 per chunk
  Then accuracy should be >= 52%
  And accuracy should be <= 60%