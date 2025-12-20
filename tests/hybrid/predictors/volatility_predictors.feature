@volatility_predictor
Feature: Volatility Predictor
  ML-based predictor for HIGH_VOL / LOW_VOL regimes

  Background:
    Given config files are available in tests/config/predictors

  @unit
  Scenario: Train predictor on historical data
    Given a VolatilityPredictor
    And market data with at least 500 bars
    When I train the predictor
    Then training should complete successfully
    And predictor should be marked as trained

  @unit
  Scenario: Predict volatility regime
    Given a trained VolatilityPredictor
    And market data with 100 bars
    When I predict volatility
    Then predictions should be 0 or 1 only
    And predictions length should match data length

  @unit
  Scenario: Insufficient data returns empty
    Given a VolatilityPredictor
    And market data with only 10 bars
    When I train the predictor
    Then predictor should not be marked as trained

  @unit
  Scenario: Features use only past data
    Given a VolatilityPredictor
    And market data with 100 bars
    When I create features for bar 50
    Then features should only use data from bars 0 to 49

@validation @slow
  Scenario: Prediction has edge over random
    Given data source is set to data/eurusd/
    And a VolatilityPredictor
    When I train and validate with temporal split
    Then volatility predictions should meet configured threshold