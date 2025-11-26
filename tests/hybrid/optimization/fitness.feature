@fitness_calculator
Feature: Fitness Calculator
  Calculate fitness scores for optimization with configurable weights and penalties

  Background:
    Given a valid UnifiedConfig is loaded with fitness configuration

  @fitness_default
  Scenario: Calculate default fitness with balanced metrics
    Given backtest results with:
      | metric         | value |
      | total_return   | 0.15  |
      | sharpe_ratio   | 1.8   |
      | max_drawdown   | 0.12  |
      | num_trades     | 50    |
      | win_rate       | 0.55  |
      | profit_factor  | 1.4   |
    When I calculate fitness with "default" function
    Then fitness score should be positive
    And fitness should include return component
    And fitness should include sharpe component
    And fitness should include drawdown component
    And fitness should include trade count component

  @fitness_sharpe_focused
  Scenario: Calculate Sharpe-focused fitness
    Given backtest results with high Sharpe ratio
    When I calculate fitness with "sharpe_focused" function
    Then fitness score should emphasize Sharpe ratio
    And drawdown penalty should be applied
    And trade bonus should be included

  @fitness_return_focused
  Scenario: Calculate return-focused fitness
    Given backtest results with high total return
    When I calculate fitness with "return_focused" function
    Then fitness score should emphasize total return
    And drawdown penalty should be applied

  @fitness_balanced
  Scenario: Calculate balanced fitness with equal weights
    Given backtest results with multiple metrics
    When I calculate fitness with "balanced" function
    Then all metrics should be weighted equally
    And normalized scores should be combined

  @fitness_risk_adjusted
  Scenario: Calculate risk-adjusted fitness
    Given backtest results with:
      | metric           | value |
      | sharpe_ratio     | 1.8   |
      | sortino_ratio    | 2.1   |
      | max_drawdown     | 0.12  |
      | max_loss_streak  | 5     |
    When I calculate fitness with "risk_adjusted" function
    Then fitness should include Sharpe and Sortino components
    And drawdown penalty should be applied
    And loss streak penalty should be applied

  @fitness_calmar_ratio
  Scenario: Include Calmar ratio in fitness calculation
    Given backtest results with:
      | metric        | value |
      | total_return  | 0.30  |
      | max_drawdown  | 0.15  |
    When I calculate fitness
    Then fitness should include Calmar ratio component
    And Calmar ratio should be return divided by max_drawdown
    And component should be weighted by config

  @fitness_recovery_time
  Scenario: Penalize long drawdown recovery periods
    Given backtest results with max_drawdown_duration of 120 days
    And acceptable drawdown duration is 60 days
    When I calculate fitness
    Then recovery time penalty should be applied
    And penalty should increase with duration
    And penalty weight should come from config

  @fitness_consistency
  Scenario: Reward consistent returns over time
    Given backtest results with monthly returns
    And returns have low standard deviation
    When I calculate fitness
    Then consistency bonus should be applied
    And volatile returns should be penalized
    And consistency weight should come from config

  @fitness_trade_duration
  Scenario: Penalize excessive trade duration
    Given backtest results with avg_holding_period of 30 days
    And optimal holding period is 5 days
    When I calculate fitness
    Then trade duration penalty should be applied
    And capital efficiency should be considered

  @fitness_expectancy
  Scenario: Include expectancy in fitness calculation
    Given backtest results with expectancy metric
    When I calculate fitness
    Then fitness should include expectancy component
    And expectancy should be weighted by config

  @fitness_market_exposure
  Scenario: Consider market exposure in fitness
    Given backtest results with market_exposure of 0.95
    And optimal exposure is 0.70
    When I calculate fitness
    Then excessive exposure penalty should be applied
    And exposure weight should come from config

  @fitness_tail_risk
  Scenario: Penalize extreme tail risk
    Given backtest results with worst single trade loss of -15%
    And acceptable worst trade is -5%
    When I calculate fitness
    Then tail risk penalty should be applied
    And extreme losses should be heavily penalized

  @fitness_max_consecutive_losses
  Scenario: Penalize long losing streaks
    Given backtest results with max_loss_streak of 12
    And acceptable loss streak is 5
    When I calculate fitness
    Then loss streak penalty should be applied
    And penalty should scale with streak length

  @fitness_transaction_cost_sensitivity
  Scenario: Consider transaction cost impact
    Given backtest results with high trade frequency
    And total_fees represent significant portion of returns
    When I calculate fitness
    Then cost efficiency should be factored in
    And strategies with high costs should be penalized

  @fitness_risk_of_ruin
  Scenario: Calculate risk of ruin metric
    Given backtest results with:
      | metric              | value |
      | max_drawdown        | 0.30  |
      | avg_loss            | -50   |
      | win_rate            | 0.45  |
    When I calculate fitness
    Then risk of ruin should be estimated
    And high risk of ruin should apply severe penalty

  @fitness_penalty_insufficient_trades
  Scenario: Apply severe penalty for insufficient trades
    Given backtest results with only 5 trades
    And minimum required trades is 10
    When I calculate fitness
    Then fitness should equal severe_penalty value
    And no other components should be calculated

  @fitness_penalty_excessive_drawdown
  Scenario: Apply severe penalty for excessive drawdown
    Given backtest results with max_drawdown of 0.45
    And max_drawdown_limit is 0.30
    When I calculate fitness
    Then fitness should equal severe_penalty value

  @fitness_penalty_negative_return
  Scenario: Apply severe penalty for negative return
    Given backtest results with total_return of -0.05
    And minimum return requirement is 0.0
    When I calculate fitness
    Then fitness should equal severe_penalty value

  @fitness_trade_count_normalization
  Scenario: Normalize trade count with optimal range
    Given optimal trade count is 50
    And backtest results with <num_trades> trades
    When I normalize trade count
    Then normalized score should be <expected_range>

    Examples:
      | num_trades | expected_range    |
      | 25         | half of maximum   |
      | 50         | maximum bonus     |
      | 100        | diminishing bonus |

  @fitness_profit_factor_normalization
  Scenario: Normalize profit factor
    Given profit factor scale configuration
    And backtest results with profit_factor <value>
    When I normalize profit factor
    Then normalized score should be <expected>

    Examples:
      | value | expected           |
      | 0.8   | zero (losing)      |
      | 1.0   | zero (break even)  |
      | 2.0   | positive           |
      | 5.0   | capped at maximum  |

  @fitness_compare_methods
  Scenario: Compare different fitness methods
    Given backtest results
    When I compare all fitness methods
    Then results should include fitness scores for all methods
    And best method should be identified
    And worst method should be identified
    And consistency score should be calculated
    And recommendation should be provided

  @fitness_method_agreement
  Scenario: Methods agree on good strategy
    Given backtest results with consistently good metrics
    When I compare fitness methods
    Then consistency should be low
    And all methods should rank highly
    And recommendation should be "use any method"

  @fitness_method_disagreement
  Scenario: Methods disagree on strategy
    Given backtest results with mixed metrics
    When I compare fitness methods
    Then consistency should be high
    And methods should rank differently
    And recommendation should suggest "strategy revision"

  @fitness_multi_objective
  Scenario: Calculate multi-objective fitness
    Given backtest results
    And objectives list: ["return", "sharpe", "drawdown", "calmar", "expectancy"]
    When I calculate multi-objective fitness
    Then objective_scores should contain score for each objective
    And combined_score should be average of objectives
    And penalty_applied flag should be present

  @fitness_multi_objective_with_penalty
  Scenario: Multi-objective fitness with penalty
    Given backtest results with insufficient trades
    And objectives list: ["return", "sharpe"]
    When I calculate multi-objective fitness
    Then penalty_applied should be true
    And combined_score should be severe_penalty

  @fitness_trade_efficiency_objective
  Scenario: Calculate trade efficiency objective
    Given backtest results with:
      | metric       | value |
      | total_return | 0.20  |
      | num_trades   | 40    |
    When I calculate multi-objective fitness with "trade_efficiency"
    Then trade_efficiency should be return per trade
    And score should be 0.005

  @fitness_validation
  Scenario: Validate fitness function with test results
    Given 20 test backtest results
    And results are classified as GOOD, AVERAGE, or BAD
    When I validate fitness function
    Then validation_score should be calculated
    And ranking_accuracy should show average ranks per class
    And recommendation should indicate function quality

  @fitness_validation_insufficient_samples
  Scenario: Validation requires minimum samples
    Given only 3 test backtest results
    And minimum validation samples is 10
    When I validate fitness function
    Then error should be raised
    And error message should indicate minimum samples needed

  @fitness_validation_excellent
  Scenario: Fitness function validates excellently
    Given test results where GOOD strategies are clearly ranked higher
    When I validate fitness function
    Then validation_score should be above 0.9
    And recommendation should be "EXCELLENT"

  @fitness_validation_poor
  Scenario: Fitness function validates poorly
    Given test results where BAD strategies rank higher than GOOD
    When I validate fitness function
    Then validation_score should be below 0.6
    And recommendation should be "POOR"
    And recommendation should suggest adjusting weights

  @fitness_configurable_weights
  Scenario: All weights are configurable
    Given fitness configuration with custom weights
    When I calculate fitness
    Then no hardcoded values should be used
    And all weights should come from config
    And severe_penalty should come from config

  @fitness_zero_hardcoded_values
  Scenario: Zero hardcoded values in calculations
    Given fitness calculator instance
    Then zero_value should come from config
    And unity_value should come from config
    And all thresholds should come from config
    And all scaling factors should come from config
    And all penalty weights should come from config
    And all new metric weights should come from config