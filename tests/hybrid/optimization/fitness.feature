@fitness_calculator
Feature: Fitness Calculator
  Calculate fitness scores using configurable metrics and penalties

  Background:
    Given config files are available in tests/config/optimization

  @fitness_basic_calculation
  Scenario: Calculate fitness from configured metrics
    Given metrics with values:
      | metric        | value |
      | total_return  | 0.20  |
      | sharpe_ratio  | 2.0   |
      | max_drawdown  | 0.10  |
      | num_trades    | 50    |
    And a FitnessCalculator instance
    When I calculate fitness
    Then metric weights should come from config
    And metric directions should come from config
    And a fitness score should be returned
    And the score should be a number
    And the score should be 6.17 with the deviation of 0.01

  @fitness_penalty_conditions
  Scenario Outline: Apply penalties when conditions violated
    Given metrics with values:
      | metric        | value      |
      | num_trades    | <trades>   |
      | total_return  | <return>   |
      | max_drawdown  | <drawdown> |
    And a FitnessCalculator instance
    When I calculate fitness
    Then severe_penalty should be returned
    And severe_penalty value should come from config
  Examples:
    | trades | return | drawdown |
    | 5      | 0.15   | 0.10     |
    | 50     | 0.15   | 0.45     |

  @fitness_config_loading
  Scenario: Fitness configuration loaded from config
    Given a FitnessCalculator instance
    Then metrics list should be loaded from config
    And each metric should have weight from config
    And each metric should have direction from config
    And penalty conditions should be loaded from config
    And severe_penalty value should be loaded from config

@fitness_maximize_minimize
Scenario: Metrics are maximized or minimized correctly
  Given metrics configured with maximize and minimize directions
  And metrics with values:
    | metric        | value |
    | total_return  | 0.20  |
    | sharpe_ratio  | 2.0   |
    | max_drawdown  | 0.10  |
    | num_trades    | 50    |
  And a FitnessCalculator instance
  When I calculate fitness
  Then the score should be 6.17 with the deviation of 0.01
  And maximize metrics should contribute positively
  And minimize metrics should contribute negatively
