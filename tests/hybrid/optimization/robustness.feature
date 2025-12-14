Feature: Robustness Analysis for Parameter Optimization
  As a trading system optimizer
  I want to analyze parameter robustness and detect overfitting
  So that I can select reliable parameter configurations for production

Background:
  Given config files are available in tests/config/optimization
  And  a RobustnessAnalyzer with standard configuration
  And robustness thresholds are set:
    | threshold                    | value |
    | cv_threshold_robust          | 0.10  |
    | cv_threshold_sensitive       | 0.25  |
    | top_performers_percentile    | 0.20  |
    | plateau_threshold            | 0.15  |
    | strong_correlation_threshold | 0.70  |

@parameter_stability
Scenario Outline: Parameter stability classification
  Given <n_results> optimization results where <param_name> has mean <mean_value> and std <std_value>
  When I analyze parameter stability
  Then parameter <param_name> should be classified as <robustness_class>

  Examples:
    | n_results | param_name      | mean_value | std_value | robustness_class |
    | 500        | atr_period      | 14.0       | 0.7       | ROBUST           |
    | 500        | atr_period      | 14.0       | 2.52      | MODERATE         |
    | 500        | atr_period      | 14.0       | 4.9       | SENSITIVE        |
    | 500        | stop_multiplier | 2.0        | 0.16      | ROBUST           |

  @parameter_stability @edge_cases
  Scenario Outline: Handle insufficient optimization results
    Given I have <n_results> optimization results
    When I analyze parameter stability
    Then the analysis should indicate insufficient data
    Examples:
      | n_results |
      | 0         |
      | 5         |

  @fitness_landscape
  Scenario Outline: Detect fitness landscape type
    Given I have <n_results> optimization results
    And top <top_percentage> of results have fitness within <tolerance> of maximum <max_fitness>
    When I analyze fitness landscape
    Then landscape type should be <landscape_type>

    Examples:
      | n_results | top_percentage | tolerance | max_fitness | landscape_type    |
      | 100       | 15%            | 5%        | 100         | PLATEAU_DOMINATED |
      | 100       | 3%             | 30%       | 100         | PEAKY             |
      | 100       | 10%            | 15%       | 100         | MIXED             |

  @robust_ranges
  Scenario Outline: Identify robust parameter ranges based on classification
    Given <n_results> optimization results where <param_name> has mean <mean_value> and std <std_value>
    When I analyze parameter stability
    And I find robust parameter ranges
    Then <param_name> should have confidence level <confidence>

    Examples:
      | n_results | param_name      | mean_value | std_value | confidence |
      | 500       | atr_period      | 14.0       | 0.7       | HIGH       |
      | 500       | position_size   | 0.02       | 0.003     | MEDIUM     |
      | 500       | entry_threshold | 0.5        | 0.25      | NONE       |

  @report
  Scenario Outline: Generate comprehensive robustness report
    Given <n_results> optimization results with <robust_pct> robust parameters
    And fitness landscape is <landscape_type>
    When I generate robustness report
    Then recommendation should be <recommendation>

    Examples:
      | n_results | robust_pct | landscape_type    | recommendation |
      | 500       | 80%        | PLATEAU_DOMINATED | EXCELLENT      |
      | 500       | 60%        | MIXED             | GOOD           |
      | 500       | 40%        | PEAKY             | CAUTION        |
      | 500       | 20%        | PEAKY             | POOR           |
