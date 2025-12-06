@strategy_interface
Feature: Strategy Interface Requirements
  All strategy implementations must provide complete strategy capabilities
  including dependency injection, component management, and execution

  Background:
    Given config files are available in tests/config/strategies

  @core_methods
  Scenario Outline: Strategy implements core interface methods
    Given a strategy of type "<strategy_type>"
    Then the strategy should have method "set_money_manager"
    And the strategy should have method "set_data_manager"
    And the strategy should have method "set_position_orchestrator"
    And the strategy should have method "add_entry_signal"
    And the strategy should have method "add_exit_signal"
    And the strategy should have method "add_optimizer"
    And the strategy should have method "add_predictor"
    And the strategy should have method "add_metric"
    And the strategy should have method "add_execution_listener"
    And the strategy should have method "run"
    And the strategy should have method "get_optimizable_parameters"

    Examples:
      | strategy_type |
      | base         |
      | chained      |

  @base_attributes
  Scenario Outline: Strategy has required attributes
    Given a strategy of type "<strategy_type>"
    Then it should have attribute "name"
    And it should have attribute "config"
    And it should have attribute "money_manager"
    And it should have attribute "data_manager"
    And it should have attribute "entry_signal"
    And it should have attribute "exit_signal"
    And it should have attribute "optimizers"
    And it should have attribute "predictors"
    And it should have attribute "runners"
    And it should have attribute "metrics"

    Examples:
      | strategy_type |
      | base         |
      | chained      |

  @dependency_injection
  Scenario Outline: Strategy accepts dependency injection
    Given a strategy of type "<strategy_type>"
    When I create mock dependencies
    And I inject MoneyManager into the strategy
    And I inject DataManager into the strategy
    And I inject PositionOrchestrator into the strategy
    Then no injection errors should occur

    Examples:
      | strategy_type |
      | base         |
      | chained      |

  @component_management
  Scenario Outline: Strategy accepts component addition
    Given a strategy of type "<strategy_type>"
    When I create mock components
    And I add entry signal to the strategy
    And I add exit signal to the strategy
    And I add optimizer to the strategy
    And I add predictor to the strategy
    And I add metric to the strategy
    Then no component addition errors should occur

    Examples:
      | strategy_type |
      | base         |
      | chained      |