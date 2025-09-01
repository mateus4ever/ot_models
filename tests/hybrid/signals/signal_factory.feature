Feature: SignalFactory Creation and Error Handling
  As a trading system orchestrator
  I want the SignalFactory to create all implemented signals successfully
  So that I can reliably instantiate any available signal with namespaced organization

  Background:
    Given the system has proper directory structure
    And I have a SignalFactory instance

  @factory @creation @success
  Scenario Outline: Successfully create all implemented signals
    Given the factory is properly initialized
    When I create a <signal_name> signal
    Then a valid signal instance should be created
    And the signal should implement SignalInterface
    And no creation errors should occur

    Examples:
      | signal_name |
      | mean_reversion.bollinger |
      | momentum.rsi |
      | trend_following.simplemovingaveragecrossover |

  @factory @creation @with_config
  Scenario Outline: Successfully create signals with configuration
    Given the factory is properly initialized
    And I have a valid signal configuration object
    When I create a <signal_name> signal with configuration
    Then a valid signal instance should be created with config
    And the configuration should be passed to the signal
    And no creation errors should occur

    Examples:
      | signal_name                                  |
      | mean_reversion.bollinger                     |
      | momentum.rsi                                 |
      | trend_following.simplemovingaveragecrossover |

  @factory @error_handling
  Scenario Outline: Handle invalid inputs appropriately
    Given the factory is properly initialized
    When I try to create a signal with <invalid_input>
    Then a ValueError should be thrown
    And the error message should be informative

    Examples:
      | invalid_input        |
      | "nonexistent.signal" |
      | None                |
      | ""                  |
      | "   "               |

  @factory @available_signals
  Scenario: Get list of available signals
    Given the factory is properly initialized
    When I request the list of available signals
    Then the list should contain "mean_reversion.bollinger"
    And the list should contain "momentum.rsi"
    And the list should contain "trend_following.simplemovingaveragecrossover"
    And the list should have exactly 3 signals

  @factory @categorization @discovery
  Scenario: Get signals by category
    Given the factory is properly initialized
    When I request signals by category "momentum"
    Then the list should contain "momentum.rsi"
    And the list should have exactly 1 signal in momentum category
    When I request signals by category "mean_reversion"
    Then the list should contain "mean_reversion.bollinger"
    And the list should have exactly 1 signal in mean_reversion category
    When I request signals by category "trend_following"
    Then the list should contain "trend_following.simplemovingaveragecrossover"
    And the list should have exactly 1 signal in trend_following category

  @factory @categorization @available_categories
  Scenario: Get list of available categories
    Given the factory is properly initialized
    When I request the list of available categories
    Then the category list should contain "momentum"
    And the category list should contain "mean_reversion"
    And the category list should contain "trend_following"
    And the category list should have exactly 3 categories