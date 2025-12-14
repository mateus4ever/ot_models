Feature: SignalFactory Creation and Error Handling
  As a trading system orchestrator
  I want the SignalFactory to create all implemented signals successfully
  So that I can reliably instantiate any available signal with namespaced organization

  Background:
    Given config files are available in tests/config/signals
    And  I have a SignalFactory instance

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

  @factory @available_signals
  Scenario: Get list of available signals
    Given the factory is properly initialized
    When I request the list of available signals
    Then the list should contain "mean_reversion.bollinger"
    And the list should contain "momentum.rsi"
    And the list should contain "trend_following.simplemovingaveragecrossover"
    And the list should have exactly 3 signals

@factory @categorization @discovery
Scenario Outline: Get signals by category
  Given the factory is properly initialized
  When I request signals by category <category>
  Then the list should contain <expected_signal>
  And the list should have exactly <signal_count> signal in <category> category

  Examples:
    | category        | expected_signal                             | signal_count |
    | momentum        | momentum.rsi                                | 1            |
    | mean_reversion  | mean_reversion.bollinger                    | 1            |
    | trend_following | trend_following.simplemovingaveragecrossover| 1            |

  @factory @categorization @available_categories
  Scenario: Get list of available categories
    Given the factory is properly initialized
    When I request the list of available categories
    Then the category list should contain "momentum"
    And the category list should contain "mean_reversion"
    And the category list should contain "trend_following"
    And the category list should have exactly 3 categories