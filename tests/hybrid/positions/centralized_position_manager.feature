Feature: Centralized Position Manager
  As a trading system with parallel bots
  I want to manage capital allocation centrally
  So that bots don't over-allocate capital and compete fairly

  Background:
    Given config files are available in tests/config/positions
    And a centralized position manager is initialized from configuration

  # ============================================================================
  # INITIALIZATION
  # ============================================================================

   Scenario: Initialize position manager with total capital
    Then available capital should be 100000
    And committed capital should be 0
    And active positions count should be 0

  # ============================================================================
  # COMMIT POSITION - SUCCESS
  # ============================================================================

 Scenario Outline: Commit capital for positions
  When bots commit amounts <amounts>
  Then all commitments should succeed
  And available capital should be <available>
  And committed capital should be <committed>
  And active positions count should be <count>

  Examples:
    | amounts                | available | committed | count |
    | [20000]                | 80000     | 20000     | 1     |
    | [15000, 25000, 10000]  | 50000     | 50000     | 3     |
    | [100000]               | 0         | 100000    | 1     |

  # ============================================================================
  # COMMIT POSITION - FAILURE
  # ============================================================================

Scenario Outline: Cannot commit invalid amounts
  Given bots have committed amounts <initial_amounts>
  When When bot tries to commit <invalid_amount>
  Then commitment should fail
  And available capital should be <available>
  And committed capital should be <committed>
  And active positions count should be <count>

  Examples:
    | initial_amounts | invalid_amount | available | committed | count |
    | [70000]         | 40000          | 30000     | 70000     | 1     |
    | []              | 0              | 100000    | 0         | 0     |
    | []              | -5000          | 100000    | 0         | 0     |

Scenario: Cannot commit same trade_id twice
  Given bots have committed amounts [20000]
  When bot attempts to commit 10000 with same trade_id
  Then the duplicate commitment should fail
  And committed capital should be 20000
  And active positions count should be 1

  # ============================================================================
  # FIRST-COME-FIRST-SERVE
  # ============================================================================

  Scenario: Capital allocation race - first come first served
    Given bots have committed amounts [60000]
    When bots try to commit amounts [50000, 30000]
    Then commitments should have results [False, True]
    And available capital should be 10000
    And active positions count should be 2

  # ============================================================================
  # RELEASE POSITION
  # ============================================================================

  Scenario Outline: Release committed positions
  Given bots have committed amounts <commit_amounts>
  When positions <release_indices> are released
  Then releases should have results <results>
  And available capital should be <available>
  And committed capital should be <committed>
  And active positions count should be <count>

  Examples:
    | commit_amounts      | release_indices | results      | available | committed | count |
    | [30000]             | [0]             | [True]       | 100000    | 0         | 0     |
    | [20000, 30000, 15000] | [0, 2]        | [True, True] | 70000     | 30000     | 1     |
    | [40000, 25000]      | [0, 1]          | [True, True] | 100000    | 0         | 0     |

  Scenario Outline: Cannot release invalid positions
    Given bots have committed amounts <commit_amounts>
    And positions <already_released> are already released
    When position <invalid_index> is released
    Then release should fail
    And available capital should be <available>

  Examples:
    | commit_amounts | already_released | invalid_index | available |
    | []             | []               | 999           | 100000    |
    | [20000]        | [0]              | 0             | 100000    |

  # ============================================================================
  # ALLOCATION SUMMARY
  # ============================================================================

  Scenario Outline: Get allocation summary
    Given bots have committed amounts <amounts>
    When allocation summary is requested
    Then summary total_capital should be <total>
    And summary available should be <available>
    And summary committed should be <committed>
    And summary available_pct should be <available_pct>
    And summary committed_pct should be <committed_pct>
    And summary active_positions should be <count>

    Examples:
      | amounts           | total  | available | committed | available_pct | committed_pct | count |
      | []                | 100000 | 100000    | 0         | 100.0         | 0.0           | 0     |
      | [25000, 35000]    | 100000 | 40000     | 60000     | 40.0          | 60.0          | 2     |

  Scenario: Allocation summary groups by bot
    Given bot "bot_1" commits multiple amounts [15000, 20000]
    And bot "bot_2" commits amounts [30000]
    When allocation summary is requested
    Then bot "bot_1" should have 2 positions totaling 35000
    And bot "bot_2" should have 1 position totaling 30000

  # ============================================================================
  # CONCURRENT ACCESS (THREAD SAFETY)
  # ============================================================================

  Scenario Outline: Concurrent capital allocation respects limits
    Given <bot_count> bots attempt to commit <amount> each simultaneously
    Then exactly <success_count> commitments should succeed
    And <fail_count> commitments should fail
    And available capital should be <available>
    And committed capital should be <committed>
    And active positions count should be <success_count>

  Examples:
    | bot_count | amount | success_count | fail_count | available | committed |
    | 10        | 15000  | 6             | 4          | 10000     | 90000     |
    | 5         | 25000  | 4             | 1          | 0         | 100000    |

  Scenario: Concurrent commit and release operations are atomic
    Given 5 bots perform mixed commit/release operations
    When 100 operations are executed concurrently
    Then capital integrity is maintained
    And available plus committed equals total capital

  # ============================================================================
  # RESET (TESTING UTILITY)
  # ============================================================================

  Scenario: Reset clears all committed positions
    Given bot "bot_1" has committed 30000 for trade "trade_001"
    And bot "bot_2" has committed 40000 for trade "trade_002"
    When position manager is reset
    Then available capital should be 100000
    And active positions count should be 0
