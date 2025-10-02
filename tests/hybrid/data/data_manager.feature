Feature: DataManager Multi-Market Data Loading and Temporal Management
  As a data management system
  I want to verify DataManager's ability to load market data and manage temporal boundaries
  So that strategies can access clean data while preventing data leakage

  Background:
    Given the system has proper directory structure
    And test data files are available in tests/data

  @data @loading @multi-market
  Scenario Outline: Multi-market data loading coordination
    Given I have <file_count> market data files in <data_directory>
    And each market file has <rows_per_file> rows without headers
    And the files follow the format: <csv_format>
    When I load multiple market data through DataManager
    Then DataManager should load all markets successfully
    And market record counts should show <expected_markets> markets with <total_records_per_market> records each
    And no data loading errors should occur
    And DataManager should cache the loaded data

    Examples:
      | scenario_type | file_count | data_directory | rows_per_file | expected_markets | total_records_per_market | csv_format                           |
      | same_market   | 3          | small          | 100           | 1                | 300                      | timestamp;open;high;low;close;volume |
      | single_file   | 1          | big            | 100000        | 1                | 100000                   | timestamp;open;high;low;close;volume |
      | diff_markets  | 3          | diff_markets   | 100           | 3                | 100                      | timestamp;open;high;low;close;volume |

  @data @temporal @initialization
  Scenario Outline: Initialize temporal boundaries for walk-forward analysis
    Given I have a market data file <data_file> in <data_directory>
    When I load the data and initialize temporal pointer with training window of <training_records> records
    Then temporal pointer should be positioned at record <now_position>
    And past data boundary should be set to record <training_records>
    And future data access should be restricted

    Examples:
      | data_file                           | data_directory | training_records | now_position |
      | DAT_ASCII_EURUSD_M1_2021_100000.csv | big            | 20000            | 20001        |

  @data @temporal @iterator @navigation
  Scenario Outline: Navigate temporal pointer with iterator methods
    Given I have <file_count> market data files in <data_directory>
    And each market file has <rows_per_file> rows without headers
    And the files follow the format: <csv_format>
    And I have loaded market data with temporal boundaries initialized
    When I query current pointer position
    Then current position should match expected location
    When I advance pointer using next(<step_size>) method
    Then pointer should move to new position correctly
    When I set pointer to absolute position <target_position>
    Then pointer should be at specified position

    Examples:
      | file_count | data_directory | rows_per_file | csv_format                           | step_size | target_position |
      | 1          | big            | 100000        | timestamp;open;high;low;close;volume | 1         | 25000          |
      | 1          | big            | 100000        | timestamp;open;high;low;close;volume | 100       | 30000          |

  @data @temporal @iterator @boundaries
  Scenario Outline: Test iterator boundary enforcement
    Given I have <file_count> market data files in <data_directory>
    And each market file has <rows_per_file> rows without headers
    And the files follow the format: <csv_format>
    And I have loaded market data with temporal boundaries initialized
    When I attempt to advance pointer beyond available data
    Then advancement should fail gracefully
    And pointer should remain at last valid position
    When I attempt to set pointer to invalid position
    Then position setting should fail with appropriate error

    Examples:
      | file_count | data_directory | rows_per_file | csv_format                           |
      | 1          | big            | 100000        | timestamp;open;high;low;close;volume |

  @data @temporal @access_control @multi_market
  Scenario Outline: Multi-market temporal data access
    Given I have <file_count> market data files in <data_directory>
    And each market file has <rows_per_file> rows without headers
    And the files follow the format: <csv_format>
    And I have loaded market data with temporal boundaries initialized
    When training algorithms request historical data
    Then past data should be returned for all loaded markets
    And data should only include records before current pointer
    When trading signals request current market data
    Then current data should be returned for all loaded markets
    And data should only include current pointer position

    Examples:
      | file_count | data_directory | rows_per_file | csv_format                           |
      | 1          | big            | 100000        | timestamp;open;high;low;close;volume |

  @data @caching @performance
  Scenario Outline: Data caching functionality
    Given I have <file_count> market data files in <data_directory>
    And each market file has <rows_per_file> rows without headers
    And the files follow the format: <csv_format>
    And I have previously loaded market data for <market_name>
    When I request the same market data again
    Then cached data should be used
    And no file loading should occur
    And cache hit should be logged

    Examples:
      | file_count | data_directory | rows_per_file | csv_format                           | market_name |
      | 3          | small          | 100           | timestamp;open;high;low;close;volume | EURUSD      |

  @data @error @handling
  Scenario Outline: Data loading error handling
    Given I have an invalid data directory <invalid_directory>
    When I attempt to load market data
    Then appropriate error messages should be logged
    And loading should fail gracefully
    And no system crash should occur

    Examples:
      | invalid_directory |
      | nonexistent/path |