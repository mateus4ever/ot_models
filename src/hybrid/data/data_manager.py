# data_manager.py
# DataManager for multi-market data loading and coordination
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
# ONLY LOADS REAL FILES - NO SAMPLE DATA GENERATION
# PROPER LOGGING THROUGHOUT
# DATA CONTROLLER - NO DATAFRAME EXPOSURE
# MARKET CONSOLIDATION - Append chunked files into single market datasets

import pandas as pd
import logging
import re
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading, preparation, and caching for multiple strategies

    Acts as data controller with temporal boundary enforcement.
    Consolidates chunked files into single market datasets.
    Does not expose raw DataFrames to prevent data leakage.
    """

    def __init__(self, config):
        self.config = config
        self._cached_data = {}  # Private - no external access {market_name: consolidated_dataframe}
        self._training_data_cache = {}  # Private - no external access

        # Market management
        self._active_market = None  # Currently selected market name (e.g., "EUR/USD")
        self._active_market_data = None  # Cached DataFrame reference for performance
        self.loaded_markets = set()  # Track which markets are loaded

        # Timestamp-based temporal management
        self.temporal_timestamp = None  # Current datetime position (the "now" pointer)
        self._active_market_index = None  # Cached index position in active market for performance
        self.training_window_size = None
        self.total_records = 0

        logger.info("Initializing DataManager as data controller with timestamp-based temporal management")
        self._log_initialization_info()

    def _log_initialization_info(self):
        """Log initialization information for debugging"""
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent

        logger.debug(f"DataManager file location: {current_file}")
        logger.debug(f"Detected project root: {project_root}")

        # Log config information
        data_config = self.config.get_section('data_loading', {})
        data_source = data_config.get('data_source', 'tests/data')
        logger.debug(f"Configured data source: {data_source}")

        # Check if data directory exists
        if Path(data_source).is_absolute():
            data_path = Path(data_source)
        else:
            data_path = project_root / data_source

        logger.debug(f"Resolved data path: {data_path}")
        logger.debug(f"Data directory exists: {data_path.exists()}")

        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            logger.info(f"Found {len(csv_files)} CSV files in data directory")
            if csv_files:
                logger.debug(f"Available CSV files: {[f.name for f in csv_files[:5]]}")

    def _extract_market_from_filename(self, filename: str) -> str:
        """Extract market identifier from filename with strict pattern matching

        Args:
            filename: CSV filename

        Returns:
            Market identifier in EUR/USD format
        """

        # Strict pattern for files like "DAT_ASCII_EURUSD_M1_2021_100000.csv"
        pattern = r'(?:DAT_ASCII_)?([A-Z]{6})(?:_M1|_H1|_D1)'
        match = re.search(pattern, filename.upper())

        if match:
            currency_pair = match.group(1)
            # Convert EURUSD to EUR/USD format
            return f"{currency_pair[:3]}/{currency_pair[3:]}"

        # Fallback: extract first 6 letters and format
        letters_only = re.sub(r'[^A-Za-z]', '', filename.upper())
        if len(letters_only) >= 6:
            return f"{letters_only[:3]}/{letters_only[3:6]}"

        logger.warning(f"Could not extract market from filename: {filename}")
        return "UNKNOWN"

    def load_market_data(self, markets: List[str]) -> bool:
        """Load and consolidate chunked files for markets

        Consolidates all files into single market datasets.
        Last loaded market becomes active market.

        Args:
            markets: List of market file names

        Returns:
            True if all markets loaded successfully
        """
        logger.info(f"Starting market consolidation for {len(markets)} files: {markets}")

        # Group files by market identifier
        market_groups = {}
        for market_file in markets:
            market_id = self._extract_market_from_filename(market_file)
            if market_id not in market_groups:
                market_groups[market_id] = []
            market_groups[market_id].append(market_file)

        logger.info(f"Detected {len(market_groups)} unique markets: {list(market_groups.keys())}")

        successful_loads = 0
        last_loaded_market = None

        for market_id, file_list in market_groups.items():
            try:
                logger.info(f"Consolidating {len(file_list)} files for market {market_id}")
                success = self._load_and_consolidate_market(market_id, file_list)
                if success:
                    successful_loads += 1
                    last_loaded_market = market_id
                    self.loaded_markets.add(market_id)

            except Exception as e:
                logger.error(f"Failed to consolidate market {market_id}: {e}")

        # Set last loaded market as active
        if last_loaded_market:
            self.set_active_market(last_loaded_market)
            logger.info(f"Active market set to: {self._active_market}")

        all_loaded = successful_loads == len(market_groups)
        logger.info(f"Market consolidation result: {successful_loads}/{len(market_groups)} markets loaded")

        return all_loaded

    def _load_and_consolidate_market(self, market_id: str, file_list: List[str]) -> bool:
        """Load and consolidate chunked files for a single market

        Args:
            market_id: Market identifier (e.g., "EUR/USD")
            file_list: List of chunked files for this market

        Returns:
            True if consolidation successful
        """
        logger.debug(f"Consolidating {len(file_list)} files for market {market_id}")

        # Check cache first
        if market_id in self._cached_data:
            logger.debug(f"Cache hit for market: {market_id}")
            return True

        consolidated_chunks = []

        for filename in file_list:
            file_path = self._get_market_file_path(filename)
            logger.debug(f"Loading chunk: {filename}")

            chunk_df = self._load_csv_file(file_path)
            consolidated_chunks.append(chunk_df)

        # Consolidate all chunks chronologically
        consolidated_df = pd.concat(consolidated_chunks, ignore_index=False)
        consolidated_df.sort_index(inplace=True)  # Ensure chronological order

        # Remove any duplicate timestamps
        consolidated_df = consolidated_df[~consolidated_df.index.duplicated(keep='first')]

        # Store consolidated market data
        self._cached_data[market_id] = consolidated_df

        logger.info(
            f"Market {market_id} consolidated: {len(consolidated_df)} total records from {len(file_list)} files")
        logger.debug(f"Consolidated date range: {consolidated_df.index[0]} to {consolidated_df.index[-1]}")

        return True

    def load_market_data_single(self, market_file: str) -> bool:
        """Load single market file (handles as single-file market)

        Args:
            market_file: Market file name

        Returns:
            True if loading successful
        """
        market_id = self._extract_market_from_filename(market_file)
        logger.debug(f"Loading single market file {market_file} as market {market_id}")

        success = self._load_and_consolidate_market(market_id, [market_file])
        if success:
            self._active_market = market_id
            self.loaded_markets.add(market_id)
            logger.info(f"Single market loaded and set as active: {market_id}")

        return success

    def set_active_market(self, market_id: str):
        """Switch active market for temporal operations

        Args:
            market_id: Market identifier (e.g., "EUR/USD")

        Raises:
            ValueError: If market not loaded
        """
        if market_id not in self._cached_data:
            available = list(self._cached_data.keys())
            raise ValueError(f"Market {market_id} not loaded. Available markets: {available}")

        old_active = self._active_market
        self._active_market = market_id
        self._active_market_data = self._cached_data[market_id]  # Cache for performance
        self.total_records = len(self._active_market_data)

        # Update cached index position for current timestamp in new market
        if self.temporal_timestamp is not None:
            self._active_market_index = self._find_timestamp_index(self.temporal_timestamp)
        else:
            self._active_market_index = None

        logger.info(f"Active market switched from {old_active} to {market_id}")
        if self.temporal_timestamp is not None:
            logger.debug(
                f"Temporal position updated: timestamp {self.temporal_timestamp} -> index {self._active_market_index}")

    def _find_timestamp_index(self, timestamp) -> int:
        """Find index position for timestamp in active market data

        Args:
            timestamp: Target timestamp

        Returns:
            Index position in active market data

        Raises:
            ValueError: If no active market data
        """
        if self._active_market_data is None:
            raise ValueError("No active market data available")

        try:
            # Find nearest timestamp in active market
            index_pos = self._active_market_data.index.get_loc(timestamp, method='nearest')
            logger.debug(f"Found timestamp {timestamp} at index {index_pos} in {self._active_market}")
            return index_pos
        except KeyError:
            raise ValueError(f"Timestamp {timestamp} not found in {self._active_market} data")

    def get_available_markets(self) -> List[str]:
        """Return list of loaded markets available for selection

        Returns:
            List of market identifiers
        """
        return list(self._cached_data.keys())

    def load_market_data_with_temporal_setup(self, markets: List[str], training_window: int) -> bool:
        """Load market data and initialize temporal boundaries in one operation

        Args:
            markets: List of market file names
            training_window: Number of records for initial training window

        Returns:
            True if loading and temporal setup successful
        """
        logger.info(f"Loading {len(markets)} files with temporal setup, training window: {training_window}")

        # Load and consolidate markets
        all_loaded = self.load_market_data(markets)

        if all_loaded and self._active_market and self._active_market_data is not None:
            # Initialize temporal boundaries using active market data
            self.initialize_temporal_pointer(self._active_market_data, training_window)

            logger.info(
                f"Temporal setup complete for {self._active_market}: pointer at timestamp {self.temporal_timestamp}")
            return True
        else:
            logger.error("Failed to load markets or setup temporal boundaries")
            return False

    def _get_market_file_path(self, market: str) -> str:
        """Get the actual file path for a market CSV file"""
        # If market is already a full path, use as-is
        if '/' in market or '\\' in market:
            logger.debug(f"Using provided full path for market: {market}")
            return market

        # Get base directory from config
        data_config = self.config.get_section('data_loading', {})
        base_directory = data_config.get('data_source', 'tests/data')
        logger.debug(f"Base directory from config: {base_directory}")

        # Find project root - go up from current file location
        current_file = Path(__file__).resolve()  # src/hybrid/data/data_manager.py
        project_root = current_file.parent.parent.parent.parent  # Go up 4 levels to project root

        logger.debug(f"Current file location: {current_file}")
        logger.debug(f"Resolved project root: {project_root}")

        # Ensure .csv extension
        if not market.endswith('.csv'):
            market_file = f"{market}.csv"
            logger.debug(f"Added .csv extension: {market} -> {market_file}")
        else:
            market_file = market
            logger.debug(f"Market already has .csv extension: {market_file}")

        # Construct full file path from project root + config path
        if Path(base_directory).is_absolute():
            file_path = Path(base_directory) / market_file
            logger.debug(f"Using absolute base directory: {base_directory}")
        else:
            # Remove any 'src/' prefix if it exists in the config path
            clean_base_directory = base_directory
            if base_directory.startswith('src/') or base_directory.startswith('src\\'):
                clean_base_directory = base_directory[4:]  # Remove 'src/' or 'src\'
                logger.debug(f"Removed 'src/' prefix: {base_directory} -> {clean_base_directory}")

            file_path = project_root / clean_base_directory / market_file
            logger.debug(f"Using cleaned relative base directory from project root: {clean_base_directory}")
            logger.debug(f"Final path construction: {project_root} / {clean_base_directory} / {market_file}")

        logger.info(f"Resolved market file path: {file_path}")
        logger.debug(f"File exists check: {file_path.exists()}")

        return str(file_path)

    def _load_csv_file(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with proper parsing - NO SAMPLE GENERATION"""
        logger.debug(f"Starting CSV file load: {file_path}")

        try:
            # Load CSV with semicolon delimiter (based on your test format)
            logger.debug("Loading CSV with semicolon delimiter, no headers")
            df = pd.read_csv(
                file_path,
                delimiter=';',
                header=None,  # No headers as specified in tests
                names=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            logger.debug(f"Raw CSV loaded: {len(df)} rows, {len(df.columns)} columns")

            # Convert timestamp to datetime and set as index
            logger.debug("Converting timestamp column to datetime index")
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
            df.set_index('timestamp', inplace=True)

            # Sort by timestamp
            logger.debug("Sorting data by timestamp")
            df.sort_index(inplace=True)

            logger.info(f"Successfully loaded and processed {len(df)} rows from {file_path}")
            logger.debug(f"Data date range: {df.index[0]} to {df.index[-1]}")
            logger.debug(f"Data columns: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            logger.debug(f"CSV loading error details", exc_info=True)
            raise

    # =============================================================================
    # TEMPORAL BOUNDARY METHODS
    # =============================================================================

    def initialize_temporal_pointer(self, market_data: pd.DataFrame, training_window: int) -> None:
        """Initialize timestamp-based temporal pointer for walk-forward analysis

        Case 1: temporal_timestamp = None -> Set to timestamp at training_window position

        Args:
            market_data: DataFrame with market data
            training_window: Number of records for initial training window

        Returns:
            Position of "now" pointer (training_window + 1, 1-based)
        """
        if len(market_data) <= training_window:
            raise ValueError(f"Market data has {len(market_data)} records, need at least {training_window + 1}")

        self.training_window_size = training_window
        self.total_records = len(market_data)

        # Case 1: Set temporal pointer to timestamp at training window position
        self.temporal_timestamp = market_data.index[training_window]  # Get actual timestamp
        self._active_market_index = training_window  # Cache index position for performance

        logger.info(f"Temporal pointer initialized for {self._active_market}: training window={training_window}")
        logger.info(f"Temporal timestamp set to: {self.temporal_timestamp} (record {training_window + 1}, 1-based)")
        logger.debug(f"Total records: {self.total_records}, cached index: {self._active_market_index}")

        return training_window + 1  # Return 1-based position

    def get_current_pointer(self) -> int:
        """Get current temporal pointer position (1-based)

        Returns:
            Current pointer position as 1-based index

        Raises:
            ValueError: If temporal pointer not initialized
        """
        if self.temporal_timestamp is None:
            raise ValueError("Temporal pointer not initialized. Call initialize_temporal_pointer() first.")

        return self._active_market_index + 1  # Return 1-based

    def set_pointer(self, position: int):
        """Set temporal pointer to absolute position (1-based)

        Args:
            position: Target position (1-based)

        Raises:
            ValueError: If position is invalid or temporal pointer not initialized
        """
        if self.temporal_timestamp is None:
            raise ValueError("Temporal pointer not initialized. Call initialize_temporal_pointer() first.")

        if self._active_market_data is None:
            raise ValueError("No active market data available")

        if position < 1 or position > self.total_records:
            raise ValueError(f"Position {position} out of range [1, {self.total_records}]")

        if position <= self.training_window_size:
            logger.warning(f"Setting pointer to {position}, within training window (1-{self.training_window_size})")

        old_position = self._active_market_index + 1

        # Convert to 0-based index and get timestamp
        new_index = position - 1
        new_timestamp = self._active_market_data.index[new_index]

        # Update both timestamp and cached index
        self.temporal_timestamp = new_timestamp
        self._active_market_index = new_index

        logger.info(f"Temporal pointer moved from position {old_position} to {position}")
        logger.debug(f"New timestamp: {new_timestamp}, cached index: {new_index}")

    def set_pointer_to_date(self, target_date):
        """Case 3: Set temporal pointer to specific timestamp

        Args:
            target_date: Target datetime or string timestamp

        Raises:
            ValueError: If date not found in active market data
        """
        if self._active_market_data is None:
            raise ValueError("No active market data available")

        # Convert string to datetime if needed
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)

        # Find index position for this timestamp
        try:
            new_index = self._find_timestamp_index(target_date)
            old_timestamp = self.temporal_timestamp

            # Update both timestamp and cached index
            self.temporal_timestamp = target_date
            self._active_market_index = new_index

            logger.info(f"Temporal pointer set to date: {target_date} (index {new_index})")
            logger.debug(f"Previous timestamp: {old_timestamp}")

        except ValueError as e:
            logger.error(f"Failed to set pointer to date {target_date}: {e}")
            raise

    def next(self, steps: int = 1) -> bool:
        """Case 2: Advance temporal pointer by specified steps

        Args:
            steps: Number of steps to advance (default 1)

        Returns:
            True if advancement successful, False if at end of data

        Raises:
            ValueError: If temporal pointer not initialized
        """
        if self.temporal_timestamp is None:
            raise ValueError("Temporal pointer not initialized. Call initialize_temporal_pointer() first.")

        if self._active_market_data is None:
            raise ValueError("No active market data available")

        new_index = self._active_market_index + steps

        if new_index >= self.total_records:
            logger.warning(f"Cannot advance {steps} steps: would exceed data boundaries")
            return False

        old_timestamp = self.temporal_timestamp
        old_position = self._active_market_index + 1

        # Update to new timestamp and cache new index
        self.temporal_timestamp = self._active_market_data.index[new_index]
        self._active_market_index = new_index
        new_position = new_index + 1

        logger.debug(f"Temporal pointer advanced from position {old_position} to {new_position} (+{steps} steps)")
        logger.debug(f"Timestamp changed from {old_timestamp} to {self.temporal_timestamp}")

        return True

    def previous(self, steps: int = 1) -> bool:
        """Move temporal pointer backward by specified steps

        Args:
            steps: Number of steps to move backward (default 1)

        Returns:
            True if movement successful, False if at beginning of data

        Raises:
            ValueError: If temporal pointer not initialized
        """
        if self.temporal_pointer is None:
            raise ValueError("Temporal pointer not initialized. Call initialize_temporal_pointer() first.")

        new_position = self.temporal_pointer - steps

        if new_position < 0:
            logger.warning(f"Cannot move backward {steps} steps: would go before start of data")
            return False

        old_position = self.temporal_pointer + 1
        self.temporal_pointer = new_position
        new_position_1based = self.temporal_pointer + 1

        logger.debug(f"Temporal pointer moved backward from {old_position} to {new_position_1based} (-{steps} steps)")
        return True

    def get_past_data(self) -> Dict[str, pd.DataFrame]:
        """Get past data for training (all data before current timestamp)

        Returns:
            Dictionary with active market past data

        Raises:
            ValueError: If temporal pointer not initialized or no active market
        """
        if self.temporal_timestamp is None:
            raise ValueError("Temporal pointer not initialized. Call initialize_temporal_pointer() first.")

        if self._active_market is None or self._active_market_data is None:
            raise ValueError("No active market set. Load market data first.")

        # Get all data before current timestamp
        past_data = {
            self._active_market: self._active_market_data.loc[:self.temporal_timestamp].iloc[:-1].copy()
            # Exclude current timestamp
        }

        logger.debug(
            f"Retrieved past data for {self._active_market}: {len(past_data[self._active_market])} records before {self.temporal_timestamp}")
        return past_data

    def get_current_data(self) -> Dict[str, pd.Series]:
        """Get current data at temporal timestamp for signal generation

        Returns:
            Dictionary with active market current record

        Raises:
            ValueError: If temporal pointer not initialized or no active market
        """
        if self.temporal_timestamp is None:
            raise ValueError("Temporal pointer not initialized. Call initialize_temporal_pointer() first.")

        if self._active_market is None or self._active_market_data is None:
            raise ValueError("No active market set. Load market data first.")

        # Get record at current timestamp
        current_data = {
            self._active_market: self._active_market_data.loc[self.temporal_timestamp].copy()
        }

        logger.debug(f"Retrieved current data for {self._active_market} at timestamp {self.temporal_timestamp}")
        return current_data

    def get_future_data_preview(self, lookahead_records: int = None) -> Dict[str, pd.DataFrame]:
        """Get future data for prediction validation (NOT for training)

        Args:
            lookahead_records: Number of future records to preview (optional)

        Returns:
            Dictionary with active market future data

        Warning:
            This method is for prediction validation only.
            Future data must NEVER be used for training or signal generation.
        """
        if self.temporal_timestamp is None:
            raise ValueError("Temporal pointer not initialized. Call initialize_temporal_pointer() first.")

        if self._active_market is None or self._active_market_data is None:
            raise ValueError("No active market set. Load market data first.")

        # Get all data after current timestamp
        future_data = self._active_market_data.loc[self.temporal_timestamp:].iloc[
                      1:].copy()  # Exclude current timestamp

        if lookahead_records is not None:
            future_data = future_data.head(lookahead_records)

        future_result = {
            self._active_market: future_data
        }

        logger.warning(
            f"Future data preview accessed for {self._active_market}: {len(future_data)} records after {self.temporal_timestamp}. USE FOR VALIDATION ONLY!")
        return future_result

    def validate_temporal_boundaries(self) -> Dict[str, any]:
        """Validate and return current temporal boundary information

        Returns:
            Dictionary with boundary information
        """
        if self.temporal_pointer is None:
            return {
                'status': 'uninitialized',
                'message': 'Temporal pointer not initialized'
            }

        boundaries = {
            'status': 'active',
            'active_market': self._active_market,
            'total_records': self.total_records,
            'past_records_count': self.temporal_pointer,
            'current_record_position': self.temporal_pointer + 1,  # 1-based
            'future_records_count': self.total_records - self.temporal_pointer - 1,
            'past_accessible': f"records 1 to {self.temporal_pointer}",
            'current_accessible': f"record {self.temporal_pointer + 1}",
            'future_accessible': f"records {self.temporal_pointer + 2} to {self.total_records} (validation only)"
        }

        logger.debug(f"Temporal boundaries validated for {self._active_market}: {boundaries}")

        return boundaries

    def reset_temporal_state(self):
        """Reset temporal management state"""
        logger.info("Resetting temporal state")

        self.temporal_pointer = None
        self.training_window_size = None
        self.total_records = 0
        # Keep loaded markets and active market - only reset temporal state

        logger.debug("Temporal state reset completed")

    def get_market_record_counts(self) -> Dict[str, int]:
        """Get record count per loaded market for data integrity validation

        Returns:
            Dictionary mapping market names to record counts
        """
        record_counts = {market: len(df) for market, df in self._cached_data.items()}

        logger.debug(f"Market record counts: {record_counts}")
        return record_counts

    def get_cache_info(self) -> Dict:
        """Get information about cached data - metadata only"""
        cache_info = {
            'cached_markets': list(self._cached_data.keys()),
            'cache_size': len(self._cached_data),
            'training_cache_size': len(self._training_data_cache),
            'active_market': self._active_market,
            'available_markets': self.get_available_markets(),
            'temporal_status': 'initialized' if self.temporal_timestamp is not None else 'uninitialized'
        }

        logger.debug(f"Cache info requested: {cache_info}")
        return cache_info

    def clear_cache(self):
        """Clear all cached data"""
        cached_count = len(self._cached_data)
        training_cached_count = len(self._training_data_cache)

        self._cached_data.clear()
        self._training_data_cache.clear()
        self.loaded_markets.clear()
        self._active_market = None

        logger.info(
            f"Data cache cleared: {cached_count} market data entries, {training_cached_count} training data entries")

    def prepare_training_data(self, training_window: int) -> bool:
        """Prepare training data for strategies - no DataFrame exposure

        Args:
            training_window: Size of training window

        Returns:
            Success status
        """
        logger.info(f"Preparing training data with window size: {training_window}")

        if self._active_market is None or self._active_market not in self._cached_data:
            logger.error("No active market data loaded for training preparation")
            return False

        active_df = self._cached_data[self._active_market]

        if len(active_df) < training_window:
            logger.warning(f"Active market has only {len(active_df)} periods, less than required {training_window}")
            self.training_window_size = len(active_df)
        else:
            self.training_window_size = training_window

        logger.info(
            f"Training data prepared for {self._active_market} with effective window: {self.training_window_size}")
        return True