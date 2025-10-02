# data_manager.py
# DataManager for multi-market data loading and coordination
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
# ONLY LOADS REAL FILES - NO SAMPLE DATA GENERATION
# PROPER LOGGING THROUGHOUT
# DATA CONTROLLER - NO DATAFRAME EXPOSURE
# MARKET CONSOLIDATION - Append chunked files into single market datasets
# STRATEGY PATTERN INTEGRATION - Supports both file paths and directory discovery

import pandas as pd
import logging
import re
from typing import Dict, List, Optional, Union
from pathlib import Path

from .data_loader import DataLoader, FilePathLoader, FileDiscoveryLoader, DirectoryScanner
from .trade_history import TradeHistory

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading, preparation, and caching for multiple strategies

    Acts as data controller with temporal boundary enforcement.
    Consolidates chunked files into single market datasets.
    Does not expose raw DataFrames to prevent data leakage.

    Supports flexible loading via Strategy pattern:
    - Direct file paths (List[str] or Dict with file_paths)
    - Directory discovery (Dict with directory_path)
    """

    def __init__(self, config):
        self.config = config
        self._cached_data = {}  # Private - no external access {market_name: consolidated_dataframe}
        self._training_data_cache = {}  # Private - no external access

        # Global temporal management - single time pointer for active market
        self.temporal_pointer = None
        self.temporal_timestamp = None  # Timestamp-based pointer
        self._active_market_index = None  # Cached index for performance
        self.training_window_size = None
        self.total_records = 0
        self.loaded_markets = set()  # Track which markets are loaded
        self._active_market = None  # Currently selected market for temporal operations

        # Get data_loading config section
        data_config = config.get_section('data_loading', {})

        self._loader_registry = {
            'filepath': FilePathLoader(data_config),
            'discovery': FileDiscoveryLoader(data_config),
            'directory_scan': DirectoryScanner(data_config)
        }

        self.trade_history = TradeHistory(config)

        logger.info("Initializing DataManager as data controller with Strategy pattern support")
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

    def load_market_data(self, source: Union[List[str], Dict, str]) -> bool:
        """Load market data using Strategy pattern or legacy interface

        Supports multiple input formats:
        1. List[str] - Legacy: list of filenames (backward compatibility)
        2. str - Single file path or directory path
        3. Dict - Strategy pattern source_config:
           - {'loader_type': 'filepath', 'file_paths': [path1, path2, ...]}
           - {'loader_type': 'discovery', 'directory_path': '/path/to/dir', 'file_pattern': '*.csv'}

        Args:
            source: Data source specification

        Returns:
            True if all markets loaded successfully
        """
        logger.info(f"Loading market data with source type: {type(source)}")

        # Convert input to standardized source_config
        source_config = self._normalize_source_input(source)
        logger.debug(f"Normalized source config: {source_config}")

        # Select appropriate loader
        loader_type = source_config.get('loader_type')
        if loader_type not in self._loader_registry:
            raise ValueError(f"Unknown loader type: {loader_type}. Available: {list(self._loader_registry.keys())}")

        loader = self._loader_registry[loader_type]

        try:
            # Load data using Strategy pattern
            market_data = loader.load(source_config)

            # Store loaded data in cache
            successful_loads = 0
            last_loaded_market = None

            for market_id, consolidated_df in market_data.items():
                self._cached_data[market_id] = consolidated_df
                self.loaded_markets.add(market_id)
                successful_loads += 1
                last_loaded_market = market_id
                logger.info(f"Successfully cached market {market_id}: {len(consolidated_df)} records")

            # Set last loaded market as active
            if last_loaded_market:
                self._active_market = last_loaded_market
                logger.info(f"Active market set to: {self._active_market}")

            all_loaded = successful_loads == len(market_data)
            logger.info(f"Market loading result: {successful_loads}/{len(market_data)} markets loaded")

            return all_loaded

        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return False

    def _normalize_source_input(self, source: Union[List[str], Dict, str]) -> Dict:
        """Convert various input formats to standardized source_config

        Decision logic:
        - Complete file path → FilePathLoader
        - Directory + filename → FileDiscoveryLoader
        - Directory only → DirectoryScanner

        Args:
            source: Input in various formats

        Returns:
            Standardized source_config dictionary
        """
        if isinstance(source, dict):
            # Already a source_config dictionary
            return source

        elif isinstance(source, str):
            # Single string - detect if complete file path or directory
            path = Path(source)

            if path.is_file() or (not path.is_dir() and ('.' in path.name)):
                # Complete file path (exists or has extension) → FilePathLoader
                return {
                    'loader_type': 'filepath',
                    'file_paths': [str(path)]
                }
            elif path.is_dir():
                # Directory only → DirectoryScanner
                return {
                    'loader_type': 'directory_scan',
                    'directory_path': str(path),
                    'recursive': True,
                    'file_pattern': '*.csv'
                }
            else:
                # Assume it's a filename to be discovered
                return {
                    'loader_type': 'discovery',
                    'filenames': [source]
                }

        elif isinstance(source, list):
            # List format - check first item to determine type
            if not source:
                raise ValueError("Empty source list provided")

            first_item = source[0]
            first_path = Path(first_item)

            # Check if items are complete file paths
            if (first_path.is_file() or
                    ('/' in first_item or '\\' in first_item) or
                    first_path.is_absolute() or
                    '.' in first_path.name):
                # Complete file paths → FilePathLoader
                return {
                    'loader_type': 'filepath',
                    'file_paths': source
                }
            else:
                # Just filenames → FileDiscoveryLoader
                return {
                    'loader_type': 'discovery',
                    'filenames': source
                }

        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def load_market_data_single(self, market_file: str) -> bool:
        """Load single market file (handles as single-file market)

        Args:
            market_file: Market file name or path

        Returns:
            True if loading successful
        """
        logger.debug(f"Loading single market file: {market_file}")
        return self.load_market_data([market_file])

    def load_market_data_with_temporal_setup(self, source: Union[List[str], Dict, str], training_window: int) -> bool:
        """Load market data and initialize temporal boundaries in one operation

        Args:
            source: Data source specification (same formats as load_market_data)
            training_window: Number of records for initial training window

        Returns:
            True if loading and temporal setup successful
        """
        logger.info(f"Loading data with temporal setup, training window: {training_window}")

        # Load and consolidate markets
        all_loaded = self.load_market_data(source)

        if all_loaded and self._active_market and self._active_market in self._cached_data:
            # Initialize temporal boundaries using active market data
            active_data = self._cached_data[self._active_market]
            self.initialize_temporal_pointer(active_data, training_window)

            logger.info(f"Temporal setup complete for {self._active_market}")
            return True
        else:
            logger.error("Failed to load markets or setup temporal boundaries")
            return False

    @property
    def _active_market_data(self) -> Optional[pd.DataFrame]:
        """Get active market data DataFrame (internal use only)"""
        if self._active_market and self._active_market in self._cached_data:
            return self._cached_data[self._active_market]
        return None

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
        logger.info(f"Active market switched from {old_active} to {market_id}")

    def get_available_markets(self) -> List[str]:
        """Return list of loaded markets available for selection

        Returns:
            List of market identifiers
        """
        return list(self._cached_data.keys())

    # =============================================================================
    # TEMPORAL BOUNDARY METHODS
    # =============================================================================

    def initialize_temporal_pointer(self, market_data: pd.DataFrame, training_window: int) -> int:
        """Initialize timestamp-based temporal pointer for walk-forward analysis

        Args:
            market_data: DataFrame with market data
            training_window: Number of records for initial training window

        Returns:
            Position of temporal pointer (1-based)
        """
        if len(market_data) <= training_window:
            raise ValueError(f"Market data has {len(market_data)} records, need at least {training_window + 1}")

        self.training_window_size = training_window
        self.total_records = len(market_data)

        # Set temporal pointer to timestamp at training window position
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

    def next(self, steps: int = 1) -> bool:
        """Advance temporal pointer by specified steps

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
        if self.temporal_timestamp is None:
            raise ValueError("Temporal pointer not initialized. Call initialize_temporal_pointer() first.")

        new_index = self._active_market_index - steps

        if new_index < 0:
            logger.warning(f"Cannot move backward {steps} steps: would go before start of data")
            return False

        old_timestamp = self.temporal_timestamp
        old_position = self._active_market_index + 1

        # Update to new timestamp and cache new index
        self.temporal_timestamp = self._active_market_data.index[new_index]
        self._active_market_index = new_index
        new_position = new_index + 1

        logger.debug(f"Temporal pointer moved backward from position {old_position} to {new_position} (-{steps} steps)")
        logger.debug(f"Timestamp changed from {old_timestamp} to {self.temporal_timestamp}")

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
        future_data = self._active_market_data.loc[self.temporal_timestamp:].iloc[1:].copy()

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
        if self.temporal_timestamp is None:
            return {
                'status': 'uninitialized',
                'message': 'Temporal pointer not initialized'
            }

        boundaries = {
            'status': 'active',
            'active_market': self._active_market,
            'total_records': self.total_records,
            'past_records_count': self._active_market_index,
            'current_record_position': self._active_market_index + 1,  # 1-based
            'future_records_count': self.total_records - self._active_market_index - 1,
            'past_accessible': f"records 1 to {self._active_market_index}",
            'current_accessible': f"record {self._active_market_index + 1}",
            'future_accessible': f"records {self._active_market_index + 2} to {self.total_records} (validation only)"
        }

        logger.debug(f"Temporal boundaries validated for {self._active_market}: {boundaries}")
        return boundaries

    def reset_temporal_state(self):
        """Reset temporal management state"""
        logger.info("Resetting temporal state")

        self.temporal_timestamp = None
        self.training_window_size = None
        self.total_records = 0
        self._active_market_index = None

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