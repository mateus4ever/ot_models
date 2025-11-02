# data_loader.py
# Abstract base class and concrete implementations for DataManager data loading strategies
# Supports extensible data source architecture with Strategy pattern

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging
import pandas as pd
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loading strategies

    Defines the contract for all data source implementations.
    Each concrete loader handles a specific data source type (files, database, streams, etc.)
    """

    def __init__(self, config):
        """Initialize data loader with configuration

        Args:
            config: Configuration object containing loader-specific settings
        """
        self.config = config
        logger.debug(f"Initializing {self.__class__.__name__}")

    @abstractmethod
    def load(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from the configured source

        Args:
            source_config: Source-specific configuration parameters

        Returns:
            Dictionary containing loaded market data:
            {
                'market_id': consolidated_dataframe,
                'market_id2': consolidated_dataframe2,
                ...
            }

        Raises:
            ValueError: If source configuration is invalid
            FileNotFoundError: If data source is not accessible
            ConnectionError: If remote data source is unavailable
        """
        pass

    def validate_config(self, source_config: Dict[str, Any]) -> bool:
        """Validate source configuration parameters

        Args:
            source_config: Configuration to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid with details
        """
        if not isinstance(source_config, dict):
            raise ValueError("Source configuration must be a dictionary")

        required_fields = self.get_required_config_fields()
        missing_fields = [field for field in required_fields if field not in source_config]

        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")

        return True

    def get_loader_info(self) -> Dict[str, Any]:
        """Return information about this loader implementation

        Returns:
            Dictionary with loader metadata
        """
        return {
            'loader_type': self.__class__.__name__,
            'supported_sources': self.get_supported_sources(),
            'required_config': self.get_required_config_fields()
        }


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

    def _load_csv_file(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with proper parsing using configured format

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        logger.debug(f"Starting CSV file load: {file_path}")

        csv_format = self._get_csv_format_config()

        try:
            df = self._read_csv_with_config(file_path, csv_format)
            df = self._normalize_dataframe(df, csv_format)

            logger.info(f"Successfully loaded and processed {len(df)} rows from {file_path}")
            logger.debug(f"Data date range: {df.index[0]} to {df.index[-1]}")
            logger.debug(f"Data columns: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            logger.debug(f"CSV loading error details", exc_info=True)
            raise

    def _get_csv_format_config(self) -> dict:
        """Get and validate CSV format configuration

        Returns:
            CSV format configuration dictionary

        Raises:
            ValueError: If required configuration is missing
        """
        csv_format = self.config.get('csv_format')
        if not csv_format:
            raise ValueError("Missing 'csv_format' section in configuration")

        # Validate required fields
        delimiter = csv_format.get('delimiter')
        has_header = csv_format.get('has_header')
        timestamp_column = csv_format.get('timestamp_column')

        if delimiter is None:
            raise ValueError("Missing 'delimiter' in csv_format configuration")
        if has_header is None:
            raise ValueError("Missing 'has_header' in csv_format configuration")
        if timestamp_column is None:
            raise ValueError("Missing 'timestamp_column' in csv_format configuration")

        return csv_format

    def _read_csv_with_config(self, file_path: str, csv_format: dict) -> pd.DataFrame:
        """Read CSV file using configuration parameters

        Args:
            file_path: Path to CSV file
            csv_format: CSV format configuration

        Returns:
            Raw DataFrame from CSV
        """
        delimiter = csv_format.get('delimiter')
        has_header = csv_format.get('has_header')
        column_names = csv_format.get('column_names')

        # Build pd.read_csv parameters dynamically
        read_params = {'delimiter': delimiter}

        if has_header:
            read_params['header'] = 0
        else:
            read_params['header'] = None
            if column_names:
                read_params['names'] = column_names
            else:
                raise ValueError("'column_names' required when has_header=false")

        logger.debug(f"Loading CSV with delimiter='{delimiter}', has_header={has_header}")
        df = pd.read_csv(file_path, **read_params)
        logger.debug(f"Raw CSV loaded: {len(df)} rows, {len(df.columns)} columns")

        return df

    def _normalize_dataframe(self, df: pd.DataFrame, csv_format: dict) -> pd.DataFrame:
        """Normalize dataframe columns and set timestamp index

        Args:
            df: Raw dataframe
            csv_format: CSV format configuration

        Returns:
            Normalized dataframe with datetime index
        """
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        logger.debug(f"Normalized columns: {list(df.columns)}")

        # Convert timestamp and set as index
        df = self._set_timestamp_index(df, csv_format)

        # Sort by timestamp
        df.sort_index(inplace=True)
        logger.debug("Sorted data by timestamp")

        return df

    def _set_timestamp_index(self, df: pd.DataFrame, csv_format: dict) -> pd.DataFrame:
        """Convert timestamp column to datetime and set as index

        Args:
            df: Dataframe with normalized columns
            csv_format: CSV format configuration

        Returns:
            Dataframe with datetime index
        """
        timestamp_column = csv_format.get('timestamp_column').lower()
        timestamp_format = csv_format.get('timestamp_format')

        if timestamp_column not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        logger.debug(f"Converting '{timestamp_column}' to datetime index")

        # Convert to datetime
        if timestamp_format:
            if timestamp_format == "ISO8601":
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            else:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column], format=timestamp_format)
        else:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Set as index
        df.set_index(timestamp_column, inplace=True)

        return df
    def _consolidate_market_data(self, market_id: str, file_paths: List[str]) -> pd.DataFrame:
        """Consolidate multiple CSV files into single market dataset

        Args:
            market_id: Market identifier for logging
            file_paths: List of file paths to consolidate

        Returns:
            Consolidated DataFrame with chronological ordering
        """
        logger.debug(f"Consolidating {len(file_paths)} files for market {market_id}")

        consolidated_chunks = []

        for file_path in file_paths:
            logger.debug(f"Loading chunk: {Path(file_path).name}")
            chunk_df = self._load_csv_file(file_path)
            consolidated_chunks.append(chunk_df)

        # Consolidate all chunks chronologically
        consolidated_df = pd.concat(consolidated_chunks, ignore_index=False)
        consolidated_df.sort_index(inplace=True)

        # Remove any duplicate timestamps
        consolidated_df = consolidated_df[~consolidated_df.index.duplicated(keep='first')]

        logger.info(
            f"Market {market_id} consolidated: {len(consolidated_df)} total records from {len(file_paths)} files")
        logger.debug(f"Consolidated date range: {consolidated_df.index[0]} to {consolidated_df.index[-1]}")

        return consolidated_df


class FilePathLoader(DataLoader):
    """Concrete loader for direct file path loading

    Loads market data from explicitly provided file paths.
    No path discovery - uses files exactly as specified.
    """

    def get_required_config_fields(self) -> List[str]:
        """Required configuration fields for file path loading"""
        return ['file_paths']

    def get_supported_sources(self) -> List[str]:
        """Supported source types"""
        return ['file_paths', 'direct_files']

    def load(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load market data from direct file paths

        Args:
            source_config: Must contain 'file_paths' key with list of file paths

        Returns:
            Dictionary mapping market IDs to consolidated DataFrames
        """
        self.validate_config(source_config)

        file_paths = source_config['file_paths']
        logger.info(f"Loading {len(file_paths)} files via direct paths")

        # Group files by market identifier
        market_groups = {}
        for file_path in file_paths:
            filename = Path(file_path).name
            market_id = self._extract_market_from_filename(filename)

            if market_id not in market_groups:
                market_groups[market_id] = []
            market_groups[market_id].append(file_path)

        logger.info(f"Detected {len(market_groups)} unique markets: {list(market_groups.keys())}")

        # Load and consolidate each market
        market_data = {}
        for market_id, paths in market_groups.items():
            try:
                consolidated_df = self._consolidate_market_data(market_id, paths)
                market_data[market_id] = consolidated_df
                logger.info(f"Successfully loaded market {market_id}")
            except Exception as e:
                logger.error(f"Failed to load market {market_id}: {e}")
                raise

        return market_data


class FileDiscoveryLoader(DataLoader):
    """Concrete loader for recursive file discovery

    Searches for files by name in configured base directory.
    Performs recursive search through all subdirectories.
    """

    def get_required_config_fields(self) -> List[str]:
        """Required configuration fields for file discovery"""
        return ['filenames']

    def get_supported_sources(self) -> List[str]:
        """Supported source types"""
        return ['file_discovery', 'recursive_search']

    def load(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load market data via recursive file discovery

        Args:
            source_config: Must contain 'filenames' key with list of filenames to find

        Returns:
            Dictionary mapping market IDs to consolidated DataFrames
        """
        self.validate_config(source_config)

        filenames = source_config['filenames']
        logger.info(f"Discovering {len(filenames)} files via recursive search")

        # Get base directory from config
        base_directory = self._get_base_directory()

        # Find file paths through recursive search
        file_paths = []
        for filename in filenames:
            found_path = self._find_file_recursively(base_directory, filename)
            file_paths.append(found_path)

        # Group files by market identifier
        market_groups = {}
        for file_path in file_paths:
            filename = Path(file_path).name
            market_id = self._extract_market_from_filename(filename)

            if market_id not in market_groups:
                market_groups[market_id] = []
            market_groups[market_id].append(file_path)

        logger.info(f"Detected {len(market_groups)} unique markets: {list(market_groups.keys())}")

        # Load and consolidate each market
        market_data = {}
        for market_id, paths in market_groups.items():
            try:
                consolidated_df = self._consolidate_market_data(market_id, paths)
                market_data[market_id] = consolidated_df
                logger.info(f"Successfully loaded market {market_id}")
            except Exception as e:
                logger.error(f"Failed to load market {market_id}: {e}")
                raise

        return market_data

    def _get_base_directory(self) -> Path:
        """Get base directory for file search from configuration

        Returns:
            Path object pointing to search root directory
        """
        data_config = self.config.get_section("data_loading", {})
        base_directory = data_config.get("data_source")

        if not base_directory:
            raise ValueError("No base directory specified in config[data_loading][data_source]")

        base_path = Path(base_directory)

        if base_path.is_absolute():
            search_root = base_path
        else:
            # Resolve relative to project root (4 levels up from this file)
            project_root = Path(__file__).resolve().parents[3]
            # Clean "src/" prefix if present
            clean_base = (
                base_path.parts[1:] if base_path.parts[0] in ('src',)
                else base_path.parts
            )
            search_root = project_root.joinpath(*clean_base)

        logger.debug(f"Using search root: {search_root}")
        return search_root

    def _find_file_recursively(self, search_root: Path, filename: str) -> str:
        """Find file recursively under search root

        Args:
            search_root: Directory to search under
            filename: Name of file to find

        Returns:
            Full path to found file

        Raises:
            FileNotFoundError: If file not found
        """
        # Ensure .csv extension
        search_filename = f"{filename}.csv" if not filename.endswith(".csv") else filename

        # Search recursively
        matches = list(search_root.rglob(search_filename))

        if not matches:
            raise FileNotFoundError(f"File '{search_filename}' not found under {search_root}")

        if len(matches) > 1:
            logger.warning(f"Multiple files named {search_filename} found, using first: {matches[0]}")

        resolved_path = matches[0].resolve()
        logger.debug(f"Found file: {resolved_path}")
        return str(resolved_path)


class DirectoryScanner(DataLoader):
    """Concrete loader for automatic directory scanning

    Scans a directory for all CSV files and loads them automatically.
    No need to specify filenames - discovers whatever CSV files exist.
    Supports recursive scanning of subdirectories.
    """

    def get_required_config_fields(self) -> List[str]:
        """Required configuration fields for directory scanning"""
        return ['directory_path']

    def get_supported_sources(self) -> List[str]:
        """Supported source types"""
        return ['directory_scan', 'auto_discovery', 'folder_scan']

    def load(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load market data by scanning directory for all CSV files

        Args:
            source_config: Must contain:
                - 'directory_path': Path to directory to scan
                - 'recursive': Optional, whether to scan subdirectories (default: True)
                - 'file_pattern': Optional, file pattern to match (default: '*.csv')

        Returns:
            Dictionary mapping market IDs to consolidated DataFrames
        """
        self.validate_config(source_config)

        directory_path = Path(source_config['directory_path'])
        recursive = source_config.get('recursive', True)
        file_pattern = source_config.get('file_pattern', '*.csv')

        logger.info(f"Scanning directory {directory_path} for {file_pattern} files (recursive: {recursive})")

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        # Find all CSV files
        if recursive:
            csv_files = list(directory_path.rglob(file_pattern))
        else:
            csv_files = list(directory_path.glob(file_pattern))

        if not csv_files:
            raise FileNotFoundError(f"No {file_pattern} files found in {directory_path} (recursive: {recursive})")

        logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

        # Convert to string paths for processing
        file_paths = [str(f) for f in csv_files]

        # Group files by market identifier
        market_groups = {}
        for file_path in file_paths:
            filename = Path(file_path).name
            market_id = self._extract_market_from_filename(filename)

            if market_id not in market_groups:
                market_groups[market_id] = []
            market_groups[market_id].append(file_path)

        logger.info(f"Detected {len(market_groups)} unique markets: {list(market_groups.keys())}")

        # Load and consolidate each market
        market_data = {}
        for market_id, paths in market_groups.items():
            try:
                consolidated_df = self._consolidate_market_data(market_id, paths)
                market_data[market_id] = consolidated_df
                logger.info(f"Successfully loaded market {market_id} from {len(paths)} files")
            except Exception as e:
                logger.error(f"Failed to load market {market_id}: {e}")
                raise

        return market_data
