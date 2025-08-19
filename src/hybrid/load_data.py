# load_data.py
# Modern data loading and preprocessing functionality

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional
from src.hybrid.config.unified_config import UnifiedConfig, get_config

logger = logging.getLogger(__name__)


class DataLoader:
    """Modern data loader with validation and preprocessing"""

    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()

    def load_all_csv(self, data_dir: Path, validate: bool = True) -> pd.DataFrame:
        """Load and combine all CSV files with comprehensive validation"""

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")

        column_names = ["datetime", "open", "high", "low", "close", "volume"]
        frames = []

        # Get file pattern from config
        file_pattern = self.config.config.get('data_loading', {}).get('file_pattern', '*.csv')
        csv_files = list(data_dir.glob(file_pattern))

        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir} matching pattern {file_pattern}")

        logger.info(f"Loading {len(csv_files)} CSV files...")

        # Use verbose setting from general config
        show_progress = self.config.config.get('general', {}).get('verbose', True)
        progress_bar = tqdm(sorted(csv_files), desc="Loading CSV files") if show_progress else sorted(csv_files)

        for csv_path in progress_bar:
            try:
                df = pd.read_csv(csv_path, sep=";", header=None, names=column_names, dtype=str)
                df["timestamp"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S", errors="coerce")

                # Convert price columns to numeric
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Data validation
                if validate:
                    df = self._validate_ohlc_data(df)

                if df.empty or df["timestamp"].isna().all():
                    logger.warning(f"Skipping {csv_path.name} - invalid data")
                    continue

                frames.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_path.name}: {e}")
                continue

        if not frames:
            raise ValueError("No valid data files could be loaded")

        # Combine and clean data
        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.drop_duplicates(subset="timestamp")
        df_all = df_all.sort_values("timestamp").dropna(subset=["timestamp"])
        df_all = df_all.set_index("timestamp")

        # Remove obvious outliers (always enabled for data quality)
        df_all = self._remove_outliers(df_all)

        logger.info(f"Loaded {len(df_all)} records from {df_all.index[0]} to {df_all.index[-1]}")
        return df_all[["open", "high", "low", "close", "volume"]]

    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC data consistency"""
        # Use reasonable defaults for minimum price threshold
        min_price_threshold = 0.0001

        # Remove rows where high < low or close/open outside high-low range
        valid_mask = (
                (df["high"] >= df["low"]) &
                (df["high"] >= df["close"]) &
                (df["high"] >= df["open"]) &
                (df["low"] <= df["close"]) &
                (df["low"] <= df["open"]) &
                (df["open"] > min_price_threshold) &
                (df["high"] > min_price_threshold) &
                (df["low"] > min_price_threshold) &
                (df["close"] > min_price_threshold)
        )

        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} invalid OHLC rows")

        return df[valid_mask]

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove price outliers using z-score"""
        outlier_threshold = 5.0  # Conservative threshold
        price_cols = ["open", "high", "low", "close"]

        for col in price_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > outlier_threshold
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    logger.warning(f"Removed {outlier_count} outliers from {col}")
                    df = df[~outlier_mask]

        return df

    def create_sample_data(self, n_points: int = None, save_to: Optional[Path] = None) -> pd.DataFrame:
        """Create sample forex-like data for testing"""

        if n_points is None:
            # Use max_records from config, default to reasonable size
            n_points = self.config.config.get('data_loading', {}).get('max_records', 100000)

        logger.info(f"Creating sample data with {n_points} points...")

        # Use random state from general config
        random_state = self.config.config.get('general', {}).get('random_state', 42)
        np.random.seed(random_state)

        # Start with base price
        base_price = 1.1000  # EUR/USD-like

        # Generate price series with some trending and ranging periods
        returns = []
        for i in range(n_points):
            # Add regime changes
            if i % 1000 == 0:  # Change regime every 1000 points
                trend_strength = np.random.choice([-0.0001, 0, 0.0001], p=[0.3, 0.4, 0.3])

            # Add noise with occasional jumps
            noise = np.random.normal(0, 0.0002)
            if np.random.random() < 0.01:  # 1% chance of larger move
                noise *= 5

            daily_return = trend_strength + noise
            returns.append(daily_return)

        # Create price series
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices[1:])  # Remove first element

        # Create OHLC data
        data = []
        for i, close in enumerate(prices):
            # Generate realistic OHLC from close price
            volatility = 0.0005  # Typical forex daily volatility

            open_price = close * (1 + np.random.normal(0, volatility / 4))
            high_price = max(open_price, close) * (1 + abs(np.random.normal(0, volatility / 2)))
            low_price = min(open_price, close) * (1 - abs(np.random.normal(0, volatility / 2)))
            volume = np.random.randint(1000, 10000)  # Random volume

            # Create timestamp
            timestamp = pd.Timestamp('2020-01-01') + pd.Timedelta(minutes=i)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        if save_to:
            save_to.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_to / 'sample_eurusd_data.csv')
            logger.info(f"Sample data saved to {save_to / 'sample_eurusd_data.csv'}")

        return df

    def load_data_with_fallback(self, data_path: str = None) -> pd.DataFrame:
        """
        Load data with automatic fallback to sample data if real data not found
        """

        if data_path is None:
            data_path = self.config.config.get('data_loading', {}).get('data_source', 'data/eurusd')

        # Resolve paths relative to project root (where backtest.py is)
        if not Path(data_path).is_absolute():
            # Get project root: go up 2 levels from src/hybrid/ to root
            project_root = Path(__file__).parent.parent.parent
            path_obj = project_root / data_path
        else:
            path_obj = Path(data_path)

        # Try to load real data first
        file_pattern = self.config.config.get('data_loading', {}).get('file_pattern', '*.csv')
        if path_obj.exists() and list(path_obj.glob(file_pattern)):
            try:
                df = self.load_all_csv(path_obj)
                logger.info(f"Successfully loaded {len(df)} records from {data_path}")

                # Apply sample size limit if configured
                max_records = self.config.config.get('data_loading', {}).get('max_records')
                if max_records and len(df) > max_records:
                    logger.info(f"Using last {max_records} records as configured")
                    df = df.tail(max_records)

                # Skip initial rows if configured (for optimization holdout)
                skip_initial = self.config.config.get('data_loading', {}).get('skip_initial_rows')
                if skip_initial and len(df) > skip_initial:
                    logger.info(f"Skipping first {skip_initial} rows as configured")
                    df = df.iloc[skip_initial:]

                return df

            except Exception as e:
                logger.error(f"Error loading data from {data_path}: {e}")
                logger.info("Falling back to sample data...")

        # Fallback to sample data
        logger.info("Creating sample data...")
        path_obj.mkdir(parents=True, exist_ok=True)
        df = self.create_sample_data(save_to=path_obj)
        logger.info(f"Sample data created with {len(df)} records")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply additional preprocessing steps
        """

        logger.info("Preprocessing data...")

        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add volume column if missing (for forex data)
        if 'volume' not in df.columns:
            df['volume'] = 1000  # Default volume for forex

        # Sort by timestamp
        df = df.sort_index()

        # Remove any remaining NaN values
        initial_length = len(df)
        df = df.dropna()
        removed_rows = initial_length - len(df)

        if removed_rows > 0:
            logger.warning(f"Removed {removed_rows} rows with NaN values")

        # Validate data integrity
        if len(df) == 0:
            raise ValueError("No valid data remaining after preprocessing")

        # Check for minimum data requirements
        min_data_periods = self.config.config.get('regime_detection', {}).get('min_data_periods', 240)
        vol_window = self.config.config.get('volatility_prediction', {}).get('feature_periods', [20])[-1]
        min_required = max(min_data_periods, vol_window) + 100

        if len(df) < min_required:
            logger.warning(f"Data length ({len(df)}) is less than recommended minimum ({min_required})")

        logger.info(f"Preprocessing complete. Final data shape: {df.shape}")
        return df


def load_and_preprocess_data(data_path: str = None, config: UnifiedConfig = None) -> pd.DataFrame:
    """
    Convenience function to load and preprocess data in one step
    """

    loader = DataLoader(config)
    df = loader.load_data_with_fallback(data_path)
    df = loader.preprocess_data(df)

    return df