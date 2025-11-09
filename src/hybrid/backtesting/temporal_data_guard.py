# src/hybrid/backtesting/temporal_data_guard.py
"""
TemporalDataGuard - Enforces strict temporal boundaries to prevent look-ahead bias

Critical for walk-forward optimization and any time-series backtesting where
training/optimization must only use data available at that point in time.
"""

import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TemporalDataGuard:
    """
    Enforces strict temporal boundaries - NO FUTURE DATA ALLOWED

    Prevents look-ahead bias by ensuring training data only includes
    information that would have been available at that point in time.

    Used in walk-forward optimization to ensure model training doesn't
    accidentally use future data.
    """

    def __init__(self, config: Dict):
        """
        Initialize temporal guard with configuration

        Args:
            config: Configuration dictionary containing walk_forward section
        """
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache temporal guard configuration values"""
        walk_forward_config = self.config.get('walk_forward', {})

        # Training window parameters
        self.pretrain_rows = walk_forward_config.get('pretrain_rows', 252)  # Initial training period
        self.min_training_rows = walk_forward_config.get('min_training_rows', 100)  # Minimum data required
        self.training_lookback_limit = walk_forward_config.get('training_lookback_limit', None)  # Max lookback

        logger.info(f"TemporalDataGuard initialized: pretrain={self.pretrain_rows}, "
                    f"min_training={self.min_training_rows}, lookback_limit={self.training_lookback_limit}")

    def get_training_data(
            self,
            full_df: pd.DataFrame,
            current_step: int
    ) -> pd.DataFrame:
        """
        Get valid training data up to current step - STRICT TEMPORAL BOUNDARY

        Returns only data that would have been available at the current point
        in the backtest. Never includes future data.

        Args:
            full_df: Complete dataset
            current_step: Current position in backtest (steps after pretrain period)

        Returns:
            Training data respecting temporal boundaries

        Raises:
            ValueError: If insufficient training data available
        """
        # Calculate absolute position in dataset
        absolute_position = self.pretrain_rows + current_step

        if absolute_position > len(full_df):
            raise ValueError(f"Current position {absolute_position} exceeds data length {len(full_df)}")

        # HARD CUTOFF: Only data up to current position
        available_data = full_df.iloc[:absolute_position]

        # Apply lookback limit if configured (rolling window)
        if self.training_lookback_limit is not None:
            lookback_start = max(0, absolute_position - self.training_lookback_limit)
            training_data = full_df.iloc[lookback_start:absolute_position]
            logger.debug(f"Applied lookback limit: using rows {lookback_start} to {absolute_position}")
        else:
            # Use all available data (expanding window)
            training_data = available_data

        # Validate minimum training data requirement
        if len(training_data) < self.min_training_rows:
            raise ValueError(
                f"Insufficient training data at step {current_step}: "
                f"{len(training_data)} rows < minimum {self.min_training_rows}"
            )

        logger.debug(f"Training data retrieved: {len(training_data)} rows ending at position {absolute_position}")
        return training_data

    def validate_no_future_data(
            self,
            training_data: pd.DataFrame,
            current_step: int,
            full_df: pd.DataFrame
    ) -> bool:
        """
        Validate that no future data leaked into training set

        Performs sanity check to ensure temporal boundaries are respected.

        Args:
            training_data: Data used for training
            current_step: Current position in backtest
            full_df: Complete dataset

        Returns:
            True if validation passes

        Raises:
            ValueError: If future data detected in training set
        """
        # Calculate current date/position
        absolute_position = self.pretrain_rows + current_step

        if absolute_position >= len(full_df):
            raise ValueError(f"Invalid position {absolute_position} for validation")

        current_date = full_df.index[absolute_position]

        # Check for any data after current date
        future_data_mask = training_data.index > current_date

        if future_data_mask.any():
            future_count = future_data_mask.sum()
            future_dates = training_data.index[future_data_mask]

            raise ValueError(
                f"TEMPORAL VIOLATION: {future_count} future data points in training at step {current_step}\n"
                f"Current date: {current_date}\n"
                f"Future dates found: {future_dates.tolist()[:5]}..."  # Show first 5
            )

        logger.debug(f"Temporal validation passed at step {current_step}")
        return True

    def get_test_data(
            self,
            full_df: pd.DataFrame,
            current_step: int,
            test_window_size: int = 1
    ) -> pd.DataFrame:
        """
        Get test data for current step

        Returns the next N rows after current training position for testing.

        Args:
            full_df: Complete dataset
            current_step: Current position in backtest
            test_window_size: Number of rows to include in test set

        Returns:
            Test data for evaluation
        """
        absolute_position = self.pretrain_rows + current_step
        test_end = absolute_position + test_window_size

        if test_end > len(full_df):
            test_end = len(full_df)

        test_data = full_df.iloc[absolute_position:test_end]

        logger.debug(f"Test data retrieved: {len(test_data)} rows from position {absolute_position}")
        return test_data

    def split_train_test(
            self,
            full_df: pd.DataFrame,
            train_size: int,
            test_size: int,
            start_position: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets with temporal awareness

        Args:
            full_df: Complete dataset
            train_size: Number of rows for training
            test_size: Number of rows for testing
            start_position: Starting position in dataset

        Returns:
            Tuple of (train_df, test_df)
        """
        train_end = start_position + train_size
        test_end = train_end + test_size

        if test_end > len(full_df):
            raise ValueError(f"Split extends beyond data: {test_end} > {len(full_df)}")

        train_data = full_df.iloc[start_position:train_end]
        test_data = full_df.iloc[train_end:test_end]

        # Validate split
        if train_data.index[-1] >= test_data.index[0]:
            raise ValueError("Temporal violation: training data overlaps with test data")

        logger.info(f"Split data: train={len(train_data)} rows, test={len(test_data)} rows")
        return train_data, test_data