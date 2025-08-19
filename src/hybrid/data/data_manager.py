# data_manager.py
# DataManager for multi-market data loading and coordination
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime
from src.hybrid.load_data import load_and_preprocess_data

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading, preparation, and caching for multiple strategies"""

    def __init__(self, config):
        self.config = config
        self.cached_data = {}
        self.training_data_cache = {}

    def load_market_data(self, markets: List[str]) -> Dict[str, pd.DataFrame]:
        """Load data for multiple markets"""
        logger.info(f"Loading data for {len(markets)} markets")

        market_data = {}
        for market in markets:
            market_data[market] = self.load_market_data_single(market)

        logger.info(f"Successfully loaded data for all {len(markets)} markets")
        return market_data

    def load_market_data_single(self, market: str) -> pd.DataFrame:
        """Load and preprocess data for a single market"""
        data_start = datetime.now()
        logger.debug(f"Loading and preprocessing data for market: {market}")

        # Check cache first
        if market in self.cached_data:
            logger.debug(f"Using cached data for market: {market}")
            return self.cached_data[market]

        # Determine data path
        data_path = self._get_market_data_path(market)

        # Load and preprocess data
        df = load_and_preprocess_data(data_path, self.config)
        data_duration = (datetime.now() - data_start).total_seconds()

        # Cache the data
        self.cached_data[market] = df

        logger.info(f"Market {market}: {len(df)} records loaded from {df.index[0]} to {df.index[-1]}")
        logger.debug(f"Data loading for {market} took: {data_duration:.1f} seconds")

        return df

    def _get_market_data_path(self, market: str) -> str:
        """Get data path for a specific market"""
        if market is None:
            data_config = self.config.get_section('data_loading', {})
            return data_config.get('data_source', 'data/eurusd')

        # If market is a full path, use as-is
        if '/' in market or '\\' in market:
            return market

        # Otherwise, construct path from base data source
        data_config = self.config.get_section('data_loading', {})
        base_data_source = data_config.get('data_source', 'data')

        # Remove any existing market name from base path
        if base_data_source.endswith('/eurusd') or base_data_source.endswith('\\eurusd'):
            base_data_source = base_data_source[:-7]  # Remove '/eurusd'

        return f"{base_data_source}/{market.lower()}"