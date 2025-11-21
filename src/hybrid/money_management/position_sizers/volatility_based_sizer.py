# position_sizers/volatility_based_sizer.py
import logging
import pandas as pd
import numpy as np
from .fixed_fractional_sizer import PositionSizingStrategy
from src.hybrid.positions.types import TradingSignal, PortfolioState

logger = logging.getLogger(__name__)


class VolatilityBasedSizer(PositionSizingStrategy):
    """Volatility-based position sizing - inverse relationship to volatility"""

    def __init__(self, config):
        """
        Initialize VolatilityBasedSizer with configuration

        Args:
            config: UnifiedConfig instance
        """
        super().__init__(config)
        self.target_volatility = self.config['target_volatility']
        self.volatility_lookback = self.config['volatility_lookback']
        self.base_risk = self.config['base_risk_per_trade']

    def calculate_size(self, signal: TradingSignal, portfolio: PortfolioState,
                       market_data: pd.DataFrame) -> int:
        # Calculate current volatility
        current_vol = self._calculate_volatility(market_data)

        if current_vol <= 0:
            current_vol = self.target_volatility

        # Volatility adjustment factor
        vol_adjustment = self.target_volatility / current_vol

        # Adjusted risk based on volatility
        adjusted_risk = self.base_risk * vol_adjustment
        adjusted_risk = max(self.config['min_risk_per_trade'],
                            min(self.config['max_risk_per_trade'], adjusted_risk))

        # Calculate position size
        risk_amount = portfolio.total_equity * adjusted_risk

        # Use ATR for stop loss calculation
        atr = self._calculate_atr(market_data)
        stop_distance = atr * self.config['stop_loss_atr_multiplier']

        if stop_distance <= 0:
            return 0

        position_size = int(risk_amount / stop_distance)

        logger.debug(f"Volatility sizing: current_vol={current_vol:.3f}, "
                     f"vol_adj={vol_adjustment:.3f}, adj_risk={adjusted_risk:.3f}")

        return position_size

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        if len(market_data) < self.volatility_lookback:
            return self.target_volatility

        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(self.volatility_lookback).std().iloc[-1]
        return volatility * np.sqrt(252)  # Annualize

    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(market_data) < period:
            return market_data['high'].iloc[-1] - market_data['low'].iloc[-1]

        high_low = market_data['high'] - market_data['low']
        high_close = abs(market_data['high'] - market_data['close'].shift(1))
        low_close = abs(market_data['low'] - market_data['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean().iloc[-1]

    def get_strategy_name(self) -> str:
        return "VolatilityBased"