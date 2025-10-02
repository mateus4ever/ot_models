# risk_managers/volatility_based_risk_manager.py
import logging
import pandas as pd
import numpy as np

from . import RiskManagementStrategy
from ..types import TradingSignal, PortfolioState, PositionDirection

logger = logging.getLogger(__name__)


class VolatilityBasedRiskManager(RiskManagementStrategy):
    """Volatility-based risk management with dynamic stop losses"""

    def __init__(self, config):
        """
        Initialize VolatilityBasedRiskManager with configuration

        Args:
            config: UnifiedConfig instance
        """
        super().__init__(config)
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        self.volatility_multiplier = self.config.get('volatility_multiplier', 2.0)
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)
        self.max_drawdown = self.config.get('max_drawdown', 0.20)

    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """Calculate stop loss using volatility"""
        volatility = self._calculate_volatility(market_data)

        # Convert annualized volatility to daily
        daily_volatility = volatility / np.sqrt(252)
        stop_distance = signal.entry_price * daily_volatility * self.volatility_multiplier

        if signal.direction == PositionDirection.LONG:
            stop_loss = signal.entry_price - stop_distance
        else:  # SHORT
            stop_loss = signal.entry_price + stop_distance

        return stop_loss

    def should_reduce_risk(self, portfolio: PortfolioState) -> bool:
        """Check if risk reduction is needed"""
        # Daily loss check
        daily_loss_pct = abs(portfolio.daily_pnl) / portfolio.total_equity
        if daily_loss_pct > self.max_daily_loss:
            return True

        # Drawdown check
        if portfolio.max_drawdown > self.max_drawdown:
            return True

        return False

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        if len(market_data) < self.volatility_lookback:
            return 0.15  # Default 15% annual volatility

        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(self.volatility_lookback).std().iloc[-1]
        return volatility * np.sqrt(252)  # Annualize