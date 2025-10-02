# risk_managers/atr_based_risk_manager.py
import logging
import pandas as pd
from . import RiskManagementStrategy
from ..types import TradingSignal, PortfolioState, PositionDirection

logger = logging.getLogger(__name__)


class ATRBasedRiskManager(RiskManagementStrategy):
    """ATR-based stop losses and risk management"""

    def __init__(self, config):
        """
        Initialize ATRBasedRiskManager with configuration

        Args:
            config: UnifiedConfig instance
        """
        super().__init__(config)
        self.atr_multiplier = self.config.get('stop_loss_atr_multiplier', 2.0)
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)
        self.max_drawdown = self.config.get('max_drawdown', 0.20)

    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """Calculate stop loss using ATR"""
        atr = self._calculate_atr(market_data)

        if signal.direction == PositionDirection.LONG:
            stop_loss = signal.entry_price - (atr * self.atr_multiplier)
        else:  # SHORT
            stop_loss = signal.entry_price + (atr * self.atr_multiplier)

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

    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(market_data) < period:
            return market_data['high'].iloc[-1] - market_data['low'].iloc[-1]

        high_low = market_data['high'] - market_data['low']
        high_close = abs(market_data['high'] - market_data['close'].shift(1))
        low_close = abs(market_data['low'] - market_data['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean().iloc[-1]