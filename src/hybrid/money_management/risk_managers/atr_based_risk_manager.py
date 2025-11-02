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

        # Get ATR-specific config section
        risk_config = self.config['risk_managers']['atr_based']

        # NO DEFAULTS - fail if missing from config
        self.atr_multiplier = risk_config['stop_loss_atr_multiplier']
        self.max_daily_loss = risk_config['max_daily_loss']
        self.max_drawdown = risk_config['max_drawdown']
        self.atr_period = risk_config['atr_period']
        self.starting_atr = risk_config['starting_atr']

    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """Calculate stop loss using ATR"""
        atr = self._calculate_atr(market_data)

        logger.info(f"ATR_DEBUG: ATR={atr:.6f}, "
                    f"Multiplier={self.atr_multiplier}, "
                    f"Data points={len(market_data)}")

        if signal.direction == PositionDirection.LONG:
            stop_loss = signal.entry_price - (atr * self.atr_multiplier)
        else:  # SHORT
            stop_loss = signal.entry_price + (atr * self.atr_multiplier)

        stop_distance = abs(signal.entry_price - stop_loss)
        logger.info(f"STOP_CALC: Entry={signal.entry_price:.5f}, "
                    f"Stop={stop_loss:.5f}, Distance={stop_distance:.6f}")

        return stop_loss

    def should_reduce_risk(self, portfolio: PortfolioState) -> bool:
        """Check if risk reduction is needed"""
        # Daily loss check
        daily_loss_pct = abs(portfolio.daily_pnl) / portfolio.total_equity
        if daily_loss_pct > self.max_daily_loss:
            return True

        # Calculate CURRENT drawdown (not historical max)
        current_drawdown = (portfolio.peak_equity - portfolio.total_equity) / portfolio.peak_equity
        if current_drawdown > self.max_drawdown:
            return True

        return False

    def _calculate_atr(self, market_data: pd.DataFrame) -> float:
        """Calculate Average True Range from configuration period using SMA."""
        high_low = market_data['high'] - market_data['low']
        high_close = abs(market_data['high'] - market_data['close'].shift(1))
        low_close = abs(market_data['low'] - market_data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        if len(market_data) < self.atr_period:
            # Use SMA of available TR values
            num_periods = len(market_data)
            weighted_sum = self.starting_atr * (self.atr_period - num_periods) + true_range.sum()
            atr = weighted_sum / self.atr_period
        else:
            # For larger datasets, use the standard rolling ATR calculation
            atr = true_range.rolling(self.atr_period).mean().iloc[-1]

        return atr

    #TODO: apply it later
    def _calculate_atr_ema(self, market_data: pd.DataFrame) -> float:
        """Calculate Average True Range from configuration period, gradually replacing starting ATR with actual TR values."""
        high_low = market_data['high'] - market_data['low']
        high_close = abs(market_data['high'] - market_data['close'].shift(1))
        low_close = abs(market_data['low'] - market_data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        if len(market_data) < self.atr_period:
            # If there are fewer periods than the ATR period, gradually replace starting ATR with actual TR values
            num_periods = len(market_data)
            weighted_sum = self.starting_atr * (self.atr_period - num_periods) + true_range.sum()
            atr = weighted_sum / self.atr_period
        else:
            # For larger datasets, use the standard rolling ATR calculation
            atr = true_range.rolling(self.atr_period).mean().iloc[-1]

        return atr