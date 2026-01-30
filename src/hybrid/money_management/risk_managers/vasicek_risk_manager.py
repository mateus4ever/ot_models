# risk_managers/vasicek_risk_manager.py
import logging
import pandas as pd
from . import RiskManagementStrategy
from src.hybrid.positions.types import TradingSignal, PortfolioState, PositionDirection

logger = logging.getLogger(__name__)


class VasicekRiskManager(RiskManagementStrategy):
    """Z-score based stop losses for Vasicek mean-reversion strategies"""

    def __init__(self, config):
        """
        Initialize VasicekRiskManager with configuration

        Args:
            config: UnifiedConfig instance
        """
        super().__init__(config)

        # Get Vasicek-specific config section
        risk_config = self.config['risk_managers']['vasicek']
        if risk_config is None:
            raise ValueError("Missing 'vasicek' section in risk_managers config")

        # NO DEFAULTS - fail if missing from config
        self.stop_loss_z_score = risk_config['parameters']['stop_loss_z_score']
        self.max_daily_loss = risk_config['parameters']['max_daily_loss']
        self.max_drawdown = risk_config['parameters']['max_drawdown']

        # Vasicek parameters (set by predictor after calibration)
        self.theta = None  # Long-term mean
        self.sigma = None  # Volatility
        self.kappa = None  # Reversion speed (for reference)

        logger.info(f"VasicekRiskManager initialized with stop_loss_z_score={self.stop_loss_z_score}")

    def set_vasicek_parameters(self, theta: float, sigma: float, kappa: float = None):
        """
        Store calibrated Vasicek parameters

        Called by predictor after calibration to provide parameters
        needed for Z-score based stop loss calculations

        Args:
            theta: Long-term mean (equilibrium level)
            sigma: Volatility
            kappa: Reversion speed (optional, for reference)
        """

        self.theta = theta
        self.sigma = sigma
        self.kappa = kappa

        logger.info(
            f"Vasicek parameters set: θ={theta:.6f}, σ={sigma:.6f}"
            f"{f', κ={kappa:.4f}' if kappa else ''}"
        )

    def calculate_stop_loss(self, signal: TradingSignal, market_data: pd.DataFrame) -> float:
        """
        Calculate stop loss using Z-score threshold

        For Vasicek spread positions:
        - LONG_SPREAD: Stop if Z-score drops too far (spread widens against us)
        - SHORT_SPREAD: Stop if Z-score rises too far (spread widens against us)

        Args:
            signal: TradingSignal with entry_z_score in metadata
            market_data: Not used for Vasicek (Z-score based, not price-based)

        Returns:
            Stop loss price
        """
        """Calculate stop loss using Vasicek model"""
        if self.theta is None or self.sigma is None:
            raise ValueError("Vasicek parameters not set - calibration required")
        if self.sigma <= 0:
            raise ValueError("Invalid sigma - must be positive")

        # Get entry Z-score from signal metadata
        entry_z_score = signal.metadata.get('entry_z_score')
        if entry_z_score is None:
            raise ValueError("Missing entry_z_score in signal metadata")

        if entry_z_score == 0.0 and 'entry_z_score' not in signal.metadata:
            logger.warning("No entry_z_score in signal metadata, using 0.0")

        logger.debug(f"Z-SCORE_DEBUG: Entry Z={entry_z_score:.2f}, "
                    f"Stop threshold={self.stop_loss_z_score}, "
                    f"θ={self.theta:.6f}, σ={self.sigma:.6f}")

        # Calculate stop loss Z-score based on direction
        if signal.direction == PositionDirection.LONG:
            # LONG_SPREAD: Entered at negative Z (spread underpriced)
            # Stop if spread widens further against us (more negative)
            stop_z_score = entry_z_score - self.stop_loss_z_score
        else:  # SHORT_SPREAD
            # SHORT_SPREAD: Entered at positive Z (spread overpriced)
            # Stop if spread widens further against us (more positive)
            stop_z_score = entry_z_score + self.stop_loss_z_score

        # Convert Z-score to absolute price
        # Price = θ + (Z × σ)
        stop_loss = self.theta + (stop_z_score * self.sigma)

        stop_distance = abs(signal.entry_price - stop_loss)
        logger.info(f"STOP_CALC: Entry={signal.entry_price:.6f}, "
                    f"Entry Z={entry_z_score:.2f}, "
                    f"Stop={stop_loss:.6f}, "
                    f"Stop Z={stop_z_score:.2f}, "
                    f"Distance={stop_distance:.6f}")

        return stop_loss

    def should_reduce_risk(self, portfolio: PortfolioState) -> bool:
        """Check if risk reduction is needed"""
        # Daily loss check
        daily_loss_pct = abs(portfolio.daily_pnl) / portfolio.total_equity
        if daily_loss_pct > self.max_daily_loss:
            logger.warning(f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.max_daily_loss:.2%}")
            return True

        # Calculate current drawdown
        current_drawdown = self._calculate_drawdown(portfolio)
        if current_drawdown > self.max_drawdown:
            logger.warning(f"Drawdown {current_drawdown:.2%} exceeds limit {self.max_drawdown:.2%}")
            return True

        return False

    def _calculate_drawdown(self, portfolio: PortfolioState) -> float:
        """Calculate current drawdown from peak"""
        if portfolio.peak_equity <= 0:
            return 0.0
        return (portfolio.peak_equity - portfolio.total_equity) / portfolio.peak_equity