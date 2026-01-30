# position_sizers/kelly_criterion_sizer.py
import logging
from typing import TYPE_CHECKING

from src.hybrid.money_management.position_sizers import PositionSizingStrategy
from src.hybrid.positions.base_trade_history import PositionOutcome

if TYPE_CHECKING:
    from src.hybrid.positions.types import TradingSignal, PortfolioState

logger = logging.getLogger(__name__)


class KellyCriterionSizer(PositionSizingStrategy):
    """Kelly Criterion position sizing based on configured win statistics"""

    def __init__(self, unified_config):
        super().__init__(unified_config)

        # Static parameters from configuration
        # TODO: check how to insert optimized parameters
        self.kelly_fraction = self.config['parameters']['kelly_fraction']
        self.kelly_lookback = self.config['parameters']['kelly_lookback']

        # TODO: max_kelly_position must be questioned
        self.max_kelly_position = self.config['parameters']['max_position_pct']
        self.kelly_win_rate = self.config['parameters']['kelly_win_rate']
        self.kelly_avg_win = self.config['parameters']['kelly_avg_win']
        self.kelly_avg_loss = self.config['parameters']['kelly_avg_loss']

        # Trade history integration parameters
        self.min_trades_threshold = self.config['parameters']['kelly_min_trades_threshold']
        self.trade_outcomes = []

        if self.kelly_lookback <= 0:
            raise ValueError("kelly_lookback must be a positive integer")

    def calculate_size(self, signal: 'TradingSignal', portfolio: 'PortfolioState',
                       stop_distance: float) -> int:
        """
        Calculate position size using Kelly Criterion

        Args:
            signal: Trading signal with entry price
            portfolio: Current portfolio state
            stop_distance: Risk distance per share (from risk manager)

        Returns:
            Number of shares to trade
        """
        # Calculate Kelly percentage using configuration parameters
        kelly_percentage = self._calculate_kelly_percentage()

        # Apply Kelly fraction and maximum limits
        final_kelly_percentage = self._apply_kelly_limits(kelly_percentage)

        # Calculate position size based on risk budget and stop distance
        position_size = self._calculate_position_from_kelly(
            portfolio, final_kelly_percentage, stop_distance
        )

        logger.debug(f"Kelly calculation: win_rate={self.kelly_win_rate:.3f}, "
                     f"raw_kelly={kelly_percentage:.3f}, "
                     f"final_kelly={final_kelly_percentage:.3f}, "
                     f"position_size={position_size}")

        return position_size

    def update_trade_result(self, position_outcome: PositionOutcome) -> None:
        """Update with completed trade result for dynamic statistics"""
        self.trade_outcomes.append(position_outcome)

        # Keep only recent trades based on lookback period
        if len(self.trade_outcomes) > self.kelly_lookback:
            self.trade_outcomes = self.trade_outcomes[-self.kelly_lookback:]

        logger.debug(f"Kelly updated with trade outcome: {position_outcome.outcome}, "
                     f"total outcomes: {len(self.trade_outcomes)}")

    def _calculate_kelly_percentage(self) -> float:
        """Calculate raw Kelly percentage using current statistics"""
        win_rate, avg_win, avg_loss = self._get_current_statistics()

        if avg_loss == 0:
            logger.warning("Average loss is zero, Kelly calculation invalid")
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly_percentage = win_rate - ((1 - win_rate) / win_loss_ratio)

        return kelly_percentage
    def _apply_kelly_limits(self, kelly_percentage: float) -> float:
        """
        Apply Kelly fraction and maximum position limits

        Args:
            kelly_percentage: Raw Kelly percentage

        Returns:
            Limited Kelly percentage
        """
        # Don't allow negative Kelly (would mean strategy has negative edge)
        if kelly_percentage <= 0:
            logger.info(f"Negative Kelly percentage {kelly_percentage:.3f}, using zero position")
            return 0.0

        # Apply Kelly fraction (conservative scaling)
        scaled_kelly = kelly_percentage * self.kelly_fraction

        # Apply maximum position limit
        final_kelly = min(scaled_kelly, self.max_kelly_position)

        return final_kelly

    def _calculate_position_from_kelly(self, portfolio: 'PortfolioState',
                                       kelly_percentage: float, stop_distance: float) -> int:
        """
        Calculate number of shares based on Kelly percentage and stop distance

        Args:
            portfolio: Current portfolio state
            kelly_percentage: Final Kelly percentage to risk
            stop_distance: Risk per share from stop distance

        Returns:
            Number of shares to trade
        """
        if kelly_percentage == 0 or stop_distance == 0:
            return 0

        # Calculate total risk budget from Kelly percentage
        risk_budget = portfolio.total_equity * kelly_percentage

        # Calculate shares based on risk per share (stop distance)
        position_size = int(risk_budget / stop_distance)

        return max(0, position_size)

    def _calculate_kelly_percentage(self) -> float:
        """Calculate raw Kelly percentage using current statistics"""
        win_rate, avg_win, avg_loss = self._get_current_statistics()

        if avg_loss == 0:
            logger.warning("Average loss is zero, Kelly calculation invalid")
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly_percentage = win_rate - ((1 - win_rate) / win_loss_ratio)

        return kelly_percentage

    def _get_current_statistics(self):
        """Bayesian approach: weight prior and observed data"""
        n_outcomes = len(self.trade_outcomes)

        if n_outcomes == 0:
            return self.kelly_win_rate, self.kelly_avg_win, self.kelly_avg_loss

        # Weight decreases as more data observed
        prior_weight = self.min_trades_threshold / (self.min_trades_threshold + n_outcomes)
        data_weight = n_outcomes / (self.min_trades_threshold + n_outcomes)

        # Calculate from actual outcomes
        wins = [o for o in self.trade_outcomes if o.outcome == 'win']
        losses = [o for o in self.trade_outcomes if o.outcome == 'loss']

        observed_win_rate = len(wins) / len(self.trade_outcomes) if self.trade_outcomes else 0
        observed_avg_win = sum(o.net_pnl for o in wins) / len(wins) if wins else 0
        observed_avg_loss = abs(sum(o.net_pnl for o in losses) / len(losses)) if losses else 0

        # DEBUG
        logger.debug(f"n_outcomes: {n_outcomes}, wins: {len(wins)}, losses: {len(losses)}")
        logger.debug(f"observed_win_rate: {observed_win_rate}, prior_weight: {prior_weight}")

        # Weighted combination
        win_rate = prior_weight * self.kelly_win_rate + data_weight * observed_win_rate
        avg_win = prior_weight * self.kelly_avg_win + data_weight * observed_avg_win
        avg_loss = prior_weight * self.kelly_avg_loss + data_weight * observed_avg_loss

        return win_rate, avg_win, avg_loss

    def _calculate_statistics_from_outcomes(self) -> tuple[float, float, float]:
        """Calculate win rate and averages from accumulated PositionOutcome objects

        Returns:
            Tuple of (win_rate, avg_win, avg_loss)
        """
        wins = [o for o in self.trade_outcomes if o.outcome == 'win']
        losses = [o for o in self.trade_outcomes if o.outcome == 'loss']

        total_trades = len(wins) + len(losses)  # Exclude break-even
        if total_trades == 0:
            # No wins or losses yet, return bootstrap values
            return self.kelly_win_rate, self.kelly_avg_win, self.kelly_avg_loss

        win_rate = len(wins) / total_trades
        avg_win = sum(o.net_pnl for o in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(o.net_pnl for o in losses) / len(losses)) if losses else 0.0

        return win_rate, avg_win, avg_loss

    def get_strategy_name(self) -> str:
        """Return strategy name for logging"""
        return "KellyCriterion"