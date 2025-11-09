# src/hybrid/backtesting/walk_forward_retraining_strategy.py
"""
WalkForwardRetrainingStrategy - Manages retraining decisions in walk-forward optimization

Determines WHEN to retrain strategy parameters during walk-forward backtesting.
Supports both fixed-frequency and adaptive (performance-based) retraining.
"""

import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WalkForwardRetrainingStrategy:
    """
    Manages retraining strategy during walk-forward backtesting

    Supports multiple retraining approaches:
    1. Fixed frequency: Retrain every N steps
    2. Adaptive: Retrain when performance degrades
    3. Hybrid: Combination of both

    Prevents over-fitting by controlling how often parameters are updated.
    """

    def __init__(self, config: Dict):
        """
        Initialize retraining strategy with configuration

        Args:
            config: Configuration dictionary containing walk_forward section
        """
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache retraining strategy configuration"""
        walk_forward_config = self.config.get('walk_forward', {})
        retraining_config = walk_forward_config.get('retraining_strategy', {})

        # Fixed frequency retraining
        self.retrain_frequency = walk_forward_config.get('retrain_frequency', 50)

        # Adaptive retraining parameters
        self.adaptive_retraining = retraining_config.get('adaptive_retraining', False)
        self.performance_threshold = retraining_config.get('performance_threshold', -0.05)  # -5% return
        self.lookback_window = retraining_config.get('lookback_window', 20)

        # Min steps between retrains (prevent thrashing)
        self.min_steps_between_retrains = retraining_config.get('min_steps_between_retrains', 10)

        # Track last retrain
        self.last_retrain_step = 0

        logger.info(f"RetrainingStrategy initialized: frequency={self.retrain_frequency}, "
                    f"adaptive={self.adaptive_retraining}, threshold={self.performance_threshold}")

    def should_retrain(
            self,
            current_step: int,
            recent_performance: Optional[List[float]] = None
    ) -> bool:
        """
        Determine if retraining should occur at current step

        Args:
            current_step: Current step in walk-forward process
            recent_performance: Recent performance metrics (e.g., returns, Sharpe ratios)

        Returns:
            True if retraining should occur, False otherwise
        """
        # Don't retrain too frequently (prevent thrashing)
        steps_since_last = current_step - self.last_retrain_step
        if steps_since_last < self.min_steps_between_retrains:
            logger.debug(f"Skipping retrain: only {steps_since_last} steps since last retrain")
            return False

        # Skip first step (no history yet)
        if current_step == 0:
            return False

        # Check fixed frequency retraining
        if self._should_retrain_by_frequency(current_step):
            logger.info(f"Retraining triggered by frequency at step {current_step}")
            self.last_retrain_step = current_step
            return True

        # Check adaptive retraining (if enabled and data available)
        if self.adaptive_retraining and recent_performance:
            if self._should_retrain_by_performance(recent_performance, current_step):
                logger.info(f"Retraining triggered by performance degradation at step {current_step}")
                self.last_retrain_step = current_step
                return True

        return False

    def _should_retrain_by_frequency(self, current_step: int) -> bool:
        """
        Check if retraining should occur based on fixed frequency

        Args:
            current_step: Current step

        Returns:
            True if frequency threshold reached
        """
        return (current_step % self.retrain_frequency == 0) and (current_step > 0)

    def _should_retrain_by_performance(
            self,
            recent_performance: List[float],
            current_step: int
    ) -> bool:
        """
        Check if retraining should occur based on performance degradation

        Args:
            recent_performance: Recent performance metrics
            current_step: Current step

        Returns:
            True if performance below threshold
        """
        if len(recent_performance) < self.lookback_window:
            logger.debug(f"Insufficient performance history: {len(recent_performance)} < {self.lookback_window}")
            return False

        # Calculate average performance over lookback window
        recent_avg = np.mean(recent_performance[-self.lookback_window:])

        # Trigger retrain if performance below threshold
        if recent_avg < self.performance_threshold:
            logger.warning(f"Performance degradation detected: {recent_avg:.4f} < {self.performance_threshold}")
            return True

        return False

    def reset(self):
        """Reset retraining state (for new walk-forward run)"""
        self.last_retrain_step = 0
        logger.info("Retraining strategy reset")

    def get_retrain_schedule(self, total_steps: int) -> List[int]:
        """
        Get planned retraining schedule for fixed-frequency mode

        Args:
            total_steps: Total number of steps in walk-forward

        Returns:
            List of step numbers where retraining will occur
        """
        if self.retrain_frequency <= 0:
            return []

        schedule = list(range(self.retrain_frequency, total_steps, self.retrain_frequency))
        logger.info(f"Retraining schedule: {len(schedule)} retrains planned over {total_steps} steps")
        return schedule

    def update_performance_threshold(self, new_threshold: float):
        """
        Dynamically update performance threshold

        Args:
            new_threshold: New threshold value
        """
        old_threshold = self.performance_threshold
        self.performance_threshold = new_threshold
        logger.info(f"Performance threshold updated: {old_threshold} -> {new_threshold}")

    def get_statistics(self) -> Dict:
        """
        Get retraining statistics

        Returns:
            Dictionary with retraining stats
        """
        return {
            'retrain_frequency': self.retrain_frequency,
            'adaptive_enabled': self.adaptive_retraining,
            'performance_threshold': self.performance_threshold,
            'lookback_window': self.lookback_window,
            'min_steps_between_retrains': self.min_steps_between_retrains,
            'last_retrain_step': self.last_retrain_step
        }


class FixedWindowRetrainingStrategy(WalkForwardRetrainingStrategy):
    """
    Simple fixed-frequency retraining strategy

    Retrains every N steps, no performance monitoring.
    Useful for consistent, predictable retraining schedule.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        # Disable adaptive retraining
        self.adaptive_retraining = False
        logger.info(f"FixedWindowRetrainingStrategy: retrain every {self.retrain_frequency} steps")


class AdaptiveRetrainingStrategy(WalkForwardRetrainingStrategy):
    """
    Performance-based adaptive retraining strategy

    Only retrains when performance degrades below threshold.
    More efficient but requires good performance metrics.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        # Enable adaptive, disable fixed frequency
        self.adaptive_retraining = True
        self.retrain_frequency = float('inf')  # Disable fixed frequency
        logger.info(f"AdaptiveRetrainingStrategy: threshold={self.performance_threshold}, "
                    f"lookback={self.lookback_window}")