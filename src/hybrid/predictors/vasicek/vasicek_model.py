"""
Vasicek Mean-Reversion Model
Models Ornstein-Uhlenbeck process for mean-reverting time series
"""
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class VasicekModel:
    """
    Vasicek Mean-Reversion Model (Ornstein-Uhlenbeck Process)

    Models: dX(t) = κ(θ - X(t))dt + σdW(t)
    Where:
        X(t) = spread value at time t
        θ = long-term mean (equilibrium level)
        κ = speed of mean reversion
        σ = volatility

    Used for mean-reversion trading strategies.
    """

    def __init__(self, config):
        """
        Initialize Vasicek model with configuration

        Args:
            config: UnifiedConfig object with model parameters
        """
        if not config:
            raise ValueError("Config is required")

        self.config = config

        # Get model parameters
        vasicek_config = config.get_section('vasicek_prediction')
        if not vasicek_config:
            raise ValueError("vasicek_prediction section must be configured in JSON config")

        # Get active profile from setting
        profile = vasicek_config.get('setting')
        if not profile:
            raise ValueError("vasicek_prediction.setting must be configured in JSON config")

        profile_config = vasicek_config.get(profile)
        if not profile_config:
            raise ValueError(f"vasicek_prediction.{profile} section must be configured in JSON config")

        params = profile_config.get('parameters')
        if not params:
            raise ValueError(f"vasicek_prediction.{profile}.parameters must be configured in JSON config")

        self.profile = profile

        # Cache config values
        self._cache_config_values()

        # Calibrated parameters (set during calibration)
        self.theta: Optional[float] = None  # Long-term mean
        self.kappa: Optional[float] = None  # Mean reversion speed
        self.sigma: Optional[float] = None  # Volatility
        self.half_life: Optional[float] = None  # Half-life of mean reversion

        # Calibration diagnostics
        self.p_value: Optional[float] = None
        self.r_squared: Optional[float] = None
        self.rejection_reason: Optional[str] = None
        self.warnings: list = []

        self.is_calibrated = False

        logger.info(f"Initialized VasicekModel with profile '{profile}'")

    def _cache_config_values(self):
        """Cache configuration values from config"""
        vasicek_config = self.config.get_section('vasicek_prediction')
        params = vasicek_config[self.profile]['parameters']
        validation_config = params['validation']
        bounds_config = params['parameter_bounds']

        # Trading parameters
        self.lookback_window = params['lookback_window']
        self.entry_threshold = params['entry_threshold']
        self.exit_threshold = params['exit_threshold']
        self.pip_multiplier = params['pip_multiplier']

        # Validation thresholds
        self.significance_level = validation_config['significance_level']
        self.max_half_life = validation_config['max_half_life']
        self.min_kappa = validation_config['min_kappa']
        self.max_kappa = validation_config['max_kappa']
        self.min_observations = validation_config['min_observations']

        # Parameter bounds for validation
        self.theta_max = bounds_config['theta_max']
        self.sigma_max = bounds_config['sigma_max']
        self.kappa_reasonable_max = bounds_config['kappa_reasonable_max']
    def calibrate(self, spread_series: pd.Series) -> Dict[str, float]:
        """
        Calibrate Vasicek parameters using OLS regression

        Uses discrete-time approximation:
        X(t+1) - X(t) = κ(θ - X(t))Δt + σ√Δt * ε

        Rearranging: ΔX(t) = α + β*X(t) + ε
        Where: α = κθΔt, β = -κΔt

        Args:
            spread_series: Time series of spread values

        Returns:
            Dictionary with calibrated parameters
        """
        logger.info("Calibrating Vasicek parameters using OLS...")

        # Calculate differences
        X_t = spread_series.values[:-1]  # X(t)
        X_t1 = spread_series.values[1:]  # X(t+1)
        delta_X = X_t1 - X_t  # ΔX(t)

        # OLS regression: ΔX(t) = α + β*X(t) + ε
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_t, delta_X)

        # Extract Vasicek parameters
        # Assuming Δt = 1 (daily data)
        beta = slope
        alpha = intercept

        # Calculate parameters
        self.kappa = -beta  # κ = -β
        self.theta = alpha / self.kappa if self.kappa != 0 else X_t.mean()  # θ = α/κ

        # Calculate volatility (residual standard deviation)
        predicted = alpha + beta * X_t
        residuals = delta_X - predicted
        self.sigma = np.std(residuals)

        # Calculate half-life: t_half = ln(2) / κ
        if self.kappa > 0:
            self.half_life = np.log(2) / self.kappa
        else:
            self.half_life = np.inf
            logger.warning("Kappa ≤ 0: No mean reversion detected!")

        self.p_value = p_value
        self.r_squared = r_value ** 2

        self.is_calibrated = True

        # Calculate fit quality
        r_squared = r_value ** 2

        calibration_results = {
            'theta': self.theta,
            'kappa': self.kappa,
            'sigma': self.sigma,
            'half_life': self.half_life,
            'r_squared': r_squared,
            'p_value': p_value,
            'observations': len(X_t)
        }

        logger.info(
            f"Vasicek calibration complete:\n"
            f"  θ (long-term mean): {self.theta:.6f}\n"
            f"  κ (reversion speed): {self.kappa:.6f}\n"
            f"  σ (volatility): {self.sigma:.6f}\n"
            f"  Half-life: {self.half_life:.2f} periods\n"
            f"  R²: {r_squared:.4f}\n"
            f"  p-value: {p_value:.4f}"
        )

        # Validate calibration
        if p_value > 0.05:
            logger.warning(
                f"Poor fit: p-value {p_value:.4f} > 0.05. "
                f"Mean reversion may not be statistically significant."
            )

        if self.half_life < 1:
            logger.warning(
                f"Very fast reversion: half-life {self.half_life:.2f} < 1 period. "
                f"May indicate overfitting."
            )

        if self.half_life > 200:
            logger.warning(
                f"Very slow reversion: half-life {self.half_life:.2f} > 200 periods. "
                f"Mean reversion too weak for trading."
            )

        return calibration_results

    def calculate_equilibrium_deviation(self, current_value: float) -> float:
        """
        Calculate deviation from equilibrium (current - theta)

        Args:
            current_value: Current spread value

        Returns:
            Deviation from long-term mean
        """
        if not self.is_calibrated:
            raise ValueError("Model not calibrated. Call calibrate() first.")

        return current_value - self.theta

    def calculate_z_score(self, current_value: float) -> float:
        """
        Calculate Z-score (standardized deviation)

        Args:
            current_value: Current spread value

        Returns:
            Z-score (number of standard deviations from mean)
        """
        if not self.is_calibrated:
            raise ValueError("Model not calibrated. Call calibrate() first.")

        deviation = self.calculate_equilibrium_deviation(current_value)
        z_score = deviation / self.sigma if self.sigma > 0 else 0.0

        return z_score

    def predict_next_value(self, current_value: float, time_step: float = 1.0) -> float:
        """
        Predict expected value at next time step

        Uses: E[X(t+Δt)|X(t)] = θ + (X(t) - θ)e^(-κΔt)

        Args:
            current_value: Current spread value
            time_step: Time step size (default: 1.0)

        Returns:
            Expected value at next time step
        """
        if not self.is_calibrated:
            raise ValueError("Model not calibrated. Call calibrate() first.")

        expected = self.theta + (current_value - self.theta) * np.exp(-self.kappa * time_step)
        return expected

    def calculate_expected_return_to_mean(self, current_value: float) -> float:
        """
        Calculate expected return (movement) towards mean in one period

        Args:
            current_value: Current spread value

        Returns:
            Expected change (positive means spread will increase)
        """
        if not self.is_calibrated:
            raise ValueError("Model not calibrated. Call calibrate() first.")

        next_value = self.predict_next_value(current_value)
        expected_return = next_value - current_value

        return expected_return

    def get_trading_threshold(self, z_threshold: float) -> float:
        """
        Convert Z-score threshold to absolute spread value

        Args:
            z_threshold: Z-score threshold (e.g., 2.0)

        Returns:
            Absolute spread value at threshold
        """
        if not self.is_calibrated:
            raise ValueError("Model not calibrated. Call calibrate() first.")

        return self.theta + (z_threshold * self.sigma)

    def is_mean_reverting(self) -> bool:
        """
        Check if time series exhibits mean reversion

        Returns:
            True if mean reversion is detected
        """
        if not self.is_calibrated:
            raise ValueError("Model not calibrated. Call calibrate() first.")

        # Check kappa is positive (mean-reverting)
        if self.kappa <= 0:
            return False

        # Check kappa is strong enough
        if self.kappa < self.min_kappa:
            return False

        # Check half-life is reasonable (not too slow)
        if self.half_life > self.max_half_life:
            return False

        return True

    def get_reversion_category(self) -> str:
        """Categorize reversion speed based on half-life."""
        if not self.is_calibrated:
            raise ValueError("Model not calibrated. Call calibrate() first.")

        if self.half_life == float('inf') or self.kappa <= 0:
            return "none"

        vasicek_config = self.config.get_section('vasicek_prediction')
        params = vasicek_config[self.profile]['parameters']

        thresholds = params.get('reversion_categories')
        if not thresholds:
            raise ValueError("reversion_categories must be configured in vasicek_prediction parameters")

        if self.half_life < thresholds['very_fast']:
            return "very_fast"
        elif self.half_life < thresholds['fast']:
            return "fast"
        elif self.half_life < thresholds['moderate']:
            return "moderate"
        elif self.half_life < thresholds['slow']:
            return "slow"
        else:
            return "very_slow"