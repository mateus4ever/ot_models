"""
Triangular Arbitrage Predictor with Vasicek Mean-Reversion
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

import pandas as pd

from src.hybrid.predictors.predictor_interface import PredictorInterface
from src.hybrid.predictors.vasicek.vasicek_model import VasicekModel

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageState:
    """Current state of arbitrage strategy"""
    position: Optional[str]  # 'LONG_SPREAD', 'SHORT_SPREAD', None
    entry_spread_pips: Optional[float]
    entry_z_score: Optional[float]


class TriangularArbitragePredictor(PredictorInterface):
    """
    Triangular Arbitrage Predictor with Vasicek Mean-Reversion Model
    """

    def __init__(self, config, vasicek_model=None):
        self.config = config

        # Get config section
        arb_config = config.get_section('triangular_arbitrage')
        if not arb_config:
            raise ValueError("triangular_arbitrage section must be configured")

        # Get active profile
        profile = arb_config.get('setting')
        if not profile:
            raise ValueError("triangular_arbitrage.setting must be configured")

        profile_config = arb_config.get(profile)
        if not profile_config:
            raise ValueError(f"triangular_arbitrage.{profile} section must be configured")

        params = profile_config.get('parameters')
        if not params:
            raise ValueError(f"triangular_arbitrage.{profile}.parameters must be configured")

        # Market configuration
        self.target_market = params.get('target_market')
        self.leg1_market = params.get('leg1_market')
        self.leg2_market = params.get('leg2_market')

        if not all([self.target_market, self.leg1_market, self.leg2_market]):
            raise ValueError(
                "Config must specify: target_market, leg1_market, leg2_market"
            )

        # Strategy parameters
        self.lookback_window = params.get('lookback_window')
        self.entry_threshold = params.get('entry_threshold')
        self.exit_threshold = params.get('exit_threshold')
        self.pip_multiplier = params.get('pip_multiplier')
        self.confidence_divisor = params.get('confidence_divisor')

        if not all([self.lookback_window, self.entry_threshold,
                    self.exit_threshold, self.pip_multiplier]):
            raise ValueError(
                "Config must specify: lookback_window, entry_threshold, "
                "exit_threshold, pip_multiplier"
            )

        # Inject or create VasicekModel
        self.vasicek_model = vasicek_model or VasicekModel(config)

        # State tracking
        self._is_trained = False
        self.state = ArbitrageState(
            position=None,
            entry_spread_pips=None,
            entry_z_score=None
        )

        logger.info(
            f"Initialized TriangularArbitragePredictor:\n"
            f"  Markets: {self.target_market} vs ({self.leg1_market} × {self.leg2_market})\n"
            f"  Entry threshold: ±{self.entry_threshold}σ\n"
            f"  Exit threshold: ±{self.exit_threshold}σ"
        )

    def get_required_markets(self) -> list:
        """Return list of markets required by this predictor"""
        return [self.target_market, self.leg1_market, self.leg2_market]

    def calculate_synthetic_price(self, leg1_price: float, leg2_price: float) -> float:
        """Calculate synthetic target price from legs"""
        return leg1_price * leg2_price

    def calculate_spread(self, target_price: float, leg1_price: float, leg2_price: float) -> float:
        """Calculate spread between actual and synthetic"""
        synthetic = self.calculate_synthetic_price(leg1_price, leg2_price)
        return target_price - synthetic

    def train(self, data_manager) -> Dict:
        """
        Train/calibrate predictor using historical data

        Steps:
            1. Get historical prices for all three markets
            2. Calculate spread time series
            3. Calibrate Vasicek model on spread series

        Args:
            data_manager: DataManager with loaded market data

        Returns:
            Dict with training results and 'success' flag
        """
        logger.info("Training triangular arbitrage predictor with Vasicek model...")

        try:
            # Verify markets are loaded
            required_markets = self.get_required_markets()
            available_markets = data_manager.get_available_markets()
            missing = set(required_markets) - set(available_markets)
            if missing:
                raise ValueError(f"Required markets not loaded: {missing}")

            # Get historical data
            past_data = data_manager.get_markets_past_data(required_markets)

            # Extract close prices
            target_prices = past_data[self.target_market]['close']
            leg1_prices = past_data[self.leg1_market]['close']
            leg2_prices = past_data[self.leg2_market]['close']

            # Calculate spread series
            spread_series = self.calculate_spread_series(
                target_prices,
                leg1_prices,
                leg2_prices
            )

            # Resample to hourly - minute data too noisy for mean reversion
            spread_series = spread_series.resample('1h').last().dropna()

            # Apply lookback window if specified
            if self.lookback_window and len(spread_series) > self.lookback_window:
                spread_series = spread_series.iloc[-self.lookback_window:]

            # Apply lookback window if specified
            if self.lookback_window and len(spread_series) > self.lookback_window:
                spread_series = spread_series.iloc[-self.lookback_window:]

            # Calibrate Vasicek model on spread
            vasicek_params = self.vasicek_model.calibrate(spread_series)

            # Validate mean reversion
            if not self.vasicek_model.is_mean_reverting():
                logger.error(
                    "Spread does NOT exhibit mean reversion! "
                    "Trading this spread is not recommended."
                )
                return {
                    'success': False,
                    'reason': 'Spread does not exhibit mean reversion'
                }

            self._is_trained = True

            # Calculate spread statistics for reporting
            spread_pips = spread_series * self.pip_multiplier

            logger.info(
                f"Training complete:\n"
                f"  Vasicek θ (mean): {vasicek_params['theta'] * self.pip_multiplier:.2f} pips\n"
                f"  Vasicek κ (reversion): {vasicek_params['kappa']:.6f}\n"
                f"  Vasicek σ (volatility): {vasicek_params['sigma'] * self.pip_multiplier:.2f} pips\n"
                f"  Half-life: {vasicek_params['half_life']:.2f} periods\n"
                f"  Spread range: [{spread_pips.min():.2f}, {spread_pips.max():.2f}] pips\n"
                f"  Observations: {vasicek_params['observations']}"
            )

            return {
                'success': True,
                'theta': vasicek_params['theta'],
                'kappa': vasicek_params['kappa'],
                'sigma': vasicek_params['sigma'],
                'half_life': vasicek_params['half_life'],
                'observations': vasicek_params['observations'],
                'spread_min_pips': spread_pips.min(),
                'spread_max_pips': spread_pips.max()
            }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self._is_trained = False
            return {
                'success': False,
                'reason': str(e)
            }

    def calculate_spread_series(self, target_prices: pd.Series, leg1_prices: pd.Series,
                                leg2_prices: pd.Series) -> pd.Series:
        """Calculate spread series between actual and synthetic"""
        synthetic_prices = leg1_prices * leg2_prices
        return target_prices - synthetic_prices
    def predict(self, data_manager) -> Dict:
        """
        Generate prediction using Vasicek model

        Args:
            data_manager: DataManager at current temporal pointer

        Returns:
            Prediction dictionary
        """
        if not self._is_trained:
            logger.warning("Predictor not trained, returning neutral signal")
            return self._create_neutral_prediction()

        try:
            # Get current prices
            required_markets = self.get_required_markets()
            current_data = data_manager.get_markets_current_data(required_markets)

            # Extract prices
            target_price = current_data[self.target_market]['close']
            leg1_price = current_data[self.leg1_market]['close']
            leg2_price = current_data[self.leg2_market]['close']

            # Calculate spread using spread calculator
            spread = self.calculate_spread(
                target_price, leg1_price, leg2_price
            )
            spread_pips = spread * self.pip_multiplier

            # Calculate Z-score using Vasicek model
            z_score = self.vasicek_model.calculate_z_score(spread)
            logger.warning(f"Z-SCORE DEBUG: spread={spread:.8f}, "
                           f"theta={self.vasicek_model.theta:.8f}, "
                           f"sigma={self.vasicek_model.sigma:.8f}, "
                           f"z={z_score:.2f}")

            # Calculate expected return to mean (Vasicek prediction)
            expected_return = self.vasicek_model.calculate_expected_return_to_mean(spread)
            expected_return_pips = expected_return * self.pip_multiplier

            # Generate signal based on Vasicek Z-score
            signal, confidence = self._generate_signal(z_score)

            # Calculate synthetic for reporting
            synthetic_price = self.calculate_synthetic_price(
                leg1_price, leg2_price
            )

            # Update state
            self._update_state(signal, spread_pips, z_score)

            # Create prediction
            prediction = {
                'signal': signal,
                'confidence': confidence,
                'z_score': z_score,
                'spread': spread,
                'spread_pips': spread_pips,
                'expected_return': expected_return,
                'expected_return_pips': expected_return_pips,
                'target_price': target_price,
                'synthetic_price': synthetic_price,
                'leg1_price': leg1_price,
                'leg2_price': leg2_price,
                'position': self.state.position,
                'entry_spread_pips': self.state.entry_spread_pips,
                'entry_z_score': self.state.entry_z_score,
                'vasicek_theta': self.vasicek_model.theta,
                'vasicek_half_life': self.vasicek_model.half_life,
                'markets': {
                    'target': self.target_market,
                    'leg1': self.leg1_market,
                    'leg2': self.leg2_market
                }
            }

            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return self._create_neutral_prediction()


    def _update_state(self, signal: str, spread_pips: float, z_score: float):
        """Update predictor state"""
        if signal in ['LONG_SPREAD', 'SHORT_SPREAD']:
            self.state.position = signal
            self.state.entry_spread_pips = spread_pips
            self.state.entry_z_score = z_score
        elif signal == 'CLOSE':
            self.state.position = None
            self.state.entry_spread_pips = None
            self.state.entry_z_score = None

    def _create_neutral_prediction(self) -> Dict:
        """Create neutral/hold prediction"""
        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'z_score': 0.0,
            'spread_pips': 0.0,
            'position': self.state.position
        }

    def reset(self):
        """Reset predictor state"""
        self.state = ArbitrageState(
            position=None,
            entry_spread_pips=None,
            entry_z_score=None
        )
        logger.debug("Predictor state reset")

    def _generate_signal(self, z_score: float) -> tuple:
        """Generate trading signal based on Vasicek Z-score"""
        if self.state.position is None:
            # No position - check for entry
            if z_score >= self.entry_threshold:
                signal = 'SHORT_SPREAD'
                confidence = min(1.0, abs(z_score) / self.confidence_divisor)
            elif z_score <= -self.entry_threshold:
                signal = 'LONG_SPREAD'
                confidence = min(1.0, abs(z_score) / self.confidence_divisor)
            else:
                signal = 'HOLD'
                confidence = 0.0
        else:
            # Have position - check for exit
            if self.state.position == 'LONG_SPREAD':
                if z_score >= -self.exit_threshold:  # Changed > to >=
                    signal = 'CLOSE'
                    confidence = 1.0
                else:
                    signal = 'HOLD'
                    confidence = 0.0

            elif self.state.position == 'SHORT_SPREAD':
                if z_score <= self.exit_threshold:  # Changed < to <=
                    signal = 'CLOSE'
                    confidence = 1.0
                else:
                    signal = 'HOLD'
                    confidence = 0.0
            else:
                signal = 'HOLD'
                confidence = 0.0

        return signal, confidence

    @property
    def is_trained(self) -> bool:
        """Whether predictor is ready to predict"""
        return self._is_trained