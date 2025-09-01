# base_strategy.py
# Base strategy implementation
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

import logging
from typing import Any, Dict
from ..strategy_interface import StrategyInterface

logger = logging.getLogger(__name__)


class BaseStrategy(StrategyInterface):
    """Base strategy implementation"""

    def __init__(self, name: str, config: Any = None):
        self.name = name
        self.config = config
        self.money_manager = None
        self.data_manager = None
        self.signals = []
        self.optimizations = []
        self.predictors = []
        self.runners = []
        self.metrics = []
        self.verificators = []

    def setMoneyManager(self, money_manager: Any) -> None:
        """Inject MoneyManager dependency"""
        self.money_manager = money_manager

    def setDataManager(self, data_manager: Any) -> None:
        """Inject DataManager dependency"""
        self.data_manager = data_manager

    def addSignal(self, signal: Any) -> None:
        """Add signal generator to strategy"""
        self.signals.append(signal)

    def addOptimization(self, optimization: Any) -> None:
        """Add optimization component to strategy"""
        self.optimizations.append(optimization)

    def addPredictor(self, predictor: Any) -> None:
        """Add predictor component to strategy"""
        self.predictors.append(predictor)

    def addRunner(self, runner: Any) -> None:
        """Add runner component to strategy"""
        self.runners.append(runner)

    def addMetric(self, metric: Any) -> None:
        """Add metric component to strategy"""
        self.metrics.append(metric)

    def addVerificator(self, verificator: Any) -> None:
        """Add verificator component to strategy"""
        self.verificators.append(verificator)

    def initialize(self, market_data: Dict) -> bool:
        """Initialize strategy with market data"""
        logger.debug(f"Initializing {self.name} strategy")
        return True

    def generate_signals(self, data: Dict) -> Any:
        """Generate trading signals"""
        logger.debug(f"Generating signals for {self.name}")
        # Simple base implementation
        return [{'signal': 'HOLD', 'confidence': 0.5}]

    def execute_trades(self, signals: Any) -> Dict:
        """Execute trades based on signals"""
        logger.debug(f"Executing trades for {self.name}")
        # Simple base implementation
        return {
            'trades': len(signals),
            'profit': 0.0,
            'strategy': self.name
        }

    def run_backtest(self, market_data: Dict) -> Dict:
        """Run complete backtest for this strategy"""
        logger.info(f"Running backtest for {self.name}")

        if not self.initialize(market_data):
            return {'error': f'Strategy {self.name} initialization failed'}

        signals = self.generate_signals(market_data)
        results = self.execute_trades(signals)

        return results