# strategy_interface.py
# Protocol-based strategy interface for trading strategies
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

from typing import Protocol, Any, Dict


class StrategyInterface(Protocol):
    """Protocol defining the interface for all trading strategies"""

    name: str
    money_manager: Any
    data_manager: Any
    signals: list
    optimizations: list
    predictors: list
    runners: list
    metrics: list
    verificators: list

    def setMoneyManager(self, money_manager: Any) -> None:
        """Inject MoneyManager dependency"""
        ...

    def setDataManager(self, data_manager: Any) -> None:
        """Inject DataManager dependency"""
        ...

    def addSignal(self, signal: Any) -> None:
        """Add signal generator to strategy"""
        ...

    def addOptimization(self, optimization: Any) -> None:
        """Add optimization component to strategy"""
        ...

    def addPredictor(self, predictor: Any) -> None:
        """Add predictor component to strategy"""
        ...

    def addRunner(self, runner: Any) -> None:
        """Add runner component to strategy"""
        ...

    def addMetric(self, metric: Any) -> None:
        """Add metric component to strategy"""
        ...

    def addVerificator(self, verificator: Any) -> None:
        """Add verificator component to strategy"""
        ...

    def initialize(self, market_data: Dict) -> bool:
        """Initialize strategy with market data"""
        ...

    def generate_signals(self, data: Dict) -> Any:
        """Generate trading signals"""
        ...

    def execute_trades(self, signals: Any) -> Dict:
        """Execute trades based on signals"""
        ...

    def run_backtest(self, market_data: Dict) -> Dict:
        """Run complete backtest for this strategy"""
        ...