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

    def set_money_manager(self, money_manager: Any) -> None:
        """Inject MoneyManager dependency"""
        ...

    def set_data_manager(self, data_manager: Any) -> None:
        """Inject DataManager dependency"""
        ...

    def add_entry_signal(self, signal: Any) -> None:
        """Add signal generator to strategy"""
        ...

    def add_exit_signal(self, signal: Any) -> None:
        """Add signal generator to strategy"""
        ...

    def add_optimizer(self, optimization: Any) -> None:
        """Add optimization component to strategy"""
        ...

    def add_predictor(self, predictor: Any) -> None:
        """Add predictor component to strategy"""
        ...

    def add_metric(self, metric: Any) -> None:
        """Add metric component to strategy"""
        ...

    def add_verificator(self, verificator: Any) -> None:
        """Add verificator component to strategy"""
        ...

    def run(self) -> Dict:
        """Run complete backtest for this strategy"""
        ...

    def set_position_orchestrator(self, position_orchestrator):
        ...

    def get_optimizable_parameters(self) -> Dict:
        ...
