# strategy_factory.py
# Factory for creating trading strategy instances
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

# strategy_factory.py
import logging
from pathlib import Path
from typing import Dict, Any

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data import DataManager
from src.hybrid.money_management import MoneyManager
from src.hybrid.positions.position_orchestrator import PositionOrchestrator
from src.hybrid.signals import SignalFactory
from src.hybrid.strategies.strategy_interface import StrategyInterface  # Changed from relative
from src.hybrid.strategies.implementation import BaseStrategy, ChainedStrategy

logger = logging.getLogger(__name__)

class StrategyFactory:
    """Factory for creating trading strategy instances"""

    def __init__(self):
        self._strategies = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default strategy types"""

        self._strategies['base'] = BaseStrategy
        self._strategies['chained'] = ChainedStrategy

    def register_strategy(self, name: str, strategy_class: type):
        """Register a new strategy type"""
        self._strategies[name] = strategy_class
        logger.debug(f"Registered strategy: {name}")

    def get_available_strategies(self) -> list:
        """Get list of available strategy names"""
        return list(self._strategies.keys())

    def _create_and_wire_signals(self, strategy: StrategyInterface, config: Any) -> None:
        """
        Create and wire signals to strategy

        Args:
            strategy: Strategy instance to wire signals to
            config: Configuration containing signal definitions
        """
        signal_factory = SignalFactory(config)
        entry_signal_name = config.get_section('strategy', {}).get('entry_signal')
        exit_signal_name = config.get_section('strategy', {}).get('exit_signal')

        if entry_signal_name:
            entry_signal = signal_factory.create_signal(entry_signal_name, config)
            strategy.add_entry_signal(entry_signal)

        if exit_signal_name:
            exit_signal = signal_factory.create_signal(exit_signal_name, config)
            strategy.add_exit_signal(exit_signal)

    def create_strategy_isolated(self, strategy_name: str, config: Any,
                                 initial_capital: float,
                                 project_root: Path) -> StrategyInterface:
        """Create strategy with isolated dependencies (for optimization)

        Args:
            project_root: Required explicit path injection
        """
        if strategy_name not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

        data_manager = DataManager(config, project_root)
        data_manager.load_market_data()

        money_manager = MoneyManager(config)
        position_orchestrator = PositionOrchestrator(config)
        position_orchestrator.set_initial_capital(initial_capital)
        money_manager.set_position_orchestrator(position_orchestrator)

        strategy_class = self._strategies[strategy_name]
        strategy = strategy_class(name=strategy_name, config=config)

        self._create_and_wire_signals(strategy, config)

        strategy.set_data_manager(data_manager)
        strategy.set_money_manager(money_manager)
        strategy.set_position_orchestrator(position_orchestrator)

        logger.info(f"Created isolated strategy: {strategy_name}")
        return strategy

    def create_strategy_shared(self,
                               strategy_name: str,
                               config: Any,
                               data_manager: DataManager,
                               money_manager: MoneyManager,
                               position_orchestrator: PositionOrchestrator) -> StrategyInterface:
        """Create strategy with shared dependencies (for backtest/trading)"""
        if strategy_name not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

        # Create strategy
        strategy_class = self._strategies[strategy_name]
        strategy = strategy_class(name=strategy_name, config=config)

        # Wire signals
        self._create_and_wire_signals(strategy, config)

        # Wire shared dependencies
        strategy.set_data_manager(data_manager)
        strategy.set_money_manager(money_manager)
        strategy.set_position_orchestrator(position_orchestrator)

        logger.info(f"Created shared strategy: {strategy_name}")
        return strategy

    def create_strategy_with_params(self, strategy_name: str, base_config: Any, params: Dict,
                                    initial_capital: float,project_root: Path) -> StrategyInterface:
        """Create strategy with optimization parameters applied"""
        config = UnifiedConfig(base_config.config_path)
        config.config = base_config.config.copy()

        if params:
            # Map params to nested structure
            nested_params = self._map_params_to_config_structure(params, config)
            config.update_config(nested_params)

        return self.create_strategy_isolated(
            strategy_name, config, initial_capital, project_root)

    @staticmethod
    def find_param_location(param_name, section, path=[]):  # ← Keep original name
        """Find ALL occurrences of a parameter in config tree"""
        locations = []

        if isinstance(section, dict):
            if 'optimizable_parameters' in section:
                if param_name in section['optimizable_parameters']:
                    param_def = section['optimizable_parameters'][param_name]
                    location = path + ['parameters', param_name]
                    param_type = param_def.get('type')
                    locations.append((location, param_type))

            for key, value in section.items():
                nested_results = StrategyFactory.find_param_location(param_name, value, path + [key])
                locations.extend(nested_results)

        return locations

    def _map_params_to_config_structure(self, params: Dict, config) -> Dict:
        """Map optimization params to ALL matching locations in config"""
        updates = {}

        for param_name, param_value in params.items():
            # Find ALL occurrences
            all_locations = StrategyFactory.find_param_location(param_name, config.config)

            if not all_locations:
                print(f"WARNING: Parameter '{param_name}' not found in config!")
                continue

            # Update ALL occurrences
            for location, param_type in all_locations:
                # Convert type if needed
                converted_value = param_value
                if param_type == 'integer':
                    converted_value = int(round(param_value))

                # Build nested update dict
                current = updates
                for key in location[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[location[-1]] = converted_value

        return updates

class StrategyFactoryCallable:
    def __init__(self, base_config, strategy_name,
                 initial_capital,project_root):
        self.base_config = base_config
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.project_root = project_root

    def __call__(self, params):
        strategy_factory = StrategyFactory()
        return strategy_factory.create_strategy_with_params(
            strategy_name=self.strategy_name,
            base_config=self.base_config,
            params=params,
            initial_capital=self.initial_capital,
            project_root=self.project_root
        )