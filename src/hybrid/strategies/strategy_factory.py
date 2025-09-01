# strategy_factory.py
# Factory for creating trading strategy instances
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

# strategy_factory.py
import sys
from pathlib import Path

import logging
from typing import Dict, Any
from src.hybrid.strategies.strategy_interface import StrategyInterface  # Changed from relative
from src.hybrid.strategies.implementation import BaseStrategy, HybridStrategy

logger = logging.getLogger(__name__)

class StrategyFactory:
    """Factory for creating trading strategy instances"""

    def __init__(self):
        self._strategies = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default strategy types"""

        self._strategies['base'] = BaseStrategy
        self._strategies['hybrid'] = HybridStrategy

    def register_strategy(self, name: str, strategy_class: type):
        """Register a new strategy type"""
        self._strategies[name] = strategy_class
        logger.debug(f"Registered strategy: {name}")

    def create_strategy(self, strategy_name: str, config: Any = None) -> StrategyInterface:
        """Create strategy instance by name"""
        if strategy_name not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

        strategy_class = self._strategies[strategy_name]
        strategy = strategy_class(name=strategy_name, config=config)

        logger.info(f"Created strategy: {strategy_name}")
        return strategy

    def get_available_strategies(self) -> list:
        """Get list of available strategy names"""
        return list(self._strategies.keys())