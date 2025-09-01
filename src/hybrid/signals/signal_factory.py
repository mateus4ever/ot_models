"""
SignalFactory - Factory pattern for creating signal instances
Auto-discovers available signals and provides unified creation interface
"""

import logging
from typing import Dict, List, Any, Type
from pathlib import Path
import importlib
import inspect

from .signal_interface import SignalInterface

logger = logging.getLogger(__name__)


class SignalFactory:
    """
    Factory for creating signal instances with auto-discovery

    Automatically discovers all signal implementations in the signals package
    and provides unified interface for signal creation and management
    """

    def __init__(self):
        self._signal_registry: Dict[str, Type[SignalInterface]] = {}
        self._discover_signals()
        logger.info(f"SignalFactory initialized with {len(self._signal_registry)} signals")

    def _discover_signals(self) -> None:
        """
        Auto-discover available signal implementations with recursive directory scanning

        Scans implementation/ and all subdirectories for signal classes
        Creates namespaced signal names like 'momentum.rsi' or 'trend_following.sma_crossover'
        """
        try:
            # Get the signals/implementation directory
            implementation_dir = Path(__file__).parent / 'implementation'

            # Recursively find all Python files in implementation and subdirectories
            for py_file in implementation_dir.rglob("*.py"):
                if py_file.name.startswith('__'):
                    continue

                # Calculate relative path from implementation directory
                relative_path = py_file.relative_to(implementation_dir)

                # Build module path (e.g., momentum.rsi_signal)
                module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
                module_path = '.'.join(module_parts)

                try:
                    # Import the module dynamically with full path
                    if module_parts[:-1]:  # Has subdirectory
                        full_module_path = f'.implementation.{module_path}'
                    else:  # Direct in implementation directory
                        full_module_path = f'.implementation.{module_path}'

                    module = importlib.import_module(full_module_path, package=__package__)

                    # Find classes that implement SignalInterface
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (obj != SignalInterface and
                                hasattr(obj, '__annotations__') and
                                self._implements_signal_interface(obj)):
                            # Create namespaced signal name
                            signal_name = self._create_namespaced_name(relative_path, name)
                            self._signal_registry[signal_name] = obj
                            logger.debug(f"Discovered signal: {signal_name} -> {obj.__name__}")

                except ImportError as e:
                    logger.warning(f"Could not import signal module {module_path}: {e}")

        except Exception as e:
            logger.error(f"Error during signal discovery: {e}")

    def _create_namespaced_name(self, relative_path: Path, class_name: str) -> str:
        """
        Create namespaced signal name from file path and class name

        Args:
            relative_path: Path relative to implementation directory
            class_name: Python class name

        Returns:
            Namespaced signal name (e.g., 'momentum.rsi', 'trend_following.sma_crossover')
        """
        # Get directory parts (e.g., ['momentum'] or ['trend_following'])
        path_parts = relative_path.parts[:-1]  # Exclude filename

        # Convert class name to signal name
        signal_base_name = self._get_signal_name(class_name)

        # Create full namespaced name
        if path_parts:
            namespace = '.'.join(path_parts)
            return f"{namespace}.{signal_base_name}"
        else:
            return signal_base_name

    def _implements_signal_interface(self, cls: Type) -> bool:
        """
        Check if class implements SignalInterface protocol

        Args:
            cls: Class to check

        Returns:
            True if class implements SignalInterface
        """
        required_methods = ['train', 'update_with_new_data', 'generate_signal', 'getMetrics']

        for method_name in required_methods:
            if not hasattr(cls, method_name) or not callable(getattr(cls, method_name)):
                return False

        return True

    def _get_signal_name(self, class_name: str) -> str:
        """
        Convert class name to signal name

        Args:
            class_name: Python class name (e.g., 'BollingerSignal')

        Returns:
            Signal name for factory (e.g., 'bollinger')
        """
        # Remove 'Signal' suffix if present and convert to lowercase
        name = class_name.replace('Signal', '').lower()
        return name

    def create_signal(self, signal_name: str, config: Dict[str, Any] = None) -> SignalInterface:
        """
        Create signal instance by name

        Args:
            signal_name: Name of signal to create (e.g., 'bollinger', 'rsi')
            config: Configuration parameters for signal initialization

        Returns:
            Configured signal instance

        Raises:
            ValueError: If signal name not found or creation fails
        """
        if signal_name not in self._signal_registry:
            raise ValueError(f"Unknown signal: {signal_name}")

        signal_class = self._signal_registry[signal_name]

        try:
            # Create signal instance with configuration - pass config as single parameter
            signal = signal_class(config)

            logger.info(f"Created signal: {signal_name} ({signal_class.__name__})")
            return signal

        except Exception as e:
            logger.error(f"Failed to create signal {signal_name}: {e}")
            raise ValueError(f"Failed to create signal {signal_name}: {e}")

    def get_available_signals(self) -> List[str]:
        """
        Get list of available signal names

        Returns:
            List of available signal names
        """
        return list(self._signal_registry.keys())

    def get_signals_by_category(self, category: str) -> List[str]:
        """
        Get all signals in a specific category

        Args:
            category: Category name (e.g., 'momentum', 'trend_following')

        Returns:
            List of signal names in the category
        """
        print(f"DEBUG get_signals_by_category: registry={self._signal_registry}")
        print(f"DEBUG get_signals_by_category: category={category}")
        category_signals = []
        for signal_name in self._signal_registry.keys():
            if '.' in signal_name and signal_name.startswith(f"{category}."):
                category_signals.append(signal_name)
        print(f"DEBUG get_signals_by_category: returning={category_signals}")
        return category_signals

    def get_available_categories(self) -> List[str]:
        """
        Get list of available signal categories

        Returns:
            List of unique categories
        """
        categories = set()
        for signal_name in self._signal_registry.keys():
            if '.' in signal_name:
                category = signal_name.split('.')[0]
                categories.add(category)
        return sorted(list(categories))

    def get_signal_info(self, signal_name: str) -> Dict[str, Any]:
        """
        Get information about a specific signal

        Args:
            signal_name: Name of signal to query

        Returns:
            Dictionary with signal information

        Raises:
            ValueError: If signal name not found
        """
        if signal_name not in self._signal_registry:
            raise ValueError(f"Unknown signal: {signal_name}")

        signal_class = self._signal_registry[signal_name]

        # Extract constructor parameters
        sig = inspect.signature(signal_class.__init__)
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                params[param_name] = {
                    'type': param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': param.default if param.default != inspect.Parameter.empty else None
                }

        return {
            'class_name': signal_class.__name__,
            'module': signal_class.__module__,
            'docstring': signal_class.__doc__ or 'No documentation available',
            'parameters': params
        }

    def validate_signal_config(self, signal_name: str, config: Dict[str, Any]) -> bool:
        if signal_name not in self._signal_registry:
            return False

        signal_class = self._signal_registry[signal_name]

        # Check required parameters directly
        # todo: do static method exist in python?
        required_params = signal_class.get_required_parameters(signal_class)
        return all(param in config for param in required_params)
