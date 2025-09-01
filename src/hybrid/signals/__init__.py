"""Signal package - interfaces and factory for trading signals"""
from .signal_interface import SignalInterface
from .signal_factory import SignalFactory

__all__ = ['SignalInterface', 'SignalFactory']