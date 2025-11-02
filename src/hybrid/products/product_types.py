# src/hybrid/products/product_types.py
from abc import ABC, abstractmethod
from enum import Enum, auto

class ProductType(Enum):
    """Types of tradable products"""
    STOCK = "stock"
    ETF = "etf"
    FOREX = "forex"
    CRYPTO = "crypto"
    OPTIONS = "options"
    CFD = "cfd"


class PositionDirection(Enum):
    """Position direction (long or short)"""
    LONG = auto()
    SHORT = auto()


class Product(ABC):
    """
    Base class for tradable products

    Subclass this to add product-specific calculations and behavior
    (e.g., options Greeks, forex pip values, leveraged ETF decay)
    """

    def __init__(self, product_type: ProductType):
        self.product_type = product_type
        self.supports_long = True
        self.supports_short = self._get_short_support()

    @abstractmethod
    def _get_short_support(self) -> bool:
        """Determine if product supports short positions"""
        pass

    def can_trade_direction(self, direction: PositionDirection) -> bool:
        """Check if this product supports the given direction"""

        # if direction.name == PositionDirection.LONG.name:
        #     return self.supports_long
        # elif direction.name == PositionDirection.SHORT.name:
        #     return self.supports_short
        # return False

        if direction is PositionDirection.LONG:
            return self.supports_long
        elif direction is PositionDirection.SHORT:
            return self.supports_short
        return False

    def calculate_position_value(self, entry_price: float, current_price: float,
                                 quantity: float, direction: PositionDirection) -> float:
        """
        Calculate current position value (can be overridden for complex products)

        Default: Simple linear calculation
        Override: Options (Greeks), Futures (contract multiplier), etc.
        """
        if direction == PositionDirection.LONG:
            return (current_price - entry_price) * quantity
        else:  # SHORT
            return (entry_price - current_price) * quantity


class Stock(Product):
    """Stock product - long only"""

    def __init__(self):
        super().__init__(ProductType.STOCK)

    def _get_short_support(self) -> bool:
        return False


class ETF(Product):
    """ETF product - supports both directions"""

    def __init__(self, is_leveraged: bool = False):
        super().__init__(ProductType.ETF)
        self.is_leveraged = is_leveraged
        self.has_daily_reset = is_leveraged

    def _get_short_support(self) -> bool:
        return True


class Forex(Product):
    """Forex product - supports both directions"""

    def __init__(self, pip_value: float = 0.0001):
        super().__init__(ProductType.FOREX)
        self.pip_value = pip_value

    def _get_short_support(self) -> bool:
        return True


class Crypto(Product):
    """Crypto product - supports both directions"""

    def __init__(self):
        super().__init__(ProductType.CRYPTO)

    def _get_short_support(self) -> bool:
        return True


class Options(Product):
    """
    Options product - placeholder for future Greeks calculations

    Override calculate_position_value() to add:
    - Delta hedging
    - Gamma scalping
    - Theta decay
    - Vega exposure
    """

    def __init__(self):
        super().__init__(ProductType.OPTIONS)
        # Future: strike, expiry, option_type (call/put), greeks, etc.

    def _get_short_support(self) -> bool:
        return True

    def calculate_position_value(self, entry_price: float, current_price: float,
                                 quantity: float, direction: PositionDirection) -> float:
        """
        Future: Calculate options value using Black-Scholes and Greeks
        For now: Use simple linear calculation as placeholder
        """
        return super().calculate_position_value(entry_price, current_price, quantity, direction)


class CFD(Product):
    """Contract for Difference - linear payoff, both directions, leveraged"""

    def __init__(self, leverage: float = 1.0):
        super().__init__(ProductType.CFD)
        self.leverage = leverage

    def _get_short_support(self) -> bool:
        return True


class ProductFactory:
    """Factory to create Product instances from string type identifiers"""

    _product_map = {
        'stock': Stock,
        'etf': ETF,
        'forex': Forex,
        'crypto': Crypto,
        'options': Options,
        'cfd': CFD,
    }

    @classmethod
    def create_product(cls, product_type: str, **kwargs) -> Product:
        """
        Create Product instance from string identifier

        Args:
            product_type: String identifier ('stock', 'etf', 'forex', 'crypto', 'options')
            **kwargs: Additional product-specific parameters (e.g., is_leveraged for ETF)

        Returns:
            Product instance

        Raises:
            ValueError: If product_type is not recognized

        Examples:
            >>> ProductFactory.create_product('stock')
            <Stock instance>
            >>> ProductFactory.create_product('etf', is_leveraged=True)
            <ETF instance with daily reset>
        """
        product_type_lower = product_type.lower()

        if product_type_lower not in cls._product_map:
            raise ValueError(f"Unknown product type: {product_type}. "
                             f"Valid types: {list(cls._product_map.keys())}")

        product_class = cls._product_map[product_type_lower]
        return product_class(**kwargs)