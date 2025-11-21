# types.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from src.hybrid.products.product_types import PositionDirection


@dataclass
class TradingSignal:
    """Trading signal with entry parameters"""
    symbol: str
    direction: PositionDirection
    signal_strength: float  # Signal strength 0.0 to 1.0
    entry_price: float
    timestamp: pd.Timestamp


@dataclass
class Position:
    """Current position information"""
    trade_id: str
    symbol: str
    direction: PositionDirection
    size: int
    entry_price: float
    current_price: float
    entry_time: datetime
    last_update: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_equity: float
    available_cash: float
    positions: Dict[str, Position]
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0