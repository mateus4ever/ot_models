# src/hybrid/results/backtest_result.py
"""
BacktestResult - Core result container with extensibility

Provides:
- Strict typing for core fields
- Lazy metric calculation
- Multiple save/load formats
- Extensibility via custom_data
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.hybrid.backtesting.metrics_calculator import MetricsCalculator
from src.hybrid.backtesting.performance_metrics import PerformanceMetrics
from src.hybrid.positions.base_trade_history import BaseTradeHistory
from src.hybrid.positions.leg_trade_history import LegTradeHistory
from src.hybrid.positions.spread_trade_history import SpreadTradeHistory


@dataclass
class BacktestResult:
    """
    Hybrid backtest result container

    Core fields are strictly typed and always present.
    Custom data can be added via custom_data dict for strategy-specific metrics.
    Metrics are lazy-calculated from trade history.
    """

    # === CORE FIELDS (Required) ===
    strategy_name: str
    market_id: str
    config: Dict[str, Any]
    trade_histories: Dict[str, BaseTradeHistory]

    timestamp: datetime = field(default_factory=datetime.now)

    # === OPTIONAL CALCULATED FIELDS ===
    equity_curve: Optional[List[float]] = None
    execution_time_seconds: Optional[float] = None

    # === EXTENSIBILITY ===
    custom_data: Dict[str, Any] = field(default_factory=dict)

    # === LAZY PROPERTIES ===
    _metrics: Optional['PerformanceMetrics'] = field(default=None, init=False, repr=False)

    @property
    def metrics(self) -> 'PerformanceMetrics':
        """
        Lazy calculate performance metrics from primary trade history

        Returns:
            PerformanceMetrics object with calculated statistics
        """
        if self._metrics is None:
            calculator = MetricsCalculator(self.config)
            self._metrics = calculator.calculate_metrics(
                self.primary_trade_history,
                self.equity_curve,
                self.config.get('backtesting')
            )
        return self._metrics

    def get_fitness(self, metric_name: str) -> float:
        """
        Get fitness value for optimizer

        Args:
            metric_name: Name of metric to use as fitness (e.g., 'sharpe_ratio', 'total_return')

        Returns:
            Numeric fitness value

        Raises:
            AttributeError: If metric doesn't exist
        """
        return getattr(self.metrics, metric_name)

    def add_custom_metric(self, name: str, value: Any) -> None:
        """
        Add strategy-specific custom metric

        Args:
            name: Metric name
            value: Metric value (any serializable type)
        """
        self.custom_data[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary

        Returns:
            Dictionary representation of result
        """
        return {
            'strategy_name': self.strategy_name,
            'market_id': self.market_id,
            'timestamp': self.timestamp.isoformat(),
            'execution_time_seconds': self.execution_time_seconds,
            'equity_curve': self.equity_curve,
            'config': self.config,
            'custom_data': self.custom_data,
            'metrics': self._metrics.to_dict() if self._metrics else None,
            'trade_counts': {name: len(h.trades) for name, h in self.trade_histories.items()}
        }

    def save_full(self, path: str) -> bool:
        """
        Save complete backtest - everything including pre-calculated metrics

        Args:
            path: File path for JSON output

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'strategy_name': self.strategy_name,
                'market_id': self.market_id,
                'timestamp': self.timestamp.isoformat(),
                'execution_time_seconds': self.execution_time_seconds,
                'equity_curve': self.equity_curve,
                'config': self.config,
                'custom_data': self.custom_data,
                'metrics': self.metrics.to_dict(),
                'trade_histories': {
                    name: {
                        'type': h.__class__.__name__,
                        'trades': list(h.trades.values())
                    }
                    for name, h in self.trade_histories.items()
                },
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'trade_counts': {name: len(h.trades) for name, h in self.trade_histories.items()},
                    'format_version': '2.0'
                }
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            return True

        except Exception as e:
            import logging
            logging.error(f"Error saving full result: {e}")
            return False

    def save_trades_only(self, path: str) -> bool:
        """
        Save all trade histories - lightweight format

        Args:
            path: File path for trade histories JSON

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'trade_histories': {
                    name: {
                        'type': h.__class__.__name__,
                        'trades': list(h.trades.values())
                    }
                    for name, h in self.trade_histories.items()
                }
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            return True
        except Exception as e:
            import logging
            logging.error(f"Error saving trades: {e}")
            return False

    def to_csv(self, path: str) -> bool:
        """
        Export trades to CSV file

        Args:
            path: File path for CSV output

        Returns:
            True if successful, False otherwise
        """
        try:
            import pandas as pd

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            all_trades = []
            for name, history in self.trade_histories.items():
                for trade in history.trades.values():
                    trade_copy = dict(trade)
                    trade_copy['history_type'] = name
                    all_trades.append(trade_copy)

            if not all_trades:
                df = pd.DataFrame(columns=['history_type', 'timestamp', 'direction', 'entry_price',
                                           'exit_price', 'quantity', 'gross_pnl', 'net_pnl'])
            else:
                df = pd.DataFrame(all_trades)

            df.to_csv(path, index=False)
            return True

        except Exception as e:
            import logging
            logging.error(f"Error exporting to CSV: {e}")
            return False

    @classmethod
    def load_from_full(cls, path: str) -> 'BacktestResult':
        """
        Load complete backtest from full save file

        Args:
            path: Path to full result JSON

        Returns:
            BacktestResult instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        # Factory for trade history types
        history_classes = {
            'LegTradeHistory': LegTradeHistory,
            'SpreadTradeHistory': SpreadTradeHistory
        }

        # Reconstruct trade histories
        trade_histories = {}
        for name, history_data in data.get('trade_histories', {}).items():
            history_type = history_data.get('type', 'LegTradeHistory')
            history_class = history_classes.get(history_type)
            if history_class is None:
                raise ValueError(f"Unknown trade history type: {history_type}")

            history = history_class(data['config'])
            for trade in history_data.get('trades', []):
                history.add_trade(trade)

            trade_histories[name] = history

        if not trade_histories:
            raise ValueError("No trade histories found in file")

        # Create result
        result = cls(
            strategy_name=data['strategy_name'],
            market_id=data['market_id'],
            trade_histories=trade_histories,
            config=data['config'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            equity_curve=data.get('equity_curve'),
            execution_time_seconds=data.get('execution_time_seconds'),
            custom_data=data.get('custom_data', {})
        )

        # Pre-load metrics if available
        if 'metrics' in data and data['metrics']:
            result._metrics = PerformanceMetrics.from_dict(data['metrics'])

        return result

    @classmethod
    def load_from_trades(cls, trades_path: str, config: Dict[str, Any],
                         strategy_name: str = "Loaded",
                         market_id: str = "Unknown") -> 'BacktestResult':
        """
        Load from trade history only, recalculate metrics on demand

        Args:
            trades_path: Path to trade history JSON
            config: Configuration dict for metric calculation
            strategy_name: Optional strategy name
            market_id: Optional market identifier

        Returns:
            BacktestResult instance (metrics calculated lazily)
        """
        path = Path(trades_path)

        if not path.exists():
            raise FileNotFoundError(f"Trades file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        history_classes = {
            'LegTradeHistory': LegTradeHistory,
            'SpreadTradeHistory': SpreadTradeHistory
        }

        trade_histories = {}
        for name, history_data in data.get('trade_histories', {}).items():
            history_type = history_data.get('type', 'LegTradeHistory')
            history_class = history_classes.get(history_type)
            if history_class is None:
                raise ValueError(f"Unknown trade history type: {history_type}")

            history = history_class(config)
            for trade in history_data.get('trades', []):
                history.add_trade(trade)

            trade_histories[name] = history

        if not trade_histories:
            raise ValueError(f"No trade histories found in: {trades_path}")

        return cls(
            strategy_name=strategy_name,
            market_id=market_id,
            trade_histories=trade_histories,
            config=config
        )

    def generate_filename(self, prefix: str = "", suffix: str = "", extension: str = ".json") -> str:
        """Generate filename with timestamp"""
        timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
        parts = [prefix, self.strategy_name, self.market_id, timestamp, suffix]
        name = "_".join(p for p in parts if p)
        return f"{name}{extension}"

    def save_with_config(self, output_dir: str) -> Dict[str, str]:
        """Save result and configuration together"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Result
        result_path = output_dir / self.generate_filename(prefix="result")
        self.save_full(str(result_path))
        saved['result'] = str(result_path)

        # Config
        config_path = output_dir / self.generate_filename(prefix="config")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        saved['config'] = str(config_path)

        # Trades CSV
        csv_path = output_dir / self.generate_filename(prefix="trades", extension=".csv")
        self.to_csv(str(csv_path))
        saved['trades_csv'] = str(csv_path)

        return saved

    def __post_init__(self):
        if not self.trade_histories:
            raise ValueError("At least one trade history required")

    def get_trade_history(self, name: str) -> BaseTradeHistory:
        """Get specific trade history by name"""
        if name not in self.trade_histories:
            raise ValueError(f"Trade history '{name}' not found")
        return self.trade_histories[name]
    @property
    def primary_trade_history(self) -> BaseTradeHistory:
        """Get primary trade history for metrics (spread if exists, else first)"""
        if 'spread' in self.trade_histories:
            return self.trade_histories['spread']
        return next(iter(self.trade_histories.values()))