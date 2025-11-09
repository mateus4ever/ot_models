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
from src.hybrid.data.trade_history import TradeHistory


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
    trade_history: TradeHistory
    config: Dict[str, Any]
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
        Lazy calculate performance metrics from trade history

        Returns:
            PerformanceMetrics object with calculated statistics
        """
        if self._metrics is None:

            calculator = MetricsCalculator(self.config)
            self._metrics = calculator.calculate_metrics(
                self.trade_history,
                self.equity_curve,
                self.config.get('backtesting', {}).get('initial_capital', 10000)
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
            'trade_count': len(self.trade_history.trades)
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
                'metrics': self.metrics.to_dict(),  # Force calculation and save
                'trades': list(self.trade_history.trades.values()),
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'trade_count': len(self.trade_history.trades),
                    'format_version': '1.0'
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
        Save just trade history - lightweight format

        Args:
            path: File path for trade history JSON

        Returns:
            True if successful, False otherwise
        """
        return self.trade_history.save_to_json(path)

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

            # Get all trades as list
            trades = list(self.trade_history.trades.values())

            if not trades:
                # Empty CSV with headers only
                df = pd.DataFrame(columns=['timestamp', 'direction', 'entry_price',
                                           'exit_price', 'quantity', 'gross_pnl', 'net_pnl'])
            else:
                df = pd.DataFrame(trades)

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

        # Reconstruct trade history
        trade_history = TradeHistory(data['config'])
        for trade in data.get('trades', []):
            trade_history.add_trade(trade)

        # Create result
        result = cls(
            strategy_name=data['strategy_name'],
            market_id=data['market_id'],
            trade_history=trade_history,
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
        trade_history = TradeHistory(config)
        success = trade_history.load_from_json(trades_path)

        if not success:
            raise ValueError(f"Failed to load trade history from: {trades_path}")

        return cls(
            strategy_name=strategy_name,
            market_id=market_id,
            trade_history=trade_history,
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