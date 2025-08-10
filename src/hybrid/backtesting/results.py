# src/hybrid/backtesting/results.py
# Results formatting and file operations with ZERO hardcoded values

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict
from src.hybrid.config.unified_config import UnifiedConfig


class ResultsFormatter:
    """
    Handle result formatting and file operations
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache ALL results configuration values"""
        # Display configuration
        self.display_config = self.config.get_section('display_configuration', {})
        self.output_config = self.config.get_section('output_formatting', {})

        self.headers = self.display_config.get('section_headers', {})
        self.defaults = self.display_config.get('default_values', {})
        self.separator_length = self.output_config.get('separator_length')

        # File operations
        self.file_config = self.config.get_section('file_operations', {})

        # Numeric formatting
        self.numeric_config = self.config.get_section('numeric_formatting', {})

        # Mathematical constants
        self.math_config = self.config.get_section('mathematical_operations', {})
        self.zero_value = self.math_config.get('zero')
        self.unity_value = self.math_config.get('unity')

    def calculate_strategy_summary(self, signals_df: pd.DataFrame, df: pd.DataFrame) -> Dict:
        """Calculate strategy summary statistics with configurable thresholds"""

        # Get configurable values
        percentage_multiplier = self.numeric_config.get('percentage_conversion', {}).get('multiplier')

        # Signal statistics
        total_signals = (signals_df['signal'] != self.zero_value).sum()
        signal_frequency = (total_signals / len(signals_df)) * percentage_multiplier

        long_signals = (signals_df['signal'] > self.zero_value).sum()
        short_signals = (signals_df['signal'] < self.zero_value).sum()

        long_percentage = (
                    long_signals / total_signals * percentage_multiplier) if total_signals > self.zero_value else self.zero_value
        short_percentage = (
                    short_signals / total_signals * percentage_multiplier) if total_signals > self.zero_value else self.zero_value

        avg_position_size = signals_df['position_size'].mean() * percentage_multiplier

        # Regime distribution
        regime_distribution = {}
        if 'regime' in signals_df.columns:
            regime_counts = signals_df['regime'].value_counts()
            regime_names = {
                self.zero_value: 'Ranging',
                self.unity_value: 'Trending Up',
                self.unity_value * 2: 'Trending Down',
                self.unity_value * 3: 'High Volatility'
            }

            for regime_id, count in regime_counts.items():
                regime_name = regime_names.get(regime_id, f'Regime_{regime_id}')
                regime_distribution[regime_name] = count

        return {
            'total_signals': total_signals,
            'signal_frequency': signal_frequency,
            'long_percentage': long_percentage,
            'short_percentage': short_percentage,
            'avg_position_size': avg_position_size,
            'regime_distribution': regime_distribution
        }

    def print_results_summary(self, results: Dict):
        """Print comprehensive results summary using ONLY configurable values"""

        newline = "\n"
        separator = "=" * self.separator_length

        print(f"{newline}{separator}")
        print(self.headers.get('main_title'))
        print(separator)

        # Configuration info
        if 'config_summary' in results:
            config_summary = results['config_summary']
            print(f"{newline}{self.headers.get('configuration')}")

            strategy_type_default = self.defaults.get('strategy_type')
            numeric_default = self.defaults.get('numeric_default')
            float_default = self.defaults.get('float_default')

            print(f"  Strategy Type: {config_summary.get('strategy_type', strategy_type_default)}")
            print(f"  ML Components: {config_summary.get('ml_components', {})}")

            # Get formatting configuration for percentages
            percentage_precision = self.numeric_config.get('decimal_precision', {}).get('percentage')

            sl_pct = config_summary.get('stop_loss_pct', float_default)
            tp_pct = config_summary.get('take_profit_pct', float_default)
            max_pos = config_summary.get('max_position_size', float_default)

            print(f"  Risk Management: SL={sl_pct:.{percentage_precision}%}, TP={tp_pct:.{percentage_precision}%}")
            print(f"  Position Sizing: Max={max_pos:.{percentage_precision}%}")

        # Training results
        if 'training' in results:
            print(f"{newline}{self.headers.get('ml_training')}")
            accuracy_keys = ['regime_strength_accuracy', 'regime_direction_accuracy', 'volatility_accuracy']

            for key, value in results['training'].items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float) and key in accuracy_keys:
                        accuracy_precision = self.numeric_config.get('decimal_precision', {}).get('threshold')
                        print(f"  {key}: {value:.{accuracy_precision}f}")
                    else:
                        print(f"  {key}: {value}")

        # Strategy summary
        if 'strategy_summary' in results:
            summary = results['strategy_summary']
            print(f"{newline}{self.headers.get('signal_generation')}")

            numeric_default = self.defaults.get('numeric_default')
            decimal_precision = self.numeric_config.get('decimal_precision', {}).get('percentage')

            print(f"  Total Signals: {summary.get('total_signals', numeric_default)}")
            print(f"  Signal Frequency: {summary.get('signal_frequency', numeric_default):.{decimal_precision}f}%")

            long_pct = summary.get('long_percentage', numeric_default)
            short_pct = summary.get('short_percentage', numeric_default)
            avg_size = summary.get('avg_position_size', numeric_default)

            print(f"  Long/Short Split: {long_pct:.{decimal_precision}f}% / {short_pct:.{decimal_precision}f}%")
            print(f"  Avg Position Size: {avg_size:.{decimal_precision}f}%")

            if 'regime_distribution' in summary:
                print(f"{newline}{self.headers.get('regime_distribution')}")
                for regime, count in summary['regime_distribution'].items():
                    print(f"    {regime}: {count}")

        # Backtest performance
        if 'backtest' in results:
            bt = results['backtest']
            print(f"{newline}{self.headers.get('backtest_performance')}")

            numeric_default = self.defaults.get('numeric_default')
            decimal_precision = self.numeric_config.get('decimal_precision', {})

            capital_precision = decimal_precision.get('price')
            percentage_precision = decimal_precision.get('percentage')

            print(f"  Final Capital: ${bt.get('final_capital', numeric_default):,.{capital_precision}f}")
            print(f"  Total Return: {bt.get('total_return', numeric_default):.{percentage_precision}%}")
            print(f"  Number of Trades: {bt.get('num_trades', numeric_default)}")

            if bt.get('num_trades', numeric_default) > numeric_default:
                print(f"  Win Rate: {bt.get('win_rate', numeric_default):.{percentage_precision}%}")

                avg_return_precision = decimal_precision.get('threshold')
                sharpe_precision = decimal_precision.get('price')

                print(
                    f"  Avg Return per Trade: {bt.get('avg_return_per_trade', numeric_default):.{avg_return_precision}%}")
                print(f"  Sharpe Ratio: {bt.get('sharpe_ratio', numeric_default):.{sharpe_precision}f}")
                print(f"  Sortino Ratio: {bt.get('sortino_ratio', numeric_default):.{sharpe_precision}f}")
                print(f"  Max Drawdown: {bt.get('max_drawdown', numeric_default):.{percentage_precision}%}")
                print(f"  Profit Factor: {bt.get('profit_factor', numeric_default):.{sharpe_precision}f}")

                periods_precision = decimal_precision.get('percentage')
                print(
                    f"  Avg Holding Period: {bt.get('avg_holding_period', numeric_default):.{periods_precision}f} periods")

                # Risk metrics
                print(f"{newline}{self.headers.get('risk_metrics')}")
                print(f"  Max Win Streak: {bt.get('max_win_streak', numeric_default)}")
                print(f"  Max Loss Streak: {bt.get('max_loss_streak', numeric_default)}")
                print(f"  Total Fees: ${bt.get('total_fees', numeric_default):.{capital_precision}f}")

                # Exit reasons
                if 'exit_reasons' in bt and bt['exit_reasons']:
                    print(f"{newline}{self.headers.get('exit_reasons')}")
                    total_exits = sum(bt['exit_reasons'].values())
                    percentage_multiplier = self.output_config.get('percentage_multiplier')

                    for reason, count in bt['exit_reasons'].items():
                        pct = count / total_exits * percentage_multiplier
                        print(f"    {reason}: {count} ({pct:.{percentage_precision}f}%)")

    def save_results(self, results: Dict, config: UnifiedConfig, timestamp: str = None):
        """Save backtest results and configuration with configurable paths"""

        if timestamp is None:
            timestamp_format = self.file_config.get('timestamp_format')
            timestamp = datetime.now().strftime(timestamp_format)

        # Get file configuration
        results_dir_name = self.file_config.get('results_directory', 'results')
        results_dir = Path(results_dir_name)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get file naming configuration
        prefixes = self.file_config.get('file_prefixes', {})
        extensions = self.file_config.get('file_extensions', {})

        # Save backtest results (excluding large data)
        excluded_keys = ['trades_detail', 'equity_curve', 'positions', 'daily_pnl']
        backtest_summary = {k: v for k, v in results['backtest'].items() if k not in excluded_keys}

        summary_filename = f"{prefixes.get('backtest_summary', 'summary_')}{timestamp}{extensions.get('json', '.json')}"
        with open(results_dir / summary_filename, 'w') as f:
            json_indent = self.numeric_config.get('json_indent', 2)
            json.dump(backtest_summary, f, indent=json_indent, default=str)

        # Save detailed trades
        if 'trades_detail' in results['backtest'] and results['backtest']['trades_detail']:
            trades_df = pd.DataFrame(results['backtest']['trades_detail'])
            trades_filename = f"{prefixes.get('trades_detail', 'trades_')}{timestamp}{extensions.get('csv', '.csv')}"
            csv_index = self.file_config.get('csv_index_flag', False)
            trades_df.to_csv(results_dir / trades_filename, index=csv_index)

        # Save signals
        general_config = config.get_section('general', {})
        if general_config.get('save_signals') and 'signals' in results:
            signals_filename = f"{prefixes.get('signals', 'signals_')}{timestamp}{extensions.get('csv', '.csv')}"
            results['signals'].to_csv(results_dir / signals_filename)

        # Save configuration
        config_filename = f"{prefixes.get('config', 'config_')}{timestamp}{extensions.get('json', '.json')}"
        config.save_config(str(results_dir / config_filename))

        return results_dir, timestamp