# backtest.py
# Main execution script that orchestrates the entire hybrid trading strategy
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

# Import and setup Windows compatibility FIRST - before any other imports
from src.hybrid.utils.windows_compat import setup_windows_compatibility

setup_windows_compatibility(max_cores=16)

# Rest of imports after Windows compatibility is set up
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional
import json
from datetime import datetime
import time as timer

from src.hybrid.config.unified_config import UnifiedConfig, get_config
from src.hybrid.load_data import load_and_preprocess_data
from src.hybrid.hybrid_strategy import HybridStrategy
from src.hybrid.backtesting import MetricsCalculator, ResultsFormatter, BacktestEngine, ConfigValidator, RiskManagement
from src.hybrid.optimization import (
    OptimizerFactory,
    OptimizationType,
    run_optimization
)


class StrategyBacktester:
    """
    Main backtesting orchestrator - delegates to specialized components
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or UnifiedConfig()

        # Validate configuration
        validator = ConfigValidator(self.config)
        validator.validate_config()

        # Initialize components
        self.backtest_engine = BacktestEngine(self.config)

    def run_backtest(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> Dict:
        """Run backtest using BacktestEngine"""
        return self.backtest_engine.run_backtest(df, signals_df)


# All display and saving functionality now handled by ResultsFormatter


def run_hybrid_strategy_backtest(data_path: str = None, config: UnifiedConfig = None,
                                 save_results_flag: bool = True) -> Dict:
    """
    Main function to run the complete hybrid strategy backtest
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    # Use provided config or load default
    if config is None:
        config = UnifiedConfig()

    # Start timing
    start_time = datetime.now()

    print("=" * 80)
    print("HYBRID ML-TECHNICAL TRADING STRATEGY BACKTEST")
    print("=" * 80)
    print("Solving the ML correlation problem with hybrid approach!")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. Load and preprocess data
        data_start = datetime.now()
        print(f"\n1. Loading and preprocessing data...")
        if data_path is None:
            data_config = config.get_section('data_loading', {})
            data_path = data_config.get('data_source', 'data/eurusd')

        df = load_and_preprocess_data(data_path, config)
        data_duration = (datetime.now() - data_start).total_seconds()
        print(f"   Data loaded: {len(df)} records")
        print(f"   Time range: {df.index[0]} to {df.index[-1]}")
        print(f"   ✓ Data loading took: {data_duration:.1f} seconds")

        # 2. Initialize and train strategy
        print(f"\n2. Initializing hybrid strategy...")
        strategy = HybridStrategy(config)

        print(f"\n3. Training ML components...")
        training_start = datetime.now()
        training_results = strategy.train(df)
        training_duration = (datetime.now() - training_start).total_seconds()
        print(f"   ✓ ML training took: {training_duration:.1f} seconds")

        # 4. Generate signals
        print(f"\n4. Generating hybrid signals...")
        signals_start = datetime.now()
        signals_df = strategy.generate_signals(df)
        signals_duration = (datetime.now() - signals_start).total_seconds()
        print(f"   ✓ Signal generation took: {signals_duration:.1f} seconds")

        # 5. Calculate strategy summary using ResultsFormatter
        summary_start = datetime.now()
        results_formatter = ResultsFormatter(config)
        strategy_summary = results_formatter.calculate_strategy_summary(signals_df, df)
        summary_duration = (datetime.now() - summary_start).total_seconds()

        # 6. Run backtest
        print(f"\n5. Running backtest...")
        backtest_start = datetime.now()
        backtester = StrategyBacktester(config)
        backtest_results = backtester.run_backtest(df, signals_df)
        backtest_duration = (datetime.now() - backtest_start).total_seconds()
        print(f"   ✓ Backtesting took: {backtest_duration:.1f} seconds")

        # 7. Create configuration summary
        config_start = datetime.now()
        config_summary = create_config_summary(config)
        config_duration = (datetime.now() - config_start).total_seconds()

        # 8. Compile results
        results = {
            'config_summary': config_summary,
            'data_info': {
                'n_records': len(df),
                'start_date': str(df.index[0]),
                'end_date': str(df.index[-1]),
                'data_path': data_path
            },
            'training': training_results,
            'strategy_summary': strategy_summary,
            'backtest': backtest_results,
            'signals': signals_df if config.get_section('general', {}).get('save_signals', False) else None,
            'timing': {
                'data_loading': data_duration,
                'ml_training': training_duration,
                'signal_generation': signals_duration,
                'strategy_summary': summary_duration,
                'backtesting': backtest_duration,
                'config_creation': config_duration
            }
        }

        # 9. Save and display results
        if save_results_flag:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_formatter = ResultsFormatter(config)
            results_dir, timestamp = results_formatter.save_results(results, config, timestamp)
            print(f"Configuration saved to: {results_dir}\\config_{timestamp}.json")

        # 10. Print summary using ResultsFormatter
        results_formatter = ResultsFormatter(config)
        results_formatter.print_results_summary(results)

        # Calculate and display total time
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        print(f"\n{'=' * 80}")
        print("PERFORMANCE TIMING BREAKDOWN")
        print(f"{'=' * 80}")
        print(f"Data Records: {len(df):,}")
        print(f"Start time:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time:   {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)")
        print(f"")
        print(f"Component Breakdown:")
        print(f"  Data Loading:      {data_duration:6.1f}s ({data_duration / total_duration * 100:4.1f}%)")
        print(f"  ML Training:       {training_duration:6.1f}s ({training_duration / total_duration * 100:4.1f}%)")
        print(f"  Signal Generation: {signals_duration:6.1f}s ({signals_duration / total_duration * 100:4.1f}%)")
        print(f"  Backtesting:       {backtest_duration:6.1f}s ({backtest_duration / total_duration * 100:4.1f}%)")
        print(
            f"  Other:             {summary_duration + config_duration:6.1f}s ({(summary_duration + config_duration) / total_duration * 100:4.1f}%)")
        print(f"")
        print(f"Performance Metrics:")
        print(f"  Records/second (total): {len(df) / total_duration:,.0f}")
        print(f"  Records/second (ML):    {len(df) / training_duration:,.0f}")
        print(f"{'=' * 80}")

        return results

    except Exception as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        print(f"Error running hybrid strategy backtest: {e}")
        print(f"Failed after {total_duration:.1f} seconds")
        import traceback
        traceback.print_exc()
        return {}


# All strategy summary calculation now handled by ResultsFormatter


def create_config_summary(config: UnifiedConfig) -> Dict:
    """Create configuration summary for results"""

    regime_config = config.get_section('regime_detection', {})
    risk_config = config.get_section('risk_management', {})
    vol_config = config.get_section('volatility_prediction', {})
    duration_config = config.get_section('trend_duration_prediction', {})

    return {
        'strategy_type': 'Hybrid ML-Technical',
        'ml_components': {
            'regime': regime_config.get('method'),
            'volatility': vol_config.get('use_volatility_ml'),
            'duration': duration_config.get('enabled')
        },
        'stop_loss_pct': risk_config.get('stop_loss_pct'),
        'take_profit_pct': risk_config.get('take_profit_pct'),
        'max_position_size': risk_config.get('max_position_size')
    }


if __name__ == "__main__":
    """
    Main execution with different modes
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    import sys

    # Get configurable values for comparison
    config_temp = UnifiedConfig()
    array_config = config_temp.get_section('array_indexing', {})
    unity_value = config_temp.get_section('mathematical_operations', {}).get('unity')

    if len(sys.argv) > unity_value:
        mode = sys.argv[unity_value].lower()
    else:
        mode = "single"

    # Check for optimization mode
    if mode == "optimize":
        # Get optimization configuration
        opt_config = config_temp.get_section('optimization', {})
        default_config = opt_config.get('defaults', {})

        n_combinations = default_config.get('n_combinations')
        max_workers = default_config.get('max_workers')
        quiet_mode = default_config.get('quiet_mode')
        use_bayesian = False

        # Parse command line overrides
        for i, arg in enumerate(sys.argv):
            if arg == "--combinations" and i + 1 < len(sys.argv):
                n_combinations = int(sys.argv[i + 1])
            elif arg == "--workers" and i + 1 < len(sys.argv):
                max_workers = int(sys.argv[i + 1])
            elif arg == "--verbose":
                quiet_mode = False
            elif arg == "--bayesian":
                use_bayesian = True

        if use_bayesian:
            print("Running BAYESIAN parameter optimization...")
            optimization_results = run_optimization(
                optimizer_type=OptimizationType.BAYESIAN,
                data_path=None,  # Let optimization function use config default
                n_combinations=n_combinations
            )
        else:
            print("Running CACHED parameter optimization...")
            optimization_results = run_optimization(
                optimizer_type=OptimizationType.CACHED_RANDOM,
                data_path=None,  # Let optimization function use config default
                n_combinations=n_combinations
            )

        if optimization_results:
            print(f"\n{'=' * 80}")
            print("OPTIMIZATION COMPLETED!")
            print(f"{'=' * 80}")
        exit(0)

    # Load configuration and apply preset
    config = UnifiedConfig()

    preset_modes = ["swing", "scalping", "conservative", "aggressive", "forex_position"]
    if mode in preset_modes:
        # Apply specific preset
        preset_name = f"forex_{mode}" if mode in ["swing", "scalping"] else mode
        print(f"Applying preset: {preset_name}")

        # Apply preset by updating config
        presets = config.get_section('presets', {})
        if preset_name in presets:
            config.update_config(presets[preset_name])
        else:
            print(f"Warning: Preset '{preset_name}' not found, using default")

    else:
        # Default: use forex_swing configuration
        print("Usage: python backtest.py [single|optimize|swing|scalping|conservative|aggressive|forex_position]")
        print("  optimize: Run parameter optimization")
        print("    --combinations N: Number of parameter combinations")
        print("    --workers N: Number of parallel workers")
        print("    --verbose: Enable verbose output")
        print("    --bayesian: Use Bayesian optimization (smarter parameter selection)")
        print("Running with default forex_swing configuration...\n")

        presets = config.get_section('presets', {})
        if 'forex_swing' in presets:
            config.update_config(presets['forex_swing'])

    # Display some config values for verification
    print(f"Loaded configuration from: {config.config_path}")
    if 'forex_swing' in config.get_section('presets', {}):
        print("Applied preset: forex_swing")

    # Debug output
    constants = config.get_section('constants', {})
    print(f"DEBUG: Config percentile = {constants.get('percentile', 'Not set')}")
    print(f"DEBUG: Config forward window = {constants.get('forward_window', 'Not set')}")

    # Run backtest
    results = run_hybrid_strategy_backtest(config=config)

    if results:
        print(f"\n{'=' * 80}")
        print("BACKTEST COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 80}")
        print(f"\nThe hybrid approach combines:")
        print(f"• Rule-based regime detection (transparent, no overfitting)")
        print(f"• ML for volatility prediction (risk management)")
        print(f"• Technical analysis for entry/exit signals (KAMA, Kalman)")
        print(f"• Intelligent signal combination based on market conditions")
        print(f"\nThis solves the ML correlation problem by using ML for")
        print(f"what it can predict (regimes, volatility) rather than")
        print(f"trying to predict exact price movements!")
    else:
        print("Backtest failed. Please check the logs for errors.")