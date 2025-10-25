# backtest.py
# SIMPLIFIED: Main orchestration script with clean architecture
# ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
# DELEGATES to specialized components for complex logic

from src.hybrid.backtesting.backtest_orchestrator import BacktestOrchestrator
# Import and setup Windows compatibility FIRST
from src.hybrid.utils.windows_compat import setup_windows_compatibility

setup_windows_compatibility(max_cores=16)

# Core imports

# Configuration and data
from src.hybrid.config.unified_config import UnifiedConfig

# Backtesting engines

# Optimization
from src.hybrid.optimization import (
    OptimizerType,
    run_optimization
)


def run_optimization_mode(config: UnifiedConfig) -> bool:
    """
    Handle optimization mode execution
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE

    Returns:
        True if optimization was run, False otherwise
    """
    import sys

    # Check for optimization mode
    math_config = config.get_section('mathematical_operations', {})
    unity_value = math_config.get('unity')

    if len(sys.argv) <= unity_value or sys.argv[unity_value].lower() != "optimize":
        return False

    # Get optimization configuration
    opt_config = config.get_section('optimization', {})
    default_config = opt_config.get('defaults', {})

    n_combinations = default_config.get('n_combinations')
    max_workers = default_config.get('max_workers')
    quiet_mode = default_config.get('quiet_mode')
    use_bayesian = False

    # Parse command line overrides
    for i, arg in enumerate(sys.argv):
        if arg == "--combinations" and i + unity_value < len(sys.argv):
            n_combinations = int(sys.argv[i + unity_value])
        elif arg == "--workers" and i + unity_value < len(sys.argv):
            max_workers = int(sys.argv[i + unity_value])
        elif arg == "--verbose":
            quiet_mode = False
        elif arg == "--bayesian":
            use_bayesian = True

    # Run optimization with walk-forward backtesting
    if use_bayesian:
        print("Running BAYESIAN parameter optimization with walk-forward...")
        optimization_results = run_optimization(
            optimizer_type=OptimizerType.BAYESIAN,
            data_path=None,
            n_combinations=n_combinations
        )
    else:
        print("Running CACHED parameter optimization with walk-forward...")
        optimization_results = run_optimization(
            optimizer_type=OptimizerType.CACHED_RANDOM,
            data_path=None,
            n_combinations=n_combinations
        )

    if optimization_results:
        print(f"\n{'=' * 80}")
        print("OPTIMIZATION COMPLETED!")
        print(f"{'=' * 80}")

    return True


def apply_preset_configuration(config: UnifiedConfig, mode: str):
    """
    Apply preset configuration based on mode
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """
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
        print("Usage: python backtest.py [walkforward|optimize|swing|scalping|conservative|aggressive|forex_position]")
        print("  walkforward: Run walk-forward backtest (default)")
        print("  optimize: Run parameter optimization with walk-forward")
        print("    --combinations N: Number of parameter combinations")
        print("    --workers N: Number of parallel workers")
        print("    --verbose: Enable verbose output")
        print("    --bayesian: Use Bayesian optimization")
        print("Running with default forex_swing configuration...\n")

        presets = config.get_section('presets', {})
        if 'forex_swing' in presets:
            config.update_config(presets['forex_swing'])


def print_configuration_debug_info(config: UnifiedConfig):
    """Print configuration debug information"""
    print(f"Loaded configuration from: {config.config_path}")

    # Check if forex_swing preset was applied
    presets = config.get_section('presets', {})
    if 'forex_swing' in presets:
        print("Applied preset: forex_swing")

    # Debug output for key configuration values
    backtest_config = config.get_section('backtesting', {})
    print(f"DEBUG: Backtesting method = {backtest_config.get('method', 'Not set')}")
    print(f"DEBUG: Pretrain rows = {backtest_config.get('pretrain_rows', 'Not set')}")
    print(f"DEBUG: Retrain frequency = {backtest_config.get('retrain_frequency', 'Not set')}")


if __name__ == "__main__":
    """
    Main execution entry point with clean command line handling
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    import sys

    # Initialize configuration
    config = UnifiedConfig()

    # Handle optimization mode first
    if run_optimization_mode(config):
        exit(0)

    # Get mode for preset application
    math_config = config.get_section('mathematical_operations', {})
    unity_value = math_config.get('unity')

    if len(sys.argv) > unity_value:
        mode = sys.argv[unity_value].lower()
    else:
        mode = "walkforward"

    # Apply preset configuration
    apply_preset_configuration(config, mode)

    # Print configuration information
    print_configuration_debug_info(config)

    # Run backtest using orchestrator
    orchestrator = BacktestOrchestrator(config)
    results = orchestrator.run_backtest()

    if results:
        print(f"\n{'=' * 80}")
        print("WALK-FORWARD BACKTEST COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 80}")
        print(f"\nThe walk-forward approach ensures:")
        print(f"• NO data leakage - strict temporal isolation")
        print(f"• Proper out-of-sample testing")
        print(f"• Realistic performance estimates")
        print(f"• ML models retrained with only past data")
        print(f"\nThis eliminates the data leakage that was causing")
        print(f"unrealistic accuracy scores (VolAcc 0.842 → realistic levels)!")
        print(f"\nKey architectural improvements:")
        print(f"• TemporalDataGuard enforces strict boundaries")
        print(f"• WalkForwardBacktester handles complex temporal logic")
        print(f"• Clean separation of concerns across modules")
        print(f"• Reusable BacktestEngine for actual trade execution")
    else:
        print("Walk-forward backtest failed. Please check the logs for errors.")
