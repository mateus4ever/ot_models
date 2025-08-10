# src/hybrid/backtesting/validator.py
# Configuration validation with ZERO hardcoded values

from src.hybrid.config.unified_config import UnifiedConfig


class ConfigValidator:
    """
    Validate configuration completeness
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config

    def validate_config(self):
        """Validate that ALL required config values are present"""

        # Get all required configuration values
        required_values = self._get_required_values()

        # Check for missing values
        missing_values = [name for name, value in required_values if value is None]

        if missing_values:
            error_config = self.config.get_section('error_handling', {})
            error_template = error_config.get('missing_config_template')
            raise ValueError(error_template.format(missing_values=missing_values))

    def _get_required_values(self):
        """Get all required configuration values"""

        # Main config sections
        backtest_config = self.config.get_section('backtesting', {})
        risk_config = self.config.get_section('risk_management', {})
        general_config = self.config.get_section('general', {})

        # Mathematical constants
        constants_config = self.config.get_section('mathematical_operations', {})

        # Array indexing
        array_config = self.config.get_section('array_indexing', {})

        # Backtesting calculation parameters
        backtest_calc_config = backtest_config.get('calculations', {})

        return [
            # Backtesting parameters
            ('initial_capital', backtest_config.get('initial_capital')),
            ('transaction_cost', backtest_config.get('transaction_cost')),
            ('slippage', backtest_config.get('slippage')),
            ('risk_free_rate', backtest_config.get('risk_free_rate')),

            # General parameters
            ('save_signals', general_config.get('save_signals')),
            ('verbose', general_config.get('verbose')),

            # Risk management parameters
            ('stop_loss_pct', risk_config.get('stop_loss_pct')),
            ('take_profit_pct', risk_config.get('take_profit_pct')),
            ('max_holding_periods', risk_config.get('max_holding_periods')),
            ('max_daily_trades', risk_config.get('max_daily_trades')),
            ('max_daily_loss_pct', risk_config.get('max_daily_loss_pct')),
            ('max_drawdown_pct', risk_config.get('max_drawdown_pct')),

            # Mathematical constants
            ('zero_value', constants_config.get('zero')),
            ('unity_value', constants_config.get('unity')),

            # Array indexing
            ('first_index', array_config.get('first_index')),
            ('second_index', array_config.get('second_index')),

            # Backtesting calculations
            ('min_trades_for_metrics', backtest_calc_config.get('min_trades_for_metrics')),
            ('days_per_year', backtest_calc_config.get('days_per_year')),
            ('loop_start_index', backtest_calc_config.get('loop_start_index')),
            ('position_long_value', backtest_calc_config.get('position_long_value')),
            ('position_short_value', backtest_calc_config.get('position_short_value')),
            ('position_neutral_value', backtest_calc_config.get('position_neutral_value')),
            ('signal_threshold', backtest_calc_config.get('signal_threshold')),
            ('size_threshold', backtest_calc_config.get('size_threshold')),
            ('min_performance_samples', backtest_calc_config.get('min_performance_samples'))
        ]