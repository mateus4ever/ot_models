# src/hybrid/backtesting/risk.py
# Risk management component with ZERO hardcoded values

from typing import Tuple, Dict, Optional
from src.hybrid.config.unified_config import UnifiedConfig


class RiskManagement:
    """
    Risk management component for trading strategy
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._cache_config_values()

    def _cache_config_values(self):
        """Cache ALL risk management configuration values"""
        # Main config sections
        self.risk_config = self.config.get_section('risk_management', {})

        # Mathematical constants
        constants_config = self.config.get_section('mathematical_operations', {})
        self.zero_value = constants_config.get('zero')
        self.unity_value = constants_config.get('unity')

        # Risk limits
        self.max_daily_trades = self.risk_config.get('max_daily_trades')
        self.max_daily_loss_pct = self.risk_config.get('max_daily_loss_pct')
        self.max_drawdown_pct = self.risk_config.get('max_drawdown_pct')

        # Position sizing
        self.max_position_size = self.risk_config.get('max_position_size')
        self.min_position_size = self.risk_config.get('min_position_size')

        # Stop loss and take profit
        self.stop_loss_pct = self.risk_config.get('stop_loss_pct')
        self.take_profit_pct = self.risk_config.get('take_profit_pct')

        # Portfolio risk
        self.max_portfolio_risk_pct = self.risk_config.get('max_portfolio_risk_pct')

        # Risk calculation parameters
        risk_calc_config = self.risk_config.get('calculations', {})
        self.risk_per_trade_pct = risk_calc_config.get('risk_per_trade_pct')
        self.volatility_multiplier = risk_calc_config.get('volatility_multiplier')
        self.correlation_adjustment = risk_calc_config.get('correlation_adjustment')

    def check_risk_limits(self, capital: float, peak_capital: float,
                          daily_loss: float, daily_trades: int) -> Tuple[bool, Optional[str]]:
        """
        Check if risk limits are breached using configurable thresholds
        Returns: (is_limit_breached, breach_reason)
        """

        # Daily trade limit
        if daily_trades >= self.max_daily_trades:
            return True, "max_daily_trades_exceeded"

        # Daily loss limit
        if daily_loss > capital * self.max_daily_loss_pct:
            return True, "max_daily_loss_exceeded"

        # Maximum drawdown limit
        current_drawdown = (peak_capital - capital) / peak_capital
        if current_drawdown > self.max_drawdown_pct:
            return True, "max_drawdown_exceeded"

        return False, None

    def calculate_position_size(self, signal_strength: float, volatility: float,
                                capital: float, price: float) -> float:
        """
        Calculate position size based on risk management rules
        ALL parameters configurable
        """

        # Base position size from signal strength
        base_size = min(abs(signal_strength), self.max_position_size)

        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjusted_size = base_size / (self.unity_value + volatility * self.volatility_multiplier)

        # Ensure minimum size
        volatility_adjusted_size = max(volatility_adjusted_size, self.min_position_size)

        # Risk-based position sizing (Kelly criterion approach)
        risk_based_size = self._calculate_risk_based_size(capital, price, volatility)

        # Take the minimum of signal-based and risk-based sizing
        final_size = min(volatility_adjusted_size, risk_based_size)

        # Apply correlation adjustment if multiple positions
        final_size *= self.correlation_adjustment

        return final_size

    def _calculate_risk_based_size(self, capital: float, price: float, volatility: float) -> float:
        """Calculate position size based on risk per trade"""

        # Calculate the amount at risk per trade
        risk_amount = capital * self.risk_per_trade_pct

        # Calculate stop loss distance
        stop_distance = price * self.stop_loss_pct

        # Position size = Risk Amount / Stop Distance
        if stop_distance > self.zero_value:
            risk_based_size = risk_amount / (stop_distance * price)
        else:
            risk_based_size = self.min_position_size

        # Cap at maximum position size
        return min(risk_based_size, self.max_position_size)

    def check_stop_loss(self, position_type: int, entry_price: float,
                        current_price: float) -> Tuple[bool, float]:
        """
        Check if stop loss should be triggered
        Returns: (should_stop, stop_price)
        """

        if position_type > self.zero_value:  # Long position
            stop_price = entry_price * (self.unity_value - self.stop_loss_pct)
            should_stop = current_price <= stop_price
        else:  # Short position
            stop_price = entry_price * (self.unity_value + self.stop_loss_pct)
            should_stop = current_price >= stop_price

        return should_stop, stop_price

    def check_take_profit(self, position_type: int, entry_price: float,
                          current_price: float) -> Tuple[bool, float]:
        """
        Check if take profit should be triggered
        Returns: (should_take_profit, take_profit_price)
        """

        if position_type > self.zero_value:  # Long position
            tp_price = entry_price * (self.unity_value + self.take_profit_pct)
            should_tp = current_price >= tp_price
        else:  # Short position
            tp_price = entry_price * (self.unity_value - self.take_profit_pct)
            should_tp = current_price <= tp_price

        return should_tp, tp_price

    def calculate_portfolio_risk(self, current_positions: Dict, capital: float) -> float:
        """Calculate current portfolio risk exposure"""

        total_risk = self.zero_value

        for position_id, position_data in current_positions.items():
            position_size = position_data.get('size', self.zero_value)
            entry_price = position_data.get('entry_price', self.zero_value)

            # Calculate risk for this position
            position_value = position_size * entry_price
            position_risk = position_value * self.stop_loss_pct

            total_risk += position_risk

        # Return as percentage of capital
        return total_risk / capital if capital > self.zero_value else self.zero_value

    def is_portfolio_risk_acceptable(self, current_positions: Dict,
                                     new_position_size: float, new_entry_price: float,
                                     capital: float) -> bool:
        """Check if adding a new position would exceed portfolio risk limits"""

        # Calculate current portfolio risk
        current_risk = self.calculate_portfolio_risk(current_positions, capital)

        # Calculate risk of new position
        new_position_value = new_position_size * new_entry_price
        new_position_risk = new_position_value * self.stop_loss_pct
        new_risk_pct = new_position_risk / capital

        # Check if total risk would exceed limit
        total_risk = current_risk + new_risk_pct

        return total_risk <= self.max_portfolio_risk_pct

    def apply_risk_filters(self, signal_strength: float, position_size: float,
                           capital: float, current_positions: Dict,
                           entry_price: float) -> Tuple[bool, float, Optional[str]]:
        """
        Apply all risk filters to a potential trade
        Returns: (trade_approved, adjusted_size, rejection_reason)
        """

        # Check portfolio risk
        if not self.is_portfolio_risk_acceptable(current_positions, position_size,
                                                 entry_price, capital):
            return False, self.zero_value, "portfolio_risk_exceeded"

        # Check minimum position size
        if position_size < self.min_position_size:
            return False, self.zero_value, "position_too_small"

        # Check maximum position size
        if position_size > self.max_position_size:
            adjusted_size = self.max_position_size
        else:
            adjusted_size = position_size

        return True, adjusted_size, None

    def get_risk_metrics(self, trades: list, equity_curve: list,
                         initial_capital: float) -> Dict:
        """Calculate comprehensive risk metrics"""

        if not trades:
            return {
                'var_95': self.zero_value,
                'expected_shortfall': self.zero_value,
                'risk_adjusted_return': self.zero_value,
                'calmar_ratio': self.zero_value
            }

        # Calculate returns
        returns = [trade['return'] for trade in trades]

        # Value at Risk (95th percentile)
        returns_sorted = sorted(returns)
        percentile_config = self.config.get_section('statistical_analysis', {})
        var_percentile = percentile_config.get('var_percentile')
        var_index = int(len(returns_sorted) * var_percentile)
        var_95 = returns_sorted[var_index] if var_index < len(returns_sorted) else self.zero_value

        # Expected Shortfall (average of losses beyond VaR)
        tail_losses = [r for r in returns if r <= var_95]
        expected_shortfall = sum(tail_losses) / len(tail_losses) if tail_losses else self.zero_value

        # Risk-adjusted return
        avg_return = sum(returns) / len(returns)
        return_std = (sum([(r - avg_return) ** 2 for r in returns]) / len(returns)) ** 0.5
        risk_adjusted_return = avg_return / return_std if return_std > self.zero_value else self.zero_value

        # Calmar ratio (annual return / max drawdown)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        calmar_ratio = total_return / max_drawdown if max_drawdown > self.zero_value else self.zero_value

        return {
            'var_95': var_95,
            'expected_shortfall': expected_shortfall,
            'risk_adjusted_return': risk_adjusted_return,
            'calmar_ratio': calmar_ratio
        }

    def _calculate_max_drawdown(self, equity_curve: list) -> float:
        """Calculate maximum drawdown from equity curve"""

        if len(equity_curve) < 2:
            return self.zero_value

        peak = equity_curve[0]
        max_dd = self.zero_value

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd