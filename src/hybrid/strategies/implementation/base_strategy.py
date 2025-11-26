# base_strategy.py
"""
Base strategy implementation with dependency injection and execution listener pattern.

Execution Listener Pattern:
The strategy emits execution events (entry/exit) to registered listeners.
This allows different execution modes without changing strategy logic:

- Backtest mode: Listener records virtual trades
- Live trading mode: Listener sends orders to broker
- Logging mode: Listener writes execution details to database

Example usage:
    class BrokerExecutor:
        def on_entry(self, symbol, price, size):
            self.broker.place_order(symbol, 'BUY', size, price)

        def on_exit(self, symbol, price, pnl):
            self.broker.close_position(symbol)

    strategy = BaseStrategy(...)
    strategy.add_execution_listener(BrokerExecutor())
    strategy.run()
"""
import logging
from typing import Dict

import pandas as pd

from src.hybrid.money_management import PositionDirection, TradingSignal
from src.hybrid.positions.position_orchestrator import PositionOrchestrator
from src.hybrid.products.product_types import ProductFactory
from src.hybrid.signals.market_signal_enum import MarketSignal
from src.hybrid.strategies import StrategyInterface

logger = logging.getLogger(__name__)


class BaseStrategy(StrategyInterface):
    """Base strategy implementation with dependency injection"""

    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.entry_signal = None  # Single entry trigger
        self.exit_signal = None  # Single exit condition (besides stop)
        self.predictors = []
        self.optimizers = []
        self.runners = []
        self.metrics = []
        self.data_manager = None
        self.money_manager = None
        self.execution_listeners = []  # For future live trading integration

        # TODO: Move to config
        self.progress_log_interval = 1000
        self.signal_log_interval = 5000

    # Dependency injection methods
    def set_data_manager(self, data_manager):
        self.data_manager = data_manager

    def set_money_manager(self, money_manager):
        self.money_manager = money_manager

    def add_entry_signal(self, signal):
        self.entry_signal= signal

    def add_exit_signal(self, signal):
        self.exit_signal = signal

    def add_predictor(self, predictor):
        self.predictors.append(predictor)

    def add_optimizer(self, optimizer):
        self.optimizers.append(optimizer)

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_execution_listener(self, listener):
        """Add listener for trade execution events (for live trading)"""
        self.execution_listeners.append(listener)

    # Main execution method
    def run(self) -> Dict:
        """Run strategy on data stream (historical or live)

        Strategy accesses data via self.data_manager which was injected.

        Returns:
            Dictionary with strategy results
        """
        logger.info(f"Running strategy: {self.name}")

        # 1. Validate and setup
        setup_result = self._validate_and_setup()
        if 'error' in setup_result:
            return setup_result

        market_id = setup_result['market_id']
        past_data = setup_result['past_data']

        # 2. Train signals
        self._train_signals(past_data)

        # 3. Process data stream (agnostic to historical vs live)
        self._process_stream(market_id)

        # 4. Calculate and return metrics
        return self._calculate_final_metrics()

    def get_optimizable_parameters(self) -> Dict:
        """
        Collect all optimizable parameters from active components

        Gathers parameters from:
        - Entry signal configuration
        - Exit signal configuration (if configured)
        - Active risk manager
        - Active position sizer

        Returns:
            Dictionary of parameter definitions with min/max ranges
        """
        params = {}

        # 1. Entry signal parameters
        strategy_config = self.config.get_section('strategy', {})
        entry_signal_name = strategy_config.get('entry_signal')
        if entry_signal_name:
            signal_params = self._get_signal_parameters(entry_signal_name)
            params.update(signal_params)

        # 2. Exit signal parameters (if configured)
        exit_signal_name = strategy_config.get('exit_signal')
        if exit_signal_name:
            signal_params = self._get_signal_parameters(exit_signal_name)
            params.update(signal_params)

        # 3. Risk management parameters
        mm_config = self.config.get_section('money_management', {})
        active_risk_mgr = mm_config.get('risk_management')
        if active_risk_mgr in mm_config.get('risk_managers', {}):
            risk_params = mm_config['risk_managers'][active_risk_mgr].get('optimizable_parameters', {})
            params.update(risk_params)

        # 4. Position sizing parameters
        active_sizer = mm_config.get('position_sizing')
        if active_sizer in mm_config.get('position_sizers', {}):
            sizer_params = mm_config['position_sizers'][active_sizer].get('optimizable_parameters', {})
            params.update(sizer_params)

        return params

    def _validate_and_setup(self) -> Dict:
        """Validate dependencies and setup strategy environment

        Returns:
            Dict with market_id and past_data, or error message
        """
        # Verify dependencies injected
        if not self.data_manager or not self.money_manager:
            return {'error': 'Missing DataManager or MoneyManager injection'}

        if not self.entry_signal:
            return {'error': 'No entry signal added to strategy'}
        if not self.exit_signal:
            return {'error': 'No exit signal added to strategy'}

        # Get market identifier
        market_id = self.data_manager._active_market
        if not market_id:
            return {'error': 'No active market in DataManager'}

        # Set temporal pointer
        signals_config = self.config.get_section('signals', {})
        if 'training_window' not in signals_config:
            return {'error': 'training_window not found in signals config'}

        training_window = signals_config['training_window']

        active_market_data = self.data_manager._active_market_data
        self.data_manager.initialize_temporal_pointer(active_market_data, training_window)

        data_config = self.config.get_section('data_management', {})
        markets_config = data_config.get('markets', {})
        market_config = markets_config.get(market_id, {})
        #todo: this must be configured
        product_type = market_config.get('product_type', 'cfd')

        self.product = ProductFactory.create_product(product_type)
        logger.info(f"Market {market_id} using product: {product_type}")


        # Get past data for training
        past_data_dict = self.data_manager.get_past_data()
        past_data = past_data_dict[market_id]

        return {
            'market_id': market_id,
            'past_data': past_data
        }

    def _train_signals(self, past_data: pd.DataFrame) -> None:
        """Train entry and exit signals on historical data

        Args:
            past_data: Historical DataFrame for training
        """
        logger.info(f"Training signals on {len(past_data)} past records")

        # Train entry signal
        if self.entry_signal and hasattr(self.entry_signal, 'train'):
            self.entry_signal.train(past_data)
            logger.info(f"Entry signal trained. Is ready: {self.entry_signal.is_ready}")

        # Train exit signal (optional)
        if self.exit_signal and hasattr(self.exit_signal, 'train'):
            self.exit_signal.train(past_data)
            logger.info(f"Exit signal trained. Is ready: {self.exit_signal.is_ready}")
    def _process_stream(self, market_id: str) -> None:
        """Process data stream"""
        current_position = None
        iteration = 0
        signal_counts = {signal: 0 for signal in MarketSignal}

        while self.data_manager.next():
            iteration += 1

            if iteration % self.progress_log_interval == 0:
                trade_count = self.position_orchestrator.trade_history.get_trade_count()
                logger.info(
                    f"Progress: {iteration}/{self.data_manager.total_records} "
                    f"({iteration / self.data_manager.total_records * 100:.1f}%), Trades: {trade_count}"
                )

            current_data_dict = self.data_manager.get_current_data()
            current_bar = current_data_dict[market_id]

            # Update entry signal
            self.entry_signal.update_with_new_data(current_bar)
            entry_signal_value = self.entry_signal.generate_signal()
            signal_counts[entry_signal_value] += 1

            # === WHEN IN POSITION ===
            if current_position:
                # 1. Check stop loss first
                if self._handle_stop_loss(current_bar, current_position, market_id):
                    current_position = None
                    continue

                # 2. Check take profit
                if self._handle_take_profit(current_bar, current_position, market_id):
                    current_position = None
                    continue

                # 3. Check exit signal (if set)
                if self.exit_signal:
                    self.exit_signal.update_with_new_data(current_bar)
                    if self._handle_exit_signal(current_bar, current_position, market_id):
                        current_position = None
                        continue

            # === WHEN NO POSITION ===
            else:
                if entry_signal_value == MarketSignal.BULLISH:
                    current_position = self._try_enter_position(current_bar, market_id, PositionDirection.LONG)

                elif entry_signal_value == MarketSignal.BEARISH:
                    current_position = self._try_enter_position(current_bar, market_id, PositionDirection.SHORT)

            if iteration % self.signal_log_interval == 0:
                logger.info(f"Signal at iteration {iteration}: {entry_signal_value}, Close: {current_bar['close']}")

        logger.info(
            f"Signal distribution - BULLISH: {signal_counts[MarketSignal.BULLISH]}, "
            f"BEARISH: {signal_counts[MarketSignal.BEARISH]}, "
            f"NEUTRAL: {signal_counts[MarketSignal.NEUTRAL]}"
        )

    def _handle_take_profit(self, current_bar, position, market_id) -> bool:
        """Check and handle take profit"""
        if 'take_profit' not in position:
            return False

        if self._check_take_profit_hit(current_bar, position):
            self.position_orchestrator.close_position(position['trade_id'], position['take_profit'], 'take_profit')
            logger.info(f"Take profit hit at {position['take_profit']}")
            return True
        return False

    def _handle_exit_signal(self, current_bar, position, market_id) -> bool:
        """Check exit signal for momentum exhaustion etc"""
        exit_value = self.exit_signal.generate_signal()

        # Exit long on bearish, exit short on bullish
        if position['direction'] == PositionDirection.LONG.name and exit_value == MarketSignal.BEARISH:
            self.position_orchestrator.close_position(position['trade_id'], current_bar['close'], 'exit_signal')
            return True
        elif position['direction'] == PositionDirection.SHORT.name and exit_value == MarketSignal.BULLISH:
            self.position_orchestrator.close_position(position['trade_id'], current_bar['close'], 'exit_signal')
            return True

        return False

    def _check_take_profit_hit(self, current_bar: pd.Series, position: Dict) -> bool:
        """Check if current price has hit the take profit

        Args:
            current_bar: Current market data
            position: Current position dictionary

        Returns:
            True if take profit hit
        """
        if 'take_profit' not in position:
            return False

        current_price = current_bar['close']
        take_profit = position['take_profit']

        if position['direction'] == PositionDirection.LONG.name:
            return current_price >= take_profit
        else:  # SHORT
            return current_price <= take_profit

    def _handle_stop_loss(self, current_bar: pd.Series, position: Dict, market_id: str) -> Dict:
        """Handle stop loss check and exit

        Returns:
            Trade dict if stop loss hit, None otherwise
        """
        if self._check_stop_loss_hit(current_bar, position):
            trade = self._exit_on_stop_loss(current_bar, position)
            self._notify_exit_signal(market_id, trade['exit'], trade['pnl'])
            logger.info(f"Position stopped out at {current_bar['close']}")
            return trade
        return None


    def _try_enter_position(self, current_bar: pd.Series, market_id: str,
                            direction: PositionDirection) -> Dict:
        """Attempt to enter a position in specified direction"""

        past_data_dict = self.data_manager.get_past_data()
        past_data = past_data_dict[market_id]

        # Create trading signal with specified direction
        # TODO: signal_strength is hardcoded
        trading_signal = TradingSignal(
            symbol=market_id,
            direction=direction,
            signal_strength=1.0,
            entry_price=current_bar['close'],
            timestamp=current_bar.name
        )

        # MoneyManager calculates size based on direction
        position_size = self.money_manager.calculate_position_size(
            trading_signal,
            past_data
        )

        if position_size > 0:
            stop_loss = self.money_manager.calculate_stop_loss(trading_signal, past_data)

            # Calculate take profit (2:1 reward/risk)
            atr_distance = abs(current_bar['close'] - stop_loss)
            if direction == PositionDirection.LONG:
                take_profit = current_bar['close'] + (atr_distance * 2)
            else:
                take_profit = current_bar['close'] - (atr_distance * 2)

            trade_id = f"{market_id}_{current_bar.name}"
            capital_required = position_size * current_bar['close']

            success = self.position_orchestrator.open_position(
                trade_id=trade_id,
                symbol=market_id,
                direction=direction,
                quantity=position_size,
                entry_price=current_bar['close'],
                capital_required=capital_required
            )

            if success:
                position = {
                    'trade_id': trade_id,
                    'entry_price': current_bar['close'],
                    'size': position_size,
                    'direction': direction.name,  # Use .name for string
                    'stop_loss': stop_loss,
                    'take_profit': take_profit  # ADD THIS
                }
                logger.debug(
                    f"Entered {direction.name} at {current_bar['close']}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
                return position

        return None

    def _try_exit_position(self, current_bar: pd.Series, position: Dict) -> Dict:
        """Attempt to exit current position"""
        exit_price = current_bar['close']
        pnl = (exit_price - position['entry_price']) * position['size']

        # Close in position orchestrator
        self.position_orchestrator.close_position(position['trade_id'], exit_price, 'signal')

        trade = {
            'trade_id': position['trade_id'],
            'entry': position['entry_price'],
            'exit': exit_price,
            'pnl': pnl,
            'size': position['size'],
            'direction': position['direction'],
            'exit_reason': 'signal'
        }

        logger.debug(f"Exited at {exit_price}, P&L={pnl:.2f}")
        return trade

    def _check_stop_loss_hit(self, current_bar: pd.Series, position: Dict) -> bool:
        """Check if current price has hit the stop loss

        Args:
            current_bar: Current market data
            position: Current position dictionary

        Returns:
            True if stop loss hit
        """
        if 'stop_loss' not in position:
            return False

        current_price = current_bar['close']
        stop_loss = position['stop_loss']

        # For LONG positions, stop is below entry
        if position['direction'] == 'LONG':
            return current_price <= stop_loss
        else:  # SHORT
            return current_price >= stop_loss

    def _exit_on_stop_loss(self, current_bar: pd.Series, position: Dict) -> Dict:
        """Exit position at stop loss"""
        stop_price = position['stop_loss']
        pnl = (stop_price - position['entry_price']) * position['size']

        # Close in position orchestrator
        self.position_orchestrator.close_position(position['trade_id'], stop_price, 'stop_loss')

        trade = {
            'trade_id': position['trade_id'],
            'entry': position['entry_price'],
            'exit': stop_price,
            'pnl': pnl,
            'size': position['size'],
            'direction': position['direction'],
            'exit_reason': 'stop_loss'
        }

        logger.info(f"STOP LOSS HIT: Entry={position['entry_price']:.4f}, Stop={stop_price:.4f}, P&L={pnl:.2f}")
        return trade
    def _notify_entry_signal(self, symbol: str, entry_price: float, size: float) -> None:
        """Notify execution listeners of entry signal (for live trading)

        Args:
            symbol: Market symbol
            entry_price: Entry price
            size: Position size
        """
        for listener in self.execution_listeners:
            if hasattr(listener, 'on_entry'):
                listener.on_entry(symbol, entry_price, size)

    def _notify_exit_signal(self, symbol: str, exit_price: float, pnl: float) -> None:
        """Notify execution listeners of exit signal (for live trading)

        Args:
            symbol: Market symbol
            exit_price: Exit price
            pnl: Profit/Loss for the trade
        """
        for listener in self.execution_listeners:
            if hasattr(listener, 'on_exit'):
                listener.on_exit(symbol, exit_price, pnl)

    def _calculate_final_metrics(self) -> Dict:
        """Calculate final strategy metrics from trade history"""
        from src.hybrid.backtesting.metrics_calculator import MetricsCalculator

        metrics_calc = MetricsCalculator(self.config)
        initial_capital = self.position_orchestrator.position_manager.total_capital

        performance_metrics = metrics_calc.calculate_metrics(
            trade_history=self.position_orchestrator.trade_history,
            equity_curve=None,
            initial_capital=initial_capital
        )

        results = performance_metrics.to_dict()
        results['strategy_name'] = self.name

        # Fix trade_details extraction
        results['trade_details'] = [
            {
                'trade_id': t.get('uuid'),
                'type': t.get('direction').name if isinstance(t.get('direction'), PositionDirection) else t.get(
                    'direction'),
                'entry': t.get('entry_price'),
                'exit': t.get('exit_price'),
                'pnl': t.get('net_pnl') or t.get('gross_pnl') or t.get('pnl'),
                'exit_reason': t.get('exit_reason', 'UNKNOWN')
            }
            for t in self.position_orchestrator.trade_history.trades.values()
            if t.get('status') == 'closed'
        ]

        return results

    def set_position_orchestrator(self, position_orchestrator: PositionOrchestrator):
        """Set position orchestrator for position management"""
        self.position_orchestrator = position_orchestrator
        logger.debug(f"PositionOrchestrator set for strategy {self.__class__.__name__}")

    def _get_signal_parameters(self, signal_name: str) -> Dict:
        """
        Extract optimizable parameters for a specific signal

        Navigates the nested signals config structure to find the signal
        and extract its optimizable_parameters section.

        Args:
            signal_name: Name of the signal (e.g., 'simplemovingaveragecrossover')

        Returns:
            Dictionary of optimizable parameters with prefixed names
            Example: {'sma_fast_period': {'min': 5, 'max': 50}, ...}
        """
        signals_config = self.config.get_section('signals', {})

        # Search through signal categories (trend_following, mean_reversion, momentum, filters)
        for category_name, category_config in signals_config.items():
            # Skip non-dict entries (like 'training_window')
            if not isinstance(category_config, dict):
                continue

            # Check if signal exists in this category
            if signal_name in category_config:
                signal_config = category_config[signal_name]
                optimizable = signal_config.get('optimizable_parameters', {})

                # Prefix parameter names with signal name to avoid collisions
                prefixed_params = {}
                for param_name, param_def in optimizable.items():
                    prefixed_name = f"{signal_name}_{param_name}"
                    prefixed_params[prefixed_name] = param_def

                return prefixed_params

        # Signal not found
        return {}