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
from typing import Dict, List, Any
import pandas as pd

from src.hybrid.money_management import PositionDirection, TradingSignal
from src.hybrid.strategies import StrategyInterface

logger = logging.getLogger(__name__)


class BaseStrategy(StrategyInterface):
    """Base strategy implementation with dependency injection"""

    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.signals = []
        self.predictors = []
        self.optimizers = []
        self.runners = []
        self.metrics = []
        self.data_manager = None
        self.money_manager = None
        self.execution_listeners = []  # For future live trading integration

    # Dependency injection methods
    def setDataManager(self, data_manager):
        self.data_manager = data_manager

    def setMoneyManager(self, money_manager):
        self.money_manager = money_manager

    def addSignal(self, signal):
        self.signals.append(signal)

    def addPredictor(self, predictor):
        self.predictors.append(predictor)

    def addOptimizer(self, optimizer):
        self.optimizers.append(optimizer)

    def addRunner(self, runner):
        self.runners.append(runner)

    def addMetric(self, metric):
        self.metrics.append(metric)

    def add_execution_listener(self, listener):
        """Add listener for trade execution events (for live trading)"""
        self.execution_listeners.append(listener)

    # Main execution method
    def run(self, market_data: Dict = None) -> Dict:
        """Run strategy on data stream (historical or live)

        Args:
            market_data: Optional market data dict (not currently used)

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
        trades = self._process_stream(market_id)

        # 4. Calculate and return metrics
        return self._calculate_final_metrics(trades)

    def _validate_and_setup(self) -> Dict:
        """Validate dependencies and setup strategy environment

        Returns:
            Dict with market_id and past_data, or error message
        """
        # Verify dependencies injected
        if not self.data_manager or not self.money_manager:
            return {'error': 'Missing DataManager or MoneyManager injection'}

        if not self.signals:
            return {'error': 'No signals added to strategy'}

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

        # Get past data for training
        past_data_dict = self.data_manager.get_past_data()
        past_data = past_data_dict[market_id]

        return {
            'market_id': market_id,
            'past_data': past_data
        }

    def _train_signals(self, past_data: pd.DataFrame) -> None:
        """Train all signals on historical data

        Args:
            past_data: Historical DataFrame for training
        """
        logger.info(f"Training signal on {len(past_data)} past records")

        for signal in self.signals:
            if hasattr(signal, 'train'):
                signal.train(past_data)
                logger.info(f"Signal trained. Is ready: {signal.is_ready}")

    def _process_stream(self, market_id: str) -> List[Dict]:
        """Process data stream - walk through time and generate trades (historical or live)

        Args:
            market_id: Market identifier

        Returns:
            List of trade dictionaries
        """
        trades = []
        current_position = None
        iteration = 0
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}

        while self.data_manager.next():
            iteration += 1

            # Log progress
            if iteration % 1000 == 0:
                logger.info(
                    f"Progress: {iteration}/{self.data_manager.total_records} records processed "
                    f"({iteration / self.data_manager.total_records * 100:.1f}%), Trades: {len(trades)}"
                )

            # Check stop loss
            if current_position:
                if self._check_stop_loss_hit(current_bar, current_position):
                    trade = self._exit_on_stop_loss(current_bar, current_position)
                    trades.append(trade)

                    # Notify listeners
                    self._notify_exit_signal(market_id, trade['exit'], trade['pnl'])

                    current_position = None
                    logger.info(f"Position stopped out at {current_bar['close']}")
                    continue  # Skip to next bar, don't check signals this bar

            current_data_dict = self.data_manager.get_current_data()
            current_bar = current_data_dict[market_id]

            # Generate signal
            signal_value = self.signals[0].generate_signal(current_bar)
            signal_counts[signal_value] += 1

            if iteration % 5000 == 0:
                logger.info(f"Signal at iteration {iteration}: {signal_value}, Close: {current_bar['close']}")

            # Try to enter position
            if signal_value == 'BUY' and not current_position:
                current_position = self._try_enter_position(current_bar, market_id)

                # Notify execution listeners (for live trading)
                if current_position:
                    self._notify_entry_signal(
                        market_id,
                        current_position['entry_price'],
                        current_position['size']
                    )

            # Try to exit position
            elif signal_value == 'SELL' and current_position:
                trade = self._try_exit_position(current_bar, current_position)
                if trade:
                    trades.append(trade)

                    # Notify execution listeners (for live trading)
                    self._notify_exit_signal(
                        market_id,
                        trade['exit'],
                        trade['pnl']
                    )
                    current_position = None

        logger.info(
            f"Signal distribution - BUY: {signal_counts['BUY']}, "
            f"SELL: {signal_counts['SELL']}, HOLD: {signal_counts['HOLD']}"
        )

        return trades

    def _try_enter_position(self, current_bar: pd.Series, market_id: str) -> Dict:
        """Attempt to enter a long position

        Args:
            current_bar: Current market data
            market_id: Market identifier

        Returns:
            Position dictionary if entered, None otherwise
        """
        # Get past data for position sizing
        past_data_dict = self.data_manager.get_past_data()
        past_data = past_data_dict[market_id]

        # Create trading signal
        trading_signal = TradingSignal(
            symbol=market_id,
            direction=PositionDirection.LONG,
            strength=1.0,
            entry_price=current_bar['close'],
            timestamp=current_bar.name
        )

        # Calculate position size via MoneyManager
        position_size = self.money_manager.calculate_position_size(
            trading_signal,
            past_data
        )

        if position_size > 0:
            stop_loss = self.money_manager.calculate_stop_loss(trading_signal, past_data)
            position = {
                'entry_price': current_bar['close'],
                'size': position_size,
                'direction': 'LONG',
                'stop_loss': stop_loss
            }
            logger.debug(f"Entered LONG at {current_bar['close']}, size={position_size}")
            return position

        return None

    def _try_exit_position(self, current_bar: pd.Series, position: Dict) -> Dict:
        """Attempt to exit current position

        Args:
            current_bar: Current market data
            position: Current position dictionary

        Returns:
            Trade dictionary with P&L
        """
        exit_price = current_bar['close']
        pnl = (exit_price - position['entry_price']) * position['size']

        trade = {
            'entry': position['entry_price'],
            'exit': exit_price,
            'pnl': pnl,
            'size': position['size']
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
        """Exit position at stop loss

        Args:
            current_bar: Current market data
            position: Current position dictionary

        Returns:
            Trade dictionary with P&L
        """
        stop_price = position['stop_loss']
        pnl = (stop_price - position['entry_price']) * position['size']

        trade = {
            'entry': position['entry_price'],
            'exit': stop_price,
            'pnl': pnl,
            'size': position['size'],
            'exit_reason': 'stop_loss'  # Track why we exited
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

    def _calculate_final_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate final strategy metrics

        Args:
            trades: List of completed trades

        Returns:
            Dictionary with strategy results
        """
        if not trades:
            return {
                'strategy': self.name,
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'trades': []
            }

        total_pnl = sum(t['pnl'] for t in trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        logger.info(
            f"Strategy complete: {len(trades)} trades, "
            f"P&L={total_pnl:.2f}, Win Rate={win_rate:.1%}"
        )

        return {
            'strategy': self.name,
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'trades': trades
        }