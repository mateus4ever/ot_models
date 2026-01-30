"""
Triangular Arbitrage Strategy
Specialized strategy for spread trading with 3-leg execution

Architecture:
- Predictor (Strategic): Answers WHAT to do (LONG_SPREAD, SHORT_SPREAD, CLOSE, HOLD)
- Signal (Tactical): Answers WHEN to do it (timing alignment, optional)

Entry modes:
- Signals not configured → trust predictor only
- Signals configured → wait for timing alignment (predictor + signal agree)
"""

import logging
from typing import Dict, Optional

import pandas as pd

from src.hybrid.backtesting import MetricsCalculator
from src.hybrid.money_management import PositionDirection
from src.hybrid.positions.position_orchestrator import PositionOrchestrator
from src.hybrid.signals.market_signal_enum import MarketSignal
from src.hybrid.strategies import StrategyInterface

logger = logging.getLogger(__name__)


class TriangularStrategy(StrategyInterface):
    """
    Strategy for triangular arbitrage spread trading

    Key differences from BaseStrategy:
    - Manages 3-leg positions (target, leg1, leg2)
    - Uses spread-based signals (LONG_SPREAD, SHORT_SPREAD, CLOSE)
    - Risk management based on Z-score, not price levels
    - Atomic execution of all 3 legs
    - Predictor-driven with optional signal timing
    """

    def __init__(self, name: str, config):
        self.name = name
        self.config = config

        # Signal-based (optional for timing alignment)
        self.entry_signal = None
        self.exit_signal = None

        # Predictor-based (required for triangular arbitrage)
        self.predictors = []

        # Common attributes (same as BaseStrategy)
        self.optimizers = []
        self.runners = []
        self.metrics = []
        self.data_manager = None
        self.money_manager = None
        self.execution_listeners = []
        self.position_orchestrator = None

        # Triangular-specific
        self.current_spread_position: Optional[Dict] = None
        self.spread_trades = []
        self.spread_trade_history = None
        self.current_trades = []
        self.progress_log_interval = 1000
        self.signal_log_interval = 5000

        # Cache config values
        arb_config = self.config.get_section('triangular_arbitrage', {})
        profile = arb_config.get('setting', 'standard')
        params = arb_config.get(profile, {}).get('parameters', {})

        self.pip_value_per_lot = params.get('pip_value_per_lot')
        if self.pip_value_per_lot is None:
            raise ValueError('pip_value_per_lot not found in triangular_arbitrage config')

        logger.info(f"Initialized TriangularStrategy: {name}")

    # =========================================================================
    # Dependency injection (same as BaseStrategy)
    # =========================================================================

    def set_spread_trade_history(self, spread_trade_history):
        self.spread_trade_history = spread_trade_history
    def set_data_manager(self, data_manager):
        self.data_manager = data_manager

    def set_money_manager(self, money_manager):
        self.money_manager = money_manager

    def set_position_orchestrator(self, position_orchestrator: PositionOrchestrator):
        self.position_orchestrator = position_orchestrator
        logger.debug(f"PositionOrchestrator set for strategy {self.__class__.__name__}")

    def add_entry_signal(self, signal):
        self.entry_signal = signal

    def add_exit_signal(self, signal):
        self.exit_signal = signal

    def add_predictor(self, predictor):
        self.predictors.append(predictor)

    def add_optimizer(self, optimizer):
        self.optimizers.append(optimizer)

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_execution_listener(self, listener):
        self.execution_listeners.append(listener)

    def get_optimizable_parameters(self) -> Dict:
        """Collect optimizable parameters from triangular arbitrage config"""
        params = {}

        arb_config = self.config.get_section('triangular_arbitrage', {})
        optimizable = arb_config.get('optimizable_parameters', {})
        params.update(optimizable)

        return params

    # =========================================================================
    # Strategy execution
    # =========================================================================

    def run(self) -> Dict:
        """Run triangular arbitrage strategy"""
        logger.info(f"Running triangular arbitrage strategy: {self.name}")

        # 1. Validate and setup - raises if safeguards missing
        setup_result = self._validate_and_setup()

        # 2. Train signals (if configured)
        if setup_result.get('past_data') is not None:
            self._train_signals(setup_result['past_data'])

        # 3. Train predictor
        predictor = self.predictors[0]
        train_result = predictor.train(self.data_manager)
        if not train_result.get('success', False):
            raise ValueError(f"Predictor training failed: {train_result.get('reason', 'Unknown')}")

        # 4. Process stream
        self._process_stream()

        # 5. Calculate metrics
        return self._calculate_final_metrics()

    def _validate_and_setup(self) -> Dict:
        """Validate dependencies and setup strategy environment"""
        if not self.data_manager:
            raise ValueError("DataManager not set")

        if not self.predictors:
            raise ValueError("No predictor added to strategy")

        if not self.position_orchestrator:
            raise ValueError("PositionOrchestrator not set")

        if not self.money_manager:
            raise ValueError("MoneyManager is required - risk management cannot be skipped")

        if not self.spread_trade_history:
            raise ValueError("SpreadTradeHistory not set")

        # Get triangular arbitrage config (required)
        arb_config = self.config.get_section('triangular_arbitrage')
        if not arb_config:
            return {'error': 'triangular_arbitrage section not found in config'}

        if 'lookback_window' not in arb_config:
            return {'error': 'lookback_window not found in triangular_arbitrage config'}

        if 'target_market' not in arb_config:
            return {'error': 'target_market not found in triangular_arbitrage config'}

        # Get required markets from predictor
        predictor = self.predictors[0]
        required_markets = predictor.get_required_markets()
        available_markets = self.data_manager.get_available_markets()

        missing = set(required_markets) - set(available_markets)
        if missing:
            return {'error': f'Required markets not loaded: {missing}'}

        # Initialize temporal pointer
        lookback = arb_config['lookback_window']
        self.data_manager.initialize_temporal_pointer(lookback)

        # Get past data for signal training (if signals configured)
        past_data = None
        if self.entry_signal or self.exit_signal:
            past_data_dict = self.data_manager.get_past_data()
            target_market = arb_config['target_market']
            past_data = past_data_dict.get(target_market)

        return {
            'past_data': past_data
        }
    def _train_signals(self, past_data: pd.DataFrame) -> None:
        """Train entry and exit signals on historical data"""
        if past_data is None:
            return

        logger.info(f"Training signals on {len(past_data)} past records")

        if self.entry_signal and hasattr(self.entry_signal, 'train'):
            self.entry_signal.train(past_data)
            logger.info(f"Entry signal trained. Is ready: {self.entry_signal.is_ready}")

        if self.exit_signal and hasattr(self.exit_signal, 'train'):
            self.exit_signal.train(past_data)
            logger.info(f"Exit signal trained. Is ready: {self.exit_signal.is_ready}")

    def _process_stream(self):
        """Process data stream for triangular arbitrage"""
        predictor = self.predictors[0]
        arb_config = self.config.get_section('triangular_arbitrage', {})
        target_market = arb_config.get('target_market')

        iteration = 0
        last_processed_hour = None
        signal_counts = {'LONG_SPREAD': 0, 'SHORT_SPREAD': 0, 'CLOSE': 0, 'HOLD': 0}
        entry_counts = {'predictor_only': 0, 'timing_aligned': 0, 'filtered': 0}

        while self.data_manager.next():
            iteration += 1

            # Only process on hourly boundaries
            timestamp = self.data_manager.temporal_timestamp
            current_hour = timestamp.replace(minute=0, second=0, microsecond=0)
            if current_hour == last_processed_hour:
                continue
            last_processed_hour = current_hour

            if iteration % self.progress_log_interval == 0:
                trade_count = self.position_orchestrator.trade_history.get_trade_count()
                logger.info(
                    f"Progress: {iteration}/{self.data_manager.total_records} "
                    f"({iteration / self.data_manager.total_records * 100:.1f}%), "
                    f"Trades: {trade_count}"
                )

            # Get current bar for signal updates (target market)
            current_data_dict = self.data_manager.get_current_data()
            current_bar = current_data_dict.get(target_market)

            # Update signals with current data (if configured)
            if self.entry_signal and current_bar is not None:
                self.entry_signal.update_with_new_data(current_bar)
            if self.exit_signal and current_bar is not None:
                self.exit_signal.update_with_new_data(current_bar)

            # Strategic: What does predictor say?
            prediction = predictor.predict(self.data_manager)
            strategic_signal = prediction['signal']
            signal_counts[strategic_signal] += 1

            # === WHEN IN POSITION ===
            if self.current_spread_position:
                logger.warning(f"OPEN POSITION: {self.current_spread_position['signal']}, "
                               f"entry_z={self.current_spread_position['entry_z_score']:.2f}")

                should_exit, exit_reason = self._should_exit(strategic_signal, prediction)
                if should_exit:
                    logger.warning(f"EXIT: reason={exit_reason}, current_z={prediction['z_score']:.2f}")
                    self._close_spread_position(prediction, reason=exit_reason)
                    self.current_spread_position = None

            # === WHEN NO POSITION ===
            else:
                if strategic_signal in ['LONG_SPREAD', 'SHORT_SPREAD']:
                    should_enter, entry_mode = self._should_enter(strategic_signal, prediction)

                    if should_enter:
                        entry_counts[entry_mode] += 1
                        self.current_spread_position = self._open_spread_position(
                            prediction, strategic_signal
                        )
                    else:
                        entry_counts['filtered'] += 1

            if iteration % self.signal_log_interval == 0:
                logger.info(
                    f"Iteration {iteration}: predictor={strategic_signal}, "
                    f"Z={prediction['z_score']:.2f}"
                )

        logger.info(
            f"Signal distribution - "
            f"LONG_SPREAD: {signal_counts['LONG_SPREAD']}, "
            f"SHORT_SPREAD: {signal_counts['SHORT_SPREAD']}, "
            f"CLOSE: {signal_counts['CLOSE']}, "
            f"HOLD: {signal_counts['HOLD']}"
        )
        logger.info(
            f"Entry decisions - "
            f"predictor_only: {entry_counts['predictor_only']}, "
            f"timing_aligned: {entry_counts['timing_aligned']}, "
            f"filtered: {entry_counts['filtered']}"
        )

    def _should_enter(self, strategic_signal: str, prediction: Dict) -> tuple:
        """
        Combine predictor + signal for entry decision

        Returns:
            (should_enter: bool, entry_mode: str)
        """
        # No entry signal configured → trust predictor only
        if not self.entry_signal:
            return True, 'predictor_only'

        # Signal configured → check timing alignment
        tactical_signal = self.entry_signal.generate_signal()

        # LONG_SPREAD needs BULLISH momentum
        if strategic_signal == 'LONG_SPREAD' and tactical_signal == MarketSignal.BULLISH:
            return True, 'timing_aligned'

        # SHORT_SPREAD needs BEARISH momentum
        if strategic_signal == 'SHORT_SPREAD' and tactical_signal == MarketSignal.BEARISH:
            return True, 'timing_aligned'

        # No alignment → filter (don't enter)
        return False, 'filtered'

    def _should_exit(self, strategic_signal: str, prediction: Dict) -> tuple:
        """
        Combine predictor + signal for exit decision

        Returns:
            (should_exit: bool, exit_reason: str)
        """
        # Predictor says close → close
        if strategic_signal == 'CLOSE':
            return True, 'signal'

        # Stop loss check
        if self._check_spread_stop_loss(prediction):
            return True, 'stop_loss'

        # Exit signal override (if configured)
        if self.exit_signal:
            exit_value = self.exit_signal.generate_signal()

            # Exit long spread if bearish momentum
            if self.current_spread_position['signal'] == 'LONG_SPREAD':
                if exit_value == MarketSignal.BEARISH:
                    return True, 'exit_signal'

            # Exit short spread if bullish momentum
            elif self.current_spread_position['signal'] == 'SHORT_SPREAD':
                if exit_value == MarketSignal.BULLISH:
                    return True, 'exit_signal'

        return False, ''

    def _open_spread_position(self, prediction: Dict, signal: str) -> Optional[Dict]:
        """Open spread position (3 legs atomically)"""
        timestamp = self.data_manager.temporal_timestamp
        markets = prediction['markets']
        position_size = self.money_manager.get_lot_size()

        logger.info(f"OPENING SPREAD: {signal} at {timestamp}, size={position_size}")

        # Determine directions
        if signal == 'LONG_SPREAD':
            directions = [PositionDirection.LONG, PositionDirection.SHORT, PositionDirection.SHORT]
        else:  # SHORT_SPREAD
            directions = [PositionDirection.SHORT, PositionDirection.LONG, PositionDirection.LONG]

        # Define legs
        legs = [
            {'symbol': markets['target'], 'price': prediction['target_price'], 'direction': directions[0]},
            {'symbol': markets['leg1'], 'price': prediction['leg1_price'], 'direction': directions[1]},
            {'symbol': markets['leg2'], 'price': prediction['leg2_price'], 'direction': directions[2]},
        ]

        total_capital = sum(position_size * leg['price'] for leg in legs)
        capital_per_leg = total_capital / 3

        # Open all legs atomically
        trade_ids = []
        for leg in legs:
            trade_id = f"{leg['symbol']}_{timestamp}"
            if self._open_leg(trade_id, leg['symbol'], leg['direction'],
                              position_size, leg['price'], capital_per_leg):
                trade_ids.append(trade_id)
            else:
                # Rollback
                for tid in trade_ids:
                    self.position_orchestrator.close_position(tid, 0.0, 'rollback')
                logger.error("Failed to open spread position atomically")
                return None

        # Success
        return {
            'signal': signal,
            'trade_ids': trade_ids,
            'entry_spread_pips': prediction['spread_pips'],
            'entry_z_score': prediction['z_score'],
            'timestamp': timestamp,
            'size': position_size
        }

    def _close_spread_position(self, prediction: Dict, reason: str = 'signal'):
        """Close spread position (all 3 legs)"""
        if not self.current_spread_position:
            return

        markets = prediction['markets']

        # Calculate SPREAD P&L (like experiment)
        entry_spread_pips = self.current_spread_position['entry_spread_pips']
        exit_spread_pips = prediction['spread_pips']
        position_size = self.current_spread_position['size']

        if self.current_spread_position['signal'] == 'LONG_SPREAD':
            pnl_pips = exit_spread_pips - entry_spread_pips
        else:  # SHORT_SPREAD
            pnl_pips = entry_spread_pips - exit_spread_pips

        pnl_gross = pnl_pips * position_size * self.pip_value_per_lot

        # Get current prices
        target_price = prediction['target_price']
        leg1_price = prediction['leg1_price']
        leg2_price = prediction['leg2_price']

        # Close all 3 legs (for position tracking)
        for trade_id in self.current_spread_position['trade_ids']:
            if markets['target'] in trade_id:
                exit_price = target_price
            elif markets['leg1'] in trade_id:
                exit_price = leg1_price
            else:
                exit_price = leg2_price

            self.position_orchestrator.close_position(trade_id, exit_price, reason)

        # Track spread trade result
        spread_trade = {
            'timestamp': self.data_manager.temporal_timestamp.isoformat() + 'Z',
            'entry_price': entry_spread_pips,
            'exit_price': exit_spread_pips,
            'quantity': position_size,
            'direction': self.current_spread_position['signal'],
            'leg_trades': self.current_spread_position['trade_ids'],
            'gross_pnl': pnl_gross,
            'status': 'closed',
            'entry_date': self.current_spread_position['timestamp'].isoformat() + 'Z',
            'exit_date': self.data_manager.temporal_timestamp.isoformat() + 'Z',
            'entry_z': self.current_spread_position['entry_z_score'],
            'exit_z': prediction['z_score'],
            'pnl_pips': pnl_pips,
            'exit_reason': reason
        }
        self.spread_trade_history.add_trade(spread_trade)

        logger.info(
            f"Closed {self.current_spread_position['signal']}: "
            f"Z {self.current_spread_position['entry_z_score']:.2f}→{prediction['z_score']:.2f}, "
            f"Spread {entry_spread_pips:.2f}→{exit_spread_pips:.2f}, "
            f"P&L ${pnl_gross:.2f} ({reason})"
        )

    def _check_spread_stop_loss(self, prediction: Dict) -> bool:
        """Check if spread has moved too far against us"""
        if not self.current_spread_position:
            return False

        stop_loss_threshold = self.money_manager.risk_manager.stop_loss_z_score
        current_z = prediction['z_score']

        # For LONG_SPREAD: Stop if Z moves too positive (spread widening)
        if self.current_spread_position['signal'] == 'LONG_SPREAD':
            return current_z > stop_loss_threshold

        # For SHORT_SPREAD: Stop if Z moves too negative (spread widening other way)
        return current_z < -stop_loss_threshold

    def _calculate_final_metrics(self) -> Dict:
        """Calculate strategy performance metrics"""
        metrics_calc = MetricsCalculator(self.config)
        initial_capital = self.position_orchestrator.position_manager.total_capital

        performance_metrics = metrics_calc.calculate_metrics(
            trade_history=self.position_orchestrator.trade_history,
            equity_curve=None,
            initial_capital=initial_capital
        )

        results = performance_metrics.to_dict()
        results['strategy_name'] = self.name

        # Leg-level details (existing)
        results['trade_details'] = [
            {
                'trade_id': t.get('uuid'),
                'symbol': t.get('symbol'),
                'direction': t.get('direction').name if hasattr(t.get('direction'), 'name') else t.get('direction'),
                'entry_price': t.get('entry_price'),
                'exit_price': t.get('exit_price'),
                'pnl': t.get('net_pnl') or t.get('gross_pnl'),
                'exit_reason': t.get('exit_reason', 'UNKNOWN')
            }
            for t in self.position_orchestrator.trade_history.trades.values()
            if t.get('status') == 'closed'
        ]

        # Spread-level metrics
        if self.spread_trade_history.get_trade_count() > 0:
            spread_stats = self.spread_trade_history.get_trade_statistics(
                lookback_periods=0)
            total_trades = spread_stats.total_positions
            win_rate = (spread_stats.winning_positions /
                        total_trades) if total_trades > 0 else 0

            results['spread_metrics'] = {
                'total_trades': total_trades,
                'total_pnl': spread_stats.total_pnl,
                'win_rate': win_rate,
                'winners': spread_stats.winning_positions,
                'losers': spread_stats.losing_positions
            }
            results['spread_trades'] = list(self.spread_trade_history.trades.values())

            # Override legacy metrics with correct spread P&L
            results['total_pnl'] = spread_stats.total_pnl
            results['win_rate'] = win_rate
            results['total_trades'] = total_trades

        return results

    def _open_leg(self, trade_id: str, symbol: str, direction: PositionDirection,
                  quantity: float, price: float, capital: float) -> bool:
        """Open single leg of spread position"""
        success = self.position_orchestrator.open_position(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=price,
            capital_required=capital
        )
        if success:
            logger.debug(f"  LEG OK: {trade_id}")
        else:
            logger.warning(f"  LEG FAILED: {trade_id}")
        return success