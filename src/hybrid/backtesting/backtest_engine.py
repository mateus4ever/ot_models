from src.hybrid.backtesting.backtest_result import BacktestResult
from src.hybrid.data.trade_history import TradeHistory
from src.hybrid.money_management import MoneyManager
from src.hybrid.strategies import StrategyFactory


class BacktestEngine:
    def __init__(self, config, data_manager, position_manager):
        self.config = config
        self.data_manager = data_manager  # Shared
        self.position_manager = position_manager  # Shared

    def run(self) -> BacktestResult:
        # 1. Create thread-local instances
        money_manager = MoneyManager(self.config)
        money_manager.set_position_manager(self.position_manager)

        trade_history = TradeHistory(self.config)

        # 2. Create and setup strategy
        strategy = StrategyFactory.create_strategy(self.config)
        strategy.setDataManager(self.data_manager)  # Shared
        strategy.setMoneyManager(money_manager)  # Thread-local
        strategy.setTradeHistory(trade_history)  # Thread-local
        strategy.setPositionManager(self.position_manager)  # Shared

        # 3. Run
        result = strategy.run()

        return result