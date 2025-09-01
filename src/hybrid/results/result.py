# result.py
class Result:
    """Simple result class for strategy execution"""

    def __init__(self, strategy_name: str, data: any):
        self.strategy_name = strategy_name
        self.data = data